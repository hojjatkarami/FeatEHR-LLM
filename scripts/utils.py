import json
import multiprocessing as mp
import os
import pickle
import random
import traceback
import warnings
from functools import partial
from typing import Any, List

import numpy as np
import openai
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample
from tqdm import tqdm

from loguru import logger
import wandb
import yaml

from scripts.tools import *


def get_llm_client(model_name):
    # Initialize the OpenAI client with the specified model

    if model_name.startswith("gpt-"):
        # For OpenAI models
        return openai.Client(
            base_url=os.getenv("OPENAI_URL"), api_key=os.getenv("OPENAI_API_KEY")
        )
    elif "/" in model_name:  # from openrouter
        return openai.Client(
            base_url=os.getenv("BASE_URL"), api_key=os.getenv("LLM_API_KEY")
        )
    elif model_name.startswith("gemini-"):
        # For Gemini models
        return openai.Client(
            base_url=os.getenv("GOOGLE_URL"), api_key=os.getenv("GOOGLE_API_KEY")
        )


def get_tools_string(list=None):
    import inspect
    import scripts.tools as tools  # Import all functions from scripts/tools.py
    import textwrap

    # Initialize an empty string to store the result
    output_string = ""

    all_tools = {
        name: obj for name, obj in inspect.getmembers(tools, inspect.isfunction)
    }

    if list is not None:
        all_tools = {item: all_tools[item] for item in list}
    # Loop through all functions in tools module
    for name, obj in all_tools.items():

        # Get the function's docstring
        doc_string = inspect.getdoc(obj)
        # Get the first line of the function's code (signature)
        signature = str(inspect.signature(obj))

        output_string += (
            f"def {name}{signature}:\n"
            f'    """\n'
            f'{textwrap.indent(doc_string.strip(), "    ")}\n'
            f'    """\n\n'
        )

    # Print the result
    return output_string


def read_data_xgboost(
    data_name,
    split,
    version,
    run_llm,
    n,
    # history,
    strategy="",
    id_questions=None,
    ignore_cache=False,
    # feature_type="numeric",
):
    PATH_RULE = "data/featehr-llm/{data}/{split}/{version}/data.csv"
    unique_ids = {}
    df_X = {}
    df_y = None

    # load data yaml file
    with open(f"configs/data/{data_name}.yaml", "r") as f:
        config = yaml.safe_load(f)
    PATH = config["path"]
    LABEL_NAME = config["label"]
    history = config["history"]

    if version in ["numeric", "tsfresh"]:

        # check if the processed raw data exists
        path_numeric = f"./data/featehr-llm/{data_name}/{split}/{version}"
        if os.path.exists(path_numeric) and (not ignore_cache):
            print(f"\t{path_numeric} exists")
            print(f"\tloading from {path_numeric}")

            # load
            with open(f"{path_numeric}/data.pkl", "rb") as f:
                X, y, common_ids = pickle.load(f)
            print(f"\tloaded {path_numeric}")
            # print length
            print(f"\t\tLoaded {len(X)} samples from {split} data. NUMERICAL")

        else:

            path_data = PATH.format(split=split)

            with open(path_data, "rb") as f:
                data_train = pickle.load(f)
            if n is not None:
                # randomly select n samples with seed 42
                random.seed(42)
                random.shuffle(data_train)
                data_train = data_train[:n]
            print(f"\tselected {len(data_train)} samples from {split} data. NUMERICAL")
            common_ids = list(range(len(data_train)))

            # get time series data
            all_dfs = [patient["ts"][0] for patient in data_train]
            df_ts = pd.concat(all_dfs, keys=range(len(all_dfs)))

            df_ts["id"] = df_ts.index.get_level_values(0)
            df_ts = df_ts.reset_index(drop=True)

            # drop ids if all variables (except id and Hours) are nan
            vars_to_check = [col for col in df_ts.columns if col not in ["id", "Hours"]]
            ids_with_all_nan = df_ts.groupby("id")[vars_to_check].apply(
                lambda x: x.isnull().all().all()
            )
            ids_to_drop = ids_with_all_nan[ids_with_all_nan].index.tolist()

            if len(ids_to_drop) > 0:
                print(f"\tDropping {len(ids_to_drop)} ids with all NaN values")
                df_ts = df_ts[~df_ts["id"].isin(ids_to_drop)]
                # update common_ids
                common_ids = [id for id in common_ids if id not in ids_to_drop]

            if history is not None:
                print(f"\tfiltering time series data with history: {history}")
                df_ts = df_ts[df_ts["Hours"] < history]

            if data_name == "mimic3":
                df_ts = process_ts_vars(df_ts)

            # remove column Hours
            df_ts = df_ts.drop(columns=["Hours"])
            df_ts = df_ts.apply(pd.to_numeric, errors="coerce")

            ids_ts = df_ts["id"].unique()

            # update comoon ids
            common_ids = set(ids_ts).intersection(set(common_ids))
            print(f"\tcommon ids after ts processing: {len(common_ids)}")
            # print("missing ids TS:", set(range(n)).difference(set(ids_ts)))
            # get demographic data, age and gender

            all_demo = [patient["covariates"][0] for patient in data_train]
            df_static = pd.DataFrame(all_demo)
            # set column names to Age, gender
            df_static.columns = ["Age", "Gender"]
            # set id
            df_static["id"] = range(len(all_dfs))

            # set label
            all_labels = [patient[LABEL_NAME][0] for patient in data_train]
            df_static[LABEL_NAME] = all_labels

            # set types
            df_static["Gender"] = df_static["Gender"].astype("bool")
            df_static[LABEL_NAME] = df_static[LABEL_NAME].astype("bool")
            static_vars = ["Age", "Gender"]

            ids_static = df_static["id"].unique()
            # print("missing ids Static:", set(range(n)).difference(set(ids_static)))

            # update common ids
            common_ids = set(ids_static).intersection(set(common_ids))
            print(f"\tcommon ids after static processing: {len(common_ids)}")

            df_ts = df_ts[df_ts["id"].isin(common_ids)].reset_index(drop=True)
            df_static = df_static[df_static["id"].isin(common_ids)].reset_index(
                drop=True
            )

            cont_vars = df_ts.columns.tolist()
            cont_vars.remove("id")

            # convert set to list
            common_ids = [int(x) for x in list(common_ids)]

            col_labels = [LABEL_NAME]
            from scripts.utils import genTSembeddings

            # return df_ts, df_static, cont_vars, col_labels, static_vars

            def tsfreshEmbeddings(df_ts, df_static, cont_vars, col_labels):
                from tsfresh import extract_features
                from tsfresh.utilities.dataframe_functions import impute
                from tsfresh.feature_extraction import (
                    EfficientFCParameters,
                    MinimalFCParameters,
                )

                df_ts["time"] = df_ts.groupby("id").cumcount()

                long_df = df_ts.melt(
                    id_vars=["id", "time"], var_name="variable", value_name="value"
                )
                long_df_clean = long_df.dropna(subset=["value"])
                y = df_static[['id']+col_labels].set_index("id").squeeze()
                # Extract features using tsfresh
                X = extract_features(
                    long_df_clean,
                    column_id="id",
                    column_sort="time",
                    column_kind="variable",
                    column_value="value",
                    default_fc_parameters=MinimalFCParameters(),
                    impute_function=impute,  # This will impute basic NaNs automatically
                )
                print(path_numeric, X.shape)

                if os.path.exists(
                    f"{path_numeric.replace('test','train')}/tsfresh_features.pkl"
                ):
                    feature_names = pd.read_pickle(
                        f"{path_numeric.replace('test','train')}/tsfresh_features.pkl"
                    )
                    X_filtered = X[feature_names]
                else:
                    from tsfresh import select_features

                    X_filtered = select_features(X, y)
                    # Save the feature names for future reference
                    feature_names = X_filtered.columns.tolist()
                    os.makedirs(path_numeric, exist_ok=True)
                    with open(f"{path_numeric}/tsfresh_features.pkl", "wb") as f:
                        pickle.dump(feature_names, f)

                return X_filtered, y

            if version == "numeric":
                X, y = genTSembeddings(
                    df_ts, df_static, cont_vars, col_labels, static_vars=static_vars
                )
            elif version == "tsfresh":
                X, y = tsfreshEmbeddings(df_ts, df_static, cont_vars, col_labels)
            # set the index of X and y with common ids
            X.index = common_ids
            y.index = common_ids
            df_X[version] = X  # .reset_index(drop=True)
            df_y = y  # .reset_index(drop=True)

            # add "_bl" to columns
            X.columns = [col + f"{version}_bl" for col in X.columns]

            # save X,y, common_ids as pickle to numerics path

            if not ignore_cache and len(X) > 1500:
                # create folder
                os.makedirs(path_numeric, exist_ok=True)

                with open(f"{path_numeric}/data.pkl", "wb") as f:
                    pickle.dump((X, y, common_ids), f)

                print(f"\t#########HEY####### saved {path_numeric}")

        return X, y, common_ids

    
    elif version in ['resnet']:
        path_cache = f"./data/featehr-llm/{data_name}/{split}/{version}"

        if os.path.exists(path_cache) and (not ignore_cache):
            print(f"\t{path_cache} exists")
            print(f"\tloading from {path_cache}")
            
            # load
            with open(f"{path_cache}/data.pkl", "rb") as f:
                X, y, common_ids = pickle.load(f)
            print(f"\tloaded {path_cache}")
            # print length
            print(f"\t\tLoaded {len(X)} samples from {split} data. RESNET")
            
            return X, None, common_ids
            
        else:


            path_data = PATH.format(split=split)

            with open(path_data, "rb") as f:
                data_train = pickle.load(f)
            if n is not None:
                # randomly select n samples with seed 42
                random.seed(42)
                random.shuffle(data_train)
                data_train = data_train[:n]
            print(f"\tselected {len(data_train)} samples from {split} data. NUMERICAL")
            common_ids = list(range(len(data_train)))

            # get time series data
            all_dfs = [patient["ts"][0] for patient in data_train]
            if data_name == "mimic3":
                # df_ts = process_ts_vars(df_ts)
                all_dfs = [process_ts_vars(df) for df in all_dfs]
            var_names = all_dfs[0].columns.tolist()
            var_names.remove("Hours")
            
            df_ts = pd.concat(all_dfs, keys=range(len(all_dfs)))

            df_ts = pd.concat(all_dfs, keys=range(len(all_dfs)))

            df_ts["id"] = df_ts.index.get_level_values(0)
            df_ts = df_ts.reset_index(drop=True)

            var_names_mean = df_ts[var_names].mean().to_dict()
            # var_names_mean

            def impute_ts(df, var_names_mean):
                """
                df: DataFrame with time series data
                var_names_mean: dictionary with variable names as keys and their mean values as values
                """
                # forwardfill for 6 hours
                df_ff = df.ffill(limit=6)
                # print(df_ff)

                # impute mean for each variable
                for var_name in var_names_mean:
                    if var_name in df_ff.columns:
                        df_ff[var_name] = df_ff[var_name].fillna(var_names_mean[var_name])

                # print(df_ff)

                return df_ff

            all_dfs = [impute_ts(df,var_names_mean) for df in tqdm(all_dfs, desc="Imputing missing values")]  # impute missing values

            labels = [patient[LABEL_NAME][0] for patient in data_train]


            def prepare_data_for_sktime(all_dfs, labels, pad_value=0):
                """
                all_dfs: list of DataFrames (each: ['Hours', 'var_1', ..., 'var_D'])
                labels: list/array of binary labels
                Returns: X_nested (sktime format), y (pd.Series)
                """
                # Drop 'Hours' and pad to max length
                processed = [df.drop(columns=['Hours']).copy() for df in all_dfs]
                var_names = processed[0].columns.tolist()
                T_max = max(len(df) for df in processed)
                def pad_df(df):
                    if len(df) >= T_max:
                        return df.iloc[:T_max]
                    pad = pd.DataFrame(pad_value, index=range(T_max - len(df)), columns=df.columns)
                    return pd.concat([df, pad], ignore_index=True)
                X_padded = [pad_df(df) for df in processed]
                # Convert to nested DataFrame
                X_nested = pd.DataFrame({v: [df[v] for df in X_padded] for v in var_names})
                return X_nested, pd.Series(labels)

            # Example usage:
            # all_dfs = [...]  # your list of DataFrames
            # labels = [...]   # your list/array of binary labels

            X_nested, y = prepare_data_for_sktime(all_dfs, labels)


            # save X_nested, y, common_ids as pickle to numerics path
            if not ignore_cache:
                # create folder
                os.makedirs(path_cache, exist_ok=True)

                with open(f"{path_cache}/data.pkl", "wb") as f:
                    pickle.dump((X_nested, y, common_ids), f)

                print(f"\t################# saved {path_cache}")

            return X_nested, None, common_ids

    elif "llm2" in version:

        # load llm data
        # append new engineered features
        with open(run_llm.format(split=split, version=version), "rb") as f:
            data_answers = pickle.load(f)

        # sort data_anwers based on keys
        data_answers = dict(sorted(data_answers.items()))

        unique_ids["llm"] = list(data_answers.keys())
        print(f"selected {len(data_answers)} samples from {split} data. LLM")

    # elif version.startswith("v"):
    else:
        
        df_rules = pd.read_csv(
            PATH_RULE.format(data=data_name, split=split, version=version)
        )

        if "_bl" not in version:
            PATH_FEATURES = "results/{data}/{version}/features.csv"
            df_features = pd.read_csv(
                PATH_FEATURES.format(data=data_name, split=split, version=version)
            )
        else:
            strategy = "bl"
            print(f"\tThis is a beseline. Strategy: {strategy} => select all features")

        if "mi_min" in strategy:
            # get features with min mi 0.02
            MI_TH = float(strategy.split("mi_min")[-1])

            features = df_features[df_features["mean_mi"] > MI_TH][
                "function_name"
            ].tolist()

            print(
                f"\tStrategy: {strategy} => selected {len(features)}/{len(df_features)} features. MI > {MI_TH}"
            )

        elif strategy == "first":
            df_features = df_features.sort_values(by=["q_id", "fun_id"])
            features = (
                df_features.groupby("q_id")
                .first()
                .reset_index()["function_name"]
                .tolist()
            )

            print(
                f"\tStrategy: {strategy} => selected {len(features)}/{len(df_features)} features"
            )

        elif "best" in strategy:
            N_TOP = int(strategy.split("best")[-1])
            df_features = df_features.sort_values(
                by=["q_id", "mean_mi"], ascending=False
            )
            features = (
                df_features.groupby("q_id")
                .head(N_TOP)
                .reset_index()["function_name"]
                .tolist()
            )

            print(
                f"\tStrategy: {strategy} => selected {len(features)}/{len(df_features)} features"
            )
        elif strategy == "":
            features = df_features["function_name"].tolist()

            print(f"\tStrategy: {strategy} => selected {len(features)} features")
        elif strategy == "bl":
            features = df_rules.columns.tolist()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if len(set(features)) < len(features):
            print(f"\t\t ####################### Repeated Feaure Names")
            features = list(set(features))

        # select columns from df_rules that are in features
        base_features = df_rules.columns.tolist()
        features_available = [f for f in features if f in base_features]
        if len(features_available) < len(features):
            print(
                f"\t##### {len(features) - len(features_available)} features not available in rules"
            )

        df_rules = df_rules[features_available]

        df_rules.columns = [col + f"_{version}" for col in df_rules.columns]

        # replace false with 0
        df_rules = df_rules.replace("False", 0)
        # convert to numeric
        df_rules = df_rules.apply(pd.to_numeric, errors="coerce")

        # report % of cells that are 'ERROR'
        # df_rules = df_rules.replace('ERROR', np.nan)
        # if df_rules.isna().sum().sum() > 0:
        #     print(
        #         f"\t% of cells that are ERROR: {df_rules.isna().sum().sum()/df_rules.size*100:.4f}"
        #     )
        #     df_rules.isna().sum()

        #     # fillna with False
        #     pd.set_option("future.no_silent_downcasting", True)

        #     # remove columns with more that 20% nan values
        #     print(
        #         "\t",
        #         df_rules.shape,
        #         "=>",
        #         df_rules.loc[:, df_rules.isnull().mean() < 0.1].shape,
        #     )
        #     df_rules = df_rules.loc[:, df_rules.isnull().mean() < 0.1]
        #     df_rules = df_rules.fillna(0)
        #     print("\tnans replaced with 0")

        if LABEL_NAME in df_rules.columns:
            df_rules = df_rules.drop(columns=[LABEL_NAME]).reset_index()
        else:
            # df_rules = df_rules.astype(int).reset_index()
            df_rules = df_rules.reset_index()

        dtypes = set(df_rules.dtypes)
        for dtype in dtypes:
            if "int" in str(dtype):
                cols = df_rules.select_dtypes(include=[dtype]).columns
                df_rules[cols] = df_rules[cols].astype(float)
            elif "float" in str(dtype):
                cols = df_rules.select_dtypes(include=[dtype]).columns
                df_rules[cols] = df_rules[cols].astype(float)
            elif "bool" in str(dtype):
                cols = df_rules.select_dtypes(include=[dtype]).columns
                df_rules[cols] = df_rules[cols].astype(int)
            else:
                print(f"\t{dtype} is unknown")
                # column names with this dtype
                print(df_rules.select_dtypes(include=[dtype]).columns)
                # drop these columns
                df_rules = df_rules.drop(
                    columns=df_rules.select_dtypes(include=[dtype]).columns
                )

        df_rules = df_rules.iloc[:n]
        common_ids = df_rules["index"].tolist()

        print(f"\tselected {len(df_rules)} samples from {split} data. RULES")

        # filter only on common ids
        df_rules = df_rules[df_rules["index"].isin(common_ids)]
        # set index
        df_rules.index = df_rules["index"]
        df_rules = df_rules.drop(columns=["index"])

        df_X["rules"] = df_rules

        return df_rules, None, common_ids

    term
    # find common ids
    common_ids = []
    for k, v in unique_ids.items():
        if len(common_ids) == 0:
            common_ids = set(v)
        else:
            common_ids = common_ids.intersection(set(v))

    print(f"\tcommon ids: {len(common_ids)}")

    for f in features:

        # sort indexes
        print(f"\tX {f} shape:", df_X[f].shape)

    print("\ty shape:", df_y.shape)

    return df_X, df_y


def get_paths_tasks(dir="./configs/data"):
    import yaml
    import glob

    # Define the directory containing the YAML config files
    config_path = f"{dir}/*.yaml"

    # Create the dictionary to store name -> path mapping
    PATHS = {}
    TASKS = {}
    # Loop through all YAML files
    for file_path in glob.glob(config_path):
        with open(file_path, "r") as f:
            try:
                data = yaml.safe_load(f)
                if "name" in data and "path" in data:
                    PATHS[data["name"]] = data["path"]
                    TASKS[data["name"]] = data["task"]
            except yaml.YAMLError as e:
                print(f"Error loading {file_path}: {e}")
    return PATHS, TASKS


import warnings

# Filter out the specific warning message
# warnings.filterwarnings("ignore", message="This process.*use of fork\(\) may lead to deadlocks")

import traceback
import io


def run_function_capture(func, *args, **kwargs):
    # Capture warnings
    warning_capture = io.StringIO()
    error_capture = io.StringIO()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Catch all warnings
        try:
            result = func(*args, **kwargs)
            warnings_list = [str(warning.message) for warning in w]
            return {"result": result, "warnings": warnings_list, "error": None}
        except Exception as e:
            # Capture full traceback
            error_capture.write(traceback.format_exc())
            return {
                "result": None,
                "warnings": [str(warning.message) for warning in w],
                "error": error_capture.getvalue(),
            }


def set_logger(folder_name, level="DEBUG"):
    from loguru import logger

    # Remove default Loguru handler
    logger.remove()

    # Define custom format (remove time info)
    custom_format = "<level>{level}</level>: <level>{message}</level>"

    # Add handler for file logs
    logger.add(
        f"{folder_name}/log.txt",
        rotation="1 MB",
        level=level,
        format=custom_format,
        enqueue=True,
    )

    # Add handler for console logs
    logger.add(
        sys.__stdout__, level=level, format=custom_format, enqueue=True
    )  # Use sys.__stdout__ to preserve print redirection

    # Redirect print() to logger
    class StreamToLogger:
        def write(self, message):
            message = message.strip()
            if message:
                logger.info(message)

        def flush(self):
            pass

    # Redirect print output
    sys.stdout = StreamToLogger()

    return


# Function to compute mutual information for each bootstrap sample
def bootstrap_mi(X, y, n_bootstrap=1000, confidence_level=0.95):
    mi_scores_bootstrap = []
    is_categorical = [False if j == np.float64 else True for j in X.dtypes.values]

    # if len(X)>5000, randomly sample 5000
    if len(X) > 5000:
        X = X.sample(n=5000, random_state=42)
        y = y[X.index]
        print(f"Randomly sampled {len(X)} samples for MI computation.")

    # # change y type to bool
    # y = y.astype("bool")

    for _ in range(n_bootstrap):
        # Resample with replacement
        X_resampled, y_resampled = resample(X, y, random_state=None)

        # Compute MI for the resampled data
        mi_scores = mutual_info_classif(
            X_resampled, y_resampled, discrete_features=np.array(is_categorical)
        )

        # Append the MI scores to the list
        mi_scores_bootstrap.append(mi_scores)

    # Convert to a numpy array for easier manipulation
    mi_scores_bootstrap = np.array(mi_scores_bootstrap)

    # Compute lower and upper bounds of the confidence interval
    lower_bound = np.percentile(
        mi_scores_bootstrap, (1 - confidence_level) / 2 * 100, axis=0
    )
    upper_bound = np.percentile(
        mi_scores_bootstrap, (1 + confidence_level) / 2 * 100, axis=0
    )

    return lower_bound, upper_bound, mi_scores_bootstrap.mean(axis=0)


def process_ts_vars(df):

    # get time series data
    cont_vars = [
        "Alanine aminotransferase",
        "Albumin",
        "Alkaline phosphate",
        "Anion gap",
        "Asparate aminotransferase",
        "Basophils",
        "Bicarbonate",
        "Bilirubin",
        "Blood urea nitrogen",
        "Calcium",
        "Chloride",
        "Creatinine",
        "Diastolic blood pressure",
        "Fraction inspired oxygen",
        "Glucose",
        "Heart Rate",
        "Hematocrit",
        "Hemoglobin",
        "Lactate",
        "Lymphocytes",
        "Mean blood pressure",
        "Mean corpuscular volume",
        "Monocytes",
        "Neutrophils",
        "Oxygen saturation",
        "Partial pressure of carbon dioxide",
        "Phosphate",
        "Platelets",
        "Potassium",
        "Prothrombin time",
        "Red blood cell count",
        "Respiratory rate",
        "Systolic blood pressure",
        "Temperature",
        "White blood cell count",
        "pH",
    ]
    cat_vars = [
        "Capillary refill rate",
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale total",
        "Glascow coma scale verbal response",
    ]
    # set cont_vars type to numeric
    df[cont_vars] = df[cont_vars].apply(pd.to_numeric, errors="coerce")

    for v in cat_vars:

        if v == "Capillary refill rate":
            # set to bool
            df[v] = df[v].astype(float)
        elif v == "Glascow coma scale eye opening":

            df[v] = df[v].map(
                {
                    "1 No Response": 1,
                    "2 To pain": 2,
                    "3 To speech": 3,
                    "4 Spontaneously": 4,
                    "To Pain": 2,
                    "To Speech": 3,
                    "Spontaneously": 4,
                }
            )

            df[v] = df[v].astype(float)
        elif v == "Glascow coma scale motor response":
            df[v] = df[v].map(
                {
                    "1 No Response": 1,
                    "No response": 1,
                    "2 Abnorm extensn": 2,
                    "Abnormal extension": 2,
                    "3 Abnorm flexion": 3,
                    "4 Flex-withdraws": 4,
                    "Flex-withdraws": 4,
                    "5 Localizes Pain": 5,
                    "Localizes Pain": 5,
                    "6 Obeys Commands": 6,
                    "Obeys Commands": 6,
                }
            )

            df[v] = df[v].astype(float)

        elif v == "Glascow coma scale total":
            df[v] = df[v].astype(float)
        elif v == "Glascow coma scale verbal response":
            df[v] = df[v].map(
                {
                    "1 No Response": 1,
                    "1.0 ET/Trach": 1,
                    "2 Incomp sounds": 2,
                    "3 Inapprop words": 3,
                    "4 Confused": 4,
                    "5 Oriented": 5,
                    "No Response": 1,
                    "No Response-ETT": 1,
                    "Confused": 4,
                    "Incomprehensible sounds": 2,
                    "Inappropriate Words": 3,
                    "Oriented": 5,
                }
            )

            df[v] = df[v].astype(float)

        # correct gloscow mis spelling
        df.rename(columns={v: v.replace("Glascow", "Glasgow")}, inplace=True)
    return df


def get_good_bad_questions(df_best):

    # out = {}
    # min_th = 0.01
    # for cat in df_best['category'].unique():
    #     out[cat] = {'goods':[],'bads':[]}

    #     df_cat = df_best[df_best['category']==cat].sort_values(by='mean_mi', ascending=False)
    #     # print(df_cat)

    #     out[cat]['goods'] = "\n"
    #     for q in df_cat[df_cat['mean_mi']>min_th]['question'].tolist():
    #         out[cat]['goods'] += f"- {q}\n"

    #     out[cat]['bads'] = "\n"
    #     for q in df_cat[df_cat['mean_mi']<min_th]['question'].tolist():
    #         out[cat]['bads'] += f"- {q}\n"

    out = {}
    for cat in df_best["category"].unique():
        df_cat = df_best[df_best["category"] == cat]

        # max between mean_mi and mean_mi_opt
        df_cat["MI_max"] = df_cat[["mean_mi", "mean_mi_opt"]].max(axis=1)

        df_cat = df_cat.sort_values(by="MI_max", ascending=False)[
            ["question", "MI_max", "mean_mi", "mean_mi_opt"]
        ].round(3)

        df_cat

        text = f"Here is the ranking of previous questions in the {cat} category (Do not generate extremely similar questions):\n\nMutual Information (MI), question\n"

        for i, row in df_cat.iterrows():
            text += f"{row['MI_max']:.3f} - {row['question']}\n"

        out[cat] = text
        print(text)

    return out


def design_feature_rubric(params):

    TASK = params["TASK"]

    n_q = params["n_q"]
    # question_database = params["question_database"]
    units_full = params["units_full"]
    model_openai = params["model_openai"]
    temperature = params["temperature"]
    max_new_tokens = params["max_new_tokens"]

    from typing import List, Literal

    from pydantic import BaseModel

    class Rubric(BaseModel):
        name: str
        variables: List[str]
        logic: str
        points: int
        clinical_relevance: str

    class Answer(BaseModel):
        features: List[Rubric]

    # Generate the JSON schema for the Answer model
    answer_schema = Answer.model_json_schema()

    system_message = "You are an AI assistant with medical expertise."

    variables_description = "\n- ".join([f"'{k}'" for k, v in units_full.items()])

    prompt = params["prompt_template"].format(
        TASK=TASK, variables_description=variables_description, n_q=n_q
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    # print(prompt)
    model = get_llm_client(model_openai)

    print("Thinking ...")
    answer = model.beta.chat.completions.parse(
        model=model_openai,  # Updated model name if necessary
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=1,
        response_format=Answer,
    )
    wandb.log(
        {
            "prompt_tokens": answer.usage.prompt_tokens,
            "completion_tokens": answer.usage.completion_tokens,
            "total_tokens": answer.usage.total_tokens,
        }
    )
    parsed_answer = json.loads(answer.choices[0].message.content)

    # logger.info(f"\tGenerated {N_Q} questions in {CATEGORY} category")
    logger.info(
        "\n".join(
            [
                "\t\t* "
                + q["name"]
                + " | "
                + f" {q['logic']} | "
                + f" {q['points']} | "
                + f" {q['clinical_relevance']} | "
                + " ".join(q["variables"])
                for q in parsed_answer["features"]
            ]
        )
    )
    df_features = pd.DataFrame(parsed_answer["features"])

    df_features["variables_exist"] = df_features["variables"].apply(
        lambda x: [1 if v in units_full.keys() else 0 for v in x]
    )
    print(df_features)

    # remove features with variables not in units_full
    df_features = df_features[
        df_features["variables_exist"].apply(lambda x: sum(x) == len(x))
    ]
    df_features = df_features.drop(columns=["variables_exist"]).reset_index(drop=True)

    # checking if variables exist in the units_full

    return df_features


def design_feature_multi(params):

    TASK = params["TASK"]

    n_q = params["n_q"]  
    # question_database = params["question_database"]
    units_full = params["units_full"]
    model_openai = params["model_openai"]
    temperature = params["temperature"]
    max_new_tokens = params["max_new_tokens"]

    from typing import List, Literal

    from pydantic import BaseModel

    class Question(BaseModel):
        feature: str
        level: str
        clinical_relevance: str
        variables: List[str]
        answer_space: str

    class Answer(BaseModel):
        features: List[Question]

    system_message = "You are an AI assistant with medical expertise."

    variables_description = "\n- ".join([f"'{k}'" for k, v in units_full.items()])

    prompt = params["prompt_template"].format(
        TASK=TASK, variables_description=variables_description, n_q=n_q
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    # print(prompt)
    model = get_llm_client(model_openai)
    print("Thinking ...")
    answer = model.beta.chat.completions.parse(
        model=model_openai,  # Updated model name if necessary
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=1,
        response_format=Answer,
    )
    wandb.log(
        {
            "prompt_tokens": answer.usage.prompt_tokens,
            "completion_tokens": answer.usage.completion_tokens,
            "total_tokens": answer.usage.total_tokens,
        }
    )
    parsed_answer = json.loads(answer.choices[0].message.content)

    # logger.info(f"\tGenerated {N_Q} questions in {CATEGORY} category")
    logger.info(
        "\n".join(
            [
                "\t\t* "
                + q["feature"]
                + " | "
                + f" {q['answer_space']} | "
                + " ".join(q["variables"])
                for q in parsed_answer["features"]
            ]
        )
    )
    df_features = pd.DataFrame(parsed_answer["features"])

    df_features["variables_exist"] = df_features["variables"].apply(
        lambda x: [1 if v in units_full.keys() else 0 for v in x]
    )
    print(df_features)

    # remove features with variables not in units_full
    df_features = df_features[
        df_features["variables_exist"].apply(lambda x: sum(x) == len(x))
    ]
    df_features = df_features.drop(columns=["variables_exist"]).reset_index(drop=True)

    # checking if variables exist in the units_full

    return df_features.iloc[:n_q]


def check_syntax(fun_names, fun_strs):
    valid_functions = {}
    for fun_name, fun_str in zip(fun_names, fun_strs):
        try:  # check the syntax
            exec(fun_str)
            valid_functions[fun_name] = fun_str
        except Exception as e:
            print(f"\tSyntax Error in function {fun_name}: {e}")
            print(fun_str)

    return valid_functions


def process_functions(answer):
    # fun_names = [json.loads(ch.message.content)['function_name'] for ch in answer.choices]

    # fun_strs = [json.loads(ch.message.content)['python_script'] for ch in answer.choices]

    fun_names = [json.loads(a)["function_name"] for a in answer]

    fun_strs = [json.loads(a)["python_script"] for a in answer]

    fun_names_new, fun_strs_new = [], []
    for ii, (fun_name, fun_str) in enumerate(zip(fun_names, fun_strs)):

        # replace fun_name in fun_str with fun_name_i
        fun_strs_new.append(fun_str.replace(fun_name, f"{fun_name}_{ii}"))
        fun_names_new.append(f"{fun_name}_{ii}")

    return fun_names_new, fun_strs_new


def process_functions2(answer):

    fun_strs = [json.loads(a)["python_script"] for a in answer]
    explanations = [json.loads(a)["explnation_of_logic"] for a in answer]
    var_names = [json.loads(a)["variables_to_use"] for a in answer]
    hparams = [json.loads(a)["hyper_params_space"] for a in answer]
    # extract function names from fun_strs
    # def function_name(X: ....
    fun_names = [fun_str.split("def")[1].split("(")[0][1:] for fun_str in fun_strs]

    fun_names_new, fun_strs_new = [], []
    for ii, (fun_name, fun_str) in enumerate(zip(fun_names, fun_strs)):

        # replace fun_name in fun_str with fun_name_i

        fun_str_new = fun_str.replace(fun_name, f"{fun_name}_{ii}")
        # # add explanation to the function as a comment
        # fun_str_new = f"# {explanations[ii]}\n" + fun_str_new
        fun_strs_new.append(fun_str_new)
        fun_names_new.append(f"{fun_name}_{ii}")

    return fun_names_new, fun_strs_new, explanations, var_names, hparams


def process_functions_multi(answer):

    fun_strs = [json.loads(a)["python_script"] for a in answer]
    explanations = [json.loads(a)["explnation_of_logic"] for a in answer]
    var_names = [json.loads(a)["variables_to_use"] for a in answer]
    # hparams = [json.loads(a)["hyper_params_space"] for a in answer]
    # extract function names from fun_strs
    # def function_name(X: ....
    fun_names = [fun_str.split("def")[1].split("(")[0][1:] for fun_str in fun_strs]

    fun_names_new, fun_strs_new = [], []
    for ii, (fun_name, fun_str) in enumerate(zip(fun_names, fun_strs)):

        # replace fun_name in fun_str with fun_name_i

        fun_str_new = fun_str.replace(fun_name, f"{fun_name}_{ii}")
        # # add explanation to the function as a comment
        # fun_str_new = f"# {explanations[ii]}\n" + fun_str_new
        fun_strs_new.append(fun_str_new)
        fun_names_new.append(f"{fun_name}_{ii}")

    return fun_names_new, fun_strs_new, explanations, var_names


def process_functions3(answer):

    # fun_strs = [json.loads(a)['answers']['function_script'] for a in answer]
    fun_strs = [q["function_script"] for a in answer for q in json.loads(a)["answers"]]
    # hparams = [json.loads(a)['answers']['hyper_params_space'] for a in answer]
    hparams = [
        q["hyper_params_space"] for a in answer for q in json.loads(a)["answers"]
    ]

    # extract function names from fun_strs
    # def function_name(X: ....
    fun_names = [fun_str.split("def")[1].split("(")[0][1:] for fun_str in fun_strs]

    fun_names_new, fun_strs_new = [], []
    for ii, (fun_name, fun_str) in enumerate(zip(fun_names, fun_strs)):

        # replace fun_name in fun_str with fun_name_i

        fun_str_new = fun_str.replace(fun_name, f"{fun_name}_{ii}")
        # # add explanation to the function as a comment
        # fun_str_new = f"# {explanations[ii]}\n" + fun_str_new
        fun_strs_new.append(fun_str_new)
        fun_names_new.append(f"{fun_name}_{ii}")

    return fun_names_new, fun_strs_new, hparams


def process_functions_single(answer):

    fun_strs = [q["function_script"] for a in answer for q in json.loads(a)["answers"]]

    fun_names = [fun_str.split("def")[1].split("(")[0][1:] for fun_str in fun_strs]

    # adding _{ii} to the function name to avoid duplicates
    fun_names_new, fun_strs_new = [], []
    for ii, (fun_name, fun_str) in enumerate(zip(fun_names, fun_strs)):

        # replace fun_name in fun_str with fun_name_i

        fun_str_new = fun_str.replace(fun_name, f"{fun_name}_{ii}")
        # # add explanation to the function as a comment
        # fun_str_new = f"# {explanations[ii]}\n" + fun_str_new
        fun_strs_new.append(fun_str_new)
        fun_names_new.append(f"{fun_name}_{ii}")

    return fun_names_new, fun_strs_new


def design_rule_multi(params):

    feature = params["feature"]
    TASK = params["TASK"]
    # category = params["category"]
    # cat_exp = params['cat_exp']
    # n_q = params['n_q']
    # question_database = params['question_database']
    variables = params["variables"]
    # var_units = [params["var_units"][v] for v in variables]
    units_full = params["units_full"]
    model_openai = params["model_openai"]
    temperature = params["temperature"]
    max_new_tokens = params["max_new_tokens"]
    n_return = params["n_return"]
    prompt_memory = params["prompt_memory"]

    from typing import List, Literal

    from pydantic import BaseModel

    # class param_space(BaseModel):
    #     param: str
    #     type: Literal["int", "float"]
    #     begin: float
    #     end: float
    #     default: float

    class AnswerComplex(BaseModel):
        # high_level_question: str
        variables_to_use: List[str]
        explnation_of_logic: str
        python_script: str
        # hyper_params_space: List[param_space]
        # function_name: str
        # variable_consistency: List[str]

    system_message = "You are an AI assistant with medical expertise"

    variables_to_use = ", ".join(
        [f"['{var}'] (unit: {units_full[var]['unit']})" for var in variables]
    )
    # variables_to_use = " - ".join(
    #     [f"'{k}'({v['unit']})" for k, v in units_full.items()]
    # )

    tools_string = params["tools_string"]

    # write to file temp.txt
    with open("temp.txt", "w") as f:
        f.write(tools_string)

    prompt = params["prompt_template"].format(
        TASK=TASK,
        feature=feature,
        variables_to_use=variables_to_use,
        tools_string=tools_string,
        prompt_memory=prompt_memory,
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    model = get_llm_client(model_openai)

    final_answer = []
    print("Thinking ...")
    answer = model.beta.chat.completions.parse(
        model=model_openai,  # Updated model name if necessary
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=n_return,
        response_format=AnswerComplex,
    )

    if wandb.run is not None:
        wandb.log(
            {
                "prompt_tokens": answer.usage.prompt_tokens,
                "completion_tokens": answer.usage.completion_tokens,
                "total_tokens": answer.usage.total_tokens,
            }
        )
    answer_all = [ch.message.content for ch in answer.choices]

    final_answer.extend(answer_all)

    (
        fun_names,
        fun_strs,
        explanations,
        var_names,
    ) = process_functions_multi(final_answer)

    valid_functions = check_syntax(fun_names, fun_strs)

    n_valid_functions = len(valid_functions)
    n_non_valid_functions = len(fun_names) - n_valid_functions

    if n_non_valid_functions > 0:
        print(f"\t\t\t{n_non_valid_functions}/{n_return} invalid functions")

    return valid_functions


def design_rule_single(params):

    # question = params['question']
    TASK = params["TASK"]

    var_name = params["var_name"]
    var_unit = params["var_unit"]

    model_openai = params["model_openai"]
    temperature = params["temperature"]
    max_new_tokens = params["max_new_tokens"]
    n_return = params["n_return"]

    prompt_good_functions = params["prompt_good_functions"]
    prompt_variable = params["prompt_variable"]

    tools_string = params["tools_string"]
    from typing import List, Literal

    from pydantic import BaseModel

    # class FinalAnswer(BaseModel):
    #     function_script: List[str]

    class AnswerComplex(BaseModel):
        # high_level_question: str
        # # # question: str
        # # # why_relevant: str
        # function_name: str
        function_script: str
        # hyper_params_space: List[param_space]
        # variable_consistency: List[str]

    class FinalAnswer(BaseModel):
        answers: List[AnswerComplex]

    system_message = "You are an AI assistant with medical expertise"

    prompt = params["prompt_template"].format(
        TASK=TASK,
        n_return=n_return,
        prompt_variable=prompt_variable,
        tools_string=tools_string,
        var_name=var_name,
        var_unit=var_unit,
        prompt_good_functions=prompt_good_functions,
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    # print(messages)

    model = get_llm_client(model_openai)

    final_answer = []
    print("Thinking ...")
    answer = model.beta.chat.completions.parse(
        model=model_openai,  # Updated model name if necessary
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=1,
        response_format=FinalAnswer,
    )

    if wandb.run is not None:
        wandb.log(
            {
                "prompt_tokens": answer.usage.prompt_tokens,
                "completion_tokens": answer.usage.completion_tokens,
                "total_tokens": answer.usage.total_tokens,
            }
        )
    answer_all = [ch.message.content for ch in answer.choices]

    final_answer.extend(answer_all)

    fun_names, fun_strs = process_functions_single(final_answer)
    valid_functions = check_syntax(fun_names, fun_strs)

    n_valid_functions = len(valid_functions)
    n_non_valid_functions = len(fun_names) - n_valid_functions

    if n_non_valid_functions > 0:
        print(f"\t\t\t{n_non_valid_functions}/{n_return} invalid functions")

    return valid_functions


def design_rule777_2(params, bad_explanations=[]):

    # question = params['question']
    TASK = params["TASK"]
    category = params["category"]
    # cat_exp = params['cat_exp']
    # n_q = params['n_q']

    # question_database = params['question_database']
    var_units = params["var_units"]

    var_name = params["var_name"]
    var_unit = params["var_unit"]

    model_openai = params["model_openai"]
    temperature = params["temperature"]
    max_new_tokens = params["max_new_tokens"]
    n_return = params["n_return"]

    prompt_good_functions = params["prompt_good_functions"]

    from typing import List, Literal

    from pydantic import BaseModel

    # main_question = question

    class AnswerComplex(BaseModel):

        function_script: str

    class FinalAnswer(BaseModel):
        answers: List[AnswerComplex]

    system_message = "You are an AI assistant with medical expertise"

    prompt_variable = params["prompt_variable"]

    prompt = f'''Your task is to {TASK}.

You are given access to the patient's *{var_name}* time series data.

Design *{n_return}* **clinically relevant** function(s) for the variable below:

**Variable:** {prompt_variable}

### Requirements:
- The functions must **go beyond basic statistics** (e.g., avoid mean, min, max, median, std).
- Each function should reflect **clinical reasoning** and provide insights relevant to the task.
- For **each function**, provide the following:

    1. **Clinical Rationale**: Explain why this question/function is clinically relevant.
    2. **Python Code**:
        - Return a **numerical** value (float, int, or boolean). Never return `NaN` or `None`; return 0 when needed.
        - Handle **missing values** appropriately.
        - Use backslashes (`\\`) to break long lines.

---

### Helper Function (already provided — do NOT redefine):


def get_all_measurements(df: pd.DataFrame, col_name: str) -> List[dict]:
    """
    Extracts measurements from a specific column in the dataframe.
    Returns a list of dictionaries containing 'time' and 'value'.
    example output: [{{'time': 0.0, 'value': 0.0}}, {{'time': 1.0, 'value': 1.0}}, ...]
    if no measurements the output is []
    """


---

### Function Template (you must follow this structure):

```python
def function_name(X: pd.DataFrame) -> int:
    \"\"\"
    Description: Clinical feature for *{var_name}* ({var_unit})

    Args:
        - X: DataFrame with time series data.

    Explanation: Describe the clinical logic here.



    Returns: A numerical value (float, int, or boolean)
    \"\"\"



    # 1. Extract time series data
    measurements = get_all_measurements(X, variable)

    # 2. Implement logic
    ...

    return ...
```

- The dataframe may contain **NaNs** — handle them carefully.
- Generate **exactly {n_return} functions**.
- Keep each function **clinically meaningful**, and avoid trivial logic.

---

{prompt_good_functions}
'''

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    # print(messages)

    model = get_llm_client(model_openai)

    final_answer = []
    if "gpt" in model_openai:
        print("R1")
        answer = model.beta.chat.completions.parse(
            model=model_openai,  # Updated model name if necessary
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=1,
            response_format=FinalAnswer,
        )
        answer_all = [ch.message.content for ch in answer.choices]

        final_answer.extend(answer_all)
        messages.append(
            {"role": "assistant", "content": answer.choices[0].message.content}
        )
        messages.append({"role": "user", "content": "Design a more complex logic"})

    return final_answer


def design_rule5(params):

    question = params["question"]
    TASK = params["TASK"]
    category = params["category"]
    # cat_exp = params['cat_exp']
    # n_q = params['n_q']
    # question_database = params['question_database']
    var_units = params["var_units"]

    model_openai = params["model_openai"]
    temperature = params["temperature"]
    max_new_tokens = params["max_new_tokens"]
    n_return = params["n_return"]

    from typing import List, Literal

    from pydantic import BaseModel

    main_question = question

    class param_space(BaseModel):
        param: str
        type: Literal["int", "float"]
        begin: float
        end: float
        default: float

    class AnswerSimple(BaseModel):
        # question: str
        variables_to_use: List[str]
        explnation_of_logic: str
        python_script: str
        hyper_params_space: List[param_space]

    class AnswerComplex(BaseModel):
        # high_level_question: str
        variables_to_use: List[str]
        explnation_of_logic: str
        python_script: str
        hyper_params_space: List[param_space]
        # function_name: str
        # variable_consistency: List[str]

    system_message = "You are an AI assistant with medical expertise"

    # if len(bad_explanations) >0:
    #     bad_exp_prompt = f"""\n*** Here is a list of bad explanations that were not predictive in this task (Learn from these bad explanations):"""
    #     for ii, exp in enumerate(bad_explanations):
    #         bad_exp_prompt += f"\nBAD EXP {ii}: {exp}"

    #     print(f'Using {len(bad_explanations)} bad explanations')
    #     print(bad_exp_prompt)
    # else:
    #     bad_exp_prompt = ""

    prompt = f"""Your task is to {TASK}.
    
    You have access to the patient's time series data. Here are the columns of the dataframe with their units:
    {var_units}

    * 'Hours' is the measurement time. 
    * Please note that there are NaN elements in the dataframe.
    """

    prompt = (
        prompt
        + f'''You are supposed to create some rules to answer this yes/no questions:
    
    {main_question}

    * Your task is to create a rule-based system based on the available variables to answer the above question.
    ** Specify the variables that you will use in the rule.
    ** Explain the logic behind the rule in natural clinical language.
    ** Write a python function to implement the rules. 
    *** Output of the function should be a boolean value.
    *** Make sure that the variable names in the rules are consistent with the dataframe columns.
    *** define the hyperparameters of the function (e.g. thresholds that are used).
    *** define a hyperparameter space for the hyperparameters (list of possible values).
    *** use backlash when splitting the code into multiple lines.
    *** IMPORTANT: please consider the missing values in the data.
    
    *** sample output:
    def function_name(X: pd.DataFrame, **hparams) -> bool:
        """
        Variables to use: var1, var2, var3, ...

        Explanation of logic: ...

        Hyperparameters: ...

        Return: True/False
        """ 
        
        # 1- Reading hyperparameters [use hparams.get('param_name', default) to read the hyperparameters]   
        ...

        # 2- Reading variables from the dataframe X. [use get_all_measurements function]
        ...



        # 3- implement the rules
        ...



        return True/False

    *** You have access to the following functions. Do not re-write the funtion in the final code.

    
    def get_all_measurements(df: pd.DataFrame, col_name: str) -> List[dict]:
    measurements = df[['Hours', col_name]].dropna()
    output = [{{'time': float(row['Hours']), 'value': float(row[col_name])}} for _, row in measurements.iterrows()]

    if len(output) == 0:
        return None
    return output


    
'''
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    model = get_llm_client(model_openai)

    if "threshold" in category:
        n_return = 1
        answer_format = AnswerSimple
    elif "trend" in category:
        n_return = 1
        answer_format = AnswerSimple
    else:
        # n_return=n_return
        answer_format = AnswerComplex

    if "gpt" in model_openai:
        answer = model.beta.chat.completions.parse(
            model=model_openai,  # Updated model name if necessary
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n_return,
            response_format=answer_format,
        )
    elif "o3" in model_openai:
        answer = model.beta.chat.completions.parse(
            model=model_openai,  # Updated model name if necessary
            messages=messages,
            # temperature=temperature,
            # max_tokens=max_new_tokens,
            # n=1,
            response_format=answer_format,
        )
    answer = [ch.message.content for ch in answer.choices]

    # messages.append({'role': 'assistant', 'content': answer[0]})

    return answer, messages


def prepro(df, df_mean, df_std, state_vars):

    df = df.copy()
    # # missingness rate old
    # df_missing = df[['id']+state_vars].groupby('id').apply(lambda x:x.isnull().sum()/x.shape[0])[state_vars].reset_index()

    # # # # missingness rate (normalized by max time)
    # # # df_missing = df[['id']+['Time']+state_vars].groupby('id').apply(lambda x:x[state_vars].isnull().sum()/max(x.Time))[state_vars].reset_index()

    # # # # change column names to "mr+{var}"
    # # # df_missing.columns = ['id']+['mr_'+var for var in state_vars]

    # mean imputation for missing values
    df[state_vars] = df[state_vars].fillna(df[state_vars].mean())

    # # standardize
    df[state_vars] = (df[state_vars] - df_mean) / df_std
    print("###########  normalizing")
    # group by id and compute statistics (min.max,mean,std) for each variable

    df2 = df.groupby("id")[state_vars].agg(["min", "max", "mean", "std"]).reset_index()
    # df2 = df.groupby('id')[state_vars].agg(['max']).reset_index()

    # combine the names first and second level column indices

    df2.columns = ["_".join(col).strip() for col in df2.columns.values]

    df2.rename(columns={"id_": "id"}, inplace=True)

    # df2 = df2.merge(df[['id','Label']].drop_duplicates(),on=['id'],how='inner')

    # # # df2 = df2.merge(df_missing,on=['id'],how='inner')

    return df2


def genTSembeddings(df_ts, df_static, cont_vars, col_labels, static_vars=[]):

    df_mean, df_std = df_ts[cont_vars].mean(), df_ts[cont_vars].std()

    df_pro_ts = prepro(df_ts, df_mean, df_std, cont_vars)

    X = df_pro_ts

    if len(static_vars) > 0:

        if "Age" in static_vars:
            df_static["Age"] = df_static["Age"].fillna(df_static["Age"].mean())

            # normalize
            df_static["Age"] = (df_static["Age"] - df_static["Age"].mean()) / df_static[
                "Age"
            ].std()

        X = X.merge(df_static.drop(col_labels, axis=1), on="id", how="inner")

    y = df_static[["id"] + col_labels]

    assert X.shape[0] == y.shape[0]
    assert X["id"].equals(y["id"])

    return X.drop("id", axis=1), y.drop("id", axis=1).squeeze()


def compute_auroc_auprc_f1(y_test, probs):
    from sklearn.metrics import (
        accuracy_score,
        auc,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
        average_precision_score,
    )

    # flatten inputs
    # y_test = y_test.values.flatten()
    probs = probs.flatten()
    preds = (probs > 0.5).astype(int)
    # print(y_test.sum(),preds.sum(),probs)
    auroc = roc_auc_score(y_test, probs)

    # precision, recall, _ = precision_recall_curve(y_test, probs)
    # auprc = auc(recall, precision)
    auprc = average_precision_score(y_test, probs)

    f1 = f1_score(y_test, preds)

    return {"AUROC": auroc, "AUPRC": auprc, "F1": f1}


def run_tabpfn(X_train, X_test, y_train, y_test, seed=42):

    from tabpfn import TabPFNClassifier

    if len(X_train) == 2:
        print(" Training ensemble model")
        # first item is the numerical features, second item is the llm features

        X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
            X_train[0], y_train, test_size=0.2, random_state=42
        )
        # X_train_val, y_train_val = X_test[0], y_test

        model1 = run_tabpfn([X_train_train], [X_train_val], y_train_train, y_train_val)

        X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
            X_train[1], y_train, test_size=0.2, random_state=42
        )
        # X_train_val, y_train_val = X_test[1], y_test

        model2 = run_tabpfn([X_train_train], [X_train_val], y_train_train, y_train_val)

        probs1 = model1["probs"]
        probs2 = model2["probs"]

        best_alpha = optimize_alpha(probs1, probs2, y_train_val)["best_alpha"]

        probs_test_m1 = model1["model"].predict_proba(X_test[0])[:, 1]
        probs_test_m2 = model2["model"].predict_proba(X_test[1])[:, 1]

        print("Best alpha:", best_alpha)
        probs_test = best_alpha * probs_test_m1 + (1 - best_alpha) * probs_test_m2
        preds = (probs_test > 0.5).astype(int)
        output = compute_auroc_auprc_f1(y_test, probs_test)

        output.update(
            {
                "model": [model1, model2],
                "probs": probs_test,
                "preds": preds,
                "y_test": y_test.values.astype(int),
            }
        )

    else:
        model1 = TabPFNClassifier()
        model1.fit(X_train, y_train.values.flatten())

        probs_test = model1.predict_proba(X_test[0])[:, 1]
        preds = model1.predict(X_test[0])

        # compute auroc, auprc, f1

        output = compute_auroc_auprc_f1(y_test, probs_test)

        output.update(
            {
                "model": model1,
                "probs": probs_test,
                "preds": preds,
                "y_test": y_test.values.astype(int),
            }
        )

    return output

def run_mlp(
    X_train,
    X_test,
    y_train,
    y_test,
    best_params=None,
    seed=42,

):

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()

    

    # Scale features for MLP (neural networks are sensitive to feature scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if best_params is None:
        model1 = MLPClassifier(
            hidden_layer_sizes=(100,),
            random_state=seed,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1
        )
    else:
        model1 = best_params

    if False:  # hyperparameter tuning
        import optuna
        from sklearn.model_selection import train_test_split

        # Split dataset
        X_train0, X_val0, y_train0, y_val0 = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42
        )

        def objective(trial):
            params = {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes", 
                    [(50,), (100,), (200,), (100, 50), (200, 100), (100, 100), (200, 100, 50)]
                ),
                "activation": trial.suggest_categorical(
                    "activation", ["relu", "tanh", "logistic"]
                ),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", ["constant", "invscaling", "adaptive"]
                ),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-4, 1e-1, log=True
                ),
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "random_state": seed,
            }

            model = MLPClassifier(**params)
            model.fit(X_train0, y_train0)
            probs_val = model.predict_proba(X_val0)[:, 1]
            
            output = compute_auroc_auprc_f1(y_val0, probs_val)
            score = output["AUPRC"]
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        print(study.best_params)
        best_params = study.best_params

    else:
        best_params = {}

    model = MLPClassifier(
        **best_params, 
        random_state=seed, 
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        activation='relu'
    )
    model.fit(X_train_scaled, y_train)

    # Store the scaler with the model for later use
    model.scaler = scaler

    return model

    
def run_lightgbm_old(
    X_train,
    X_test,
    y_train,
    y_test,
    best_params=None,
    seed=42,
):

    from lightgbm import LGBMClassifier

    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()

    # # check for nans and return error
    # assert not X_train.isnull().values.any(), "X_train contains NaN values"
    # assert not X_test.isnull().values.any(), "X_test contains NaN values"
    # assert not y_train.isnull().values.any(), "y_train contains NaN values"
    # assert not y_test.isnull().values.any(), "y_test contains NaN values"
    if best_params is None:

        model1 = LGBMClassifier(verbose=-1, random_state=seed, max_depth=2)
    else:
        model1 = best_params

    # # drop columns with similar names
    # X_train[0] = X_train[0].loc[:,~X_train[0].columns.duplicated()]
    # X_test[0] = X_test[0].loc[:,~X_test[0].columns.duplicated()]

    if False:  # hyperparameter tuning
        import lightgbm
        import optuna
        from sklearn.model_selection import train_test_split

        # Split dataset
        # X = X_train
        # y = y_train.values.flatten()
        X_train0, X_val0, y_train0, y_val0 = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Convert to LightGBM dataset
        dtrain = lightgbm.Dataset(X_train0, label=y_train0)
        dval = lightgbm.Dataset(X_val0, label=y_val0, reference=dtrain)

        def objective(trial):
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.2, log=True
                ),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            }

            model = lightgbm.train(
                params, dtrain, valid_sets=[dval], num_boost_round=100
            )
            probs_val = model.predict(X_val0)
            # print(preds,'################')
            # term
            # probs_val = model.predict(X_val)[:,1]
            output = compute_auroc_auprc_f1(y_val0, probs_val)
            score = output["AUPRC"]
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        print(study.best_params)
        best_params = study.best_params

    else:
        best_params = {}


    model = LGBMClassifier(
        **best_params, verbose=-1, random_state=seed, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    return model




def run_lightgbm(
    X_train,
    X_test,
    y_train,
    y_test,
    best_params=None,
    seed=42,
):

    from lightgbm import LGBMClassifier

    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()





    model = LGBMClassifier(
       verbose=-1, random_state=seed, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    return model

def run_catboost(
    X_train,
    X_test,
    y_train,
    y_test,
    best_params=None,
    seed=42,
):
    from catboost import CatBoostClassifier

    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()

    if best_params is None:
        model = CatBoostClassifier(
            verbose=0,
            random_seed=seed,
            auto_class_weights="Balanced"
        )
    else:
        model = best_params

    model.fit(X_train, y_train)

    return model
def run_resnet(
    X_train,
    X_test,
    y_train,
    y_test,
    best_params=None,
    seed=42,
):
    from sktime.classification.deep_learning import ResNetClassifier

    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()

    if best_params is None:
        model = ResNetClassifier(n_epochs=20, batch_size=16, verbose=True, random_state=seed)
    else:
        model = best_params

    model.fit(X_train, y_train)

    return model


def run_xgboost(
    X_train,
    X_test,
    y_train,
    y_test,
    best_params=None,
    seed=42,

):

    from xgboost import XGBClassifier

    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()

    if best_params is None:

        model1 = XGBClassifier(
            random_state=seed, max_depth=2, verbosity=0, eval_metric="logloss"
        )
    else:
        model1 = best_params

    if False:  # hyperparameter tuning
        import xgboost as xgb
        import optuna
        from sklearn.model_selection import train_test_split

        # Split dataset
        X_train0, X_val0, y_train0, y_val0 = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Convert to XGBoost DMatrix
        dtrain = xgb.DMatrix(X_train0, label=y_train0)
        dval = xgb.DMatrix(X_val0, label=y_val0)

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "booster": "gbtree",
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.2, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            }

            model = xgb.train(
                params,
                dtrain,
                evals=[(dval, "validation")],
                num_boost_round=100,
                verbose_eval=False,
            )
            probs_val = model.predict(dval)

            output = compute_auroc_auprc_f1(y_val0, probs_val)
            score = output["AUPRC"]
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        print(study.best_params)
        best_params = study.best_params

    else:
        best_params = {}

    # if feature_selction == "perm":
    #     from s3_compute_pi import run_perm_imp4

    #     baseline_features = [f for f in X_train.columns if f.endswith("_bl")]
    #     llm_features = [f for f in X_train.columns if not f.endswith("_bl")]

    #     if len(llm_features) > 0:
    #         dff = run_perm_imp4(X_train, y_train)
    #         # print(dff)
    #         # dff
    #         # select features with positive importance
    #         selected_features = dff[dff["Importance"] > 0.00]["Feature"].tolist()

    #         # remove baseline features
    #         selected_features = [f for f in selected_features if not f.endswith("_bl")]
    #         if len(baseline_features) > 0:
    #             selected_features = baseline_features + selected_features
    #         if len(selected_features) == 0:
    #             print("No features selected")
    #         else:
    #             print(
    #                 f"\nUsing selected features: {len(selected_features)} of {X_train.shape[1]}"
    #             )
    #             X_train, X_test = X_train[selected_features], X_test[selected_features]

    model = XGBClassifier(
        **best_params, random_state=seed, verbosity=0, eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    return model


def run_ensemble(X_train, X_test, y_train, y_test):

    from sklearn.model_selection import train_test_split

    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    model1 = run_lightgbm(X_train_train, X_train_val, y_train_train, y_train_val)

    model2 = run_tabpfn(X_train_train, X_train_val, y_train_train, y_train_val)

    probs1 = model1["probs"]
    probs2 = model2["probs"]

    best_alpha = optimize_alpha(probs1, probs2, y_train_val)["best_alpha"]

    probs_test_m1 = model1["model"].predict_proba(X_test)[:, 1]
    probs_test_m2 = model2["model"].predict_proba(X_test)[:, 1]

    probs_test = best_alpha * probs_test_m1 + (1 - best_alpha) * probs_test_m2

    output = compute_auroc_auprc_f1(y_test, probs_test)

    return output


def tune_lightgbm(X_train, y_train):
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import RandomizedSearchCV

    param_dist = {
        "num_leaves": np.arange(10, 200, 10),
        "max_depth": np.arange(3, 15, 2),
        # 'learning_rate': np.logspace(-3, 0, 10),
        # 'n_estimators': np.arange(50, 1000, 10),
        # 'reg_lambda': np.logspace(-3, 2, 10),
    }

    clf = LGBMClassifier(verbose=-1)
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=30,  # Number of random samples
        scoring="roc_auc",
        cv=5,  # 5-fold cross-validation
        verbose=1,
        n_jobs=-4,
        random_state=42,
    )

    random_search.fit(X_train, y_train.values.flatten())
    print("Best parameters found:", random_search.best_params_)

    return random_search.best_estimator_


def feature_ablation(X_base, X_f, y, n_bootstraps=5):

    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    # Generate synthetic dataset with 100 features
    # X, y = make_classification(n_samples=1000, n_features=101, random_state=42)
    # Convert to DataFrame for easier feature selection
    # check X_f is panda series
    assert isinstance(X_f, pd.DataFrame), "X_f should be a pandas DataFrame"
    X = pd.concat([X_base, X_f], axis=1)

    # X = pd.DataFrame(X, columns=feature_names)

    # Select base features (first 100) and the new feature (Feature_100)
    base_features = X_base.columns.tolist()  # All columns except the last one
    new_feature = X_f.columns.tolist()  # The last column
    feature_names = base_features + [new_feature]

    # Bootstrapping to compute confidence intervals
    # n_bootstraps = 100
    epsilon_values = []

    np.random.seed(42)  # Ensure reproducibility

    for _ in range(n_bootstraps):

        # Split data into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + n_bootstraps
        )

        # Train baseline model (without new feature)
        model_base = lgb.LGBMClassifier(random_state=42)
        model_base.fit(X_train[base_features], y_train)

        # Train extended model (with new feature)
        extended_features = base_features + new_feature
        model_extended = lgb.LGBMClassifier(random_state=42)
        model_extended.fit(X_train[extended_features], y_train)

        # Resample test set with replacement
        indices = np.random.choice(len(X_test), len(X_test), replace=True)
        X_test_resampled, y_test_resampled = X_test.iloc[indices], y_test.iloc[indices]

        # # Compute accuracy for baseline and extended model
        # metric_base = accuracy_score(y_test_resampled, model_base.predict(X_test_resampled[base_features]))
        # metric_extended = accuracy_score(y_test_resampled, model_extended.predict(X_test_resampled[extended_features]))

        # Compute AUROC for baseline and extended model

        from sklearn.metrics import roc_auc_score

        metric_base = roc_auc_score(
            y_test_resampled,
            model_base.predict_proba(X_test_resampled[base_features])[:, 1],
        )
        metric_extended = roc_auc_score(
            y_test_resampled,
            model_extended.predict_proba(X_test_resampled[extended_features])[:, 1],
        )

        # Store performance difference
        epsilon_values.append(metric_extended - metric_base)

    # Compute mean and confidence interval (95% CI)
    epsilon_mean = np.mean(epsilon_values)
    epsilon_ci = np.percentile(epsilon_values, [2.5, 97.5])  # 95% CI

    # Print results
    print(f"Feature '{new_feature}' Importance (Epsilon): {epsilon_mean:.4f}")
    print(f"95% Confidence Interval: [{epsilon_ci[0]:.4f}, {epsilon_ci[1]:.4f}]")

    # return min, max, mean
    return epsilon_values
