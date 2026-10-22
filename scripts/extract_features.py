import json
import multiprocessing as mp
import pickle
import random
from functools import partial
from typing import Any, List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from scripts.utils import *
from scripts.tools import *

from dotenv import load_dotenv


def eval_functions(
    n,
    process_func,
    params={},
    return_tensor=False,
):


    global total_errors
    total_errors = 0


    partial_func = partial(process_func, **params)

    with mp.Pool(N_PROCESSES) as pool:

        all_results = list(
            tqdm(pool.imap(partial_func, range(0, n)), total=n, leave=True)
        )

    df = pd.DataFrame(all_results)
    # save to cdv
    
    XX = df.drop(columns=["label"])
    yy = df["label"]

    # replace non-numerics with ERROR
    XX = XX.apply(
        lambda x: pd.to_numeric(x, errors="coerce")
        if x.dtype == "object"
        else x.astype(float)
    )

    # Replace inf and -inf with NaN
    XX.replace([np.inf, -np.inf], np.nan, inplace=True)

    errors = XX.apply(lambda x: x == "ERROR").sum()

    # count 'ERROR' in XX
    total_errors = XX.apply(lambda x: x == "ERROR").sum().sum()
    if total_errors > 0:
        print(f"Total errors in the data: {total_errors}")
        # replace with False
        pd.set_option("future.no_silent_downcasting", True)
        XX = XX.replace("ERROR", False)

    # count nan values in XX
    total_nan = XX.isnull().sum().sum()
    if total_nan > 0:
        print(f"Total NaN values in the data: {total_nan}")
        XX = XX.fillna(False)

    return XX



def process_ts_vars_extract(i):

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
    df = data[i]["ts"][0]
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

    return data[i]




def process_patient(i):

    # print(function_names)
    global total_errors, data, function_names, HISTORY, LABEL
    global opt_args

    df = data[i]["ts"][0]
    df = df[df["Hours"] < HISTORY]
    age = int(data[i]["covariates"][0][0])
    gender = int(data[i]["covariates"][0][1])
    covariates = {"age": age, "gender": gender}
    label = int(data[i][LABEL][0])

    label = int(data[i][LABEL][0])

    new_features = {}
    for func in function_names:

        try:
            new_features[func] = eval(f"{func}(df,covariates, **{opt_args[func]})")
        except Exception as e:
            print(func, e)
            new_features[func] = "ERROR"
            total_errors += 1

    return {**new_features, "label": label}


def process_patient_single1(i):

    # print(function_names)
    global total_errors, data, function_names, HISTORY, LABEL
    global opt_args


    df = data[i]["ts"][0]
    df = df[df["Hours"] < HISTORY]



    label = int(data[i][LABEL][0])

    new_features = {}
    for func in function_names:

        try:
            new_features[func] = eval(f"{func}(df, **{opt_args[func]})")
        except Exception as e:
            new_features[func] = "ERROR"
            total_errors += 1

    return {**new_features, "label": label}


def process_patient_single2(i):

    # print(function_names)
    global total_errors, data, function_names, HISTORY, LABEL
    global opt_args


    df = data[i]["ts"][0]
    df = df[df["Hours"] < HISTORY]



    label = int(data[i][LABEL][0])

    new_features = {}
    for func in function_names:
        for var in all_vars:
            try:
                new_features[var + "_" + func] = eval(f"{func}(df,variable='{var}')")
            except Exception as e:
                print(function_names, e)
                new_features[var + "_" + func] = "ERROR"
                total_errors += 1

    return {**new_features, "label": label}


@hydra.main(config_path="../configs", config_name="extract", version_base=None)
def main(args):

    global total_errors, HISTORY, LABEL

    global opt_args, function_names, data
    functions = []
    opt_args = {}
    function_names = []

    print(args.data)
    DATA = args.data.name
    LABEL = args.data.label
    HISTORY = args.data.history
    DATA_PATH = args.data.path

    VERSION = args.version
    N_MIN = args.n_min
    N_MAX = args.n_max

    SPLIT = args.split

    FIELD = args.field
    N_LARGEST = args.n_largest

    ### NEW
    csv_file = f"./results/{DATA}/{VERSION}/features.csv"

    df = pd.read_csv(csv_file).dropna(subset=["mean_mi"])
    # keep mean_mi above 0.001
    df = df[df['mean_mi']>0.0001]
    
    print(df.columns)

    print("# W/O Optimization")
    df2 = (
        df.groupby(["round", "category", "q_id"])
        .apply(
            lambda x: x.nlargest(1, "mean_mi")[["mean_mi"]],
            include_groups=False,
        )
        .reset_index()
        .rename(columns={"mean_mi": "W/O Optimization"})
    )


    selected_cols = [
        "function_name",
        "mean_mi",
        "python_script",
        "category",
    ]

    if FIELD == "mi_opt":
        selected_cols.append("mi_opt")
        selected_cols.append("mi_init")

        # selected_cols.append("hparams")
        selected_cols.append("hparams_opt")
        selected_cols.append("python_script_opt")

    if "gain_auroc" in df.columns:
        selected_cols.append("gain_auroc")
        selected_cols.append("gain_auprc")
    df2 = (
        df.groupby(["round", "q_id"])
        .apply(
            lambda x: x.nlargest(N_LARGEST, FIELD)[selected_cols],
            include_groups=False,
        )
        .reset_index()
        .sort_values(by=FIELD, ascending=False)
    )

    if df2.empty:
        raise ValueError(
            "No features selected: filtered results are empty. "
            "Check mean_mi threshold and FIELD selection."
        )

    selected_cols = df2.columns.tolist()
    selected_cols.remove("python_script")

    print(df2[selected_cols].head(20).round(4))

    #

    for _, row in df2.iterrows():

        func = {}
        func["name"] = row["function_name"]
        function_names.append(row["function_name"])

        if ("gain" in FIELD) and (row["gain_auroc"] >= 0.1):
            func["args"] = row["hparams"]
            opt_args.update({func["name"]: row["hparams"]})
        if ("opt" in FIELD) and (row["mi_opt"] > (row["mi_init"])):
            func["python_script"] = row["python_script_opt"]
            func["args"] = row["hparams_opt"]
            opt_args.update({func["name"]: row["hparams_opt"]})
            print(
                f"########## Optimized {func['name']} with {row['mi_opt']:.4f} was {row['mi_init']:.4f}"
            )
        else:
            func["python_script"] = row["python_script"]
            func["args"] = {}
            opt_args.update({func["name"]: {}})
        # func['hparams'] = row['hparams']

        functions.append(func)


    # execute function scripts
    for f in functions[:]:
        f["name"]
        # print(f['python_script'])
        exec(f["python_script"], globals())

    # Loading the data
    path_data = DATA_PATH.format(split=SPLIT)

    print(f"[info] Loading the data from {path_data}...")
    with open(path_data, "rb") as f:
        data = pickle.load(f)

    # random sample N_MAX patients with seed
    if len(data) < N_MAX:
        print(f"Number of patients is more than {N_MAX}. We set N_MAX to {len(data)}")
        N_MAX = len(data)

    # randomly select n samples with seed 42
    random.seed(42)
    random.shuffle(data)
    data = data[:N_MAX]

    print(f"[info] Loaded {N_MAX} patients from {DATA}")

    if DATA == "mimic3":

        with mp.Pool(N_PROCESSES) as pool:
            data = list(
                tqdm(
                    pool.imap(process_ts_vars_extract, range(N_MIN, N_MAX)),
                    total=N_MAX - N_MIN,
                )
            )

    params = {}

    if df2["category"].iloc[0] == "single":
        process_func = process_patient_single1
    elif df2["category"].iloc[0] == "multi":
        process_func = process_patient
    else:
        raise ValueError("Unknown category")
        
    df_results = eval_functions(
        N_MAX,
        process_func,
        params,
        return_tensor=True,
    )
    
    
    os.makedirs(f"{DATA_DIR}/featehr-llm/{DATA}/{SPLIT}/{VERSION}/", exist_ok=True)
    
    df_results.to_csv(f"{DATA_DIR}/featehr-llm/{DATA}/{SPLIT}/{VERSION}/data.csv", index=False)
    print(f"Results saved to {DATA_DIR}/featehr-llm/{DATA}/{SPLIT}/{VERSION}/data.csv")

if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except Exception:
        pass
    random.seed(42)
    global data, function_names, total_errors, history, all_vars

    N_PROCESSES = mp.cpu_count()  # Use all available CPU cores
    print(f"Using {N_PROCESSES} processses for multiprocessing.")

    load_dotenv(override=True)

    DATA_DIR = os.getenv("DATA_DIR", "./data")
    main()
