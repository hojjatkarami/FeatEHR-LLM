# Standard library imports
import os
import sys
import json
import pickle
import random
from functools import partial
import multiprocessing as mp

# Third-party imports
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# Local application imports
from scripts.utils import *

import wandb

from loguru import logger


def run_featllm(args, X_train, y_train, X_test, y_test, units_full):
    DATA = args.data.name
    SHOT = args.baseline.shot
    NUM_QUERY = args.baseline.num_query
    SEED=42
    API_KEY = os.getenv("LLM_API_KEY")



    X_train.shape, y_train.shape, X_test.shape, y_test.shape
    label_list = ['no','yes']
    yy_train = np.array([label_list[i] for i in y_train.values]).astype(object)
    yy_test = np.array([label_list[i] for i in y_test.values]).astype(object)

    target_attr = y_train.name
    feature_names = X_train.columns.tolist()
    is_cat = [True if "Genderr" in f else False for f in feature_names]

    X_all = pd.concat([X_train, X_test], axis=0)


    def prepare_data(X_train, y_train,default_target_attribute, shot=SHOT, seed=42):
        # default_target_attribute = y_train.name

        assert(shot <= 128) # We only consider the low-shot regimes here
        X_new_train = X_train.copy()
        X_new_train[default_target_attribute] = y_train
        sampled_list = []
        total_shot_count = 0
        remainder = shot % len(np.unique(y_train))
        for _, grouped in X_new_train.groupby(default_target_attribute):
            sample_num = shot // len(np.unique(y_train))
            if remainder > 0:
                sample_num += 1
                remainder -= 1
            grouped = grouped.sample(sample_num, random_state=seed)
            sampled_list.append(grouped)
        X_balanced = pd.concat(sampled_list)
        X_train = X_balanced.drop([default_target_attribute], axis=1)
        y_train = X_balanced[default_target_attribute].to_numpy()

        return X_train, y_train


    X_train_sh,  y_train_sh = prepare_data(X_train, yy_train,target_attr, shot=SHOT)

    # writing metadata
    metadata = {}
    for var in units_full.keys():
        metadata.update({

            var: f"{var} ({units_full[var]['unit']})"
        })
    # save to json
    with open(f"./FeatLLM/data/{DATA}-metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)




    from FeatLLM import utils as utils_featllm

    ask_file_name = './FeatLLM/templates/ask_llm.txt'
    meta_data_name = f"./FeatLLM/data/{DATA}-metadata.json"
    templates, feature_desc = utils_featllm.get_prompt_for_asking(
            DATA, X_all, X_train_sh, y_train_sh, label_list, target_attr, ask_file_name, 
            meta_data_name, is_cat, num_query=NUM_QUERY
    )


    # creating rules
    _DIVIDER = "\n\n---DIVIDER---\n\n"
    _VERSION = "\n\n---VERSION---\n\n"

    rule_file_name = f'./FeatLLM/rules/rule-{DATA}-{SHOT}-{SEED}.out'
    if True:#os.path.isfile(rule_file_name) == False:
        results = utils_featllm.query_gpt(templates, API_KEY, max_tokens=1500, temperature=0.5, model = 'gpt-4o-mini')
        with open(rule_file_name, 'w') as f:
            total_rules = _DIVIDER.join(results)
            f.write(total_rules)
    else:
        with open(rule_file_name, 'r') as f:
            total_rules_str = f.read().strip()
            results = total_rules_str.split(_DIVIDER)


    # parse rules to the program code

    parsed_rules = utils_featllm.parse_rules(results, label_list)

    saved_file_name = f'./FeatLLM/rules/function-{DATA}-{SHOT}-{SEED}.out'    
    if True:#os.path.isfile(saved_file_name) == False:
        function_file_name = './FeatLLM/templates/ask_for_function.txt'
        fct_strs_all = []
        for parsed_rule in tqdm(parsed_rules):
            fct_templates = utils_featllm.get_prompt_for_generating_function(
                parsed_rule, feature_desc, function_file_name
            )
            fct_results = utils_featllm.query_gpt(fct_templates, API_KEY, max_tokens=1500, temperature=0, model = 'gpt-4o-mini')
            fct_strs = [fct_txt.split('<start>')[1].split('<end>')[0].strip() for fct_txt in fct_results]
            fct_strs_all.append(fct_strs)

        with open(saved_file_name, 'w') as f:
            total_str = _VERSION.join([_DIVIDER.join(x) for x in fct_strs_all])
            f.write(total_str)
    else:
        with open(saved_file_name, 'r') as f:
            total_str = f.read().strip()
            fct_strs_all = [x.split(_DIVIDER) for x in total_str.split(_VERSION)]


    # Get function names and strings
    fct_names = []
    fct_strs_final = []
    for fct_str_pair in fct_strs_all:
        fct_pair_name = []
        if 'def' not in fct_str_pair[0]:
            continue

        for fct_str in fct_str_pair:
            fct_pair_name.append(fct_str.split('def')[1].split('(')[0].strip())
        fct_names.append(fct_pair_name)
        fct_strs_final.append(fct_str_pair)




    # covnert to binary vectors
    executable_list, X_train_all_dict, X_test_all_dict = utils_featllm.convert_to_binary_vectors(fct_strs_final, fct_names, label_list, X_train, X_test)

    def get_features(X_train_all_dict):
        """
        Get features from the X_train_all_dict and return a DataFrame
        """
        all_features = {}
        for i in X_train_all_dict.keys():
            len(X_train_all_dict[i]['yes'].sum(1).tolist())
            
            all_features.update({
                f'f_{i}': X_train_all_dict[i]['yes'].sum(1).int().tolist(),
            })
        return pd.DataFrame(all_features)

    df_train_new = get_features(X_train_all_dict)
    df_test_new = get_features(X_test_all_dict)

    # print the shapes
    print("FeatLLM train shape:", df_train_new.shape)
    print("FeatLLM test shape:", df_test_new.shape)

    # Create directories if they do not exist
    os.makedirs(f"data/featehr-llm/{DATA}/test/v_bl_featllm", exist_ok=True)
    os.makedirs(f"data/featehr-llm/{DATA}/train/v_bl_featllm", exist_ok=True)

    # Save the data
    df_test_new.to_csv(f"data/featehr-llm/{DATA}/test/v_bl_featllm/data.csv", index=False)
    df_train_new.to_csv(f"data/featehr-llm/{DATA}/train/v_bl_featllm/data.csv", index=False)

    print("Data saved to featehr-llm folder")

    return


def run_caafe(args, X_train, y_train, X_test, y_test, units_full):
    ITER = args.baseline.iter
    DATA = args.data.name


    from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
    # from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets
    # from sklearn.ensemble import RandomForestClassifier

    import os
    # import openai
    # import torch
    from caafe import data
    from sklearn.metrics import accuracy_score
    # from tabpfn.scripts import tabular_metrics
    from functools import partial
    from caafe.preprocessing import make_datasets_numeric
    from lightgbm import LGBMClassifier
    from scripts.utils import compute_auroc_auprc_f1

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")

    # for CAAFE
    dataset_description = f'''** Electronic Health Reocrds database**
This database consists of a few clinical time series features during the first 48 hours in the ICU. The target concept is "in-hospital mortality" (i.e., true when patient has died in the hospital).  
s
Attribute Information: 'column_name' (unit)
{"\n".join([f"{i+1}. '{k}'({v['unit']})" for i,(k,v) in enumerate(units_full.items())])}

'''


    df_train  = pd.concat([X_train, y_train], axis=1).fillna(0)
    df_test  = pd.concat([X_test, y_test], axis=1).fillna(0)
    target_column_name = y_train.name

    # remove _bl from column names

    df_train.columns = df_train.columns.str.replace('_bl', '', regex=False)
    df_test.columns = df_test.columns.str.replace('_bl', '', regex=False)


    # convert dataframes to numeric
    df_train, df_test = make_datasets_numeric(df_train, df_test, target_column_name)
    # df_train = df_train.apply(pd.to_numeric, errors='coerce')
    # df_test = df_test.apply(pd.to_numeric, errors='coerce')

    train_x, train_y = data.get_X_y(df_train, target_column_name)
    test_x, test_y = data.get_X_y(df_test, target_column_name)


    clf_no_feat_eng = LGBMClassifier(verbose=-1, random_state=42)

    clf_no_feat_eng.fit(X_train, y_train)

    probs_test = clf_no_feat_eng.predict_proba(X_test)[:, 1]

    metrics_init = compute_auroc_auprc_f1(y_test, probs_test)
    print('Metrics initial',metrics_init)

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    print(df_train.shape, df_test.shape)



    caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                            llm_model="gpt-4o-mini",
                            iterations=ITER)

    caafe_clf.fit_pandas(df_train,
                        target_column_name=target_column_name,
                        dataset_description=dataset_description)


    df_test_new = caafe_clf.preprocess(df_test)
    df_train_new = caafe_clf.preprocess(df_train)

    # print the shapes
    print("CAAFE train shape:", df_train_new.shape)
    print("CAAFE test shape:", df_test_new.shape)

    # Create directories if they do not exist
    os.makedirs(f"data/featehr-llm/{DATA}/test/v_bl_caafe", exist_ok=True)
    os.makedirs(f"data/featehr-llm/{DATA}/train/v_bl_caafe", exist_ok=True)

    # Save the data
    df_test_new.to_csv(f"data/featehr-llm/{DATA}/test/v_bl_caafe/data.csv", index=False)
    df_train_new.to_csv(f"data/featehr-llm/{DATA}/train/v_bl_caafe/data.csv", index=False)



    return



@hydra.main(config_path="../configs", config_name="bl", version_base=None)
def main(args: DictConfig):
    DATA = args.data.name
    N_MAX = args.n_max
    # load original DATA
    
    XX, yy, common_ids = (
        {"train": {}, "test": {}},
        {"train": {}, "test": {}},
        {"train": {}, "test": {}},
    )

    

    XX['train']['numeric'], yy['train']['numeric'], _ = read_data_xgboost(
        data_name=DATA,
        split="train",
        version="numeric",
        run_llm=None,
        n=N_MAX,
    )

    XX['test']['numeric'], yy['test']['numeric'], _ = read_data_xgboost(
        data_name=DATA,
        split="test",
        version="numeric",
        run_llm=None,
        n=N_MAX,
    )

    # load units full
    with open(args.data.units, "r") as f:
        units_full = json.load(f)

    print("Dataset loaded")


    # for CAAFE Only
    dataset_description = f'''** Electronic Health Reocrds database**
        This database consists of a few clinical time series features during the first 48 hours in the ICU. The target concept is "in-hospital mortality" (i.e., true when patient has died in the hospital).  

        Attribute Information: 'column_name' (unit)
        {"\n".join([f"{i+1}. '{k}'({v['unit']})" for i,(k,v) in enumerate(units_full.items())])}

        '''

    X_train = XX['train']['numeric'].fillna(0).astype(float)
    X_test = XX['test']['numeric'].fillna(0).astype(float)
    y_train = yy['train']['numeric'].astype(int)
    y_test = yy['test']['numeric'].astype(int)

    # remove _bl from column names
    X_train.columns = X_train.columns.str.replace('_bl', '')
    X_test.columns = X_test.columns.str.replace('_bl', '')






    # start wandb
    wandb.init(
        project=args.wandb.project,
        name= args.wandb.name,
        config=OmegaConf.to_container(args),
    )


    if args.baseline.name=="featllm":
        run_featllm(args, X_train, y_train, X_test, y_test, units_full)
    elif args.baseline.name=="caafe":
        run_caafe(args, X_train, y_train, X_test, y_test, units_full)
    elif args.baseline.name=="autofeat":

        import numpy as np
        import pandas as pd
        from sklearn.datasets import load_breast_cancer, load_iris, load_wine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        from autofeat import AutoFeatClassifier
                
        units={}
        feateng_steps=args.baseline.feateng_steps

        # select 10 vars only
        X_train = X_train.iloc[:, :]
        y_train = y_train.iloc[:]
        X_test = X_test.iloc[:, :]


        # run autofeat
        afreg = AutoFeatClassifier(verbose=1, feateng_steps=feateng_steps, units=units)
        # fit autofeat on less data, otherwise ridge reg model with xval will overfit on new features
        X_train_tr = afreg.fit_transform(X_train, y_train)
        X_test_tr = afreg.transform(X_test)
        print("autofeat new features:", len(afreg.new_feat_cols_))
        print("autofeat Acc. on training data:", accuracy_score(y_train, afreg.predict(X_train_tr)))
        print("autofeat Acc. on test data:", accuracy_score(y_test, afreg.predict(X_test_tr)))

        df_train_new = X_train_tr.drop(columns = X_train.columns)
        df_test_new = X_test_tr.drop(columns = X_test.columns)

        # print the shapes
        print("AutoFeat train shape:", df_train_new.shape)
        print("AutoFeat test shape:", df_test_new.shape)
        
        # Create directories if they do not exist
        os.makedirs(f"data/featehr-llm/{DATA}/test/v_bl_autofeat", exist_ok=True)
        os.makedirs(f"data/featehr-llm/{DATA}/train/v_bl_autofeat", exist_ok=True)

        # Save the data
        df_test_new.to_csv(f"data/featehr-llm/{DATA}/test/v_bl_autofeat/data.csv", index=False)
        df_train_new.to_csv(f"data/featehr-llm/{DATA}/train/v_bl_autofeat/data.csv", index=False)
   
    elif args.baseline.name=="openfe":

        N_FEATURES = args.baseline.n_features
        N_JOBS = min(mp.cpu_count(),args.baseline.n_jobs)

        import pandas as pd
        from sklearn.datasets import fetch_california_housing
        from openfe import OpenFE, tree_to_formula, transform
        from sklearn.model_selection import train_test_split
        import lightgbm as lgb
        from sklearn.metrics import mean_squared_error


        # feature generation
        ofe = OpenFE()
        ofe.fit(data=X_train, label=y_train, n_jobs=N_JOBS, verbose=False)


        # transform data
        df_train_new, df_test_new = transform(X_train, X_test, ofe.new_features_list[:N_FEATURES], n_jobs=N_JOBS)

        df_train_new = df_train_new.drop(columns=X_train.columns)
        df_test_new = df_test_new.drop(columns=X_test.columns)

        # print the shapes
        print("OpenFE train shape:", df_train_new.shape)
        print("OpenFE test shape:", df_test_new.shape)

        # Create directories if they do not exist
        os.makedirs(f"{DATA_DIR}/featehr-llm/{DATA}/test/v_bl_openfe", exist_ok=True)
        os.makedirs(f"{DATA_DIR}/featehr-llm/{DATA}/train/v_bl_openfe", exist_ok=True)

        # Save the data
        df_test_new.to_csv(f"{DATA_DIR}/featehr-llm/{DATA}/test/v_bl_openfe/data.csv", index=False)
        df_train_new.to_csv(f"{DATA_DIR}/featehr-llm/{DATA}/train/v_bl_openfe/data.csv", index=False)
    else:
        print("baseline not defined:", args.baseline.name)
       
    return


if __name__ == "__main__":

    sys.path.append(os.path.abspath(os.getcwd()))
    print(sys.path)


    load_dotenv(override=True)

    
    DATA_DIR = os.getenv("DATA_DIR", "./data")

    random.seed(42)

    main()
