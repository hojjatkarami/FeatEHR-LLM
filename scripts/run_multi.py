"""
Multi-process feature engineering and rule generation for EHR data.

This module handles feature design, rule generation, and evaluation using LLMs
with multi-processing support for improved performance.
"""

# Standard library imports
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import traceback
import warnings
from functools import partial
from typing import Any, Dict, List, Tuple

# Third-party imports
import hydra
import numpy as np
import openai
import pandas as pd
import wandb
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Local application imports
from scripts.utils import (
    bootstrap_mi,
    design_feature_multi,
    design_rule_multi,
    get_tools_string,
    process_ts_vars,
    set_logger,
)

from scripts.tools import *

# Constants
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_N_BOOTSTRAP = 50
MIN_MUTUAL_INFORMATION = 0.001
MAX_ERROR_THRESHOLD = 0.5
ERROR_MARKER = "ERROR"

# Global variables (set during initialization)
data = None
function_names = []
total_errors = 0
HISTORY = None
LABEL = None
df_features = None
units_full = None
tools_string = None
prompt_memory = None
folder_name = None
TASK = None
MODEL_OPENAI = None
TEMPERATURE = None
MAX_NEW_TOKENS = None
N_RETURN = None
DESIGN_RULE_TEMPLATE = None
N_MAX = None
CATEGORY = None
all_errors = {}
all_warnings = {}


def process_rules(question_index: int) -> List[Dict[str, Any]]:
    """
    Process rules for a single feature question.
    
    Args:
        question_index: Index of the question in df_features
        
    Returns:
        List of rule data dictionaries
    """
    global df_features, units_full, tools_string, prompt_memory, folder_name
    global TASK, MODEL_OPENAI, TEMPERATURE, MAX_NEW_TOKENS, N_RETURN
    global DESIGN_RULE_TEMPLATE, CATEGORY

    question = df_features.iloc[question_index]

    print(f"\t\t{'=' * 104}")
    print(
        f"\t\t # {question_index + 1}/{len(df_features)} | "
        f"{question['feature']} | {question['variables']}\n"
        f"Why important?  # {question['clinical_relevance']}"
    )

    row_data = {
        "round": 0,
        "q_id": question_index,
        "feature": question["feature"],
        "category": f"multi/{question['level']}",
        "variables": question["variables"],
        "clinical_relevance": question["clinical_relevance"],
    }

    params = {
        "feature": row_data["feature"],
        "variables": row_data["variables"],
        "category": row_data["category"],
        "TASK": TASK,
        "model_openai": MODEL_OPENAI,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_return": N_RETURN,
        "prompt_good_functions": "",
        "units_full": units_full,
        "prompt_template": DESIGN_RULE_TEMPLATE,
        "tools_string": tools_string,
        "prompt_memory": prompt_memory,
    }

    valid_functions = _generate_valid_functions(params)
    _register_functions_in_namespace(valid_functions)

    rule_data = _create_rule_data_entries(
        valid_functions, row_data, CATEGORY, question_index
    )
    
    return rule_data


def _generate_valid_functions(params: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate valid functions with retry logic.
    
    Args:
        params: Parameters for rule design
        
    Returns:
        Dictionary of valid function names to code
    """
    valid_functions = {}
    while len(valid_functions) < 1:
        try:
            valid_functions = design_rule_multi(params)
        except Exception as e:
            logger.error(f"Error generating functions: {e}")
            raise

    return valid_functions


def _register_functions_in_namespace(valid_functions: Dict[str, str]) -> None:
    """
    Register generated function scripts in the global namespace for evaluation.
    
    Args:
        valid_functions: Dictionary of function names to code
    """
    try:
        for function_code in valid_functions.values():
            exec(function_code, globals())
    except Exception as e:
        logger.warning(f"Error executing functions: {e}")


def _create_rule_data_entries(
    valid_functions: Dict[str, str],
    row_data: Dict[str, Any],
    category: str,
    question_id: int
) -> List[Dict[str, Any]]:
    """
    Create rule data entries from valid functions.
    
    Args:
        valid_functions: Dictionary of function names to code
        row_data: Base row data dictionary
        category: Category name
        question_id: Question identifier
        
    Returns:
        List of rule data dictionaries
    """
    rule_data = []
    
    for function_id, (function_name, function_code) in enumerate(valid_functions.items()):
        rule_data.append({
            "round": 0,
            "category": category,
            "q_id": question_id,
            "feature": row_data["feature"],
            "fun_id": function_id,
            "function_name": function_name,
            "clinical_relevance": row_data["clinical_relevance"],
            "variables": row_data["variables"],
            "python_script": function_code,
        })
    
    return rule_data


def process_feature(item: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Process a single feature by generating its summary.
    
    Args:
        item: Tuple of (feature_name, feature_content)
        
    Returns:
        Tuple of (feature_name, processed_content)
    """
    feature_name, content = item
    
    summary = _generate_script_summary(content['python_script'])
    
    return feature_name, {
        'mean_mi': content['mean_mi'],
        'script': content['python_script'],
        'summary': summary,
    }


def _generate_script_summary(script: str) -> str:
    """
    Generate a summary of the given script using LLM.
    
    Args:
        script: Python script to summarize
        
    Returns:
        Summary text
    """
    prompt = (
        f"Explain the logic of this algorithm in a short paragraph "
        f"so that it is replicable:\n\n{script}\n"
    )
    
    messages = [
        {"role": "system", "content": "You're a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    
    client = openai.Client(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("LLM_API_KEY")
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5,
        max_tokens=1000,
        n=1,
    )
    
    return response.choices[0].message.content


def process_patient(patient_index: int) -> Dict[str, Any]:
    """
    Process a single patient's data through all functions.
    
    Args:
        patient_index: Index of the patient in the data list
        
    Returns:
        Dictionary of feature values and label
    """
    global total_errors, data, function_names, HISTORY, LABEL
    global all_errors, all_warnings

    patient_data = data[patient_index]
    df = patient_data["ts"][0]
    df = df[df["Hours"] < HISTORY]
    
    age = int(patient_data["covariates"][0][0])
    gender = int(patient_data["covariates"][0][1])
    covariates = {'age': age, 'gender': gender}
    label = int(patient_data[LABEL][0])

    assert "Hours" in df.columns, "DataFrame must contain 'Hours' column"
    new_features = _compute_patient_features(
        df, covariates, function_names, all_errors, all_warnings
    )
    
    return {**new_features, "label": label}


def _compute_patient_features(
    df: pd.DataFrame,
    covariates: Dict[str, int],
    function_names: List[str],
    all_errors: Dict[str, List[str]],
    all_warnings: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Compute feature values for a patient.
    
    Args:
        df: Patient time series data
        covariates: Patient covariates (age, gender)
        function_names: List of function names to execute
        all_errors: Dictionary to track errors
        all_warnings: Dictionary to track warnings
        
    Returns:
        Dictionary of computed feature values
    """
    global total_errors
    
    new_features = {}
    
    for func_name in function_names:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                new_features[func_name] = eval(f"{func_name}(df, covariates)")
                
        except Warning as w:
            logger.debug(
                f"Warning in function {func_name}:\n{str(w)}\n"
                f"{traceback.format_exc()}"
            )
            new_features[func_name] = eval(f"{func_name}(df, covariates)")
            
            if str(w) not in all_warnings[func_name]:
                all_warnings[func_name].append(str(w))
                
        except Exception as e:
            logger.debug(
                f"Error in function {func_name}:\n{str(e)}\n"
                f"{traceback.format_exc()}"
            )
            
            if str(e) not in all_errors[func_name]:
                all_errors[func_name].append(str(e))
                
            new_features[func_name] = ERROR_MARKER
            total_errors += 1
            
    return new_features


def eval_functions(
    num_samples: int,
    params: Dict[str, Any] = None,
    return_tensor: bool = False,
) -> pd.DataFrame:
    """
    Evaluate functions on a sample of patients.
    
    Args:
        num_samples: Number of samples to evaluate
        params: Additional parameters (unused, kept for compatibility)
        return_tensor: Whether to return feature tensor
        
    Returns:
        DataFrame with evaluation metrics or tuple of (DataFrame, features)
    """
    global total_errors
    total_errors = 0

    num_processes = mp.cpu_count()
    partial_func = partial(process_patient)

    with mp.Pool(num_processes) as pool:
        all_results = list(
            tqdm(
                pool.imap(partial_func, range(num_samples)),
                total=num_samples,
                leave=False
            )
        )

    df = pd.DataFrame(all_results)

    features_df = df.drop(columns=["label"])
    labels = df["label"]

    features_df = _clean_feature_dataframe(features_df)
    
    lower_bound, upper_bound, mean_mi = bootstrap_mi(
        features_df, labels,
        n_bootstrap=DEFAULT_N_BOOTSTRAP,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL
    )
    
    dist_true = features_df.sum() / len(features_df)
    errors = features_df.apply(lambda x: x == ERROR_MARKER).sum()

    df_output = pd.DataFrame({
        "Feature": features_df.columns,
        "mean_mi": mean_mi,
        "dist_true": dist_true.values,
        "errors": errors.values,
    }).sort_values(by="mean_mi", ascending=False)

    df_output["errors"] = df_output["errors"] / num_samples
    df_output["error_y"] = (upper_bound - lower_bound) / 2


    # # make sure mean_mi>0.001
    # df_output = df_output[df_output["mean_mi"] > MIN_MUTUAL_INFORMATION]


    
    if return_tensor:
        return df_output, features_df
    else:
        return df_output


def _clean_feature_dataframe(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean feature dataframe by handling errors, NaN, and inf values.
    
    Args:
        features_df: Raw feature dataframe
        
    Returns:
        Cleaned feature dataframe
    """
    # Convert to numeric, replace errors
    features_df = features_df.apply(
        lambda x: pd.to_numeric(x, errors="coerce")
        if x.dtype == "object"
        else x.astype(float)
    )

    # Replace inf values with NaN
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Handle ERROR markers
    total_errors_count = features_df.apply(lambda x: x == ERROR_MARKER).sum().sum()
    if total_errors_count > 0:
        logger.warning(f"Total errors in the data: {total_errors_count}")
        pd.set_option("future.no_silent_downcasting", True)
        features_df = features_df.replace(ERROR_MARKER, False)

    # Handle NaN values
    total_nan = features_df.isnull().sum().sum()
    # percentage of nan values
    percentage_nan = total_nan / features_df.size * 100
    if total_nan > 0:
        logger.warning(f"Total NaN values in the data: {total_nan} ({percentage_nan:.2f}%)")
        features_df = features_df.fillna(False)

    return features_df


def select_random_feature(
    df_rule_good: pd.DataFrame,
    metric: str
) -> pd.DataFrame:
    """
    Select a random feature from the dataframe and sort by the given metric.
    
    Args:
        df_rule_good: DataFrame with good rules
        metric: Metric to sort by
        
    Returns:
        DataFrame with selected feature
    """
    unique_features = df_rule_good["q_id"].unique().tolist()
    random.seed(None)
    selected_q_id = random.choice(unique_features)
    random.seed(42)
    
    print(f"selected: {selected_q_id}, {unique_features}")

    selected_df = df_rule_good[df_rule_good["q_id"] == selected_q_id]
    return selected_df.sort_values(by=metric, ascending=False).head(1)


def generate_prompt_good_functions(
    df_rule_good: pd.DataFrame,
    function_names: List[str],
    metric: str
) -> str:
    """
    Generate a prompt string for good functions from previous rounds.
    
    Args:
        df_rule_good: DataFrame with good rules
        function_names: List of function names
        metric: Metric name to display
        
    Returns:
        Formatted prompt string
    """
    prompt = (
        "## Here are the list of good functions from previous rounds "
        "(Do not copy):\n"
        "Mutual Information with the label, python script\n"
    )
    
    for func_name in function_names:
        score = df_rule_good[df_rule_good["function_name"] == func_name][metric].values[0]
        script = df_rule_good[df_rule_good["function_name"] == func_name]["python_script"].values[0]
        prompt += f"{score:.3f}, {script}\n"
        
    return prompt


def _load_or_generate_summaries(data_name: str, version: str, args: DictConfig) -> Dict[str, Dict[str, Any]]:
    """
    Load existing summaries or generate new ones for a version.
    
    Args:
        data_name: Name of the dataset
        version: Version identifier
        args: Configuration arguments
        
    Returns:
        Dictionary of feature summaries
    """
    summary_path = f"results/{data_name}/{version}/best_summary.json"
    
    if os.path.exists(summary_path):
        print(f"Loading summary from {version}...")
        with open(summary_path, "r") as f:
            return json.load(f)
    
    print(f"Summary file does not exist. Creating...")
    
    features_path = f"results/{data_name}/{version}/features.csv"
    if not os.path.exists(features_path):
        print(f"Version {version} does not exist. Please check the version.")
        return {}
    
    df = pd.read_csv(features_path)
    df = df.sort_values(by=['q_id', 'mean_mi'], ascending=[True, False])
    df = df.groupby('q_id').head(1)
    df = df[df['mean_mi'] > args.memory.mi_th]
    df = df.sort_values(by=['mean_mi'], ascending=False).head(args.memory.max_features)
    
    dict_features = df.set_index('feature')[['python_script', 'mean_mi']].to_dict(orient='index')
    
    num_workers = min(mp.cpu_count(), len(dict_features))
    
    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_feature, dict_features.items()),
                total=len(dict_features)
            )
        )
    
    prompts_summary = dict(results)
    
    with open(summary_path, "w") as f:
        json.dump(prompts_summary, f, indent=4)
    
    return prompts_summary


def _generate_memory_prompt(all_summaries: Dict[str, Dict[str, Any]], max_features: int) -> str:
    """
    Generate memory prompt from feature summaries.
    
    Args:
        all_summaries: Dictionary of all feature summaries
        max_features: Maximum number of features to include
        
    Returns:
        Formatted memory prompt string
    """
    if not all_summaries:
        return ""
    
    sorted_summaries = dict(
        sorted(
            all_summaries.items(),
            key=lambda item: item[1]["mean_mi"],
            reverse=True
        )
    )
    
    top_summaries = dict(list(sorted_summaries.items())[:max_features])
    
    prompt = (
        "### Here are the list of good features from previous experience "
        "(Do not copy):\nFeature, summary\n"
    )
    
    for idx, (feature, content) in enumerate(top_summaries.items(), 1):
        prompt += f"F{idx}. {feature}, {content['summary']}\n\n"
    
    return prompt


def _setup_memory(data_name: str, version: str, args: DictConfig) -> str:
    """
    Setup memory from previous versions.
    
    Args:
        data_name: Name of the dataset
        version: Current version
        args: Configuration arguments
        
    Returns:
        Memory prompt string
    """
    if not args.memory.enabled:
        return ""
    
    print(f"Using memory from versions {args.memory.versions}")
    
    all_summaries = {}
    for memory_version in args.memory.versions:
        version_summaries = _load_or_generate_summaries(data_name, memory_version, args)
        print(f"Loaded {len(version_summaries)} features from {memory_version} for memory")
        all_summaries.update(version_summaries)
    
    memory_prompt = _generate_memory_prompt(all_summaries, args.memory.max_features)
    
    if memory_prompt:
        memory_file_path = f"results/{data_name}/{version}/memory.txt"
        with open(memory_file_path, "w") as f:
            f.write(memory_prompt)
    
    return memory_prompt


def _load_and_prepare_data(args: DictConfig) -> Tuple[List[Any], int]:
    """
    Load and prepare patient data.
    
    Args:
        args: Configuration arguments
        
    Returns:
        Tuple of (data_list, number_of_samples)
    """
    data_path = args.data.path.format(split="train")
    print(f"Loading the data from {data_path}...")
    
    with open(data_path, "rb") as f:
        patient_data = pickle.load(f)
    
    num_samples = min(len(patient_data), args.n_max)
    
    if len(patient_data) < args.n_max:
        print(f"Number of patients is less than {args.n_max}. Setting N_MAX to {len(patient_data)}")
    
    random.seed(42)
    random.shuffle(patient_data)
    patient_data = patient_data[:num_samples]
    
    print(f"Loaded {num_samples} patients from {args.data.name}")
    
    if args.data.name == "mimic3":
        print(f"\tProcessing time series data...")
        for i in tqdm(range(num_samples), desc="Processing data", leave=False):
            patient_data[i]["ts"][0] = process_ts_vars(patient_data[i]["ts"][0])
    
    return patient_data, num_samples


def _generate_features(args: DictConfig) -> pd.DataFrame:
    """
    Generate initial features using LLM.
    
    Args:
        args: Configuration arguments
        
    Returns:
        DataFrame with generated features
    """
    params = {
        "TASK": args.data.task,
        "n_q": args.n_q,
        "model_openai": args.llm.model,
        "temperature": args.llm.temperature,
        "max_new_tokens": args.llm.max_new_tokens,
        "units_full": units_full,
        "prompt_template": args.prompts.design_features,
    }
    
    features_df = design_feature_multi(params)
    print(f"\tGenerated {args.n_q} features in {args.type} category")
    print(features_df)
    
    return features_df


def _generate_and_evaluate_rules(args: DictConfig, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate rules and evaluate them.
    
    Args:
        args: Configuration arguments
        features_df: DataFrame with features
        
    Returns:
        DataFrame with evaluated rules
    """
    global function_names, all_errors, all_warnings
    
    print(f"\tRule Generation")
    
    all_question_indices = features_df.index.tolist()
    partial_func = partial(process_rules)
    num_processes = min(args.n_processes, mp.cpu_count())
    
    with mp.Pool(num_processes) as pool:
        rule_data_nested = list(pool.imap(partial_func, all_question_indices))
    
    rule_data_flat = [
        rule for feature_rules in rule_data_nested for rule in feature_rules
    ]
    df_rule_data = pd.DataFrame(rule_data_flat)
    
    function_names = []
    all_errors = {}
    all_warnings = {}

    for _, row in df_rule_data.iterrows():
        func_name = row["function_name"]
        python_script = row["python_script"]
        
        function_names.append(func_name)
        exec(python_script, globals())
        
        all_errors[func_name] = []
        all_warnings[func_name] = []

    print("\t\t\tEvaluating functions ...")
    
    try:
        df_eval = eval_functions(N_MAX)
        print(df_eval.round(4))
        
        df_eval = df_eval[df_eval["mean_mi"] > args.mi_threshold]
        df_eval = df_eval[df_eval["errors"] < args.mi_error_th]
        print(df_eval.round(4))
        
    except Exception as e:
        logger.warning(f"ERROR in the evaluation:\n{e}")
        raise

    df_rule_data = df_rule_data.merge(
        df_eval[["Feature", "mean_mi", "dist_true", "error_y"]],
        left_on="function_name",
        right_on="Feature",
        how="left",
    ).sort_values(by="mean_mi", ascending=False)

    df_rule_data["errors"] = df_rule_data["function_name"].map(all_errors)
    df_rule_data["warnings"] = df_rule_data["function_name"].map(all_warnings)
    
    return df_rule_data


@hydra.main(config_path="../configs", config_name="multi", version_base=None)
def main(args: DictConfig) -> None:
    """
    Main function to orchestrate feature generation and evaluation.
    
    Args:
        args: Hydra configuration
    """
    global data, HISTORY, LABEL
    global df_features, units_full, tools_string, prompt_memory, folder_name
    global TASK, MODEL_OPENAI, TEMPERATURE, MAX_NEW_TOKENS, N_RETURN
    global DESIGN_RULE_TEMPLATE, N_MAX, CATEGORY

    # Initialize wandb
    wandb.init(
        project=args.wandb.project,
        name=args.wandb.name,
        config=OmegaConf.to_container(args),
    )

    # Set global configuration variables
    DATA = args.data.name
    LABEL = args.data.label
    HISTORY = args.data.history
    TASK = args.data.task
    CATEGORY = args.type
    VERSION = args.version
    N_MAX = args.n_max
    MODEL_OPENAI = args.llm.model
    MAX_NEW_TOKENS = args.llm.max_new_tokens
    TEMPERATURE = args.llm.temperature
    N_RETURN = args.llm.n_return

    print("Prompt for designing the features:\n", args.prompts.design_features)
    DESIGN_RULE_TEMPLATE = args.prompts.design_rule
    print("Prompt for designing the rules:\n", args.prompts.design_rule)

    tools_string = get_tools_string(args.tools.list)

    # Setup folders
    folder_name = f"results/{DATA}/{VERSION}"
    os.makedirs(folder_name, exist_ok=True)
    print(f"Saving results to: {folder_name}/")

    # # Setup memory
    # prompt_memory = _setup_memory(DATA, VERSION, args)

    # Setup logging
    set_logger(folder_name, level=args.log.level)

    # Load data
    data, N_MAX = _load_and_prepare_data(args)

    # Load variable units
    print(f"Loading the variable units...")
    with open(args.data.units, "r") as f:
        units_full = json.load(f)

    # Feature and rule generation
    round_number = 0
    print(f"{'#' * 39} Round {round_number}")
    
    df_features = _generate_features(args)
    df_features.to_csv(f"{folder_name}/features.csv", index=False)
    
    df_rule_data = _generate_and_evaluate_rules(args, df_features)

    # Save results
    with open(f"{folder_name}/args.json", "w") as f:
        json.dump(OmegaConf.to_container(args), f)

    df_rule_data.to_csv(f"{folder_name}/features.csv", index=False)
    print(f"\t\tSaving results to {folder_name}/features.csv")


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
    print(sys.path)

    load_dotenv(override=True)
    random.seed(42)

    main()
