"""
Single-variable feature engineering and rule generation for EHR data.

This module handles feature design, rule generation, and evaluation for single
variables using LLMs with multi-processing support for improved performance.
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
import pandas as pd
import wandb
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Local application imports
from scripts.utils import *
from scripts.tools import *

# Constants
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_N_BOOTSTRAP = 50
MIN_MUTUAL_INFORMATION = 0.001
MAX_ERROR_THRESHOLD = 0.1
ERROR_MARKER = "ERROR"
MIN_VALID_FUNCTIONS = 1

# Global variables (set during initialization)
data = None
function_names = []
total_errors = 0
HISTORY = None
LABEL = None
units_full = None
prompt_good_functions = None
tools_string = None
TASK = None
MODEL_OPENAI = None
TEMPERATURE = None
MAX_NEW_TOKENS = None
N_RETURN = None
DESIGN_RULE_TEMPLATE = None
CATEGORY = None
all_errors = {}
all_warnings = {}


def process_rules(var_name: str) -> List[Dict[str, Any]]:
    """
    Process rules for a single variable.
    
    Args:
        var_name: Name of the variable to process
        
    Returns:
        List of rule data dictionaries
    """
    global units_full, prompt_good_functions, tools_string
    global TASK, MODEL_OPENAI, TEMPERATURE, MAX_NEW_TOKENS, N_RETURN
    global DESIGN_RULE_TEMPLATE, CATEGORY

    var_index = list(units_full.keys()).index(var_name)
    var_unit = units_full[var_name]["unit"]

    print(f"\t\t{'=' * 104}")
    print(f"\t\t[info] Generating functions for:")
    print(f"\t\t # {var_index + 1}/{len(units_full)}. {var_name} (unit: {var_unit})")

    row_data = {
        "var_name": var_name,
        "var_unit": var_unit,
        "category": CATEGORY,
        "variable": var_name,
        "q_id": var_index,
    }

    prompt_variable = _create_variable_prompt(var_name, units_full[var_name])

    params = {
        "var_name": row_data["var_name"],
        "var_unit": row_data["var_unit"],
        "TASK": TASK,
        "model_openai": MODEL_OPENAI,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_return": N_RETURN,
        "prompt_good_functions": prompt_good_functions,
        "prompt_variable": prompt_variable,
        "prompt_template": DESIGN_RULE_TEMPLATE,
        "tools_string": tools_string,
    }

    valid_functions = _generate_valid_functions(params)
    _register_functions_in_namespace(valid_functions)

    rule_data = _create_rule_data_entries(
        valid_functions, row_data, CATEGORY, var_name
    )

    return rule_data


def _create_variable_prompt(var_name: str, var_stats: Dict[str, Any]) -> str:
    """
    Create a descriptive prompt for a variable including its statistics.
    
    Args:
        var_name: Name of the variable
        var_stats: Dictionary containing variable statistics
        
    Returns:
        Formatted variable prompt string
    """
    return (
        f"{var_name} ["
        f"unit:{var_stats['unit']}, "
        f"mean(std):{var_stats['mean']}({var_stats['std']}), "
        f"quantile-2.5%: {var_stats['quantile2.5']}, "
        f"quantile-97.5%: {var_stats['quantile97.5']}]"
    )


def _generate_valid_functions(params: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate valid functions with retry logic.
    
    Args:
        params: Parameters for rule design
        
    Returns:
        Dictionary of valid function names to code
    """
    min_functions = max(MIN_VALID_FUNCTIONS, params["n_return"] - 2)
    valid_functions = {}
    
    while len(valid_functions) < min_functions:
        try:
            valid_functions = design_rule_single(params)
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
    var_name: str
) -> List[Dict[str, Any]]:
    """
    Create rule data entries from valid functions.
    
    Args:
        valid_functions: Dictionary of function names to code
        row_data: Base row data dictionary
        category: Category name
        var_name: Variable name
        
    Returns:
        List of rule data dictionaries
    """
    rule_data = []
    
    for function_id, (function_name, function_code) in enumerate(valid_functions.items()):
        rule_data.append({
            "round": 0,
            "category": category,
            "q_id": row_data["q_id"],
            "feature": var_name,
            "fun_id": function_id,
            "function_name": function_name,
            "variables": [var_name],
            "python_script": function_code,
        })

    return rule_data


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
    label = int(patient_data[LABEL][0])

    new_features = _compute_patient_features(df, function_names, all_errors)

    return {**new_features, "label": label}


def _compute_patient_features(
    df: pd.DataFrame,
    function_names: List[str],
    all_errors: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Compute feature values for a patient.
    
    Args:
        df: Patient time series data
        function_names: List of function names to execute
        all_errors: Dictionary to track errors
        
    Returns:
        Dictionary of computed feature values
    """
    global total_errors
    
    new_features = {}
    
    for func_name in function_names:
        try:
            new_features[func_name] = eval(f"{func_name}(df)")
            
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
    # Convert to numeric
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
    if total_nan > 0:
        logger.warning(f"Total NaN values in the data: {total_nan}")
        features_df = features_df.fillna(False)

    return features_df


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
        print(f"\t[info] Processing time series data...")
        for i in tqdm(range(num_samples), desc="Processing data", leave=False):
            patient_data[i]["ts"][0] = process_ts_vars(patient_data[i]["ts"][0])

    return patient_data, num_samples


def _setup_new_round(folder_name: str, data_name: str, version: str) -> Tuple[List[str], str, str]:
    """
    Setup for a new round using previous results.
    
    Args:
        folder_name: Folder containing previous results
        data_name: Dataset name
        version: Version identifier
        
    Returns:
        Tuple of (function_names, prompt_good_functions, new_folder_name)
    """
    assert os.path.exists(folder_name), "No previous round found"
    assert os.path.exists(f"{folder_name}/perm_imp.csv"), "Permutation importance file not found"

    df_imp = pd.read_csv(f"{folder_name}/perm_imp.csv")
    df_imp = df_imp[df_imp["Importance"] > 0.01]
    
    function_names = df_imp["Feature"].tolist()
    print(f"\t[info] NEW ROUND with {len(function_names)} important features")
    print(function_names)

    df_rule_data = pd.read_csv(f"{folder_name}/rule_data_{data_name}_{version}.csv")

    good_functions = {}
    prompt = (
        "## Here are the list of good functions from previous rounds:\n"
        "permutation importance score, python script\n"
    )
    
    for func_name in function_names:
        good_functions[func_name] = df_rule_data[
            df_rule_data["function_name"] == func_name
        ]["python_script"].values[0]
        
        score = df_imp[df_imp["Feature"] == func_name]["Importance"].values[0]
        prompt += f"{score:.3f}, {good_functions[func_name]}\n"

    print(prompt)

    new_folder = folder_name + "_r"
    
    return function_names, prompt, new_folder


def _generate_and_evaluate_rules(
    args: DictConfig,
    units_full: Dict[str, Any],
    num_samples: int
) -> pd.DataFrame:
    """
    Generate rules and evaluate them.
    
    Args:
        args: Configuration arguments
        units_full: Dictionary of variable units and statistics
        num_samples: Number of samples for evaluation
        
    Returns:
        DataFrame with evaluated rules
    """
    global function_names, all_errors, all_warnings

    print(f"\t[info] Rule Generation")

    var_names = list(units_full.keys())
    partial_func = partial(process_rules)
    num_processes = min(args.n_processes.rules, mp.cpu_count())
    
    with mp.Pool(num_processes) as pool:
        rule_data_nested = list(pool.imap(partial_func, var_names))

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

    print("\t\t\t[info] Evaluating functions ...")
    
    try:
        df_eval = eval_functions(num_samples)
        print(df_eval.round(4))

        df_eval = df_eval[df_eval["mean_mi"] > args.mi_threshold]
        df_eval = df_eval[df_eval["errors"] < args.mi_error_th]
        print(df_eval.round(4))

        df_rule_data = df_rule_data.merge(
            df_eval[["Feature", "mean_mi", "dist_true", "error_y"]],
            left_on="function_name",
            right_on="Feature",
            how="left",
        ).sort_values(by="mean_mi", ascending=False)
        
    except Exception as e:
        logger.error(f"ERROR in the evaluation:\n{e}")
        raise

    df_rule_data["errors"] = df_rule_data["function_name"].map(all_errors)
    df_rule_data["warnings"] = df_rule_data["function_name"].map(all_warnings)

    return df_rule_data


@hydra.main(config_path="../configs", config_name="single", version_base=None)
def main(args: DictConfig) -> None:
    """
    Main function to orchestrate feature generation and evaluation.
    
    Args:
        args: Hydra configuration
    """
    global data, function_names, HISTORY, LABEL
    global units_full, prompt_good_functions, tools_string
    global TASK, MODEL_OPENAI, TEMPERATURE, MAX_NEW_TOKENS, N_RETURN
    global DESIGN_RULE_TEMPLATE, CATEGORY

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

    DESIGN_RULE_TEMPLATE = args.prompts.design_rule
    print("Prompt for designing the rules:\n", args.prompts.design_rule)

    tools_string = get_tools_string(args.tools.list)

    # Setup folders
    folder_name = f"results/{DATA}/{VERSION}"
    os.makedirs(folder_name, exist_ok=True)
    print(f"Saving results to: {folder_name}/")

    set_logger(folder_name, level=args.log.level)

    # Load data
    data, N_MAX = _load_and_prepare_data(args)

    # Load variable units
    print(f"Loading the variable units...")
    with open(args.data.units, "r") as f:
        units_full = json.load(f)

    # Setup for new round or start fresh
    prompt_good_functions = ""

    # Generate and evaluate rules
    df_rule_data = _generate_and_evaluate_rules(args, units_full, N_MAX)

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
