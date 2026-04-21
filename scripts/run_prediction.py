"""
Prediction pipeline for EHR data using various ML models.
Supports multiple models (LightGBM, XGBoost, CatBoost, MLP, ResNet, TabPFN)
with bootstrap confidence intervals.
"""

# Standard library imports
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
from functools import partial
from typing import Dict, List, Optional, Tuple

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
from scripts.utils import (
    compute_auroc_auprc_f1,
    read_data_xgboost,
    run_catboost,
    run_lightgbm,
    run_mlp,
    run_resnet,
    run_tabpfn,
    tune_lightgbm,
)

# Constants
DEFAULT_CONFIDENCE = 0.95
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_RANDOM_SEED = 42
TRAIN_TEST_SPLIT_RATIO = 0.2


def bootstrap_metric(
    i_boot: int, y_test: np.ndarray, probs_test: np.ndarray, seed: int
) -> Optional[Dict[str, float]]:
    """
    Perform a single bootstrap iteration with random resampling.
    
    Args:
        i_boot: Bootstrap iteration index
        y_test: True labels
        probs_test: Predicted probabilities
        seed: Random seed
        
    Returns:
        Dictionary with AUROC, AUPRC, F1 scores, or None if invalid sample
    """
    rng = np.random.RandomState(seed + i_boot)
    indices = rng.choice(np.arange(len(y_test)), size=len(y_test), replace=True)

    # Skip samples that lack both classes
    if len(np.unique(y_test[indices])) < 2:
        return None

    y_true_boot = y_test[indices]
    y_prob_boot = probs_test[indices]

    return compute_auroc_auprc_f1(y_true_boot, y_prob_boot)


def stratified_bootstrap_metric(
    i_boot: int, y_test: np.ndarray, probs_test: np.ndarray, seed: int
) -> Dict[str, float]:
    """
    Perform a single stratified bootstrap iteration preserving class balance.
    
    Args:
        i_boot: Bootstrap iteration index
        y_test: True labels
        probs_test: Predicted probabilities
        seed: Random seed
        
    Returns:
        Dictionary with AUROC, AUPRC, F1 scores
    """
    rng = np.random.RandomState(seed + i_boot)

    # Get indices for each class
    pos_indices = np.where(y_test == 1)[0]
    neg_indices = np.where(y_test == 0)[0]

    # Sample with replacement from each class
    pos_sampled = rng.choice(pos_indices, size=len(pos_indices), replace=True)
    neg_sampled = rng.choice(neg_indices, size=len(neg_indices), replace=True)

    # Combine and shuffle
    stratified_indices = np.concatenate([pos_sampled, neg_sampled])
    rng.shuffle(stratified_indices)

    y_true_boot = y_test[stratified_indices]
    y_prob_boot = probs_test[stratified_indices]

    return compute_auroc_auprc_f1(y_true_boot, y_prob_boot)


def ci_bounds(scores: List[float], confidence: float = DEFAULT_CONFIDENCE) -> float:
    """
    Calculate confidence interval error bounds.
    
    Args:
        scores: List of metric scores from bootstrap
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Half-width of confidence interval (error margin)
    """
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    lower = np.percentile(scores, lower_percentile)
    upper = np.percentile(scores, upper_percentile)

    return (upper - lower) / 2


def run_bootstrap_parallel(
    y_test: np.ndarray, 
    probs_test: np.ndarray, 
    n_boot: int = DEFAULT_N_BOOTSTRAP, 
    seed: int = DEFAULT_RANDOM_SEED
) -> Tuple[List[float], List[float], List[float]]:
    """
    Run stratified bootstrap in parallel to compute confidence intervals.
    
    Args:
        y_test: True labels
        probs_test: Predicted probabilities
        n_boot: Number of bootstrap iterations
        seed: Random seed
        
    Returns:
        Tuple of (AUROC scores, AUPRC scores, F1 scores)
    """
    with mp.Pool(processes=mp.cpu_count()) as pool:
        func = partial(
            stratified_bootstrap_metric, 
            y_test=y_test, 
            probs_test=probs_test, 
            seed=seed
        )
        results = pool.map(func, range(n_boot))

    # Remove None results from invalid resamples
    results = [r for r in results if r is not None]

    auroc_scores = [r["AUROC"] for r in results]
    auprc_scores = [r["AUPRC"] for r in results]
    f1_scores = [r["F1"] for r in results]

    return auroc_scores, auprc_scores, f1_scores


def prepare_data_for_model(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series,
    model: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data by handling NaN values based on model type.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model: Model name
        
    Returns:
        Cleaned data tuple (X_train, X_test, y_train, y_test)
    """
    if model not in ['resnet']:
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        y_train = y_train.fillna(0)
        y_test = y_test.fillna(0)

    print(
        f"Data shapes - Train: {X_train.shape}, Test: {X_test.shape}, "
        f"Labels - Train: {y_train.shape}, Test: {y_test.shape}"
    )
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    version: str,
    best_params: Optional[Dict] = None,
    seed: int = DEFAULT_RANDOM_SEED
):
    """
    Train a model and return the trained model object.
    
    Args:
        model_name: Name of the model to train
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        version: Data version string
        best_params: Hyperparameters for the model
        seed: Random seed
        
    Returns:
        Trained model object
    """
    if model_name == "lightgbm" or (model_name == "resnet" and version != "resnet"):
        return run_lightgbm(X_train, X_test, y_train, y_test, best_params, seed)
    
    elif model_name == "xgboost":
        return run_xgboost(X_train, X_test, y_train, y_test, best_params, seed)
    
    elif model_name == "catboost":
        return run_catboost(X_train, X_test, y_train, y_test, best_params, seed)
    
    elif model_name == "mlp":
        return run_mlp(X_train, X_test, y_train, y_test, best_params, seed)
    
    elif model_name == "resnet" and version == "resnet":
        return run_resnet(X_train, X_test, y_train, y_test, seed)
    
    elif model_name == "tabpfn":
        return run_tabpfn(X_train, X_test, y_train, y_test, seed)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_metrics_with_ci(
    y_test: np.ndarray, 
    probs_test: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics with confidence intervals using bootstrap.
    
    Args:
        y_test: True labels
        probs_test: Predicted probabilities
        
    Returns:
        Dictionary with metrics and their confidence intervals
    """
    auroc_scores, auprc_scores, f1_scores = run_bootstrap_parallel(
        y_test, probs_test, n_boot=DEFAULT_N_BOOTSTRAP, seed=DEFAULT_RANDOM_SEED
    )

    return {
        "AUROC": np.mean(auroc_scores),
        "AUPRC": np.mean(auprc_scores),
        "F1": np.mean(f1_scores),
        "AUROC_CI": ci_bounds(auroc_scores),
        "AUPRC_CI": ci_bounds(auprc_scores),
        "F1_CI": ci_bounds(f1_scores),
    }


def sample_training_data(
    X_train_full: pd.DataFrame, 
    y_train_full: pd.Series, 
    sample_size: int
) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
    """
    Sample training data to specified size.
    
    Args:
        X_train_full: Full training features
        y_train_full: Full training labels
        sample_size: Desired sample size
        
    Returns:
        Tuple of (sampled X, sampled y, sampled indices)
    """
    actual_size = min(sample_size, len(X_train_full))
    
    if sample_size < len(X_train_full):
        sampled_indices = X_train_full.sample(
            n=actual_size, random_state=DEFAULT_RANDOM_SEED
        ).index
    else:
        sampled_indices = X_train_full.index
    
    X_train = X_train_full.loc[sampled_indices].reset_index(drop=True)
    y_train = y_train_full.loc[sampled_indices].reset_index(drop=True)
    
    return X_train, y_train, sampled_indices


def create_run2(
    data: str,
    version: str,
    model: str,
    feature: str,
    size: int,
    n_cv: int = 0,
    selected_features: List = None,
    feature_selection: str = "",
) -> List[Dict]:
    """
    Run model training and evaluation pipeline.
    
    Args:
        data: Dataset name
        version: Feature version
        model: Model name
        feature: Feature type
        size: Training sample size
        n_cv: Number of cross-validation folds (0 for train/test split)
        selected_features: List of selected features
        feature_selection: Feature selection method
        
    Returns:
        List of result dictionaries
    """
    global X, y

    hp_tune = False
    all_results = []

    # Sample training data
    X_train, y_train, sampled_indices = sample_training_data(
        X["train"][version], y["train"]["numeric"], size
    )

    # Hyperparameter tuning (if enabled)
    best_model = tune_lightgbm(X_train, y_train) if hp_tune else None

    # Test split evaluation
    if n_cv == 0:
        X_test = X["test"][version].reset_index(drop=True)
        y_test = y["test"]["numeric"].reset_index(drop=True)
        
        X_train, X_test, y_train, y_test = prepare_data_for_model(
            X_train, X_test, y_train, y_test, model
        )

        for i_rep in range(1):
            trained_model = train_and_evaluate_model(
                model, X_train, X_test, y_train, y_test, version, best_model, i_rep
            )

            probs_test = trained_model.predict_proba(X_test)[:, 1]
            metric_utility = compute_metrics_with_ci(y_test, probs_test)

            result = {
                "data": data,
                "version": version,
                "model": model,
                "feature": feature,
                "size": size,
                "begin": i_rep,
                "end": i_rep + 1,
                "AUROC": metric_utility["AUROC"],
                "AUPRC": metric_utility["AUPRC"],
                "F1": metric_utility["F1"],
                "AUROC_CI": metric_utility["AUROC_CI"],
                "AUPRC_CI": metric_utility["AUPRC_CI"],
                "F1_CI": metric_utility["F1_CI"],
                "d": X_train.shape[1],
            }
            all_results.append(result)

    # Cross-validation evaluation
    else:
        edges = np.linspace(0, len(y_train), n_cv + 1).astype(int)

        for begin, end in zip(edges[:-1], edges[1:]):
            X_test_cv = X_train.iloc[begin:end]
            y_test_cv = y_train.iloc[begin:end]
            X_train_cv = X_train.drop(index=range(begin, end))
            y_train_cv = y_train.drop(index=range(begin, end))

            if model == "lightgbm":
                metric_utility = run_lightgbm(
                    [X_train_cv], [X_test_cv], y_train_cv, y_test_cv, best_model
                )
            elif model == "tabpfn":
                metric_utility = run_tabpfn(
                    [X_train_cv], [X_test_cv], y_train_cv, y_test_cv
                )

            result = {
                "data": data,
                "version": version,
                "model": model,
                "feature": feature,
                "size": size,
                "begin": begin,
                "end": end,
                "AUROC": metric_utility["AUROC"],
                "AUPRC": metric_utility["AUPRC"],
                "F1": metric_utility["F1"],
                "d": X_train.shape[1],
            }
            all_results.append(result)

    return all_results


def load_data_for_version(
    data_name: str,
    version: str,
    strategy: str,
    n_max: int,
    n_max_test: int,
    ignore_cache: bool
) -> Tuple[Dict, Dict, Dict]:
    """
    Load data for a specific version.
    
    Args:
        data_name: Dataset name
        version: Feature version
        strategy: Data loading strategy
        n_max: Maximum training samples
        n_max_test: Maximum test samples
        ignore_cache: Whether to ignore cached data
        
    Returns:
        Tuple of (X_train, y_train, common_ids_train) and (X_test, y_test, common_ids_test)
    """
    print(f"[INFO] Loading *{data_name}* data with version *{version}*")
    
    try:
        X_train, y_train, common_ids_train = read_data_xgboost(
            data_name=data_name,
            split="train",
            version=version,
            strategy=strategy,
            run_llm=None,
            n=n_max,
            ignore_cache=ignore_cache,
        )
    except Exception as e:
        print(f"Train data not available: {e}")
        return None, None, None

    try:
        X_test, y_test, common_ids_test = read_data_xgboost(
            data_name=data_name,
            split="test",
            version=version,
            strategy=strategy,
            run_llm=None,
            n=n_max_test,
            ignore_cache=ignore_cache,
        )
    except Exception as e:
        print(f"Test data not available: {e}")
        return None, None, None

    return (X_train, y_train, common_ids_train), (X_test, y_test, common_ids_test)


def align_data_by_common_ids(X: Dict, y: Dict, common_ids: Dict, versions: List[str]):
    """
    Align all data splits by common sample IDs.
    
    Args:
        X: Feature data dictionary
        y: Label data dictionary
        common_ids: Common IDs dictionary
        versions: List of data versions
    """
    for split in ["train", "test"]:
        all_common_ids = set()
        
        for version in common_ids[split].keys():
            if len(all_common_ids) == 0:
                all_common_ids = set(common_ids[split][version])
            else:
                all_common_ids = all_common_ids.intersection(common_ids[split][version])

        all_common_ids = list(all_common_ids)

        for version in versions:
            if "+" not in version:
                X[split][version] = X[split][version].loc[all_common_ids]
                if y[split][version] is not None:
                    y[split][version] = y[split][version].loc[all_common_ids]
                else:
                    y[split][version] = y[split]["numeric"].loc[all_common_ids]
            else:
                version_list = version.split("+")
                X[split][version] = pd.concat(
                    [X[split][v] for v in version_list], axis=1
                )
                y[split][version] = y[split]["numeric"]

                if len(X[split][version].columns[X[split][version].columns.duplicated()]) > 0:
                    print(f"[INFO] Dropping duplicated columns in {version} data")
                    X[split][version] = X[split][version].loc[
                        :, ~X[split][version].columns.duplicated()
                    ]


def validate_data_shapes(X: Dict, y: Dict, versions: List[str]):
    """
    Validate that train and test data have matching shapes.
    
    Args:
        X: Feature data dictionary
        y: Label data dictionary
        versions: List of data versions
    """
    for version in versions:
        print(
            f"[INFO] *{version}* data: Train shape: {X['train'][version].shape}, "
            f"{y['train'][version].shape}, Test shape: {X['test'][version].shape}, "
            f"{y['test'][version].shape}"
        )
        
        assert (
            X["train"][version].shape[1] == X["test"][version].shape[1]
        ), f"Train and test have different column counts for {version}"

        assert (
            X["train"][version].shape[0] == y["train"]["numeric"].shape[0]
        ), f"Train data and labels have different sample counts for {version}"

        assert (
            X["test"][version].shape[0] == y["test"]["numeric"].shape[0]
        ), f"Test data and labels have different sample counts for {version}"


def rebalance_train_test_split(X: Dict, y: Dict, versions: List[str]):
    """
    Rebalance train/test split to maintain 80/20 ratio.
    
    Args:
        X: Feature data dictionary
        y: Label data dictionary
        versions: List of data versions
    """
    size_train = len(X["train"]["numeric"])
    size_test = len(X["test"]["numeric"])
    size_to_move = int((size_train + size_test) * TRAIN_TEST_SPLIT_RATIO) - size_test
    
    print(f"Train size: {size_train}, Test size: {size_test}, To Move: {size_to_move}")

    if size_to_move <= 0:
        print("[INFO] No need to rebalance train/test split")
        return

    print(f"Moving {size_to_move} samples from train to test")

    ids_to_move = random.sample(X["train"]["numeric"].index.tolist(), size_to_move)

    y_test_numeric = pd.concat(
        [y["test"]["numeric"], y["train"]["numeric"].loc[ids_to_move]],
        axis=0,
        ignore_index=True,
    )
    y_train_numeric = y["train"]["numeric"].drop(index=ids_to_move)

    for version in versions:
        X["test"][version] = pd.concat(
            [X["test"][version], X["train"][version].loc[ids_to_move]],
            axis=0,
            ignore_index=True,
        )
        y["test"][version] = y_test_numeric

        X["train"][version] = X["train"][version].drop(index=ids_to_move)
        y["train"][version] = y_train_numeric

    print(f"[NEW] Train size: {len(X['train']['numeric'])}, Test size: {len(X['test']['numeric'])}")


def save_results_to_csv(results: List[Dict], data_name: str, configs: Dict):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        data_name: Dataset name
        configs: Configuration dictionary
    """
    if len(results) == 0:
        print("[INFO] No results to save")
        return

    results_config = {
        "data": data_name,
        "feature_selection": configs["feature_selection"],
        "th_mi": configs["th_mi"],
        "th_perm": configs["th_perm"],
        "n_cv": configs["n_cv"],
        "test-size": configs.get("test-size", "Ignore"),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    df = pd.DataFrame(results)
    for key, value in results_config.items():
        df[key] = value

    # Move 'd' column to the end
    df = df[[col for col in df.columns if col != "d"] + ["d"]]

    output_path = f"results/{data_name}_perf.csv"
    
    if not os.path.exists(output_path):
        df.to_csv(output_path, index=False)
    else:
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(output_path, index=False)

    print(f"[INFO] Results saved to {output_path}")


@hydra.main(config_path="../configs", config_name="prediction", version_base=None)
def main(args: DictConfig):
    """
    Main execution pipeline for model training and evaluation.
    
    Args:
        args: Hydra configuration object
    """
    global X, y

    # Extract configuration
    versions = args.versions
    strategy = args.strategy
    data_name = args.data.name
    n_max = args.n_max
    n_max_test = args.n_max_test
    model = args.model
    tag = args.tag

    configs = {
        "compute_perf": True,
        "feature_selection": tag,
        "compute_perm_imp": False,
        "compute_mi": False,
        "th_mi": 0.02,
        "th_perm": 0.00,
        "imp_cache": True,
        "n_cv": 0,
        "test-size": None,
    }

    results = []

    # Initialize data structures
    X = {"train": {}, "test": {}}
    y = {"train": {}, "test": {}}
    common_ids = {"train": {}, "test": {}}

    # Load data for all versions
    print(f"N_MAX for {data_name}: {n_max}")
    
    print(versions)
    single_versions = []
    for version in versions:
        if "+" not in version:
            single_versions.append(version)
        else:
            version_list = version.split("+")
            for v in version_list:
                if v not in single_versions:
                    single_versions.append(v)
    print(single_versions)
    for version in single_versions:
        if version in X["train"].keys():
            print(f"[INFO] Skipping {version}. Already loaded.")
            continue

        train_data, test_data = load_data_for_version(
            data_name, version, strategy, n_max, n_max_test, args.ignore_cache
        )
        
        if train_data[0] is None or test_data[0] is None:
            continue

        X["train"][version], y["train"][version], common_ids["train"][version] = train_data
        X["test"][version], y["test"][version], common_ids["test"][version] = test_data
    print(X["train"].keys())
    # Align data by common IDs
    align_data_by_common_ids(X, y, common_ids, versions)
    print(X["train"].keys())


    # Validate data shapes
    validate_data_shapes(X, y, versions)

    # Rebalance train/test split
    rebalance_train_test_split(X, y, versions)

    # Calculate training sizes
    size_train = len(X["train"]["numeric"])
    sizes = {data_name: [int(x * size_train) for x in args.sizes]}
    print(f"Training sizes: {sizes}")

    # Train models for all versions
    pbar = tqdm(versions, desc="Processing versions", leave=True)

    for version in pbar:
        pbar.set_description(f"Training on {version}")

        if configs["compute_perf"]:
            for size in sizes[data_name]:
                new_result = create_run2(
                    data_name,
                    version,
                    model,
                    "OLD",
                    size,
                    n_cv=configs["n_cv"],
                    selected_features=[],
                    feature_selection=configs["feature_selection"],
                )
                
                # Add strategy to version name if applicable
                if (
                    strategy != ""
                    and new_result[0]["version"] != "numeric"
                    and "_bl" not in new_result[0]["version"]
                ):
                    for r in new_result:
                        r["version"] = f"{r['version']}({strategy})"
                
                results.extend(new_result)

    # Save results
    save_results_to_csv(results, data_name, configs)


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
    print(sys.path)

    load_dotenv(override=True)
    random.seed(DEFAULT_RANDOM_SEED)

    main()
