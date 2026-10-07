"""
Preprocessing script for PhysioNet 2012 Mortality Prediction Challenge (P12).

Converts raw/tabular P12 patient trajectories into standardized FeatEHR-LLM format:
- trainDataset.pkl / testDataset.pkl (list of patient dicts)
- units.json (variable metadata, units, and summary statistics)

Usage:
    uv run python -m scripts.preprocess_p12 --data_dir ./data/p12-real/split0 --output_dir ./data/p12-real/split0
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_patient_dataset(
    df_ts: pd.DataFrame, df_static: pd.DataFrame, patient_ids: List[int]
) -> List[Dict]:
    """
    Constructs FeatEHR-LLM patient records for a list of patient IDs.
    """
    # Rename timepoint column to Hours if needed
    if "timepoint" in df_ts.columns and "Hours" not in df_ts.columns:
        df_ts = df_ts.rename(columns={"timepoint": "Hours"})

    df_ts_grouped = dict(tuple(df_ts.groupby("id")))
    static_dict = df_static.set_index("id").to_dict(orient="index")

    dataset = []
    for pid in patient_ids:
        if pid not in df_ts_grouped or pid not in static_dict:
            continue

        p_ts = df_ts_grouped[pid].drop(columns=["id"]).reset_index(drop=True)
        stat = static_dict[pid]

        age = float(stat.get("Age", 0.0))
        gender = float(stat.get("Gender", 0.0))
        outcome = int(stat.get("outcome", 0))

        record = {
            "sid": int(pid),
            "hadm_id": [int(pid)],
            "covariates": [[age, gender]],
            "ts": [p_ts],
            "label_ihm": [outcome],
        }
        dataset.append(record)

    return dataset


def generate_units_json(df_ts: pd.DataFrame) -> Dict[str, Dict]:
    """
    Computes statistical metadata (mean, std, quantiles) for all time series variables.
    """
    exclude_cols = {"id", "timepoint", "Hours"}
    var_cols = [c for c in df_ts.columns if c not in exclude_cols]

    units_meta = {}
    for col in var_cols:
        series = df_ts[col].dropna()
        if len(series) == 0:
            continue

        units_meta[col] = {
            "full_name": col,
            "unit": "standard",
            "description": f"Physiological measurement: {col}",
            "mean": float(np.round(series.mean(), 2)),
            "std": float(np.round(series.std(), 2)),
            "quantile2.5": float(np.round(np.percentile(series, 2.5), 2)),
            "quantile97.5": float(np.round(np.percentile(series, 97.5), 2)),
            "n_avg": float(np.round(len(series) / df_ts["id"].nunique(), 1)),
            "h_avg": 48.0,
        }
    return units_meta


def main():
    parser = argparse.ArgumentParser(description="Preprocess P12 dataset for FeatEHR-LLM")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/p12-real/split0",
        help="Directory containing df_ts.csv and df_static.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/p12-real/split0",
        help="Directory to save preprocessed pkl and json files",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ts_path = os.path.join(args.data_dir, "df_ts.csv")
    static_path = os.path.join(args.data_dir, "df_static.csv")

    print(f"Loading time series from: {ts_path}")
    df_ts = pd.read_csv(ts_path)
    print(f"Loading static data from: {static_path}")
    df_static = pd.read_csv(static_path)

    all_ids = df_static["id"].unique().tolist()
    labels = df_static.set_index("id").loc[all_ids, "outcome"].values

    train_ids, test_ids = train_test_split(
        all_ids, test_size=args.test_size, random_state=args.seed, stratify=labels
    )

    print(f"Building train dataset ({len(train_ids)} patients)...")
    train_dataset = build_patient_dataset(df_ts, df_static, train_ids)
    train_out = os.path.join(args.output_dir, "trainDataset.pkl")
    with open(train_out, "wb") as f:
        pickle.dump(train_dataset, f)
    print(f"Saved: {train_out}")

    print(f"Building test dataset ({len(test_ids)} patients)...")
    test_dataset = build_patient_dataset(df_ts, df_static, test_ids)
    test_out = os.path.join(args.output_dir, "testDataset.pkl")
    with open(test_out, "wb") as f:
        pickle.dump(test_dataset, f)
    print(f"Saved: {test_out}")

    units_out = os.path.join(args.output_dir, "units.json")
    if not os.path.exists(units_out):
        print(f"Generating units metadata...")
        units_meta = generate_units_json(df_ts)
        with open(units_out, "w") as f:
            json.dump(units_meta, f, indent=2)
        print(f"Saved: {units_out}")

    print("P12 preprocessing finished successfully.")


if __name__ == "__main__":
    main()
