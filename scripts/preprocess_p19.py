"""
Preprocessing script for PhysioNet 2019 Sepsis Prediction Challenge (P19).

Converts raw/tabular P19 patient stays into standardized FeatEHR-LLM format:
- trainDataset.pkl / testDataset.pkl (list of patient dicts)
- units.json (variable metadata, units, and summary statistics)

Usage:
    uv run python -m scripts.preprocess_p19 --data_dir ./data/p19v2-real/split0 --output_dir ./data/p19v2-real/split0
"""

import argparse
import json
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_patient_dataset_p19(
    df_ts: pd.DataFrame, df_static: pd.DataFrame, record_ids: List[int]
) -> List[Dict]:
    """
    Constructs FeatEHR-LLM patient records for a list of P19 RecordIDs.
    """
    id_col = "RecordID" if "RecordID" in df_ts.columns else "id"
    time_col = "Time" if "Time" in df_ts.columns else "Hours"

    # Rename Time to Hours
    df_ts = df_ts.rename(columns={time_col: "Hours"})

    df_ts_grouped = dict(tuple(df_ts.groupby(id_col)))
    static_dict = df_static.set_index(id_col).to_dict(orient="index")

    dataset = []
    for rid in record_ids:
        if rid not in df_ts_grouped or rid not in static_dict:
            continue

        p_ts = df_ts_grouped[rid].drop(columns=[id_col]).reset_index(drop=True)
        stat = static_dict[rid]

        age = float(stat.get("Age", 0.0))
        gender = float(stat.get("Gender", 0.0))
        label = int(stat.get("Label", stat.get("outcome", 0)))

        record = {
            "sid": int(rid),
            "hadm_id": [int(rid)],
            "covariates": [[age, gender]],
            "ts": [p_ts],
            "label_ihm": [label],
        }
        dataset.append(record)

    return dataset


def generate_units_json_p19(df_ts: pd.DataFrame) -> Dict[str, Dict]:
    """
    Generates summary statistics for P19 time series variables.
    """
    exclude_cols = {"RecordID", "id", "Time", "Hours"}
    var_cols = [c for c in df_ts.columns if c not in exclude_cols]
    id_col = "RecordID" if "RecordID" in df_ts.columns else "id"

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
            "n_avg": float(np.round(len(series) / df_ts[id_col].nunique(), 1)),
            "h_avg": 48.0,
        }
    return units_meta


def main():
    parser = argparse.ArgumentParser(description="Preprocess P19 dataset for FeatEHR-LLM")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/p19v2-real/split0",
        help="Directory containing df_ts.csv and df_static.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/p19v2-real/split0",
        help="Directory to save preprocessed pkl and json files",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ts_path = os.path.join(args.data_dir, "df_ts.csv")
    static_path = os.path.join(args.data_dir, "df_static.csv")

    print(f"Loading P19 time series: {ts_path}")
    df_ts = pd.read_csv(ts_path)
    print(f"Loading P19 static data: {static_path}")
    df_static = pd.read_csv(static_path)

    id_col = "RecordID" if "RecordID" in df_static.columns else "id"
    label_col = "Label" if "Label" in df_static.columns else "outcome"

    all_ids = df_static[id_col].unique().tolist()
    labels = df_static.set_index(id_col).loc[all_ids, label_col].values

    train_ids, test_ids = train_test_split(
        all_ids, test_size=args.test_size, random_state=args.seed, stratify=labels
    )

    print(f"Building P19 train dataset ({len(train_ids)} patients)...")
    train_dataset = build_patient_dataset_p19(df_ts, df_static, train_ids)
    train_out = os.path.join(args.output_dir, "trainDataset.pkl")
    with open(train_out, "wb") as f:
        pickle.dump(train_dataset, f)
    print(f"Saved: {train_out}")

    print(f"Building P19 test dataset ({len(test_ids)} patients)...")
    test_dataset = build_patient_dataset_p19(df_ts, df_static, test_ids)
    test_out = os.path.join(args.output_dir, "testDataset.pkl")
    with open(test_out, "wb") as f:
        pickle.dump(test_dataset, f)
    print(f"Saved: {test_out}")

    units_out = os.path.join(args.output_dir, "units.json")
    if not os.path.exists(units_out):
        print("Generating units metadata...")
        units_meta = generate_units_json_p19(df_ts)
        with open(units_out, "w") as f:
            json.dump(units_meta, f, indent=2)
        print(f"Saved: {units_out}")

    print("P19 preprocessing finished successfully.")


if __name__ == "__main__":
    main()
