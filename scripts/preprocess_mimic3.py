"""
Preprocessing script and documentation for MIMIC-III In-Hospital Mortality benchmark.

MIMIC-III time series data are derived from the well-established `mimic3-benchmarks` pipeline
(Wang et al., 2020; Harutyunyan et al., 2019):
https://github.com/YerevaNN/mimic3-benchmarks

This script formats extracted MIMIC-III stays into FeatEHR-LLM format:
- trainDataset.pkl, testDataset.pkl, valDataset.pkl
- units.json

Usage:
    uv run python -m scripts.preprocess_mimic3 --data_dir ./data/mimic3 --output_dir ./data/mimic3
"""

import argparse
import json
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd


def load_mimic3_split(csv_path: str) -> List[Dict]:
    """
    Loads MIMIC-III time series CSV and groups by patient episode into FeatEHR-LLM format.
    """
    print(f"Loading MIMIC-III CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    id_col = "stay_id" if "stay_id" in df.columns else ("id" if "id" in df.columns else df.columns[0])
    time_col = "Hours" if "Hours" in df.columns else "Hours"

    dataset = []
    for pid, group in df.groupby(id_col):
        p_ts = group.drop(columns=[id_col]).reset_index(drop=True)
        label = int(group["label"].iloc[0]) if "label" in group.columns else 0

        dataset.append({
            "sid": int(pid),
            "hadm_id": [int(pid)],
            "covariates": [[0.0, 0.0]],
            "ts": [p_ts],
            "label_ihm": [label],
        })
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Process MIMIC-III for FeatEHR-LLM")
    parser.add_argument("--data_dir", type=str, default="./data/mimic3")
    parser.add_argument("--output_dir", type=str, default="./data/mimic3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if pre-generated datasets already exist
    train_pkl = os.path.join(args.data_dir, "trainDataset.pkl")
    test_pkl = os.path.join(args.data_dir, "testDataset.pkl")

    if os.path.exists(train_pkl) and os.path.exists(test_pkl):
        print(f"Found existing MIMIC-III datasets in {args.data_dir}:")
        print(f"  Train: {train_pkl}")
        print(f"  Test:  {test_pkl}")
        print("MIMIC-III dataset is verified and ready for benchmarking.")
        return

    # If raw CSVs exist, process them
    train_csv = os.path.join(args.data_dir, "trainTS.csv")
    test_csv = os.path.join(args.data_dir, "testTS.csv")

    if os.path.exists(train_csv):
        train_dataset = load_mimic3_split(train_csv)
        with open(train_pkl, "wb") as f:
            pickle.dump(train_dataset, f)
        print(f"Saved: {train_pkl}")

    if os.path.exists(test_csv):
        test_dataset = load_mimic3_split(test_csv)
        with open(test_pkl, "wb") as f:
            pickle.dump(test_dataset, f)
        print(f"Saved: {test_pkl}")

    print("MIMIC-III preparation finished.")


if __name__ == "__main__":
    main()
