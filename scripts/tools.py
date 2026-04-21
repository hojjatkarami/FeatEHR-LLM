from typing import Any, List

import numpy as np
import pandas as pd


import sys


def get_all_measurements(df: pd.DataFrame, col_name: str) -> List[dict]:
    """
    Extracts measurements from a specific column in the dataframe.
    Returns a list of dictionaries containing 'time' and 'value'.
    example output: [{'time': 0.0, 'value': 0.0}, {'time': 1.0, 'value': 1.0}, ...]
    if no measurements the output is []
    """
    measurements = df[["Hours", col_name]].dropna()
    output = [
        {"time": float(row["Hours"]), "value": float(row[col_name])}
        for _, row in measurements.iterrows()
    ]

    return output if output else []


def is_variable_increasing(df: pd.DataFrame, col_name: str) -> bool:
    """
    Checks if a variable shows an overall increasing trend.
    Returns True if the variable is increasing over time, otherwise False.
    Clinical Importance: Identifies worsening trends, such as rising lactate indicating poor perfusion.
    """
    measurements = get_all_measurements(df, col_name)
    if len(measurements) < 2:
        return False

    times, values = zip(*[(m["time"], m["value"]) for m in measurements])
    return np.polyfit(times, values, 1)[0] > 0  # Positive slope


def is_variable_decreasing(df: pd.DataFrame, col_name: str) -> bool:
    """
    Checks if a variable shows an overall decreasing trend.
    Returns True if the variable is decreasing over time, otherwise False.
    Clinical Importance: Identifies improving trends, such as decreasing CRP indicating reduced inflammation.
    """
    measurements = get_all_measurements(df, col_name)
    if len(measurements) < 2:
        return False

    times, values = zip(*[(m["time"], m["value"]) for m in measurements])
    return np.polyfit(times, values, 1)[0] < 0  # Negative slope


def is_variable_unstable(df: pd.DataFrame, col_name: str) -> bool:
    """
    Determines if a variable is fluctuating significantly over time.
    Returns True if the standard deviation is high relative to the mean.
    Clinical Importance: Detects instability in vital signs, such as fluctuating blood pressure indicating shock.
    """
    measurements = get_all_measurements(df, col_name)
    if len(measurements) < 2:
        return False

    values = [m["value"] for m in measurements]
    return np.std(values) / np.mean(values) > 0.3  # Coefficient of variation > 30%


def count_missing_intervals(df: pd.DataFrame, col_name: str) -> int:
    """
    Counts the number of missing measurement intervals.
    Returns the number of gaps in the time series where values are missing.
    Clinical Importance: Large gaps in measurements may indicate poor monitoring.
    """
    measurements = get_all_measurements(df, col_name)
    if len(measurements) < 2:
        return 0

    times = sorted([m["time"] for m in measurements])
    diffs = np.diff(times)
    threshold = np.median(diffs) * 2  # Define a gap as 2x the median time interval
    return np.sum(diffs > threshold)


def is_variable_outlier_prone(df: pd.DataFrame, col_name: str) -> bool:
    """
    Checks if a variable has extreme outliers using the IQR method.
    Returns True if there are significant outliers.
    Clinical Importance: Identifies erratic values, such as sudden spikes in glucose levels.
    """
    measurements = get_all_measurements(df, col_name)
    if len(measurements) < 2:
        return False

    values = np.array([m["value"] for m in measurements])
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return np.any((values < lower_bound) | (values > upper_bound))
