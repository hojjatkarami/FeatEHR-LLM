from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def get_all_measurements(df: pd.DataFrame, col_name: str) -> List[Dict[str, float]]:
    """
    Extracts chronological measurements for a specific column in the patient dataframe.
    
    Handles irregularly sampled EHR time series by dropping NaN values and returning
    only the observed timestamp-value pairs.
    
    Args:
        df: Patient time series DataFrame containing a 'Hours' column and physiological variables.
        col_name: Name of the variable/column to extract.
        
    Returns:
        List of dictionaries containing 'time' and 'value', sorted chronologically.
        Example: [{'time': 0.0, 'value': 72.0}, {'time': 1.5, 'value': 75.0}, ...]
        Returns [] if column does not exist or has no valid measurements.
    """
    if col_name not in df.columns or "Hours" not in df.columns:
        return []

    measurements = df[["Hours", col_name]].dropna()
    if measurements.empty:
        return []

    output = [
        {"time": float(row["Hours"]), "value": float(row[col_name])}
        for _, row in measurements.iterrows()
    ]
    return sorted(output, key=lambda m: m["time"])


def is_variable_increasing(df: pd.DataFrame, col_name: str) -> bool:
    """
    Checks if a variable shows an overall statistically increasing trend over time.
    
    Clinical Importance: Identifies worsening physiological states (e.g., rising lactate
    indicating inadequate tissue perfusion or rising creatinine indicating renal failure).
    
    Args:
        df: Patient time series DataFrame.
        col_name: Name of the variable to analyze.
        
    Returns:
        True if the variable has a positive linear slope over time, otherwise False.
    """
    measurements = get_all_measurements(df, col_name)
    if len(measurements) < 2:
        return False

    times = [m["time"] for m in measurements]
    values = [m["value"] for m in measurements]

    # Guard against identical timestamps
    if len(set(times)) < 2:
        return False

    try:
        slope = np.polyfit(times, values, 1)[0]
        return bool(slope > 0)
    except (np.linalg.LinAlgError, ValueError):
        return False


def is_variable_decreasing(df: pd.DataFrame, col_name: str) -> bool:
    """
    Checks if a variable shows an overall statistically decreasing trend over time.
    
    Clinical Importance: Identifies physiological decline or response to therapy
    (e.g., dropping platelet counts in DIC/sepsis, or declining blood pressure in shock).
    
    Args:
        df: Patient time series DataFrame.
        col_name: Name of the variable to analyze.
        
    Returns:
        True if the variable has a negative linear slope over time, otherwise False.
    """
    measurements = get_all_measurements(df, col_name)
    if len(measurements) < 2:
        return False

    times = [m["time"] for m in measurements]
    values = [m["value"] for m in measurements]

    if len(set(times)) < 2:
        return False

    try:
        slope = np.polyfit(times, values, 1)[0]
        return bool(slope < 0)
    except (np.linalg.LinAlgError, ValueError):
        return False


def is_variable_unstable(df: pd.DataFrame, col_name: str) -> bool:
    """
    Determines if a variable is fluctuating significantly over time (coefficient of variation > 30%).
    
    Clinical Importance: Detects hemodynamic or metabolic instability in vital signs
    (e.g., labile blood pressure or erratic glucose levels in the ICU).
    
    Args:
        df: Patient time series DataFrame.
        col_name: Name of the variable to analyze.
        
    Returns:
        True if the coefficient of variation (std / |mean|) exceeds 0.3, otherwise False.
    """
    measurements = get_all_measurements(df, col_name)
    if len(measurements) < 2:
        return False

    values = np.array([m["value"] for m in measurements], dtype=float)
    mean_val = np.mean(values)
    std_val = np.std(values)

    if np.abs(mean_val) < 1e-8:
        # If mean is near zero, check absolute standard deviation
        return bool(std_val > 0.3)

    cv = std_val / np.abs(mean_val)
    return bool(cv > 0.3)


def count_missing_intervals(df: pd.DataFrame, col_name: str) -> int:
    """
    Counts the number of unusually large measurement gaps in an irregular time series.
    
    A gap is defined as a measurement interval exceeding twice the median observation interval.
    Clinical Importance: Captures monitoring gaps, observation irregularity, and structural sparsity.
    
    Args:
        df: Patient time series DataFrame.
        col_name: Name of the variable to analyze.
        
    Returns:
        Integer count of missing intervals/gaps.
    """
    measurements = get_all_measurements(df, col_name)
    if len(measurements) < 2:
        return 0

    times = sorted([m["time"] for m in measurements])
    diffs = np.diff(times)
    if len(diffs) == 0:
        return 0

    median_diff = np.median(diffs)
    if median_diff <= 0:
        return 0

    threshold = median_diff * 2.0
    return int(np.sum(diffs > threshold))


def is_variable_outlier_prone(df: pd.DataFrame, col_name: str) -> bool:
    """
    Checks if a variable exhibits extreme physiological outliers using the IQR criterion.
    
    Identifies values falling beyond [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Clinical Importance: Detects extreme physiological crises or acute spikes (e.g. hyperkalemia crisis, hypertensive urgency).
    
    Args:
        df: Patient time series DataFrame.
        col_name: Name of the variable to analyze.
        
    Returns:
        True if at least one outlier is detected, otherwise False.
    """
    measurements = get_all_measurements(df, col_name)
    if len(measurements) < 2:
        return False

    values = np.array([m["value"] for m in measurements], dtype=float)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    if iqr <= 1e-8:
        # All values identical or nearly identical
        return False

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return bool(np.any((values < lower_bound) | (values > upper_bound)))
