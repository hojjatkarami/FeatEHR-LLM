"""
Unit tests for specialized temporal querying tools in scripts.tools.

Verifies correct behavior across irregular EHR sampling, missing values,
and edge cases.
"""

import os
import sys

# Ensure repository root is on sys.path for running tests directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from scripts.tools import (
    count_missing_intervals,
    get_all_measurements,
    is_variable_decreasing,
    is_variable_increasing,
    is_variable_outlier_prone,
    is_variable_unstable,
)


def test_get_all_measurements_basic():
    df = pd.DataFrame({
        "Hours": [0.0, 1.0, 2.5, 5.0],
        "HR": [70.0, np.nan, 85.0, 90.0],
    })
    measurements = get_all_measurements(df, "HR")
    assert len(measurements) == 3
    assert measurements[0] == {"time": 0.0, "value": 70.0}
    assert measurements[1] == {"time": 2.5, "value": 85.0}
    assert measurements[2] == {"time": 5.0, "value": 90.0}


def test_get_all_measurements_empty_and_missing():
    df = pd.DataFrame({"Hours": [0.0, 1.0], "HR": [np.nan, np.nan]})
    assert get_all_measurements(df, "HR") == []
    assert get_all_measurements(df, "NON_EXISTENT") == []


def test_is_variable_increasing():
    # Clearly increasing series
    df_inc = pd.DataFrame({
        "Hours": [0.0, 2.0, 4.0, 10.0],
        "Lactate": [1.1, 1.8, 2.5, 4.2],
    })
    assert is_variable_increasing(df_inc, "Lactate") is True
    assert is_variable_decreasing(df_inc, "Lactate") is False


def test_is_variable_decreasing():
    # Clearly decreasing series
    df_dec = pd.DataFrame({
        "Hours": [0.0, 3.0, 8.0, 12.0],
        "MAP": [95.0, 85.0, 72.0, 58.0],
    })
    assert is_variable_decreasing(df_dec, "MAP") is True
    assert is_variable_increasing(df_dec, "MAP") is False


def test_trends_insufficient_data():
    df_single = pd.DataFrame({"Hours": [1.0], "HR": [80.0]})
    assert is_variable_increasing(df_single, "HR") is False
    assert is_variable_decreasing(df_single, "HR") is False


def test_is_variable_unstable():
    # Stable series (CV < 0.3)
    df_stable = pd.DataFrame({
        "Hours": [0.0, 1.0, 2.0, 3.0],
        "Temp": [37.0, 37.1, 36.9, 37.2],
    })
    assert is_variable_unstable(df_stable, "Temp") is False

    # Highly volatile series (CV > 0.3)
    df_unstable = pd.DataFrame({
        "Hours": [0.0, 1.0, 2.0, 3.0],
        "Glucose": [50.0, 250.0, 60.0, 300.0],
    })
    assert is_variable_unstable(df_unstable, "Glucose") is True


def test_count_missing_intervals():
    # Regular spacing (1h apart) -> 0 gap intervals
    df_regular = pd.DataFrame({
        "Hours": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "HR": [70.0, 72.0, 75.0, 71.0, 73.0, 70.0],
    })
    assert count_missing_intervals(df_regular, "HR") == 0

    # Irregular spacing with a large gap: diffs = [1, 1, 12, 1, 1] -> median diff is 1, gap at 12
    df_gap = pd.DataFrame({
        "Hours": [0.0, 1.0, 2.0, 14.0, 15.0, 16.0],
        "HR": [70.0, 72.0, 75.0, 71.0, 73.0, 70.0],
    })
    assert count_missing_intervals(df_gap, "HR") == 1


def test_is_variable_outlier_prone():
    # Normal distribution, no outliers
    df_normal = pd.DataFrame({
        "Hours": list(range(10)),
        "K": [4.0, 4.0, 4.1, 4.1, 4.0, 4.0, 4.1, 4.1, 4.0, 4.0],
    })
    assert is_variable_outlier_prone(df_normal, "K") is False

    # Extreme spike: Potassium = 8.5 mmol/L (lethal hyperkalemia)
    df_outlier = pd.DataFrame({
        "Hours": list(range(10)),
        "K": [4.0, 4.1, 4.2, 3.9, 4.0, 4.1, 4.2, 3.8, 4.0, 8.5],
    })
    assert is_variable_outlier_prone(df_outlier, "K") is True


if __name__ == "__main__":
    test_get_all_measurements_basic()
    test_get_all_measurements_empty_and_missing()
    test_is_variable_increasing()
    test_is_variable_decreasing()
    test_trends_insufficient_data()
    test_is_variable_unstable()
    test_count_missing_intervals()
    test_is_variable_outlier_prone()
    print("All unit tests passed successfully!")
