# Specialized Programmatic Temporal Tools (`scripts.tools`)

## Overview

In irregularly sampled Electronic Health Record (EHR) time series, patients have non-uniform observation timestamps, varying sequence lengths, and structural sparsity (unobserved variables). Standard automated feature engineering methods typically assume dense tabular rows or regular time intervals, and LLMs prompted without assistance frequently synthesize naive pandas/numpy operations that fail or crash on missing values.

To resolve this challenge, **FeatEHR-LLM** equips the LLM with a specialized programmatic toolset (`scripts.tools`). These tools serve as foundational querying primitives that the LLM invokes inside generated Python functions.

---

## Tool Functions Reference

### 1. `get_all_measurements(df: pd.DataFrame, col_name: str) -> List[Dict[str, float]]`
- **Purpose:** Extracts chronological observations of a specific variable for a patient record, automatically discarding unobserved timestamps (`NaN`s).
- **Return Format:** Sorted list of dictionaries: `[{'time': float, 'value': float}, ...]`. Returns `[]` if no valid observations exist.
- **Clinical Role:** Provides safe, zero-crash variable access across sparse records.

### 2. `is_variable_increasing(df: pd.DataFrame, col_name: str) -> bool`
- **Purpose:** Computes a first-order linear trend ($\text{polyfit}(t, x, 1)$) over non-uniform observation times and checks whether the slope is positive ($> 0$).
- **Clinical Role:** Identifies acute clinical worsening trends, such as:
  - Rising arterial lactate levels indicating worsening septic shock or tissue hypoperfusion.
  - Rising serum creatinine indicating acute kidney injury (AKI).

### 3. `is_variable_decreasing(df: pd.DataFrame, col_name: str) -> bool`
- **Purpose:** Computes the linear slope over non-uniform observation times and checks whether the slope is negative ($< 0$).
- **Clinical Role:** Identifies acute decompensation or therapeutic response:
  - Declining mean arterial pressure (MAP) signaling distributive or cardiogenic shock.
  - Declining platelet counts signaling disseminated intravascular coagulation (DIC).

### 4. `is_variable_unstable(df: pd.DataFrame, col_name: str) -> bool`
- **Purpose:** Determines if a physiological signal displays severe temporal volatility, defined by a Coefficient of Variation ($\text{CV} = \sigma / |\mu| > 0.30$).
- **Clinical Role:** Detects hemodynamic instability or metabolic lability:
  - Fluctuating systolic blood pressure reflecting autonomic dysfunction or unstable titration of vasopressors.
  - Volatile blood glucose in diabetic ketoacidosis or septic shock.

### 5. `count_missing_intervals(df: pd.DataFrame, col_name: str) -> int`
- **Purpose:** Quantifies the number of unusually large gaps between consecutive measurements, where a gap is defined as $\Delta t > 2 \times \text{median}(\Delta t)$.
- **Clinical Role:** Encodes observation frequency and informative sampling:
  - Long monitoring gaps often reflect clinical assessments that the patient is stable.
  - Sudden reductions in gaps indicate intensive nurse/physician monitoring.

### 6. `is_variable_outlier_prone(df: pd.DataFrame, col_name: str) -> bool`
- **Purpose:** Evaluates whether measurements fall outside the interquartile range criterion: $[Q_1 - 1.5 \times \text{IQR},\, Q_3 + 1.5 \times \text{IQR}]$.
- **Clinical Role:** Detects extreme crises (e.g., hyperkalemia spikes $> 6.5\text{ mmol/L}$ risking fatal arrhythmias, hypertensive crises $> 200\text{ mmHg}$).

---

## Usage in Generated Feature Functions

During univariate and multivariate feature engineering, the LLM incorporates these tools directly into synthesized feature logic:

```python
from scripts.tools import get_all_measurements, is_variable_increasing, is_variable_unstable

def shock_index_instability(X: pd.DataFrame, covariates: dict = None) -> float:
    """
    Computes whether shock index (HR / SBP) shows worsening instability.
    """
    hr_obs = get_all_measurements(X, "HR")
    sbp_obs = get_all_measurements(X, "SysABP")
    
    if len(hr_obs) < 2 or len(sbp_obs) < 2:
        return 0.0
    
    # Check if HR is increasing while blood pressure is declining
    hr_rising = is_variable_increasing(X, "HR")
    hr_unstable = is_variable_unstable(X, "HR")
    
    return float(hr_rising and hr_unstable)
```
