# FeatEHR-LLM: Leveraging Large Language Models for Feature Engineering in Electronic Health Records

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Paper:** *FeatEHR-LLM: Leveraging Large Language Models for Feature Engineering in Electronic Health Records*  
**Authors:** Hojjat Karami, David Atienza, Jean-Philippe Thiran, Anisoara Ionescu (EPFL)  
**Code Repository:** [https://github.com/hojjatkarami/FeatEHR-LLM](https://github.com/hojjatkarami/FeatEHR-LLM)

---

## Abstract

Feature engineering plays a pivotal role in machine learning by transforming raw data into informative inputs that enhance model performance. Traditional approaches, while effective, often demand domain expertise, rely on predefined transformations, and struggle with scalability. Recent advances in Large Language Models (LLMs) offer a promising alternative, leveraging their internal knowledge and code generation capabilities to automate and enrich feature creation. However, their application to Electronic Health Records (EHR)—particularly irregularly sampled time series—remains underexplored due to challenges such as variable measurement frequencies and missing values.

We introduce **FeatEHR-LLM**, a framework that harnesses LLMs for feature engineering in EHR data, focusing on irregularly sampled time series. FeatEHR-LLM guides LLMs to propose clinically meaningful features through univariate and multivariate transformations. Crucially, the framework equips the LLM with specialized programmatic tools to query and aggregate irregular temporal data without requiring patient-level records in prompts. Across eight clinical classification tasks on four ICU datasets, our approach outperforms standard methods, yielding up to a 6-point improvement in AUROC.

---

## Repository Structure

```
FeatEHR-LLM/
├── configs/                  # Hydra configuration files
│   ├── data/                 # Dataset configurations (P12, P19, MIMIC-III, eICU)
│   ├── tools/                # Toolset definitions (minimal.yaml, all.yaml)
│   ├── single.yaml           # Univariate feature generation configuration
│   ├── multi.yaml            # Multivariate feature generation configuration
│   └── prediction.yaml       # Downstream prediction & LightGBM tuning config
├── docs/
│   └── tools.md              # Detailed documentation of programmatic temporal tools
├── scripts/
│   ├── tools.py              # Specialized temporal querying tools
│   ├── run_single.py         # Univariate feature generation pipeline
│   ├── run_multi.py          # Multivariate feature generation pipeline
│   ├── extract_features.py   # Vectorized feature execution over patient cohorts
│   ├── run_prediction.py     # Downstream LightGBM training + bootstrap CI evaluation
│   ├── preprocess_p12.py     # Automated preprocessing for PhysioNet 2012
│   ├── preprocess_p19.py     # Automated preprocessing for PhysioNet 2019
│   ├── preprocess_mimic3.py  # Preprocessing wrapper for MIMIC-III benchmark
│   ├── run.sh                # End-to-end execution script
│   └── ablation.sh           # Ablation study execution script
├── tests/
│   └── test_tools.py         # Unit tests verifying temporal querying tools
├── prepare_eicu.ipynb        # eICU data preparation using FIDDLE
├── pyproject.toml            # Project dependencies and packaging
├── uv.lock                   # Deterministic lockfile
└── README.md
```

---

## Setup Instructions

### 1. Environment Installation (using `uv`)

We use [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency resolution:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.12 and sync virtual environment
uv python install 3.12
uv sync
```

Verify installation and run unit tests:
```bash
uv run python tests/test_tools.py
# Output: All unit tests passed successfully!
```

### 2. Environment Variables & API Keys

Create a `.env` file in the repository root:
```bash
cp .env.example .env
```
Populate `.env` with your API keys:
```env
GEMINI_API_KEY="your-gemini-api-key"
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
WANDB_API_KEY="your-wandb-api-key"  # Optional, for experiment tracking
```

---

## Datasets and Preprocessing

FeatEHR-LLM is evaluated across **4 ICU datasets** encompassing **8 clinical tasks**:

| Dataset | Clinical Task | Cohort Size | Variables | Observation Horizon |
|:---|:---|:---:|:---:|:---:|
| **P12** | In-Hospital Mortality | ~12,000 | 35 | 48 hours |
| **P19** | Sepsis Prediction | ~40,000 | 34 | First 48 hours |
| **MIMIC-III** | In-Hospital Mortality | ~29,000 | 41 | 48 hours |
| **eICU** | Mortality / ARF / Shock | 77k – 145k | 95 | 12h / 48h |

### Downloading Preprocessed Datasets
Preprocessed datasets can be downloaded directly and placed in `./data/`:
- **P12:** [Download P12](https://drive.google.com/drive/folders/1x0KJT6wnOD3cWBGBN6o7qO67Y-dRqNQs?usp=drive_link) $\rightarrow$ place in `data/p12-real/split0/`
- **P19:** [Download P19](https://drive.google.com/drive/folders/1St_oVmj-64qygP3wFalgXNBhsQ76E8JP?usp=drive_link) $\rightarrow$ place in `data/p19v2-real/split0/`
- **MIMIC-III:** [Download MIMIC-III](https://drive.google.com/drive/folders/1jSqeJATcwVthZwzPvTqKMXT7VeQFSg4v?usp=drive_link) $\rightarrow$ place in `data/mimic3/`
- **eICU:** Follow [`prepare_eicu.ipynb`](prepare_eicu.ipynb) to process raw eICU-CRD CSVs with FIDDLE.

If data is stored in a separate directory, create a symlink:
```bash
ln -s /path/to/your/data ./data
```

### Preprocessing Raw Challenges (from Scratch)
To construct the datasets from raw challenge CSVs:
```bash
# Preprocess PhysioNet 2012 Challenge:
uv run python -m scripts.preprocess_p12 --data_dir ./data/p12-real/split0 --output_dir ./data/p12-real/split0

# Preprocess PhysioNet 2019 Challenge:
uv run python -m scripts.preprocess_p19 --data_dir ./data/p19v2-real/split0 --output_dir ./data/p19v2-real/split0

# Process MIMIC-III benchmark:
uv run python -m scripts.preprocess_mimic3 --data_dir ./data/mimic3 --output_dir ./data/mimic3
```

---

## Programmatic Temporal Tools (`scripts.tools`)

To enable LLMs to reason over irregularly sampled, sparse time series without crashing, FeatEHR-LLM provides 6 temporal querying primitives. For full mathematical descriptions and clinical rationales, see [`docs/tools.md`](docs/tools.md).

1. `get_all_measurements(df, col_name)`: Returns chronological `[{'time': t, 'value': x}, ...]`, dropping missing timestamps.
2. `is_variable_increasing(df, col_name)`: Evaluates positive linear slope over non-uniform observation timestamps.
3. `is_variable_decreasing(df, col_name)`: Evaluates negative linear slope over non-uniform observation timestamps.
4. `is_variable_unstable(df, col_name)`: Flags severe volatility ($\text{CV} = \sigma / |\mu| > 0.30$).
5. `count_missing_intervals(df, col_name)`: Counts gaps exceeding $2\times$ the median sampling interval.
6. `is_variable_outlier_prone(df, col_name)`: Detects acute physiological crises using the IQR criterion ($1.5 \times \text{IQR}$).

---

## Step-by-Step Reproduction Guide

### Step 1: Feature Generation (via LLM)

Run univariate and multivariate feature proposal routines:

```bash
# 1. Single-variable feature generation (P12)
uv run python -m scripts.run_single \
    data=p12 \
    version="v1" \
    llm.model="gemini-2.0-flash"

# 2. Multi-variable feature generation (P12)
uv run python -m scripts.run_multi \
    data=p12 \
    version="v1" \
    llm.model="gemini-2.0-flash"
```
*Generated Python functions are written to `results/{data}/{version}/features.csv`.*

### Step 2: Feature Extraction

Execute the synthesized functions across patient records in parallel:

```bash
for split in "train" "test"; do
    uv run python -m scripts.extract_features \
        data=p12 \
        split="${split}" \
        version="v1"
done
```
*Extracted feature matrices are saved in `data/featehr-llm/{data}/{split}/{version}/data.csv`.*

### Step 3: Downstream Prediction & Stratified Bootstrap

Train LightGBM classifiers with hyperparameter tuning and estimate 95% bootstrap confidence intervals ($n_{boot}=1000$):

```bash
uv run python -m scripts.run_prediction \
    data=p12 \
    tag=final \
    versions=final
```

### Reproduce Paper Ablation Studies

To reproduce the ablation study (Table 3):
```bash
bash scripts/ablation.sh
```

---

## Computational Efficiency & Resource Footprint

Unlike deep sequence models that require 12–48 GPU hours per dataset, FeatEHR-LLM operates on lightweight metadata prompts, achieving:
- **Local GPU Hours:** **0 hours** (pure CPU execution for extraction and training).
- **Peak RAM:** **< 4.0 GB** (in contrast to > 16 GB for `tsfresh`).
- **LLM Token Cost:** **< $0.20 per dataset** (prompted once per schema, not per patient).
- **Wall-Clock Training Time:** **10 – 30 seconds** with LightGBM.

---

## Citation

If you use FeatEHR-LLM in your research, please cite our manuscript:

```bibtex
@article{karami2024featehrllm,
  title={FeatEHR-LLM: Leveraging Large Language Models for Feature Engineering in Electronic Health Records},
  author={Karami, Hojjat and Atienza, David and Thiran, Jean-Philippe and Ionescu, Anisoara},
  journal={Procedia Computer Science},
  year={2024}
}
```
