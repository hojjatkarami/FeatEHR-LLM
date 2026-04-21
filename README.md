# FeatEHR-LLM

**Paper:** FeatEHR-LLM: Leveraging Large Language Models for Feature Engineering in Electronic Health Records

## Abstract

Feature engineering plays a pivotal role in machine learning by transforming raw data into informative inputs that enhance model performance. Traditional approaches, while effective, often demand domain expertise, rely on predefined transformations, and struggle with scalability. Recent advances in Large Language Models (LLMs) offer a promising alternative, leveraging their internal knowledge and code generation capabilities to automate and enrich feature creation. However, their application to Electronic Health Records (EHR)—particularly irregularly sampled time series—remains underexplored due to challenges such as variable measurement frequencies and missing values.

We introduce **FeatEHR-LLM**, a framework that harnesses LLMs for feature engineering in EHR data, focusing on irregularly sampled time series. FeatEHR-LLM guides LLMs to propose clinically meaningful features through univariate and multivariate transformations. Across eight clinical classification tasks, our approach outperforms standard methods, yielding up to a 6-point improvement in AUROC. These results highlight the potential of LLMs to transform clinical time series data into actionable insights, enabling more accurate and efficient healthcare predictions.

**Code:** https://github.com/hojjatkarami/FeatEHR-LLM

---

## Overview

FeatEHR-LLM is a feature extraction and prediction framework for EHR (Electronic Health Records) using LLMs and baseline methods.

## Setup Instructions

### 1. Environment Setup

Install `uv` if you do not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```


```bash
uv python install 3.12
uv sync
```

### 2. Data Symlink and Datasets

```bash
ln -s /home/hokarami/data .
```
- See `prepare_eicu.ipynb` for how to prepare the eICU data.
- Download preprocessed datasets:
    - [mimic3](https://drive.google.com/drive/folders/1jSqeJATcwVthZwzPvTqKMXT7VeQFSg4v?usp=drive_link)
    - [p12](https://drive.google.com/drive/folders/1x0KJT6wnOD3cWBGBN6o7qO67Y-dRqNQs?usp=drive_link)
    - [p19](https://drive.google.com/drive/folders/1St_oVmj-64qygP3wFalgXNBhsQ76E8JP?usp=drive_link)
- Place all data in `data/` and update dataset-specific paths in `configs/data/*`.
### 3. Configuration

- Create a `.env` file in the root directory
- Use `.env.example` as a template

### 4. Activate Environment

```bash
source .venv/bin/activate
```

## Quick Start

### Step 1: Run univariate and multivariate feature generation

```bash
# Univariate feature generation
uv run python -m scripts.run_single \
    data="p12" \
    version="test-uni-gflash-25" \
    llm.model="google/gemini-2.5-flash"

# Multivariate feature generation
uv run python -m scripts.run_multi \
    data="p12" \
    version="test-multi-gflash-25" \
    llm.model="google/gemini-2.5-flash"
```

### Step 2: Run feature extraction

```bash

for version in "test-uni-gflash-25" "test-multi-gflash-25"; do
    for split in "train" "test"; do
        uv run python -m scripts.extract_features \
            data="p12" \
            split="${split}" \
            version="${version}"
    done
done

```

## Main Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run.sh` | Main entry point for training and experiments |
| `scripts/extract_features.py` | Extract features from EHR data |
| `scripts/run_prediction.py` | Run prediction tasks |
| `scripts/run_single.py` | Run single-task feature generation |
| `scripts/run_multi.py` | Run multi-task feature generation |
| `scripts/run_zero.py` | Run zero-shot experiments |
| `scripts/run_bl.py` | Run baseline methods |

## Usage Examples

### Run Full Pipeline

To run the complete workflow (feature design → feature extraction → prediction):

```bash
uv run bash scripts/run.sh
```

### Detailed Workflow

#### A. Feature Design

Generate feature functions using LLM with single-task and multi-task approaches:

```bash
# Single-variable features
uv run python -m scripts.run_single \
    data="$DATA" \
    version=s1 \
    llm.model="$LLM_MODEL"

# Multi-variable features
uv run python -m scripts.run_multi \
    data="$DATA" \
    version=m1 \
    llm.model="$LLM_MODEL"
```

**Config files:** `configs/single.yaml`, `configs/multi.yaml`

**Output:** Features are written to `results/{data}/{s1|m1}/features.csv`

#### B. Feature Extraction

Extract features from train and test splits:

```bash
uv run python -m scripts.extract_features \
    data="$DATA" \
    split=train \
    version=m1
```

**Output:** Extracted features are written to `data/featehr-llm/{data}/{split}/{version}/data.csv`

#### C. Prediction

Run downstream prediction tasks using feature combinations:

```bash
uv run python -m scripts.run_prediction \
    data="$DATA" \
    tag=final \
    versions=final
```

**Feature combinations** (defined in `configs/versions/final.yaml`):
- `s1`: Single-variable features
- `numeric+s1`: Single-variable + baseline features
- `m1`: Multi-variable features
- `numeric+m1`: Multi-variable + baseline features
- `numeric+s1+m1`: All feature types combined

### Other Common Commands

Extract features:
```bash
uv run python -m scripts.extract_features --config configs/extract.yaml
```

Run prediction:
```bash
uv run python -m scripts.run_prediction --config configs/prediction.yaml
```

Run baselines:
```bash
uv run bash baselines/run.sh
```

Run ablation study:
```bash
uv run bash scripts/ablation.sh
```

## Configuration Files

All configurations are in the `configs/` directory. Modify YAML files to customize experiments.

## Outputs & Results

- **Results:** Saved in `results/` and `outputs/` directories
- **Logs and intermediate files:** Stored in `wandb/` and `outputs/`

## Utilities

- `scripts/tools.py`, `scripts/utils.py`: Helper scripts for data processing and experiments

## Citation

If you use FeatEHR-LLM, please cite the corresponding paper.

---

For more details, see comments in scripts and config files.
