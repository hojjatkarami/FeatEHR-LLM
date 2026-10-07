#!/bin/bash
set -e

# Navigate to project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"


# Datasets to process
DATAS=(
    "p12"
    # "mimic3"
    # "p19"
    # "eicu_arf12"
    # "eicu_shock12"
    # "eicu_mo48"
    # "eicu_shock48"
    # "eicu_arf48"
    # "eicu_lung24"
)

BLS=(
    
    # "caafe"
    # "featllm"

    "openfe"
    # "autofeat"
)


for DATA in "${DATAS[@]}"; do



    ############# Zeroshot Baselines #############
        # # python -m scripts.run_zero \
        # #     data="$DATA" \
        # #     n_max=1000 \
        # #     llm.model="gemini-2.0-flash" #"o4-mini"

        # python -m scripts.run_prediction data="$DATA" tag=exp6-zeroshot n_max_test=1000 ignore_cache=True versions=exp6-zeroshot strategy=""

        # python -m scripts.run_prediction data="$DATA" tag=exp5-zeroshot n_max_test=1000 ignore_cache=True versions=exp5-zeroshot strategy=""



    ############# Feature Engineering Baselines #############
    for BL in "${BLS[@]}"; do

        python -m scripts.run_bl \
            data="$DATA" \
            baseline="$BL" 
            
    done



done