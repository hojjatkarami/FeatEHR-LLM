#!/bin/bash

# set -e  # Exit on error

# # Navigate to project directory and activate conda environment
# cd /home/hokarami/code/FeatEHR-LLM
# # conda env create -f environment.yml
# # source activate featehr-llm
source .venv/bin/activate

# =============================================================================
# Configuration
# =============================================================================

# Datasets to process
DATAS=(
    "p12"
    # "mimic3"
    # "p19"
    # "eicu_mo48"
    # "eicu_shock48"
    # "eicu_arf48"
    # "eicu_arf12"
    # "eicu_shock12"
    # "eicu_lung24"
)

# Multi-variable feature versions (5 replicates)
MULTI_VERSIONS=("m1" )

# Single-variable feature versions (5 replicates)
SINGLE_VERSIONS=( "s1" )



# LLM model configuration
# LLM_MODEL="gpt-4o-mini"  # or "o4-mini" or "gemini-2.0-flash" 
LLM_MODEL="google/gemini-2.5-flash"  # 
PREFIX="gflash-25"
# =============================================================================
# Main Processing Loop
# =============================================================================

for DATA in "${DATAS[@]}"; do
    echo "=========================================="
    echo "Processing dataset: $DATA"
    echo "=========================================="

    # -------------------------------------------------------------------------
    # Extract Multi-variable Features
    # -------------------------------------------------------------------------
    echo "Extracting multi-variable features..."

    for VERSION in "${MULTI_VERSIONS[@]}"; do
        FULL_VERSION="${PREFIX}-${VERSION}"
        echo "  Processing version: $FULL_VERSION"

        # Generate rules using LLM
        python -m scripts.run_multi \
            data="$DATA" \
            version="$FULL_VERSION" \
            llm.model="$LLM_MODEL"

        # # Extract features from train set
        # python -m scripts.extract_features \
        #     data="$DATA" \
        #     split=train \
        #     version="$FULL_VERSION"

        # # Extract features from test set
        # python -m scripts.extract_features \
        #     data="$DATA" \
        #     split=test \
        #     version="$FULL_VERSION"
    done

    # -------------------------------------------------------------------------
    # Extract Single-variable Features
    # -------------------------------------------------------------------------
    echo "Extracting single-variable features..."

    for VERSION in "${SINGLE_VERSIONS[@]}"; do
        FULL_VERSION="${PREFIX}-${VERSION}"
        echo "  Processing version: $FULL_VERSION"

        # Generate rules using LLM
        python -m scripts.run_single \
            data="$DATA" \
            version="$FULL_VERSION" \
            llm.model="$LLM_MODEL"

        # # Extract features from train set
        # python -m scripts.extract_features \
        #     data="$DATA" \
        #     split=train \
        #     version="$FULL_VERSION"

        # # Extract features from test set
        # python -m scripts.extract_features \
        #     data="$DATA" \
        #     split=test \
        #     version="$FULL_VERSION"
    done

    # -------------------------------------------------------------------------
    # Train Classifier and Evaluate
    # -------------------------------------------------------------------------
    echo "Training classifier and evaluating..."

    # python -m scripts.run_prediction \
    #     data="$DATA" \
    #     tag=final \
    #     versions=final \
    #     strategy=""

    echo "Completed dataset: $DATA"
    echo ""
done

echo "=========================================="
echo "All datasets processed successfully!"
echo "=========================================="


