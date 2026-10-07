#!/bin/bash
set -e

# Navigate to project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"         


DATAS=(
    # "eicu_arf4"
    # "eicu_arf12"
    # "eicu_shock4"
    # "eicu_shock12"

    # "p12"
    # "mimic3"
    "p19"
)

for DATA in "${DATAS[@]}"; do


    # uv run python -m scripts.run_prediction data=$DATA tag=ab22 versions=ab2 strategy=first          
    # uv run python -m scripts.run_prediction data=$DATA tag=ab22 versions=ab2 strategy=best1
    # uv run python -m scripts.run_prediction data=$DATA tag=ab22 versions=ab2 strategy=best3          
    uv run python -m scripts.run_prediction data=$DATA tag=ab22 versions=ab2 strategy=""         



done
