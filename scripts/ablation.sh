#!/bin/bash

cd /home/hokarami/code/aaa/deepseek/paper4
source activate paper4

# python -m scripts.run_prediction data=p12 tag=ablation1 versions=mr strategy="first"


# python -m scripts.run_prediction data=mimic3 tag=ablation1 versions=mr strategy=""
# python -m scripts.run_prediction data=p19 tag=ablation1 versions=mr strategy=""

# ablation
# python -m scripts.run_prediction data=p12 tag=ablation1 versions=sr strategy=first          
# python -m scripts.run_prediction data=p12 tag=ablation1 versions=sr strategy=best5          
# python -m scripts.run_prediction data=p12 tag=ablation1 versions=sr strategy=""         


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


    # python -m scripts.run_prediction data=$DATA tag=ab22 versions=ab2 strategy=first          
    # python -m scripts.run_prediction data=$DATA tag=ab22 versions=ab2 strategy=best1
    # python -m scripts.run_prediction data=$DATA tag=ab22 versions=ab2 strategy=best3          
    python -m scripts.run_prediction data=$DATA tag=ab22 versions=ab2 strategy=""         



done
