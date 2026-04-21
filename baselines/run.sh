#!/bin/bash

cd /home/hokarami/code/FeatEHR-LLM/baselines
source activate featehr-llm


# # install CAAFE
# git clone https://github.com/noahho/CAAFE.git
# cd CAAFE
# pip install -e .
# pip install openml
# cd ..


# # install OpenFE
# git clone https://github.com/IIIS-Li-Group/OpenFE.git
# cd OpenFE
# pip install -e .
# cd ..


# # install autofeat
# pip install autofeat



cd /home/hokarami/code/FeatEHR-LLM


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