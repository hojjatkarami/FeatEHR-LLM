# Standard library imports
import os
import sys
import json
import pickle
import random
from functools import partial
import multiprocessing as mp

# Third-party imports
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# Local application imports
from scripts.utils import *

from loguru import logger
import wandb

def process_patient(i):

    global data, TASK, MODEL_OPENAI, TEMPERATURE, MAX_NEW_TOKENS

    from pydantic import BaseModel

    class RiskAnalysis(BaseModel):
        reasoning: str
        risk: int

    model = get_llm_client(MODEL_OPENAI)


    label_ihm = int(data[i]["label_ihm"][0])

    text = ts_serialize(data[i])

    question = f"""# Your task is to {TASK}.

# Here is the patient's data:
{text}
        
# Please analyze the patient's data and evaluate the risk (from 0 to 100). Let's think step by step.
"""
    messages = [
        {"role": "system", "content": "You are an AI doctor."},
        # *fewshot_samples,
        {"role": "user", "content": question},
    ]

    answer = call_structured_llm(
        model=model,
        model_name=MODEL_OPENAI,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        response_format=RiskAnalysis,
    )

    if wandb.run is not None:
        wandb.log(
            {
                "prompt_tokens": answer.usage.prompt_tokens,
                "completion_tokens": answer.usage.completion_tokens,
                "total_tokens": answer.usage.total_tokens,
            }
        )

    messages.append(
        {"role": "assistant", "content": dict(answer.choices[0].message.parsed)}
    )

    output = {
        "question": question,
        "reasoning": dict(answer.choices[0].message.parsed)["reasoning"],
        "risk": dict(answer.choices[0].message.parsed)["risk"],
        "label_ihm": label_ihm,
    }
    return output


def ts_serialize(item, dict_map=None):

    global HISTORY, var_names

    HOUR_STEP = 12
    HOUR_AGG = 6

    df = item["ts"][0].copy()

    age = int(item["covariates"][0][0])

    # filter based on MAX_TIME
    df = df[df["Hours"] < HISTORY]
    # print(df.Hours)

    if dict_map is not None:
        df.rename(columns=dict_map, inplace=True)
        var_names = dict_map.values()
    # set precision of numerical values to 2
    df = df.round(2)

    # take integer part of Hours
    df["Hours"] = df["Hours"].astype(int)

    # aggregate into 4-hour time bins and keep last
    df["Hours"] = (df["Hours"].floordiv(HOUR_AGG) + 1) * HOUR_AGG

    df = df.groupby(df["Hours"]).last().reset_index()

    text = f"Patient is a {age} years old."
    for hours in range(0, 100000, HOUR_STEP):
        # print(hours)
        df_slice = df[(df["Hours"] >= hours) & (df["Hours"] < hours + HOUR_STEP)]
        if len(df_slice) == 0:
            break
        # text += f"Hours {hours}-{hours+HOUR_STEP}:\n\n "
        for _, row in df_slice.iterrows():
            current_hour = np.round(row["Hours"], 1)
            # print(current_hour)
            text += f"\n\nFrom hour {current_hour-HOUR_AGG}  to {current_hour}:\n "
            for var in var_names:
                if not pd.isnull(row[var]):

                    # if string
                    if isinstance(row[var], str):
                        text += f"{var} is '{row[var]}', "
                    else:
                        text += f"{var} is '{np.round(row[var],2)}', "
    # print(text)

    return text


def eval_zeroshot(n):

    N_PROCESSES = min(20, mp.cpu_count())  # Use all available CPU cores

    partial_func = partial(process_patient)

    with mp.Pool(N_PROCESSES) as pool:

        # all_results = list(pool.imap(partial_func, range(0, n)))
        all_results = list(
            tqdm(pool.imap(process_patient, range(0, n)), total=n, leave=False)
        )
    df = pd.DataFrame(all_results)

    return df



@hydra.main(config_path="../configs", config_name="zero", version_base=None)
def main(args):

    # start wandb
    wandb.init(
        project=args.wandb.project,
        name=args.wandb.name,
        config=OmegaConf.to_container(args),
    )

    global data, total_errors, var_names, HISTORY, TASK
    global MODEL_OPENAI, TEMPERATURE, MAX_NEW_TOKENS

    DATA = args.data.name
    HISTORY = args.data.history
    DATA_PATH = args.data.path
    TASK = args.data.task

    VERSION = args.version

    N_MAX = args.n_max
    MODEL_OPENAI = args.llm.model
    MAX_NEW_TOKENS = args.llm.max_new_tokens
    TEMPERATURE = args.llm.temperature

    # Create a folder to save the results
    print(f"[info] Saving results to: results/{DATA}/{VERSION}/")
    folder_name = f"data/featehr-llm/{DATA}/bl_{VERSION}"
    # create if not exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    set_logger(folder_name, level=args.log.level)

    # Loading the data
    path_data = DATA_PATH.format(split=args.split)

    logger.info(f"Loading the data from {path_data}...")
    with open(path_data, "rb") as f:
        data = pickle.load(f)

    # random sample N_MAX patients with seed
    if len(data) < N_MAX:
        logger.info(
            f"Number of patients is more than {N_MAX}. We set N_MAX to {len(data)}"
        )
        N_MAX = len(data)

    # randomly select n samples with seed 42
    random.seed(42)
    random.shuffle(data)
    data = data[:N_MAX]

    logger.info(f"Loaded {N_MAX} patients from {DATA}")

    if DATA == "mimic3":
        logger.info(f"\t[info] Processing time series data...")
        for i in tqdm(range(N_MAX), desc="Processing data", leave=False):
            data[i]["ts"][0] = process_ts_vars(data[i]["ts"][0])

    logger.info(f"Loading the variable units...")

    # load units full
    with open(args.data.units, "r") as f:
        units_full = json.load(f)

    var_names = list(units_full.keys())

    df_eval = eval_zeroshot(N_MAX)

    # save to csv
    df_eval.to_csv(f"{folder_name}/data.csv", index=False)
    return


if __name__ == "__main__":

    sys.path.append(os.path.abspath(os.getcwd()))
    print(sys.path)

    global data, total_errors, var_names, HISTORY, TASK

    load_dotenv(override=True)

    random.seed(42)

    main()
