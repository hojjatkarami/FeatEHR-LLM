#!/usr/bin/env python3
"""
End-to-End Execution Pipeline for FeatEHR-LLM.
Automates Feature Generation -> Vectorized Extraction -> Downstream Prediction & Comparison.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv

# Ensure environment is loaded
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
load_dotenv(find_dotenv(usecwd=True), override=True)

# Table 2 Reference Values for comparison
TABLE_2_REFERENCES = {
    "p12": {
        "BL": {"auroc": 83.6, "ci": 2.0},
        "FeatEHR-LLM": {"auroc": 85.9, "ci": 1.9},
        "OpenFE": {"auroc": 85.0, "ci": 1.9},
        "TF": {"auroc": 85.6, "ci": 2.0},
        "CAAFE": {"auroc": 84.3, "ci": 2.0},
        "FeatLLM": {"auroc": 84.6, "ci": 2.0},
    },
    "p19": {
        "BL": {"auroc": 86.4, "ci": 1.5},
        "FeatEHR-LLM": {"auroc": 92.1, "ci": 1.3},
    },
    "mimic3": {
        "BL": {"auroc": 91.0, "ci": 1.3},
        "FeatEHR-LLM": {"auroc": 94.1, "ci": 1.0},
    },
    "eicu_mo48": {
        "BL": {"auroc": 81.8, "ci": 0.9},
        "FeatEHR-LLM": {"auroc": 83.8, "ci": 0.9},
    },
}


def test_api_connection(model_name: str) -> bool:
    """Verifies that the LLM client can successfully reach the API endpoint."""
    print(f"\n[1/5] Verifying API connectivity for model '{model_name}'...")
    try:
        from scripts.utils import get_llm_client
        client = get_llm_client(model_name)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Respond strictly with the single word READY."}],
            max_tokens=100,
        )
        msg = response.choices[0].message
        reply = (msg.content or getattr(msg, "reasoning", "") or "OK").strip()
        print(f"  ✓ API Connection successful! Model response: {reply}")
        return True
    except Exception as e:
        print(f"\n❌ API Connection failed with error: {e}")
        print("\nPlease check your .env configuration:")
        print("  - If using DeepSeek official API (deepseek-chat): ensure DEEPSEEK_API_KEY is set in .env")
        print("  - If using OpenRouter (deepseek/...): ensure LLM_API_KEY is set in .env")
        return False


def run_command(cmd: list, desc: str) -> bool:
    """Executes a command and streams output."""
    print(f"\n--- {desc} ---")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Failed: {desc} (exit code {result.returncode})")
        return False
    print(f"✓ Completed: {desc}")
    return True


def run_pipeline(
    data: str = "p12",
    model: str = None,
    version_tag: str = "v1",
    mode: str = "single",
    skip_gen: bool = False,
    skip_extract: bool = False,
    skip_eval: bool = False,
):
    load_dotenv(find_dotenv(usecwd=True), override=True)
    if not model:
        model = os.getenv("LLM_MODEL", "deepseek-chat")

    # Sanitize version prefix from model name
    clean_model_name = model.split("/")[-1].replace(".", "").replace("-", "_")
    version = f"{clean_model_name}_{mode[:1]}_{version_tag}"
    print("=" * 60)
    print(f"FeatEHR-LLM Automated Pipeline")
    print(f"  Dataset:      {data}")
    print(f"  LLM Model:    {model}")
    print(f"  Mode:         {mode}")
    print(f"  Version Tag:  {version}")
    print("=" * 60)

    # 1. Test API Connectivity
    if not skip_gen:
        if not test_api_connection(model):
            return

    # 2. Feature Generation
    if not skip_gen:
        gen_script = "scripts.run_single" if mode == "single" else "scripts.run_multi"
        print(f"\n[2/5] Proposing features using {model} ({mode})...")
        cmd_gen = [
            sys.executable,
            "-m",
            gen_script,
            f"data={data}",
            f"version={version}",
            f"llm.model={model}",
        ]
        if not run_command(cmd_gen, f"Feature Generation ({mode})"):
            return

    # 3. Feature Extraction (Train & Test)
    if not skip_extract:
        print(f"\n[3/5] Extracting features over patient cohort...")
        for split in ["train", "test"]:
            cmd_extract = [
                sys.executable,
                "-m",
                "scripts.extract_features",
                f"data={data}",
                f"split={split}",
                f"version={version}",
            ]
            if not run_command(cmd_extract, f"Feature Extraction ({split} set)"):
                return

    # 4. Downstream Evaluation (Prediction + Bootstrap CI)
    if not skip_eval:
        print(f"\n[4/5] Evaluating LightGBM downstream classifier...")
        # Create temporary version config for evaluation: [numeric, version, numeric+version]
        version_config_dir = Path("configs/versions")
        version_config_dir.mkdir(parents=True, exist_ok=True)
        eval_config_name = f"eval_{clean_model_name}"
        eval_config_path = version_config_dir / f"{eval_config_name}.yaml"
        with open(eval_config_path, "w") as f:
            f.write(f"- numeric\n- {version}\n- numeric+{version}\n")

        cmd_eval = [
            sys.executable,
            "-m",
            "scripts.run_prediction",
            f"data={data}",
            f"tag={version}",
            f"versions={eval_config_name}",
        ]
        if not run_command(cmd_eval, "Downstream Model Training & Evaluation"):
            return

    # 5. Report Results & Comparison with Table 2
    print(f"\n[5/5] Generating Performance Comparison Report...")
    perf_file = Path(f"results/{data}_perf.csv")
    if not perf_file.exists():
        print(f"❌ Results file not found at {perf_file}")
        return

    df_perf = pd.read_csv(perf_file)
    # Filter for the relevant versions
    df_eval = df_perf[(df_perf["data"] == data) & (df_perf["size"] == df_perf["size"].max())].copy()

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS REPORT: {data.upper()} BENCHMARK (Table 2)")
    print("=" * 70)

    # Format output table
    summary_rows = []
    for _, row in df_eval.iterrows():
        v = row["version"]
        auroc = row["AUROC"] * 100.0
        ci = row["AUROC_CI"] * 100.0
        auprc = row["AUPRC"] * 100.0
        f1 = row["F1"] * 100.0
        summary_rows.append({
            "Configuration": v,
            "AUROC (95% CI)": f"{auroc:.2f} ± {ci:.2f}",
            "AUPRC": f"{auprc:.2f}",
            "F1 Score": f"{f1:.2f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_markdown(index=False))

    # Compare with Table 2 baseline
    if data in TABLE_2_REFERENCES:
        ref_bl = TABLE_2_REFERENCES[data]["BL"]["auroc"]
        ref_featehr = TABLE_2_REFERENCES[data]["FeatEHR-LLM"]["auroc"]
        print("\n--- Reference Benchmarks from Manuscript Table 2 ---")
        print(f"  • Baseline (BL) AUROC:               {ref_bl:.1f}%")
        print(f"  • Previous FeatEHR-LLM AUROC:        {ref_featehr:.1f}%")

        # Find the new FeatEHR-LLM (numeric + generated) row
        combo_rows = df_eval[df_eval["version"] == f"numeric+{version}"]
        if not combo_rows.empty:
            new_auroc = combo_rows.iloc[-1]["AUROC"] * 100.0
            delta_bl = new_auroc - ref_bl
            delta_prev = new_auroc - ref_featehr
            print("\n--- Comparison & Verdict ---")
            print(f"  • New Model (+FeatEHR-LLM):          {new_auroc:.2f}%")
            print(f"  • Improvement over Baseline (BL):    {delta_bl:+.2f} percentage points")
            print(f"  • Difference vs Manuscript LLM:      {delta_prev:+.2f} percentage points")
            if delta_prev > 0:
                print(f"  🎉 VERDICT: The new model IMPROVES upon the previous manuscript result!")
            elif abs(delta_prev) <= 0.5:
                print(f"  ⚖️ VERDICT: The new model achieves COMPETITIVE / EQUIVALENT performance to the manuscript result.")
            else:
                print(f"  ⚠️ VERDICT: The new model performs slightly below the previous model setting.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FeatEHR-LLM Automated Testing Pipeline")
    parser.add_argument("--data", default="p12", help="Dataset (p12, p19, mimic3, eicu_mo48)")
    parser.add_argument("--model", default=None, help="LLM model name (defaults to LLM_MODEL in .env)")
    parser.add_argument("--tag", default="v1", help="Version tag")
    parser.add_argument("--mode", default="single", choices=["single", "multi"], help="Feature mode")
    parser.add_argument("--skip-gen", action="store_true", help="Skip feature generation")
    parser.add_argument("--skip-extract", action="store_true", help="Skip feature extraction")
    parser.add_argument("--skip-eval", action="store_true", help="Skip downstream evaluation")
    args = parser.parse_args()

    run_pipeline(
        data=args.data,
        model=args.model,
        version_tag=args.tag,
        mode=args.mode,
        skip_gen=args.skip_gen,
        skip_extract=args.skip_extract,
        skip_eval=args.skip_eval,
    )
