#!/usr/bin/env python3
"""
Reproduce Table 7 (Appendix E): Downstream evaluation of tied OLMo-1B
models with and without input gradient scaling (5x) at step 10K.

Usage:
    python experiments/6_gradient_scaling/Appendix_E/reproduce_table7.py
    python experiments/6_gradient_scaling/Appendix_E/reproduce_table7.py --batch-size 16
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import pandas as pd

from lm_eval.evaluator import simple_evaluate

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # Appendix_E -> 6_gradient_scaling -> experiments -> repo root
EXP_ROOT = REPO_ROOT / "experiments"

sys.path.insert(0, str(REPO_ROOT))
from utils.olmo_native_lm import OLMoNativeLM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #

MODEL_CONFIGS = {
    "tied": {
        "checkpoint_dir": str(EXP_ROOT / "4_norm_frequency" / "OLMo-1B-tied"),
        "description": "OLMo-1B tied (10k steps)",
    },
    "tied-emb5": {
        "checkpoint_dir": str(
            EXP_ROOT / "6_gradient_scaling" / "OLMo-1B-tied-emb5-10000"
        ),
        "description": "OLMo-1B tied, 5x input grad scaling (10k steps)",
    },
}

# --------------------------------------------------------------------------- #
# Downstream tasks
# --------------------------------------------------------------------------- #

DEFAULT_TASKS = [
    "wikitext",
    "piqa",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "openbookqa",
    "blimp",
]

PRIORITY_METRICS = [
    "wikitext/word_perplexity,none",
    "wikitext/byte_perplexity,none",
    "wikitext/bits_per_byte,none",
    "piqa/acc,none",
    "piqa/acc_norm,none",
    "hellaswag/acc,none",
    "hellaswag/acc_norm,none",
    "winogrande/acc,none",
    "arc_easy/acc,none",
    "arc_easy/acc_norm,none",
    "arc_challenge/acc,none",
    "arc_challenge/acc_norm,none",
    "boolq/acc,none",
    "openbookqa/acc,none",
    "openbookqa/acc_norm,none",
    "blimp/acc,none",
]


def extract_results(raw_results: dict) -> dict:
    """Pull out the key metrics from lm-eval's results dict."""
    extracted = {}
    if "results" not in raw_results:
        return extracted

    for task_name, task_results in raw_results["results"].items():
        for metric_key, value in task_results.items():
            if metric_key == "alias":
                continue
            clean_key = f"{task_name}/{metric_key}"
            if isinstance(value, (int, float)):
                extracted[clean_key] = value
    return extracted


def run_evaluation(
    model_names: list[str],
    tasks: list[str],
    batch_size: int,
    device: str,
    output_dir: str,
    num_fewshot: int | None = None,
    limit: float | None = None,
):
    """Run evaluation for selected models and tasks."""
    all_results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        cfg = MODEL_CONFIGS[model_name]
        logger.info("=" * 70)
        logger.info(f"Evaluating: {cfg['description']} ({model_name})")
        logger.info("=" * 70)

        lm = OLMoNativeLM(
            checkpoint_dir=cfg["checkpoint_dir"],
            device=device,
            batch_size=batch_size,
            dtype=torch.bfloat16,
        )

        eval_kwargs = {
            "model": lm,
            "tasks": tasks,
        }
        if num_fewshot is not None:
            eval_kwargs["num_fewshot"] = num_fewshot
        if limit is not None:
            eval_kwargs["limit"] = limit

        raw = simple_evaluate(**eval_kwargs)

        model_out = output_path / f"{model_name}_results.json"
        # lm-eval returns non-serializable objects; keep only what we need
        serializable = {
            "results": raw.get("results", {}),
            "n-shot": raw.get("n-shot", {}),
            "config": {
                "model": model_name,
                "description": cfg["description"],
                "tasks": tasks,
                "batch_size": batch_size,
            },
        }
        with open(model_out, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"Raw results saved to {model_out}")

        all_results[model_name] = extract_results(raw)

        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def format_comparison_table(all_results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from all model results."""
    all_metrics = set()
    for metrics in all_results.values():
        all_metrics.update(metrics.keys())

    ordered_metrics = [m for m in PRIORITY_METRICS if m in all_metrics]
    remaining = sorted(all_metrics - set(ordered_metrics))
    ordered_metrics.extend(remaining)

    rows = {}
    for model_name, metrics in all_results.items():
        rows[model_name] = {m: metrics.get(m) for m in ordered_metrics}

    df = pd.DataFrame(rows).T
    df.columns = [c.replace(",none", "").replace(",stderr", " (stderr)") for c in df.columns]
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tied OLMo-1B models on perplexity and downstream tasks"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=list(MODEL_CONFIGS.keys()),
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Tasks to evaluate (default: wikitext + downstream suite)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (task default if not set)",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit number of examples per task (for testing)",
    )
    args = parser.parse_args()

    logger.info(f"Models: {args.models}")
    logger.info(f"Tasks:  {args.tasks}")
    logger.info(f"Batch:  {args.batch_size}")
    logger.info(f"Device: {args.device}")

    all_results = run_evaluation(
        model_names=args.models,
        tasks=args.tasks,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )

    if all_results:
        df = format_comparison_table(all_results)
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(df.to_string())

        output_path = Path(args.output_dir)
        combined_path = output_path / "comparison.csv"
        df.to_csv(combined_path)
        logger.info(f"\nComparison table saved to {combined_path}")

        combined_json = output_path / "all_results.json"
        with open(combined_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Combined results saved to {combined_json}")


if __name__ == "__main__":
    main()
