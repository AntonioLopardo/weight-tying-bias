#!/usr/bin/env python3
"""
Reproduce Figure 7 (Appendix C): Tuned lens and logit lens KL divergence
across layers for Qwen3-4B (tied) vs Qwen3-8B (untied).

Both models have 36 layers, enabling direct layer-by-layer comparison.

Usage:
    source .venv/bin/activate
    python reproduce_figure7.py
    # Output: figure7_qwen3_comparison.png
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tuned_lens.nn.lenses import TunedLens, LogitLens
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

from utils.tuned_lens_utils import (
    compute_bias_per_layer, load_model_and_tokenizer, SAMPLE_TEXTS,
)

TRAINED_LENSES_DIR = SCRIPT_DIR.parent / "trained_lenses"

MODELS = [
    {
        "name": "Qwen/Qwen3-4B",
        "label": "Qwen3-4B (tied)",
    },
    {
        "name": "Qwen/Qwen3-8B",
        "label": "Qwen3-8B (untied)",
    },
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = {}

    for model_config in MODELS:
        model_name = model_config["name"]
        label = model_config["label"]

        print(f"\n{'='*60}\nProcessing: {model_name}\n{'='*60}")

        model, tokenizer = load_model_and_tokenizer(model_name)
        print(f"Layers: {model.config.num_hidden_layers}")

        local_lens_path = str(TRAINED_LENSES_DIR / model_name)
        if not Path(local_lens_path).exists():
            print(f"WARNING: Tuned lens not found at {local_lens_path}")
            print("Please run ../train_qwen3_lenses.sh first")
            del model, tokenizer
            torch.cuda.empty_cache()
            continue

        print(f"Loading tuned lens from {local_lens_path}")
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id=local_lens_path, weights_only=True).to(device)
        logit_lens = LogitLens.from_model(model).to(device)

        tuned_kl, logit_kl = compute_bias_per_layer(model, tokenizer, tuned_lens, SAMPLE_TEXTS, device, logit_lens=logit_lens)

        results[label] = {"tuned_kl": tuned_kl, "logit_kl": logit_kl, "num_layers": model.config.num_hidden_layers}

        del model, tuned_lens, logit_lens, tokenizer
        torch.cuda.empty_cache()

    if len(results) < 2:
        print("\nNot enough models processed. Ensure tuned lenses are trained.")
        return

    # Both models have 36 layers — plot with absolute layer indices
    colors = ["#1f77b4", "#d62728"]
    markers = ["o", "s"]
    linestyles = ["-", "--"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, lens_type in zip(axes, ["tuned_kl", "logit_kl"]):
        for i, (label, data) in enumerate(results.items()):
            kl = data[lens_type]
            n = len(kl)
            x = np.arange(n)
            ax.plot(x, kl, marker=markers[i], color=colors[i], linestyle=linestyles[i],
                    label=label, markersize=5, linewidth=2, markevery=max(1, n // 10))

        num_layers = list(results.values())[0]["num_layers"]
        tick_labels = ["input"] + [str(j) for j in range(1, num_layers + 1)]
        step = max(1, (num_layers + 1) // 10)
        ax.set_xticks(list(range(num_layers + 1))[::step])
        ax.set_xticklabels(tick_labels[::step])

        title = "Tuned Lens" if lens_type == "tuned_kl" else "Logit Lens"
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("KL Divergence (bits)", fontsize=12)
        ax.set_title(f"{title}: Qwen3-4B vs Qwen3-8B", fontsize=13)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = "figure7_qwen3_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
