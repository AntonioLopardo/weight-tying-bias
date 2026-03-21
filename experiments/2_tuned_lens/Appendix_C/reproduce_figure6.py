#!/usr/bin/env python3
"""
Reproduce Figure 6 (Appendix C): Tuned lens and logit lens KL divergence
across layers for Pythia-2.8B (untied) vs GPT-Neo-2.7B (tied).

Usage:
    source .venv/bin/activate
    python reproduce_figure6.py
    # Output: figure6_pythia_vs_gptneo.png
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from tuned_lens.nn.lenses import TunedLens, LogitLens
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent.parent))

from utils.tuned_lens_utils import (
    compute_bias_per_layer, load_model_and_tokenizer, SAMPLE_TEXTS,
)

TRAINED_LENSES_DIR = SCRIPT_DIR.parent / "trained_lenses"

MODELS = [
    {
        "name": "EleutherAI/pythia-2.8b",
        "label": "Pythia-2.8B (untied)",
        "lens_path": str(TRAINED_LENSES_DIR / "EleutherAI" / "pythia-2.8b"),
    },
    {
        "name": "EleutherAI/gpt-neo-2.7B",
        "label": "GPT-Neo-2.7B (tied)",
        "lens_path": str(TRAINED_LENSES_DIR / "EleutherAI" / "gpt-neo-2.7B"),
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

        lens_path = model_config["lens_path"]
        print(f"Loading tuned lens from {lens_path}...")
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id=lens_path, weights_only=True).to(device)
        logit_lens = LogitLens.from_model(model).to(device)

        tuned_kl, logit_kl = compute_bias_per_layer(model, tokenizer, tuned_lens, SAMPLE_TEXTS, device, logit_lens=logit_lens)

        results[label] = {"tuned_kl": tuned_kl, "logit_kl": logit_kl, "num_layers": model.config.num_hidden_layers}

        del model, tuned_lens, logit_lens, tokenizer
        torch.cuda.empty_cache()

    # Plot
    colors = ["#E63946", "#457B9D"]
    markers = ["o", "s"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, lens_type in zip(axes, ["tuned_kl", "logit_kl"]):
        for i, (label, data) in enumerate(results.items()):
            kl = data[lens_type]
            x = np.linspace(0, 1, len(kl))
            ax.plot(x, kl, marker=markers[i], color=colors[i], label=label, markersize=6, linewidth=2,
                    markevery=max(1, len(kl) // 10))

        title = "Tuned Lens" if lens_type == "tuned_kl" else "Logit Lens"
        ax.set_xlabel("Relative Depth (0=input, 1=output)", fontsize=12)
        ax.set_ylabel("KL Divergence (bits)", fontsize=12)
        ax.set_title(f"{title}: Pythia-2.8B vs GPT-Neo-2.7B", fontsize=13)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = "figure6_pythia_vs_gptneo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
