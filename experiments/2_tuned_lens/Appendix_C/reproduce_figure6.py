#!/usr/bin/env python3
"""
Reproduce Figure 6 (Appendix C): Tuned lens and logit lens KL divergence
across layers for Pythia-2.8B (untied) vs GPT-Neo-2.7B (tied).

Usage:
    source /home/vec_norm/.venv/bin/activate
    python reproduce_figure6.py
    # Output: figure6_pythia_vs_gptneo.png
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens, LogitLens
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
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

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This is a simple sentence to test the language model's ability to predict tokens.",
    "In the field of machine learning, neural networks have become increasingly important for a wide variety of tasks including natural language processing.",
    "The development of large language models has revolutionized how we interact with artificial intelligence systems.",
    "Scientists have discovered new evidence that suggests the universe may be expanding at a faster rate than previously thought.",
    "The economic impact of climate change continues to be a major concern for policymakers around the world.",
    "Recent advances in quantum computing have opened up new possibilities for solving complex computational problems.",
    "The history of human civilization is marked by periods of great innovation and technological advancement.",
    "Music has been an integral part of human culture for thousands of years, serving both social and artistic purposes.",
    "The study of philosophy helps us understand fundamental questions about existence, knowledge, and ethics.",
    "Modern medicine has made remarkable progress in treating diseases that were once considered incurable.",
]


def compute_kl_divergence(log_p, log_q, dim=-1):
    p = log_p.exp()
    return torch.sum(p * (log_p - log_q), dim=dim)


def compute_bias_per_layer(model, tokenizer, tuned_lens, logit_lens, texts, device="cuda", max_length=512):
    num_layers = model.config.num_hidden_layers
    tuned_kl_sum = torch.zeros(num_layers + 1, device=device)
    logit_kl_sum = torch.zeros(num_layers + 1, device=device)
    total_tokens = 0

    model.eval()
    tuned_lens.eval()

    with torch.no_grad():
        for text in tqdm(texts, desc="Processing texts"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            final_log_probs = torch.log_softmax(outputs.logits, dim=-1)
            hidden_states = outputs.hidden_states
            num_tokens = inputs.input_ids.shape[1]
            total_tokens += num_tokens

            for layer_idx in range(num_layers + 1):
                h = hidden_states[layer_idx]
                lens_idx = max(0, layer_idx - 1) if layer_idx > 0 else 0

                tuned_log_probs = torch.log_softmax(tuned_lens(h, idx=lens_idx), dim=-1)
                logit_log_probs = torch.log_softmax(logit_lens(h, idx=lens_idx), dim=-1)

                tuned_kl_sum[layer_idx] += compute_kl_divergence(final_log_probs, tuned_log_probs).sum()
                logit_kl_sum[layer_idx] += compute_kl_divergence(final_log_probs, logit_log_probs).sum()

    nats_to_bits = 1.0 / np.log(2)
    return (
        (tuned_kl_sum / total_tokens).cpu().numpy() * nats_to_bits,
        (logit_kl_sum / total_tokens).cpu().numpy() * nats_to_bits,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = {}

    for model_config in MODELS:
        model_name = model_config["name"]
        label = model_config["label"]

        print(f"\n{'='*60}\nProcessing: {model_name}\n{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        print(f"Layers: {model.config.num_hidden_layers}")

        lens_path = model_config["lens_path"]
        print(f"Loading tuned lens from {lens_path}...")
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id=lens_path, weights_only=True).to(device)
        logit_lens = LogitLens.from_model(model).to(device)

        tuned_kl, logit_kl = compute_bias_per_layer(model, tokenizer, tuned_lens, logit_lens, SAMPLE_TEXTS, device)

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
