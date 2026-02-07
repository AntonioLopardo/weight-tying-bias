"""
Reproduce Figure 1: Token-Level Alignment Histograms

Compares the tied embedding matrix (OLMo-1B) to the untied input and output
matrices (OLMo-1B-0724) after linear alignment. Produces a histogram of
per-token cosine similarities showing that the tied matrix is much more
aligned with the untied output than the untied input.

Paper values:
  Input → Tied:  mean 0.525
  Output → Tied: mean 0.719
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent


def align_and_compute_cosine(source, target):
    """Apply linear transformation and compute token-level cosine similarities."""
    # Center
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    # Linear: target_centered = source_centered @ M
    M, _, _, _ = np.linalg.lstsq(source_centered, target_centered, rcond=None)

    # Transform source
    source_transformed = source_centered @ M

    # Compute cosine similarities
    source_norm = source_transformed / (np.linalg.norm(source_transformed, axis=1, keepdims=True) + 1e-8)
    target_norm = target_centered / (np.linalg.norm(target_centered, axis=1, keepdims=True) + 1e-8)
    cosine_sim = (source_norm * target_norm).sum(axis=1)

    return cosine_sim


def main():
    # Load untied model (OLMo-1B-0724)
    print("Loading untied model (OLMo-1B-0724)...")
    model_untied = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-1B-0724-hf",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    untied_input = model_untied.model.embed_tokens.weight.data.cpu().numpy()
    untied_output = model_untied.lm_head.weight.data.cpu().numpy()
    del model_untied
    torch.cuda.empty_cache()

    # Load tied model (OLMo-1B)
    print("Loading tied model (OLMo-1B)...")
    model_tied = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-1B-hf",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    tied = model_tied.model.embed_tokens.weight.data.cpu().numpy()
    del model_tied
    torch.cuda.empty_cache()

    # Compute alignments
    print("Computing alignments...")
    cos_output_tied = align_and_compute_cosine(untied_output, tied)
    cos_input_tied = align_and_compute_cosine(untied_input, tied)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(cos_input_tied, bins=80, alpha=0.7, label='Input \u2192 Tied',
            color='#648FFF', edgecolor='black', linewidth=1.0)
    ax.hist(cos_output_tied, bins=80, alpha=0.7, label='Output \u2192 Tied',
            color='#FFB000', edgecolor='black', linewidth=1.0)

    ax.axvline(cos_input_tied.mean(), color='#648FFF', linestyle='--', linewidth=2.5)
    ax.axvline(cos_output_tied.mean(), color='#FFB000', linestyle='--', linewidth=2.5)

    ax.set_xlabel('Cosine Similarity', fontsize=32)
    ax.set_ylabel('Number of Tokens', fontsize=32)
    ax.legend(fontsize=22, loc='upper left', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)

    ax.tick_params(axis='both', which='major', labelsize=26)

    plt.tight_layout()

    output_path = SCRIPT_DIR / 'token_level_alignment.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    print("\nMean cosine similarities:")
    print(f"  Input \u2192 Tied:  {cos_input_tied.mean():.3f}")
    print(f"  Output \u2192 Tied: {cos_output_tied.mean():.3f}")
    print(f"  Difference:    {cos_output_tied.mean() - cos_input_tied.mean():.3f}")


if __name__ == "__main__":
    main()
