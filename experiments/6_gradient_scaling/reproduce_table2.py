#!/usr/bin/env python3
"""
Reproduce Table 2: Cosine similarity (after Procrustes alignment) between
tied OLMo-1B embedding matrices and the input/output matrices of untied
OLMo-1B-0724 at step 10K.

Usage:
    source .venv/bin/activate
    python reproduce_table2.py
"""

import os
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Real vocab size (embedding_size 50304 includes padding tokens)
REAL_VOCAB = 50280

# Official untied reference on HuggingFace
UNTIED_HF_MODEL = "allenai/OLMo-1B-0724-hf"
UNTIED_HF_REVISION = "step10000-tokens20B"


def load_tied_embeddings_from_checkpoint(checkpoint_dir: str) -> torch.Tensor:
    """Load the shared embedding matrix from a local OLMo tied checkpoint."""
    model_path = os.path.join(checkpoint_dir, "model.pt")
    print(f"Loading: {model_path}")
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    emb = state["transformer.wte.weight"].clone().float()
    print(f"  Embeddings: {tuple(emb.shape)} (tied)")
    return emb[:REAL_VOCAB]


def load_untied_embeddings_from_hf(model_id: str, revision: str) -> dict:
    """Load input and output embedding matrices from a HuggingFace model."""
    from transformers import AutoModelForCausalLM

    print(f"Loading: {model_id} @ {revision}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.float32, low_cpu_mem_usage=True,
    )
    sd = model.state_dict()
    input_emb = sd["model.embed_tokens.weight"].float()
    output_emb = sd["lm_head.weight"].float()
    print(f"  Input: {tuple(input_emb.shape)}, Output: {tuple(output_emb.shape)}")
    del model
    return {
        "input_emb": input_emb[:REAL_VOCAB],
        "output_emb": output_emb[:REAL_VOCAB],
    }


def procrustes_cosine_similarity(source: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean cosine similarity after Procrustes (orthogonal) alignment.

    Aligns `source` to `target` via the orthogonal Procrustes solution,
    then reports mean per-token cosine similarity.
    """
    source_c = source - source.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)

    M = target_c.T @ source_c
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vt

    aligned = source_c @ W.T

    aligned_norm = aligned / (aligned.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target_c / (target_c.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = (aligned_norm * target_norm).sum(dim=1)

    return cos_sim.mean().item()


def main():
    print("=" * 60)
    print("Reproducing Table 2 (Section 6)")
    print("=" * 60)

    # Load tied models (local checkpoints)
    tied_baseline = load_tied_embeddings_from_checkpoint(
        os.path.join(SCRIPT_DIR, "OLMo-1B-tied-no-scale-10000")
    )
    tied_scaled = load_tied_embeddings_from_checkpoint(
        os.path.join(SCRIPT_DIR, "OLMo-1B-tied-emb5-10000")
    )

    # Load untied reference (official HuggingFace checkpoint)
    untied = load_untied_embeddings_from_hf(UNTIED_HF_MODEL, UNTIED_HF_REVISION)
    untied_input = untied["input_emb"]
    untied_output = untied["output_emb"]

    # Compute Procrustes alignment
    print("\nComputing Procrustes alignment...")
    results = {
        "baseline_vs_input": procrustes_cosine_similarity(tied_baseline, untied_input),
        "baseline_vs_output": procrustes_cosine_similarity(tied_baseline, untied_output),
        "scaled_vs_input": procrustes_cosine_similarity(tied_scaled, untied_input),
        "scaled_vs_output": procrustes_cosine_similarity(tied_scaled, untied_output),
    }

    # Print results
    print("\n" + "=" * 60)
    print("TABLE 2 (Step 10K)")
    print("=" * 60)
    print(f"\n{'Model':<25} {'vs Untied Input':>16} {'vs Untied Output':>17}")
    print("-" * 60)
    print(f"{'Tied (no scaling)':<25} {results['baseline_vs_input']:>16.3f} {results['baseline_vs_output']:>17.3f}")
    print(f"{'Tied (input x5)':<25} {results['scaled_vs_input']:>16.3f} {results['scaled_vs_output']:>17.3f}")

    print(f"\n{'EXPECTED (paper)':<25} {'vs Untied Input':>16} {'vs Untied Output':>17}")
    print("-" * 60)
    print(f"{'Tied (no scaling)':<25} {'0.216':>16} {'0.384':>17}")
    print(f"{'Tied (input x5)':<25} {'0.222':>16} {'0.369':>17}")


if __name__ == "__main__":
    main()
