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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.embedding_utils import load_embeddings_from_checkpoint, procrustes_cosine_similarity

# Real vocab size (embedding_size 50304 includes padding tokens)
REAL_VOCAB = 50280

# Official untied reference on HuggingFace
UNTIED_HF_MODEL = "allenai/OLMo-1B-0724-hf"
UNTIED_HF_REVISION = "step10000-tokens20B"


def main():
    from utils.embedding_utils import load_embeddings_from_hf

    print("=" * 60)
    print("Reproducing Table 2 (Section 6)")
    print("=" * 60)

    # Load tied models (local checkpoints)
    tied_baseline = load_embeddings_from_checkpoint(
        os.path.join(SCRIPT_DIR, "OLMo-1B-tied-no-scale-10000"), vocab_size=REAL_VOCAB
    )
    tied_scaled = load_embeddings_from_checkpoint(
        os.path.join(SCRIPT_DIR, "OLMo-1B-tied-emb5-10000"), vocab_size=REAL_VOCAB
    )

    # Load untied reference (official HuggingFace checkpoint)
    untied = load_embeddings_from_hf(UNTIED_HF_MODEL, revision=UNTIED_HF_REVISION)
    untied_input = untied["input_emb"][:REAL_VOCAB]
    untied_output = untied["output_emb"][:REAL_VOCAB]

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
