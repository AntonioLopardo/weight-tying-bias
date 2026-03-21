#!/usr/bin/env python3
"""
Reproduce Table 6 (Appendix E): Cosine similarity (after Procrustes alignment)
between tied OLMo-1B embedding matrices (with gradient scaling) and the
input/output matrices of untied OLMo-1B-0724 at step 1K.

Usage:
    source .venv/bin/activate
    python reproduce_table6.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", ".."))

from utils.embedding_utils import load_embeddings_from_checkpoint, procrustes_cosine_similarity

# Real vocab size (embedding_size 50304 includes padding tokens)
REAL_VOCAB = 50280

# Official untied reference on HuggingFace
UNTIED_HF_MODEL = "allenai/OLMo-1B-0724-hf"
UNTIED_HF_REVISION = "step1000-tokens2B"


def main():
    from utils.embedding_utils import load_embeddings_from_hf

    print("=" * 60)
    print("Reproducing Table 6 (Appendix E, Step 1K)")
    print("=" * 60)

    # Load tied models (local checkpoints)
    models = [
        ("Tied (no scaling)", "OLMo-1B-tied-no-scale-1000"),
        ("Tied (input x2)", "OLMo-1B-tied-emb2-1000"),
        ("Tied (input x10)", "OLMo-1B-tied-emb10-1000"),
    ]

    tied_embs = {}
    for label, dirname in models:
        tied_embs[label] = load_embeddings_from_checkpoint(
            os.path.join(SCRIPT_DIR, dirname), vocab_size=REAL_VOCAB
        )

    # Load untied reference (official HuggingFace checkpoint)
    untied = load_embeddings_from_hf(UNTIED_HF_MODEL, revision=UNTIED_HF_REVISION)
    untied_input = untied["input_emb"][:REAL_VOCAB]
    untied_output = untied["output_emb"][:REAL_VOCAB]

    # Compute Procrustes alignment
    print("\nComputing Procrustes alignment...")
    results = {}
    for label, emb in tied_embs.items():
        vs_input = procrustes_cosine_similarity(emb, untied_input)
        vs_output = procrustes_cosine_similarity(emb, untied_output)
        results[label] = (vs_input, vs_output)

    # Print results
    print("\n" + "=" * 60)
    print("TABLE 6 (Step 1K)")
    print("=" * 60)
    print(f"\n{'Model':<25} {'vs Untied Input':>16} {'vs Untied Output':>17}")
    print("-" * 60)
    for label, (vs_in, vs_out) in results.items():
        print(f"{label:<25} {vs_in:>16.3f} {vs_out:>17.3f}")

    print(f"\n{'EXPECTED (paper)':<25} {'vs Untied Input':>16} {'vs Untied Output':>17}")
    print("-" * 60)
    print(f"{'Tied (no scaling)':<25} {'0.172':>16} {'0.197':>17}")
    print(f"{'Tied (input x2)':<25} {'0.173':>16} {'0.197':>17}")
    print(f"{'Tied (input x10)':<25} {'0.173':>16} {'0.190':>17}")


if __name__ == "__main__":
    main()
