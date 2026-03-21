#!/usr/bin/env python3
"""
Reproduce Table 5 (Appendix B): KNN@10 overlap between tied and untied
embedding matrices across three model families.

KNN@10 overlap: for each token, compute its 10 nearest neighbors (by cosine
similarity) in two embedding matrices independently, then measure the
fraction of shared neighbors. Averaged over all tokens.

This metric works even when embedding dimensions differ (Qwen3-4B vs 8B)
because neighbors are identified by token index, not vector distance.

Usage:
    source .venv/bin/activate
    python reproduce_table5.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

K = 10  # Number of nearest neighbors


def load_embeddings(model_id: str) -> dict:
    """Load input and output embedding matrices from a HuggingFace model."""
    print(f"Loading: {model_id}")
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    state_dict = model.state_dict()
    weight_tying = getattr(config, "tie_word_embeddings", True)

    # Find embedding keys (handles OLMo, Pythia, GPT-Neo, Qwen naming)
    input_emb = None
    output_emb = None
    for key in state_dict.keys():
        kl = key.lower()
        if any(p in kl for p in ["embed_tokens.weight", "wte.weight", "embed_in.weight"]):
            input_emb = state_dict[key]
        if any(p in kl for p in ["lm_head.weight", "embed_out.weight"]):
            output_emb = state_dict[key]

    if weight_tying or output_emb is None:
        output_emb = input_emb

    print(f"  Tied: {weight_tying}  Input: {tuple(input_emb.shape)}  Output: {tuple(output_emb.shape)}")
    del model
    return {
        "input_emb": input_emb.clone().float(),
        "output_emb": output_emb.clone().float(),
        "weight_tying": weight_tying,
    }


def knn_at_k(emb: torch.Tensor, k: int) -> torch.Tensor:
    """Compute KNN indices for every token in the embedding matrix.

    Returns a (V, k) tensor of neighbor indices.
    """
    # L2-normalise for cosine similarity
    emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)

    # Compute full cosine similarity matrix in chunks to control memory
    V = emb_norm.shape[0]
    chunk_size = 2048
    neighbors = torch.empty(V, k, dtype=torch.long)

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        sim = emb_norm[start:end] @ emb_norm.T          # (chunk, V)
        sim[torch.arange(end - start), torch.arange(start, end)] = -float("inf")  # exclude self
        neighbors[start:end] = sim.topk(k, dim=1).indices

    return neighbors


def knn_overlap(emb_a: torch.Tensor, emb_b: torch.Tensor, k: int = K,
                indices: torch.Tensor | None = None) -> float:
    """Compute mean KNN@k overlap between two embedding matrices.

    Args:
        emb_a: First embedding matrix (V, d1)
        emb_b: Second embedding matrix (V, d2) — dimensions may differ
        k: Number of neighbors
        indices: Optional token indices to restrict to (for vocab alignment)

    Returns:
        Mean overlap in [0, 1]
    """
    if indices is not None:
        emb_a = emb_a[indices]
        emb_b = emb_b[indices]

    nn_a = knn_at_k(emb_a, k)  # (V, k)
    nn_b = knn_at_k(emb_b, k)  # (V, k)

    # For each token, compute |intersection| / k
    V = nn_a.shape[0]
    overlaps = torch.zeros(V)
    for i in range(V):
        set_a = set(nn_a[i].tolist())
        set_b = set(nn_b[i].tolist())
        overlaps[i] = len(set_a & set_b) / k

    return overlaps.mean().item()


def get_aligned_vocab_indices(tokenizer_a, tokenizer_b):
    """Find tokens present in both tokenizers and return aligned index pairs.

    Returns:
        indices_a: indices into tokenizer_a's vocabulary
        indices_b: indices into tokenizer_b's vocabulary
    """
    vocab_a = tokenizer_a.get_vocab()  # {token_str: index}
    vocab_b = tokenizer_b.get_vocab()

    common_tokens = set(vocab_a.keys()) & set(vocab_b.keys())
    print(f"  Vocab alignment: {len(vocab_a)} x {len(vocab_b)} -> {len(common_tokens)} common tokens")

    # Sort for reproducibility
    common_tokens = sorted(common_tokens)
    indices_a = torch.tensor([vocab_a[t] for t in common_tokens], dtype=torch.long)
    indices_b = torch.tensor([vocab_b[t] for t in common_tokens], dtype=torch.long)

    return indices_a, indices_b


def compute_group(tied_emb, untied_input, untied_output, label,
                  indices_tied=None, indices_untied=None):
    """Compute the 3 KNN@10 comparisons for a model pair."""
    print(f"\n  {label}:")

    if indices_tied is not None:
        # Remap to aligned subsets
        t = tied_emb[indices_tied]
        ui = untied_input[indices_untied]
        uo = untied_output[indices_untied]
    else:
        min_v = min(tied_emb.shape[0], untied_input.shape[0])
        t = tied_emb[:min_v]
        ui = untied_input[:min_v]
        uo = untied_output[:min_v]

    r1 = knn_overlap(t, ui, K)
    print(f"    Tied vs Untied Input:          {r1:.3f}")

    r2 = knn_overlap(t, uo, K)
    print(f"    Tied vs Untied Output:         {r2:.3f}")

    r3 = knn_overlap(ui, uo, K)
    print(f"    Untied Input vs Untied Output: {r3:.3f}")

    return r1, r2, r3


def main():
    print("=" * 60)
    print("Reproducing Table 5 (Appendix B) — KNN@10 Overlap")
    print("=" * 60)

    # ── Group 1: OLMo-1B (tied) vs OLMo-1B-0724 (untied) ──────────
    print("\n[1/3] OLMo-1B (tied) vs OLMo-1B-0724 (untied)")
    olmo_tied = load_embeddings("allenai/OLMo-1B-hf")
    olmo_untied = load_embeddings("allenai/OLMo-1B-0724-hf")

    olmo_r = compute_group(
        olmo_tied["input_emb"], olmo_untied["input_emb"], olmo_untied["output_emb"],
        "OLMo-1B",
    )
    del olmo_tied, olmo_untied

    # ── Group 2: Qwen3-4B (tied) vs Qwen3-8B (untied) ─────────────
    print("\n[2/3] Qwen3-4B (tied) vs Qwen3-8B (untied)")
    qwen_tied = load_embeddings("Qwen/Qwen3-4B")
    qwen_untied = load_embeddings("Qwen/Qwen3-8B")

    qwen_r = compute_group(
        qwen_tied["input_emb"], qwen_untied["input_emb"], qwen_untied["output_emb"],
        "Qwen3",
    )
    del qwen_tied, qwen_untied

    # ── Group 3: GPT-Neo-2.7B (tied) vs Pythia-2.8B (untied) ──────
    print("\n[3/3] GPT-Neo-2.7B (tied) vs Pythia-2.8B (untied)")
    gptneo = load_embeddings("EleutherAI/gpt-neo-2.7B")
    pythia = load_embeddings("EleutherAI/pythia-2.8b")

    # Align vocabularies
    tok_neo = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    tok_py = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    idx_neo, idx_py = get_aligned_vocab_indices(tok_neo, tok_py)

    eleuther_r = compute_group(
        gptneo["input_emb"], pythia["input_emb"], pythia["output_emb"],
        "GPT-Neo / Pythia (aligned vocab)",
        indices_tied=idx_neo, indices_untied=idx_py,
    )
    del gptneo, pythia

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 5 — KNN@10 Overlap")
    print("=" * 60)

    header = f"{'Comparison':<45} {'KNN@10':>8}"
    sep = "-" * 55

    print(f"\n{header}")
    print(sep)

    labels = [
        ("OLMo-1B (tied) vs OLMo-1B-0724 (untied)", olmo_r),
        ("Qwen3-4B (tied) vs Qwen3-8B (untied)", qwen_r),
        ("GPT-Neo-2.7B (tied) vs Pythia-2.8B (untied)", eleuther_r),
    ]

    for group_label, (r1, r2, r3) in labels:
        print(f"\n  {group_label}")
        print(f"    {'Tied vs Untied Input':<41} {r1:>8.3f}")
        print(f"    {'Tied vs Untied Output':<41} {r2:>8.3f}")
        print(f"    {'Untied Input vs Untied Output':<41} {r3:>8.3f}")

    # ── Expected values from paper ─────────────────────────────────
    print(f"\n{sep}")
    print("EXPECTED (paper):")
    print(sep)
    expected = [
        ("OLMo-1B", [(0.496, 0.733, 0.455)]),
        ("Qwen3", [(0.366, 0.710, 0.366)]),
        ("GPT-Neo / Pythia", [(0.372, 0.408, 0.611)]),
    ]
    for group_label, vals in expected:
        r1, r2, r3 = vals[0]
        print(f"\n  {group_label}")
        print(f"    {'Tied vs Untied Input':<41} {r1:>8.3f}")
        print(f"    {'Tied vs Untied Output':<41} {r2:>8.3f}")
        print(f"    {'Untied Input vs Untied Output':<41} {r3:>8.3f}")


if __name__ == "__main__":
    main()
