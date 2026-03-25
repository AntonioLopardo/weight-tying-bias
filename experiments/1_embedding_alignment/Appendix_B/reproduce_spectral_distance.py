#!/usr/bin/env python3
"""
Spectral distance between embedding spaces via omnibus embedding.

Implements the spectral distance from:
    "Comparing Foundation Models using Data Kernels" (arxiv 2305.05126)

Algorithm:
  1. Build k-NN adjacency matrices A_a, A_b (symmetric, hollow)
  2. Construct omnibus matrix M = [[A_a, (A_a+A_b)/2], [(A_a+A_b)/2, A_b]]
  3. Adjacency Spectral Embedding (ASE): compute top eigenpairs of M via eigsh
  4. Latent positions Z = U * sqrt(|S|), split into Z_a, Z_b
  5. Spectral distance = ||Z_a - Z_b||_2 / min(||Z_a||_2, ||Z_b||_2)

Usage:
    python reproduce_spectral_distance.py
    python reproduce_spectral_distance.py --k 64 --n-components 128
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../../.."))
from utils.embedding_utils import load_embeddings_from_hf


def build_knn_adj(emb: torch.Tensor, k: int, device: torch.device) -> sparse.csr_matrix:
    """Build symmetric hollow k-NN adjacency matrix from an embedding matrix."""
    V = emb.shape[0]
    emb_norm = emb.to(device) / (emb.to(device).norm(dim=1, keepdim=True) + 1e-8)

    rows = np.empty(V * k, dtype=np.int32)
    cols = np.empty(V * k, dtype=np.int32)

    chunk_size = 4096
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        sim = emb_norm[start:end] @ emb_norm.T
        sim[torch.arange(end - start), torch.arange(start, end)] = -float("inf")
        topk_idx = sim.topk(k, dim=1).indices.cpu().numpy()  # (chunk, k)

        chunk_range = np.arange(start, end)
        sl = slice(start * k, end * k)
        rows[sl] = np.repeat(chunk_range, k)
        cols[sl] = topk_idx.ravel()

    A_directed = sparse.csr_matrix((np.ones(V * k, dtype=np.float32), (rows, cols)), shape=(V, V))
    A = (A_directed + A_directed.T) / 2
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _spectral_norm(Z: np.ndarray) -> float:
    return torch.linalg.svdvals(torch.from_numpy(Z).float()).max().item()


def spectral_distance(emb_a: torch.Tensor, emb_b: torch.Tensor,
                      k: int = 64, n_components: int = 128,
                      device: torch.device = torch.device("cpu")) -> float:
    """Compute spectral distance between two embedding spaces.

    Args:
        emb_a: (V, d1) embedding matrix
        emb_b: (V, d2) embedding matrix (same V, dimensions may differ)
        k: number of nearest neighbors for graph construction
        n_components: ASE embedding dimension

    Returns:
        Spectral distance: ||Z_a - Z_b||_2 / min(||Z_a||_2, ||Z_b||_2)
    """
    V = emb_a.shape[0]
    assert emb_b.shape[0] == V, f"Vocab mismatch: {emb_a.shape[0]} vs {emb_b.shape[0]}"

    print(f"  Building k-NN graphs (k={k}, V={V}, device={device})...")
    A_a = build_knn_adj(emb_a, k, device)
    A_b = build_knn_adj(emb_b, k, device)

    mean_ab = (A_a + A_b) * 0.5
    M = sparse.bmat([[A_a, mean_ab], [mean_ab, A_b]], format="csr")

    print(f"  ASE: eigsh on {M.shape[0]}x{M.shape[1]} sparse matrix (d={n_components})...")
    eigenvalues, eigenvectors = eigsh(M, k=n_components, which="LM")

    Z = eigenvectors * np.sqrt(np.abs(eigenvalues))
    Z_a, Z_b = Z[:V], Z[V:]

    diff_norm = _spectral_norm(Z_a - Z_b)
    min_norm = min(_spectral_norm(Z_a), _spectral_norm(Z_b))
    return diff_norm / (min_norm + 1e-8)


def get_aligned_vocab_indices(model_id_a: str, model_id_b: str):
    """Return (indices_a, indices_b) for tokens common to both tokenizers."""
    tok_a = AutoTokenizer.from_pretrained(model_id_a)
    tok_b = AutoTokenizer.from_pretrained(model_id_b)
    vocab_a = tok_a.get_vocab()
    vocab_b = tok_b.get_vocab()
    common = sorted(set(vocab_a) & set(vocab_b))
    print(f"  Vocab alignment: {len(vocab_a)} x {len(vocab_b)} -> {len(common)} common tokens")
    idx_a = torch.tensor([vocab_a[t] for t in common], dtype=torch.long)
    idx_b = torch.tensor([vocab_b[t] for t in common], dtype=torch.long)
    return idx_a, idx_b


def compute_group(tied, untied, label, k, n_components, device,
                  idx_tied=None, idx_untied=None):
    """Run the 3 spectral distance comparisons for one model pair.

    Returns (d_tied_input, d_tied_output, d_input_output).
    """
    emb_t  = tied["input_emb"]
    emb_ui = untied["input_emb"]
    emb_uo = untied["output_emb"]

    if idx_tied is not None:
        emb_t  = emb_t[idx_tied]
        emb_ui = emb_ui[idx_untied]
        emb_uo = emb_uo[idx_untied]

    print(f"\n  {label} — Tied vs Untied Input")
    d1 = spectral_distance(emb_t, emb_ui, k, n_components, device)
    print(f"  => {d1:.6f}")

    print(f"\n  {label} — Tied vs Untied Output")
    d2 = spectral_distance(emb_t, emb_uo, k, n_components, device)
    print(f"  => {d2:.6f}")

    print(f"\n  {label} — Untied Input vs Untied Output")
    d3 = spectral_distance(emb_ui, emb_uo, k, n_components, device)
    print(f"  => {d3:.6f}")

    return d1, d2, d3


def main():
    parser = argparse.ArgumentParser(
        description="Compute spectral distance between embedding spaces (arxiv 2305.05126)"
    )
    parser.add_argument("--k", type=int, default=64,
                        help="Number of nearest neighbors for graph construction (default: 64)")
    parser.add_argument("--n-components", type=int, default=128,
                        help="ASE embedding dimension (default: 128)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for k-NN computation, e.g. cuda:0 (default: auto)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 60)
    print("Spectral Distance — Omnibus Embedding (arxiv 2305.05126)")
    print(f"  k={args.k}, n_components={args.n_components}, device={device}")
    print("=" * 60)

    # ── Group 1: OLMo-1B (tied) vs OLMo-1B-0724 (untied) ──────────
    print("\n[1/3] OLMo-1B (tied) vs OLMo-1B-0724 (untied)")
    olmo_tied   = load_embeddings_from_hf("allenai/OLMo-1B-hf")
    olmo_untied = load_embeddings_from_hf("allenai/OLMo-1B-0724-hf")
    olmo_r = compute_group(olmo_tied, olmo_untied, "OLMo-1B",
                           args.k, args.n_components, device)
    del olmo_tied, olmo_untied

    # ── Group 2: Qwen3-4B (tied) vs Qwen3-8B (untied) ─────────────
    print("\n[2/3] Qwen3-4B (tied) vs Qwen3-8B (untied)")
    qwen_tied   = load_embeddings_from_hf("Qwen/Qwen3-4B")
    qwen_untied = load_embeddings_from_hf("Qwen/Qwen3-8B")
    qwen_r = compute_group(qwen_tied, qwen_untied, "Qwen3",
                           args.k, args.n_components, device)
    del qwen_tied, qwen_untied

    # ── Group 3: GPT-Neo-2.7B (tied) vs Pythia-2.8B (untied) ──────
    print("\n[3/3] GPT-Neo-2.7B (tied) vs Pythia-2.8B (untied)")
    gptneo = load_embeddings_from_hf("EleutherAI/gpt-neo-2.7B")
    pythia = load_embeddings_from_hf("EleutherAI/pythia-2.8b")
    idx_neo, idx_py = get_aligned_vocab_indices("EleutherAI/gpt-neo-2.7B", "EleutherAI/pythia-2.8b")
    eleuther_r = compute_group(gptneo, pythia, "GPT-Neo / Pythia",
                               args.k, args.n_components, device,
                               idx_tied=idx_neo, idx_untied=idx_py)
    del gptneo, pythia

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS — Spectral Distance (lower = more similar)")
    print("=" * 60)

    groups = [
        ("OLMo-1B (tied) vs OLMo-1B-0724 (untied)", olmo_r),
        ("Qwen3-4B (tied) vs Qwen3-8B (untied)",     qwen_r),
        ("GPT-Neo-2.7B (tied) vs Pythia-2.8B (untied)", eleuther_r),
    ]
    row_labels = ["Tied vs Untied Input", "Tied vs Untied Output", "Untied Input vs Untied Output"]

    for group_label, (d1, d2, d3) in groups:
        print(f"\n  {group_label}")
        print(f"    {'Tied vs Untied Input':<41} {d1:>10.6f}")
        print(f"    {'Tied vs Untied Output':<41} {d2:>10.6f}  **")
        print(f"    {'Untied Input vs Untied Output':<41} {d3:>10.6f}")

    print()
    for group_label, (d1, d2, d3) in groups:
        if d2 < d1 and d2 < d3:
            print(f"  {group_label.split('(')[0].strip()}: Output(U) closest to Tied — consistent with output-bias hypothesis.")
        else:
            print(f"  WARN {group_label}: ordering does not match expected pattern.")


if __name__ == "__main__":
    main()
