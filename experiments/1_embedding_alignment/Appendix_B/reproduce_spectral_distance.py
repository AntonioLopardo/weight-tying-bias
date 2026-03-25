#!/usr/bin/env python3
"""
Spectral distance between embedding spaces via omnibus embedding.

Implements the spectral distance from:
    "Comparing Foundation Models using Data Kernels" (arxiv 2305.05126)

Algorithm:
  1. Build k-NN adjacency matrices A_a, A_b (symmetric, hollow)
  2. Construct omnibus matrix M = [[A_a, (A_a+A_b)/2], [(A_a+A_b)/2, A_b]]
  3. Adjacency Spectral Embedding (ASE): compute top eigenpairs of M via eigsh
  4. Latent positions Z = U * sqrt(|S|), split into Z_a (first V) and Z_b (last V)
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

    Implements the omnibus embedding + ASE approach from arxiv 2305.05126.

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

    print(f"  ASE: eigsh on {M.shape[0]}×{M.shape[1]} sparse matrix (d={n_components})...")
    eigenvalues, eigenvectors = eigsh(M, k=n_components, which="LM")

    Z = eigenvectors * np.sqrt(np.abs(eigenvalues))
    Z_a, Z_b = Z[:V], Z[V:]

    diff_norm = _spectral_norm(Z_a - Z_b)
    min_norm = min(_spectral_norm(Z_a), _spectral_norm(Z_b))
    return diff_norm / (min_norm + 1e-8)


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

    print("\nLoading OLMo models...")
    tied = load_embeddings_from_hf("allenai/OLMo-1B-hf")
    untied = load_embeddings_from_hf("allenai/OLMo-1B-0724-hf")

    emb_tied = tied["input_emb"]
    emb_ui = untied["input_emb"]
    emb_uo = untied["output_emb"]
    del tied, untied

    comparisons = [
        ("Input (U) → Output (U)", emb_ui, emb_uo),
        ("Output (U) → Tied",      emb_uo, emb_tied),
        ("Input (U) → Tied",       emb_ui, emb_tied),
    ]

    print("\nComputing spectral distances...")
    results = []
    for i, (label, a, b) in enumerate(comparisons, 1):
        print(f"\n[{i}/{len(comparisons)}] {label}")
        d = spectral_distance(a, b, k=args.k, n_components=args.n_components, device=device)
        print(f"  => {d:.6f}")
        results.append((label, d))

    print("\n" + "=" * 60)
    print("RESULTS — Spectral Distance (lower = more similar)")
    print("=" * 60)
    print(f"\n  {'Comparison':<35} {'Distance':>10}")
    print("  " + "-" * 47)
    for label, d in results:
        print(f"  {label:<35} {d:>10.6f}")

    d_in_out, d_out_tied, d_in_tied = [d for _, d in results]
    print()
    if d_out_tied < d_in_out and d_out_tied < d_in_tied:
        print("Output(U) is closer to Tied than Input(U) — consistent with output-bias hypothesis.")
    else:
        print("WARN: Ordering does not match expected pattern. Try adjusting --k or --n-components.")


if __name__ == "__main__":
    main()
