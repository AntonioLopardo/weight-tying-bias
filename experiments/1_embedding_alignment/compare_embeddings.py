"""
Embedding Similarity Analysis Script

Inspired by "Emerging Cross-lingual Structure in Pretrained Language Models" (ACL 2020)
https://aclanthology.org/2020.acl-main.536.pdf

This script compares input and output embedding matrices from a language model checkpoint,
computing various similarity metrics to understand their relationship.

The paper shows that embedding spaces tend to have similar structure and can be aligned
effectively. This analysis extends to comparing tied vs untied embeddings.

Supports:
- Local OLMo checkpoints (--checkpoint path)
- HuggingFace models (--hf-model model_id)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def load_from_huggingface(model_id: str, revision: str = None) -> tuple[dict, dict]:
    """Load model from HuggingFace and extract state dict."""
    from transformers import AutoModelForCausalLM, AutoConfig
    
    print(f"Loading model from HuggingFace: {model_id}")
    if revision:
        print(f"  Revision: {revision}")
    
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    state_dict = model.state_dict()
    
    # Convert HF config to our format
    config_dict = {
        "model": {
            "weight_tying": config.tie_word_embeddings if hasattr(config, "tie_word_embeddings") else True,
            "d_model": config.hidden_size if hasattr(config, "hidden_size") else config.d_model,
            "vocab_size": config.vocab_size,
        }
    }
    
    return state_dict, config_dict, "huggingface"


def load_checkpoint(checkpoint_path: str) -> tuple[dict, dict, str]:
    """Load model checkpoint and config."""
    checkpoint_dir = Path(checkpoint_path)
    
    # Load config
    config_path = checkpoint_dir / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load model state dict
    model_path = checkpoint_dir / "model.pt"
    state_dict = torch.load(model_path, map_location="cpu")
    
    return state_dict, config, "olmo"


def extract_embeddings(state_dict: dict, config: dict, model_format: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract input and output embedding matrices from state dict.
    
    Returns:
        input_emb: (vocab_size, d_model) - input embedding matrix
        output_emb: (vocab_size, d_model) - output projection weight matrix
    """
    # Check if weight tying is used
    weight_tying = config.get("model", {}).get("weight_tying", True)
    
    if model_format == "huggingface":
        # HuggingFace format - handle various model architectures
        # Input embedding patterns: embed_tokens, wte, embed_in
        input_emb = None
        for key in state_dict.keys():
            if any(p in key.lower() for p in ["embed_tokens.weight", "wte.weight", "embed_in.weight"]):
                input_emb = state_dict[key]
                print(f"Found input embedding at: {key}")
                break
        
        if input_emb is None:
            # Fallback: find any embedding key
            emb_keys = [k for k in state_dict.keys() if "embed" in k.lower() and "weight" in k and "out" not in k.lower()]
            if emb_keys:
                input_emb = state_dict[emb_keys[0]]
                print(f"Found input embedding at: {emb_keys[0]}")
            else:
                raise KeyError(f"Could not find input embedding. Keys: {list(state_dict.keys())[:10]}...")
        
        if weight_tying:
            print("Note: weight_tying=True, input and output embeddings are the same matrix")
            output_emb = input_emb
        else:
            # Output projection patterns: lm_head, ff_out, embed_out
            output_emb = None
            for key in state_dict.keys():
                if any(p in key.lower() for p in ["lm_head.weight", "ff_out.weight", "embed_out.weight"]):
                    output_emb = state_dict[key]
                    print(f"Found output embedding at: {key}")
                    break
            
            if output_emb is None:
                # Fallback: find any output/head key
                lm_keys = [k for k in state_dict.keys() if any(p in k.lower() for p in ["lm_head", "ff_out", "embed_out", "output"]) and "weight" in k]
                if lm_keys:
                    output_emb = state_dict[lm_keys[0]]
                    print(f"Found output embedding at: {lm_keys[0]}")
                else:
                    raise KeyError(f"Could not find output embedding. Keys: {list(state_dict.keys())[:10]}...")
    else:
        # Native OLMo checkpoint format
        input_emb = state_dict["transformer.wte.weight"]
        
        if weight_tying:
            print("Note: weight_tying=True, input and output embeddings are the same matrix")
            output_emb = input_emb
        else:
            # Output projection: transformer.ff_out.weight
            output_emb = state_dict["transformer.ff_out.weight"]
    
    return input_emb, output_emb


def compute_cosine_similarity_stats(emb1: torch.Tensor, emb2: torch.Tensor) -> dict:
    """Compute per-row cosine similarity between two embedding matrices."""
    # Normalize rows
    emb1_norm = emb1 / (emb1.norm(dim=1, keepdim=True) + 1e-8)
    emb2_norm = emb2 / (emb2.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute per-token cosine similarity
    cos_sim = (emb1_norm * emb2_norm).sum(dim=1)
    
    return {
        "mean": cos_sim.mean().item(),
        "std": cos_sim.std().item(),
        "min": cos_sim.min().item(),
        "max": cos_sim.max().item(),
        "median": cos_sim.median().item(),
        "raw": cos_sim,
    }


def compute_global_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> dict:
    """Compute global similarity metrics between embedding matrices."""
    # Flatten and compute correlation
    flat1 = emb1.flatten().float()
    flat2 = emb2.flatten().float()
    
    # Pearson correlation
    mean1, mean2 = flat1.mean(), flat2.mean()
    cov = ((flat1 - mean1) * (flat2 - mean2)).mean()
    std1, std2 = flat1.std(), flat2.std()
    pearson = (cov / (std1 * std2 + 1e-8)).item()
    
    # Frobenius norm of difference
    diff_norm = (emb1 - emb2).norm().item()
    emb1_norm = emb1.norm().item()
    emb2_norm = emb2.norm().item()
    relative_diff = diff_norm / ((emb1_norm + emb2_norm) / 2)
    
    return {
        "pearson_correlation": pearson,
        "frobenius_diff_norm": diff_norm,
        "relative_frobenius_diff": relative_diff,
        "emb1_frobenius_norm": emb1_norm,
        "emb2_frobenius_norm": emb2_norm,
    }


def compute_singular_value_alignment(emb1: torch.Tensor, emb2: torch.Tensor, top_k: int = 50) -> dict:
    """Compare singular values of the two embedding matrices.
    
    This helps understand if the matrices have similar spectral properties,
    which is relevant for understanding representational similarity.
    """
    # Convert to float for SVD
    emb1_f = emb1.float()
    emb2_f = emb2.float()
    
    # Compute SVD (only singular values)
    _, s1, _ = torch.linalg.svd(emb1_f, full_matrices=False)
    _, s2, _ = torch.linalg.svd(emb2_f, full_matrices=False)
    
    # Take top-k singular values
    s1_topk = s1[:top_k]
    s2_topk = s2[:top_k]
    
    # Normalize
    s1_norm = s1_topk / s1_topk[0]
    s2_norm = s2_topk / s2_topk[0]
    
    # Compute correlation of singular value spectra
    correlation = torch.corrcoef(torch.stack([s1_norm, s2_norm]))[0, 1].item()
    
    # L2 distance between normalized spectra
    spectral_diff = (s1_norm - s2_norm).norm().item()
    
    return {
        "sv_correlation": correlation,
        "sv_spectral_diff": spectral_diff,
        "sv_top1_ratio": (s1[0] / s2[0]).item(),
        "sv_top10_input": s1[:10].tolist(),
        "sv_top10_output": s2[:10].tolist(),
    }


def compute_procrustes_alignment(emb1: torch.Tensor, emb2: torch.Tensor) -> dict:
    """Compute Procrustes alignment between embedding matrices.
    
    Finds the orthogonal transformation that best aligns emb1 to emb2.
    This is related to the cross-lingual embedding alignment methods
    discussed in the paper (Mikolov et al., 2013; Conneau et al., 2017).
    """
    # Center the matrices
    emb1_f = emb1.float()
    emb2_f = emb2.float()
    emb1_centered = emb1_f - emb1_f.mean(dim=0, keepdim=True)
    emb2_centered = emb2_f - emb2_f.mean(dim=0, keepdim=True)
    
    # Compute SVD of the cross-covariance matrix
    # W = U @ V^T where U, _, V = svd(emb2.T @ emb1)
    M = emb2_centered.T @ emb1_centered
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vt  # Optimal orthogonal mapping
    
    # Apply transformation
    emb1_aligned = emb1_centered @ W.T
    
    # Compute alignment error
    alignment_error = (emb1_aligned - emb2_centered).norm().item()
    
    # Compute cosine similarity after alignment
    emb1_aligned_norm = emb1_aligned / (emb1_aligned.norm(dim=1, keepdim=True) + 1e-8)
    emb2_centered_norm = emb2_centered / (emb2_centered.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim_aligned = (emb1_aligned_norm * emb2_centered_norm).sum(dim=1)
    
    return {
        "alignment_error": alignment_error,
        "cos_sim_after_alignment_mean": cos_sim_aligned.mean().item(),
        "cos_sim_after_alignment_std": cos_sim_aligned.std().item(),
    }


def compute_subspace_overlap(emb1: torch.Tensor, emb2: torch.Tensor, k: int = 100) -> dict:
    """Compute overlap between principal subspaces.
    
    Uses canonical correlation analysis (CCA) style metrics to measure
    how much the top-k principal components of each matrix overlap.
    """
    emb1_f = emb1.float()
    emb2_f = emb2.float()
    
    # Center
    emb1_centered = emb1_f - emb1_f.mean(dim=0, keepdim=True)
    emb2_centered = emb2_f - emb2_f.mean(dim=0, keepdim=True)
    
    # Get top-k principal components
    _, _, V1 = torch.linalg.svd(emb1_centered, full_matrices=False)
    _, _, V2 = torch.linalg.svd(emb2_centered, full_matrices=False)
    
    V1_k = V1[:k]  # (k, d_model)
    V2_k = V2[:k]  # (k, d_model)
    
    # Compute canonical correlations (singular values of V1 @ V2.T)
    cross = V1_k @ V2_k.T  # (k, k)
    canonical_corrs = torch.linalg.svdvals(cross)
    
    # Subspace overlap metrics
    return {
        "mean_canonical_correlation": canonical_corrs.mean().item(),
        "min_canonical_correlation": canonical_corrs.min().item(),
        "top10_canonical_correlations": canonical_corrs[:10].tolist(),
    }


def analyze_token_distribution(cos_sim: torch.Tensor, n_bins: int = 20) -> dict:
    """Analyze the distribution of per-token cosine similarities."""
    # Create histogram
    hist = torch.histogram(cos_sim.float(), bins=n_bins, range=(-1.0, 1.0))
    
    # Find percentiles
    percentiles = [10, 25, 50, 75, 90]
    pct_values = {
        f"p{p}": torch.quantile(cos_sim.float(), p / 100).item() 
        for p in percentiles
    }
    
    return {
        "histogram_counts": hist.hist.tolist(),
        "histogram_edges": hist.bin_edges.tolist(),
        **pct_values,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare input and output embedding matrices from an OLMo checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the unsharded checkpoint directory",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="HuggingFace model ID (e.g., allenai/OLMo-1B-0724-hf)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="HuggingFace model revision (e.g., step1000-tokens4B)",
    )
    parser.add_argument(
        "--top-k-sv",
        type=int,
        default=50,
        help="Number of top singular values to compare",
    )
    parser.add_argument(
        "--subspace-k",
        type=int,
        default=100,
        help="Dimension of subspace for overlap analysis",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )
    
    args = parser.parse_args()
    
    # Load model from HuggingFace or local checkpoint
    if args.hf_model:
        state_dict, config, model_format = load_from_huggingface(args.hf_model, args.revision)
        source = f"HuggingFace: {args.hf_model}" + (f" (revision: {args.revision})" if args.revision else "")
    elif args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        state_dict, config, model_format = load_checkpoint(args.checkpoint)
        source = f"Local: {args.checkpoint}"
    else:
        # Default to local checkpoint
        default_path = "/home/vec_norm/OLMo/checkpoints/OLMo-1B-0724-reproduce/step2000-unsharded"
        print(f"Loading checkpoint from: {default_path}")
        state_dict, config, model_format = load_checkpoint(default_path)
        source = f"Local: {default_path}"
    
    # Extract embeddings
    print(f"\nSource: {source}")
    print("Extracting embeddings...")
    input_emb, output_emb = extract_embeddings(state_dict, config, model_format)
    
    print(f"Input embedding shape: {input_emb.shape}")
    print(f"Output embedding shape: {output_emb.shape}")
    weight_tying = config.get('model', {}).get('weight_tying', True)
    print(f"Weight tying: {weight_tying}")
    
    # Check if they're the same tensor (weight tying)
    if input_emb.data_ptr() == output_emb.data_ptr():
        print("\nNote: Input and output embeddings share the same memory (weight tying).")
        print("All similarity metrics will be perfect (1.0).")
    
    # Compute similarity metrics
    print("\n" + "="*60)
    print("EMBEDDING SIMILARITY ANALYSIS")
    print("="*60)
    
    # 1. Per-token cosine similarity
    print("\n1. Per-Token Cosine Similarity:")
    print("-" * 40)
    cos_stats = compute_cosine_similarity_stats(input_emb, output_emb)
    print(f"   Mean:   {cos_stats['mean']:.6f}")
    print(f"   Std:    {cos_stats['std']:.6f}")
    print(f"   Median: {cos_stats['median']:.6f}")
    print(f"   Min:    {cos_stats['min']:.6f}")
    print(f"   Max:    {cos_stats['max']:.6f}")
    
    # 2. Global similarity
    print("\n2. Global Similarity Metrics:")
    print("-" * 40)
    global_stats = compute_global_similarity(input_emb, output_emb)
    print(f"   Pearson correlation:      {global_stats['pearson_correlation']:.6f}")
    print(f"   Frobenius diff norm:      {global_stats['frobenius_diff_norm']:.4f}")
    print(f"   Relative Frobenius diff:  {global_stats['relative_frobenius_diff']:.6f}")
    print(f"   Input Frobenius norm:     {global_stats['emb1_frobenius_norm']:.4f}")
    print(f"   Output Frobenius norm:    {global_stats['emb2_frobenius_norm']:.4f}")
    
    # 3. Singular value analysis
    print(f"\n3. Singular Value Analysis (top-{args.top_k_sv}):")
    print("-" * 40)
    sv_stats = compute_singular_value_alignment(input_emb, output_emb, top_k=args.top_k_sv)
    print(f"   SV spectrum correlation:  {sv_stats['sv_correlation']:.6f}")
    print(f"   SV spectral difference:   {sv_stats['sv_spectral_diff']:.6f}")
    print(f"   Top-1 SV ratio (in/out):  {sv_stats['sv_top1_ratio']:.6f}")
    print(f"   Top-5 SVs (input):  {[f'{v:.2f}' for v in sv_stats['sv_top10_input'][:5]]}")
    print(f"   Top-5 SVs (output): {[f'{v:.2f}' for v in sv_stats['sv_top10_output'][:5]]}")
    
    # 4. Procrustes alignment
    print("\n4. Procrustes Alignment (Orthogonal):")
    print("-" * 40)
    proc_stats = compute_procrustes_alignment(input_emb, output_emb)
    print(f"   Alignment error:              {proc_stats['alignment_error']:.4f}")
    print(f"   Cos sim after alignment (μ):  {proc_stats['cos_sim_after_alignment_mean']:.6f}")
    print(f"   Cos sim after alignment (σ):  {proc_stats['cos_sim_after_alignment_std']:.6f}")
    
    # 5. Subspace overlap
    print(f"\n5. Subspace Overlap (k={args.subspace_k}):")
    print("-" * 40)
    subspace_stats = compute_subspace_overlap(input_emb, output_emb, k=args.subspace_k)
    print(f"   Mean canonical correlation:   {subspace_stats['mean_canonical_correlation']:.6f}")
    print(f"   Min canonical correlation:    {subspace_stats['min_canonical_correlation']:.6f}")
    print(f"   Top-5 canonical correlations: {[f'{v:.4f}' for v in subspace_stats['top10_canonical_correlations'][:5]]}")
    
    # 6. Distribution analysis
    print("\n6. Cosine Similarity Distribution:")
    print("-" * 40)
    dist_stats = analyze_token_distribution(cos_stats["raw"])
    print(f"   10th percentile:  {dist_stats['p10']:.6f}")
    print(f"   25th percentile:  {dist_stats['p25']:.6f}")
    print(f"   50th percentile:  {dist_stats['p50']:.6f}")
    print(f"   75th percentile:  {dist_stats['p75']:.6f}")
    print(f"   90th percentile:  {dist_stats['p90']:.6f}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    
    # Save results if output path specified
    if args.output:
        import json
        
        results = {
            "source": source,
            "checkpoint": args.checkpoint,
            "hf_model": args.hf_model,
            "revision": args.revision,
            "input_embedding_shape": list(input_emb.shape),
            "output_embedding_shape": list(output_emb.shape),
            "weight_tying": weight_tying,
            "cosine_similarity": {k: v for k, v in cos_stats.items() if k != "raw"},
            "global_similarity": global_stats,
            "singular_values": sv_stats,
            "procrustes": proc_stats,
            "subspace_overlap": subspace_stats,
            "distribution": dist_stats,
        }
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

