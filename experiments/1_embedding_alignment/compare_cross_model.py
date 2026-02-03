"""
Cross-Model Embedding Comparison Script

Compares embeddings from two different models:
- OLMo-1B-0724-hf (untied: separate input and output embeddings)
- OLMo-1B-hf (tied: shared input/output embedding)

Inspired by "Emerging Cross-lingual Structure in Pretrained Language Models" (ACL 2020)
https://aclanthology.org/2020.acl-main.536.pdf
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig


def load_embeddings(model_id: str) -> dict:
    """Load embeddings from a HuggingFace model."""
    print(f"\nLoading: {model_id}")
    
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    state_dict = model.state_dict()
    weight_tying = config.tie_word_embeddings if hasattr(config, "tie_word_embeddings") else True
    
    # Extract embeddings
    input_emb = state_dict["model.embed_tokens.weight"]
    
    if weight_tying:
        output_emb = input_emb
    else:
        output_emb = state_dict["lm_head.weight"]
    
    print(f"  Weight tying: {weight_tying}")
    print(f"  Input shape: {input_emb.shape}")
    print(f"  Output shape: {output_emb.shape}")
    
    # Clear model to free memory
    del model
    
    return {
        "model_id": model_id,
        "input_emb": input_emb.clone(),
        "output_emb": output_emb.clone(),
        "weight_tying": weight_tying,
    }


def compute_similarity_stats(emb1: torch.Tensor, emb2: torch.Tensor, name1: str, name2: str) -> dict:
    """Compute similarity metrics between two embedding matrices."""
    # Normalize rows for cosine similarity
    emb1_norm = emb1 / (emb1.norm(dim=1, keepdim=True) + 1e-8)
    emb2_norm = emb2 / (emb2.norm(dim=1, keepdim=True) + 1e-8)
    
    # Per-token cosine similarity
    cos_sim = (emb1_norm * emb2_norm).sum(dim=1)
    
    # Global Pearson correlation
    flat1 = emb1.flatten().float()
    flat2 = emb2.flatten().float()
    mean1, mean2 = flat1.mean(), flat2.mean()
    cov = ((flat1 - mean1) * (flat2 - mean2)).mean()
    std1, std2 = flat1.std(), flat2.std()
    pearson = (cov / (std1 * std2 + 1e-8)).item()
    
    # Frobenius norms
    diff_norm = (emb1 - emb2).norm().item()
    emb1_fnorm = emb1.norm().item()
    emb2_fnorm = emb2.norm().item()
    
    # Procrustes alignment
    emb1_f = emb1.float()
    emb2_f = emb2.float()
    emb1_centered = emb1_f - emb1_f.mean(dim=0, keepdim=True)
    emb2_centered = emb2_f - emb2_f.mean(dim=0, keepdim=True)
    
    M = emb2_centered.T @ emb1_centered
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    
    emb1_aligned = emb1_centered @ W.T
    emb1_aligned_norm = emb1_aligned / (emb1_aligned.norm(dim=1, keepdim=True) + 1e-8)
    emb2_centered_norm = emb2_centered / (emb2_centered.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim_aligned = (emb1_aligned_norm * emb2_centered_norm).sum(dim=1)
    
    return {
        "comparison": f"{name1} vs {name2}",
        "cos_sim_mean": cos_sim.mean().item(),
        "cos_sim_std": cos_sim.std().item(),
        "cos_sim_median": cos_sim.median().item(),
        "pearson": pearson,
        "frobenius_diff": diff_norm,
        "frobenius_1": emb1_fnorm,
        "frobenius_2": emb2_fnorm,
        "cos_sim_after_procrustes_mean": cos_sim_aligned.mean().item(),
        "cos_sim_after_procrustes_std": cos_sim_aligned.std().item(),
    }


def print_comparison(stats: dict):
    """Print comparison results."""
    print(f"\n{'='*60}")
    print(f"Comparison: {stats['comparison']}")
    print(f"{'='*60}")
    print(f"  Per-token Cosine Similarity:")
    print(f"    Mean:   {stats['cos_sim_mean']:.6f}")
    print(f"    Std:    {stats['cos_sim_std']:.6f}")
    print(f"    Median: {stats['cos_sim_median']:.6f}")
    print(f"  Global Metrics:")
    print(f"    Pearson correlation:  {stats['pearson']:.6f}")
    print(f"    Frobenius diff norm:  {stats['frobenius_diff']:.4f}")
    print(f"    Norm (first):         {stats['frobenius_1']:.4f}")
    print(f"    Norm (second):        {stats['frobenius_2']:.4f}")
    print(f"  After Procrustes Alignment:")
    print(f"    Cosine sim (mean):    {stats['cos_sim_after_procrustes_mean']:.6f}")
    print(f"    Cosine sim (std):     {stats['cos_sim_after_procrustes_std']:.6f}")


def main():
    # Load both models
    untied = load_embeddings("allenai/OLMo-1B-0724-hf")
    tied = load_embeddings("allenai/OLMo-1B-hf")
    
    print("\n" + "="*60)
    print("CROSS-MODEL EMBEDDING COMPARISON")
    print("="*60)
    print("\nComparing embeddings between:")
    print(f"  - OLMo-1B-0724-hf (untied: separate input/output)")
    print(f"  - OLMo-1B-hf (tied: shared embedding)")
    
    # Comparison 1: Untied Input vs Tied
    stats1 = compute_similarity_stats(
        untied["input_emb"], tied["input_emb"],
        "Untied-Input (0724)", "Tied (original)"
    )
    print_comparison(stats1)
    
    # Comparison 2: Untied Output vs Tied
    stats2 = compute_similarity_stats(
        untied["output_emb"], tied["input_emb"],
        "Untied-Output (0724)", "Tied (original)"
    )
    print_comparison(stats2)
    
    # Comparison 3: Untied Input vs Untied Output (within 0724)
    stats3 = compute_similarity_stats(
        untied["input_emb"], untied["output_emb"],
        "Untied-Input (0724)", "Untied-Output (0724)"
    )
    print_comparison(stats3)
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"\n{'Comparison':<45} {'Cos(μ)':<10} {'Pearson':<10} {'Procrustes':<10}")
    print("-"*75)
    for s in [stats1, stats2, stats3]:
        name = s['comparison'][:43]
        print(f"{name:<45} {s['cos_sim_mean']:<10.4f} {s['pearson']:<10.4f} {s['cos_sim_after_procrustes_mean']:<10.4f}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()

