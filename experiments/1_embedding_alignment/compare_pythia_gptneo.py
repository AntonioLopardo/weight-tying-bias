"""
Cross-Model Embedding Comparison: Pythia-2.8B vs GPT-Neo-2.7B

Pythia-2.8B uses untied embeddings, GPT-Neo-2.7B uses tied embeddings.
Both are from EleutherAI and have similar sizes.
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
    
    # Find embedding keys
    input_emb = None
    output_emb = None
    
    for key in state_dict.keys():
        if any(p in key.lower() for p in ["embed_tokens.weight", "wte.weight", "embed_in.weight"]):
            input_emb = state_dict[key]
            print(f"  Found input: {key}")
        if any(p in key.lower() for p in ["lm_head.weight", "embed_out.weight"]):
            output_emb = state_dict[key]
            print(f"  Found output: {key}")
    
    if weight_tying or output_emb is None:
        output_emb = input_emb
    
    print(f"  Weight tying: {weight_tying}")
    print(f"  Input shape: {input_emb.shape}")
    print(f"  Output shape: {output_emb.shape}")
    
    del model
    
    return {
        "model_id": model_id,
        "input_emb": input_emb.clone(),
        "output_emb": output_emb.clone(),
        "weight_tying": weight_tying,
    }


def compute_similarity(emb1: torch.Tensor, emb2: torch.Tensor, name1: str, name2: str) -> dict:
    """Compute similarity metrics between two embedding matrices."""
    # Handle different vocab sizes - use common subset
    min_vocab = min(emb1.shape[0], emb2.shape[0])
    emb1 = emb1[:min_vocab]
    emb2 = emb2[:min_vocab]
    
    # Normalize for cosine similarity
    emb1_norm = emb1 / (emb1.norm(dim=1, keepdim=True) + 1e-8)
    emb2_norm = emb2 / (emb2.norm(dim=1, keepdim=True) + 1e-8)
    
    # Per-token cosine similarity
    cos_sim = (emb1_norm * emb2_norm).sum(dim=1)
    
    # Pearson correlation
    flat1 = emb1.flatten().float()
    flat2 = emb2.flatten().float()
    mean1, mean2 = flat1.mean(), flat2.mean()
    cov = ((flat1 - mean1) * (flat2 - mean2)).mean()
    std1, std2 = flat1.std(), flat2.std()
    pearson = (cov / (std1 * std2 + 1e-8)).item()
    
    # Procrustes alignment
    emb1_f = emb1.float()
    emb2_f = emb2.float()
    emb1_c = emb1_f - emb1_f.mean(dim=0, keepdim=True)
    emb2_c = emb2_f - emb2_f.mean(dim=0, keepdim=True)
    
    M = emb2_c.T @ emb1_c
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    
    emb1_aligned = emb1_c @ W.T
    emb1_al_norm = emb1_aligned / (emb1_aligned.norm(dim=1, keepdim=True) + 1e-8)
    emb2_c_norm = emb2_c / (emb2_c.norm(dim=1, keepdim=True) + 1e-8)
    cos_aligned = (emb1_al_norm * emb2_c_norm).sum(dim=1)
    
    return {
        "comparison": f"{name1} vs {name2}",
        "vocab_used": min_vocab,
        "cos_sim_mean": cos_sim.mean().item(),
        "cos_sim_std": cos_sim.std().item(),
        "pearson": pearson,
        "procrustes_mean": cos_aligned.mean().item(),
        "procrustes_std": cos_aligned.std().item(),
        "norm1": emb1.norm().item(),
        "norm2": emb2.norm().item(),
    }


def print_comparison(stats: dict):
    """Print comparison results."""
    print(f"\n{'='*60}")
    print(f"{stats['comparison']}")
    print(f"{'='*60}")
    print(f"  Vocab tokens used: {stats['vocab_used']}")
    print(f"  Per-token Cosine Similarity:")
    print(f"    Mean: {stats['cos_sim_mean']:.6f}")
    print(f"    Std:  {stats['cos_sim_std']:.6f}")
    print(f"  Pearson correlation: {stats['pearson']:.6f}")
    print(f"  After Procrustes:")
    print(f"    Mean: {stats['procrustes_mean']:.6f}")
    print(f"    Std:  {stats['procrustes_std']:.6f}")
    print(f"  Frobenius norms: {stats['norm1']:.2f} vs {stats['norm2']:.2f}")


def main():
    # Load both models
    pythia = load_embeddings("EleutherAI/pythia-2.8b")
    gptneo = load_embeddings("EleutherAI/gpt-neo-2.7B")
    
    print("\n" + "="*70)
    print("CROSS-MODEL EMBEDDING COMPARISON: Pythia-2.8B vs GPT-Neo-2.7B")
    print("="*70)
    
    print("\n### Model Configurations ###")
    print(f"\nPythia-2.8B:")
    print(f"  Weight tying: {pythia['weight_tying']}")
    print(f"  Vocab size: {pythia['input_emb'].shape[0]}")
    print(f"  Hidden dim: {pythia['input_emb'].shape[1]}")
    
    print(f"\nGPT-Neo-2.7B:")
    print(f"  Weight tying: {gptneo['weight_tying']}")
    print(f"  Vocab size: {gptneo['input_emb'].shape[0]}")
    print(f"  Hidden dim: {gptneo['input_emb'].shape[1]}")
    
    # Same hidden dim, can compare directly
    assert pythia['input_emb'].shape[1] == gptneo['input_emb'].shape[1], "Hidden dims must match"
    
    # Comparisons
    comparisons = []
    
    # 1. Pythia Input vs GPT-Neo (tied)
    stats1 = compute_similarity(
        pythia["input_emb"], gptneo["input_emb"],
        "Pythia Input", "GPT-Neo (tied)"
    )
    comparisons.append(stats1)
    print_comparison(stats1)
    
    # 2. Pythia Output vs GPT-Neo (tied)
    stats2 = compute_similarity(
        pythia["output_emb"], gptneo["input_emb"],
        "Pythia Output", "GPT-Neo (tied)"
    )
    comparisons.append(stats2)
    print_comparison(stats2)
    
    # 3. Pythia Input vs Pythia Output (within model)
    stats3 = compute_similarity(
        pythia["input_emb"], pythia["output_emb"],
        "Pythia Input", "Pythia Output"
    )
    comparisons.append(stats3)
    print_comparison(stats3)
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Comparison':<35} {'Cos(μ)':<10} {'Pearson':<10} {'Procrustes':<10}")
    print("-"*65)
    for s in comparisons:
        name = s['comparison'][:33]
        print(f"{name:<35} {s['cos_sim_mean']:<10.4f} {s['pearson']:<10.4f} {s['procrustes_mean']:<10.4f}")
    
    # Per-token norm correlation analysis
    print("\n### Per-Token Norm Correlations ###")
    norms_pythia_in = pythia['input_emb'][:50257].norm(dim=1)
    norms_pythia_out = pythia['output_emb'][:50257].norm(dim=1)
    norms_gptneo = gptneo['input_emb'].norm(dim=1)
    
    corr_in_neo = torch.corrcoef(torch.stack([norms_pythia_in, norms_gptneo]))[0, 1].item()
    corr_out_neo = torch.corrcoef(torch.stack([norms_pythia_out, norms_gptneo]))[0, 1].item()
    corr_in_out = torch.corrcoef(torch.stack([norms_pythia_in, norms_pythia_out]))[0, 1].item()
    
    print(f"  Pythia Input vs GPT-Neo:   {corr_in_neo:.4f}")
    print(f"  Pythia Output vs GPT-Neo:  {corr_out_neo:.4f}")
    print(f"  Pythia Input vs Output:    {corr_in_out:.4f}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()

