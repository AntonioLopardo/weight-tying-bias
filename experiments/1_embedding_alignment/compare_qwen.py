"""
Cross-Model Embedding Comparison: Qwen3-4B vs Qwen3-8B

Qwen3-4B uses weight tying, Qwen3-8B does not.
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
    
    # Find embedding keys (Qwen uses different naming)
    input_key = None
    output_key = None
    for k in state_dict.keys():
        if "embed_tokens" in k:
            input_key = k
        if "lm_head" in k:
            output_key = k
    
    input_emb = state_dict[input_key]
    output_emb = state_dict[output_key] if output_key and not weight_tying else input_emb
    
    print(f"  Weight tying: {weight_tying}")
    print(f"  Input shape: {input_emb.shape}")
    print(f"  Output shape: {output_emb.shape}")
    print(f"  Hidden dim: {input_emb.shape[1]}")
    
    del model
    
    return {
        "model_id": model_id,
        "input_emb": input_emb.clone(),
        "output_emb": output_emb.clone(),
        "weight_tying": weight_tying,
        "hidden_dim": input_emb.shape[1],
    }


def compute_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> dict:
    """Compute similarity metrics - handles different hidden dims by projecting."""
    # Normalize for cosine similarity
    emb1_norm = emb1 / (emb1.norm(dim=1, keepdim=True) + 1e-8)
    emb2_norm = emb2 / (emb2.norm(dim=1, keepdim=True) + 1e-8)
    
    # Per-token cosine similarity (only if same hidden dim)
    if emb1.shape[1] == emb2.shape[1]:
        cos_sim = (emb1_norm * emb2_norm).sum(dim=1)
        cos_mean = cos_sim.mean().item()
        cos_std = cos_sim.std().item()
    else:
        cos_mean = None
        cos_std = None
    
    return {
        "cos_sim_mean": cos_mean,
        "cos_sim_std": cos_std,
        "emb1_norm": emb1.norm().item(),
        "emb2_norm": emb2.norm().item(),
        "emb1_shape": list(emb1.shape),
        "emb2_shape": list(emb2.shape),
    }


def main():
    # Load both Qwen3 models
    qwen4b = load_embeddings("Qwen/Qwen3-4B")
    qwen8b = load_embeddings("Qwen/Qwen3-8B")
    
    print("\n" + "="*70)
    print("QWEN3 EMBEDDING COMPARISON SUMMARY")
    print("="*70)
    
    print("\n### Model Configurations ###")
    print(f"\nQwen3-4B:")
    print(f"  Weight tying: {qwen4b['weight_tying']}")
    print(f"  Vocab size: {qwen4b['input_emb'].shape[0]}")
    print(f"  Hidden dim: {qwen4b['hidden_dim']}")
    print(f"  Embedding Frobenius norm: {qwen4b['input_emb'].norm().item():.2f}")
    
    print(f"\nQwen3-8B:")
    print(f"  Weight tying: {qwen8b['weight_tying']}")
    print(f"  Vocab size: {qwen8b['input_emb'].shape[0]}")
    print(f"  Hidden dim: {qwen8b['hidden_dim']}")
    print(f"  Input Frobenius norm: {qwen8b['input_emb'].norm().item():.2f}")
    print(f"  Output Frobenius norm: {qwen8b['output_emb'].norm().item():.2f}")
    
    # Within-model comparison for 8B (since it's untied)
    print("\n### Within-Model Analysis (Qwen3-8B) ###")
    emb8b_in_norm = qwen8b['input_emb'] / (qwen8b['input_emb'].norm(dim=1, keepdim=True) + 1e-8)
    emb8b_out_norm = qwen8b['output_emb'] / (qwen8b['output_emb'].norm(dim=1, keepdim=True) + 1e-8)
    cos_8b = (emb8b_in_norm * emb8b_out_norm).sum(dim=1)
    
    print(f"  Input vs Output cosine similarity:")
    print(f"    Mean:   {cos_8b.mean().item():.6f}")
    print(f"    Std:    {cos_8b.std().item():.6f}")
    print(f"    Median: {cos_8b.median().item():.6f}")
    
    # Procrustes alignment for 8B
    emb1 = qwen8b['input_emb'].float()
    emb2 = qwen8b['output_emb'].float()
    emb1_c = emb1 - emb1.mean(dim=0, keepdim=True)
    emb2_c = emb2 - emb2.mean(dim=0, keepdim=True)
    
    M = emb2_c.T @ emb1_c
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    
    emb1_aligned = emb1_c @ W.T
    emb1_al_norm = emb1_aligned / (emb1_aligned.norm(dim=1, keepdim=True) + 1e-8)
    emb2_c_norm = emb2_c / (emb2_c.norm(dim=1, keepdim=True) + 1e-8)
    cos_aligned = (emb1_al_norm * emb2_c_norm).sum(dim=1)
    
    print(f"  After Procrustes alignment:")
    print(f"    Cosine mean: {cos_aligned.mean().item():.6f}")
    print(f"    Cosine std:  {cos_aligned.std().item():.6f}")
    
    # Cross-model comparison (same vocab, different hidden dims)
    print("\n### Cross-Model Analysis ###")
    print(f"  Both models share vocab size: {qwen4b['input_emb'].shape[0]}")
    print(f"  Different hidden dimensions: 4B={qwen4b['hidden_dim']}, 8B={qwen8b['hidden_dim']}")
    print("  (Direct cosine comparison not possible due to different dimensions)")
    
    # Compare per-token norms
    norms_4b = qwen4b['input_emb'].norm(dim=1)
    norms_8b_in = qwen8b['input_emb'].norm(dim=1)
    norms_8b_out = qwen8b['output_emb'].norm(dim=1)
    
    # Correlation of per-token norms
    norm_corr_4b_8b_in = torch.corrcoef(torch.stack([norms_4b, norms_8b_in]))[0, 1].item()
    norm_corr_4b_8b_out = torch.corrcoef(torch.stack([norms_4b, norms_8b_out]))[0, 1].item()
    norm_corr_8b_in_out = torch.corrcoef(torch.stack([norms_8b_in, norms_8b_out]))[0, 1].item()
    
    print(f"\n  Per-token norm correlations:")
    print(f"    Qwen3-4B (tied) vs Qwen3-8B (input):  {norm_corr_4b_8b_in:.4f}")
    print(f"    Qwen3-4B (tied) vs Qwen3-8B (output): {norm_corr_4b_8b_out:.4f}")
    print(f"    Qwen3-8B input vs output:            {norm_corr_8b_in_out:.4f}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()

