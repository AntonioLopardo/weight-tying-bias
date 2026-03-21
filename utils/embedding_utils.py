"""Shared utilities for loading and comparing embedding matrices."""

import os
import torch
from typing import Optional


def load_embeddings_from_hf(model_id: str, revision: Optional[str] = None) -> dict:
    """Load input and output embedding matrices from a HuggingFace model.

    Returns dict with keys: input_emb, output_emb, weight_tying, model_id.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    kwargs = {"low_cpu_mem_usage": True, "torch_dtype": torch.float32}
    if revision:
        kwargs["revision"] = revision

    print(f"Loading: {model_id}" + (f" @ {revision}" if revision else ""))
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    config = model.config
    sd = model.state_dict()

    # Search for input embedding key
    input_emb = None
    for key in ["model.embed_tokens.weight", "transformer.wte.weight",
                "gpt_neox.embed_in.weight", "transformer.embd.wte.weight"]:
        if key in sd:
            input_emb = sd[key].float()
            break

    if input_emb is None:
        for key in sd:
            if "embed" in key.lower() and "weight" in key and "layer" not in key.lower():
                input_emb = sd[key].float()
                break

    # Search for output embedding key
    output_emb = None
    for key in ["lm_head.weight", "embed_out.weight"]:
        if key in sd:
            output_emb = sd[key].float()
            break

    if output_emb is None:
        for key in sd:
            if ("lm_head" in key or "embed_out" in key) and "weight" in key:
                output_emb = sd[key].float()
                break

    # Detect weight tying
    weight_tying = getattr(config, "tie_word_embeddings", True)
    if input_emb is not None and output_emb is not None:
        weight_tying = torch.equal(input_emb, output_emb)

    print(f"  Input: {tuple(input_emb.shape)}, Output: {tuple(output_emb.shape)}, Tied: {weight_tying}")
    del model
    return {
        "model_id": model_id,
        "input_emb": input_emb,
        "output_emb": output_emb,
        "weight_tying": weight_tying,
    }


def load_embeddings_from_checkpoint(checkpoint_dir: str, vocab_size: Optional[int] = None) -> torch.Tensor:
    """Load the shared embedding matrix from a local OLMo checkpoint.

    Args:
        checkpoint_dir: Directory containing model.pt
        vocab_size: If set, truncate to this many rows (removes padding tokens)
    """
    model_path = os.path.join(checkpoint_dir, "model.pt")
    print(f"Loading: {model_path}")
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    emb = state["transformer.wte.weight"].clone().float()
    print(f"  Embeddings: {tuple(emb.shape)} (tied)")
    if vocab_size is not None:
        emb = emb[:vocab_size]
    return emb


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


def mean_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute mean per-token cosine similarity between two embedding matrices."""
    emb1_norm = emb1 / (emb1.norm(dim=1, keepdim=True) + 1e-8)
    emb2_norm = emb2 / (emb2.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = (emb1_norm * emb2_norm).sum(dim=1)
    return cos_sim.mean().item()
