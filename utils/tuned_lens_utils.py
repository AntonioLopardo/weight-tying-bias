"""Shared utilities for tuned lens analysis."""

import torch
import numpy as np
from tqdm import tqdm
from typing import Optional

NATS_TO_BITS = 1.0 / np.log(2)


def load_eval_texts(min_char_length=20):
    """Load WikiText-2 test set for tuned lens evaluation."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return [t for t in ds["text"] if len(t.strip()) >= min_char_length]


def compute_kl_divergence(log_p: torch.Tensor, log_q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute KL(P || Q) from log probabilities."""
    p = log_p.exp()
    return torch.sum(p * (log_p - log_q), dim=dim)


def compute_bias_per_layer(model, tokenizer, tuned_lens, texts, device="cuda",
                           max_length=512, logit_lens=None):
    """Compute per-layer KL divergence using tuned lens (and optionally logit lens).

    Returns:
        If logit_lens is None: numpy array of tuned lens KL values per layer (in bits)
        If logit_lens is provided: tuple of (tuned_kl, logit_kl) numpy arrays (in bits)
    """
    num_layers = model.config.num_hidden_layers
    tuned_kl_sum = torch.zeros(num_layers + 1, device=device)
    logit_kl_sum = torch.zeros(num_layers + 1, device=device) if logit_lens is not None else None
    total_tokens = 0

    model.eval()
    tuned_lens.eval()

    with torch.no_grad():
        for text in tqdm(texts, desc="Processing texts"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            final_log_probs = torch.log_softmax(outputs.logits, dim=-1)
            hidden_states = outputs.hidden_states
            num_tokens = inputs.input_ids.shape[1]
            total_tokens += num_tokens

            for layer_idx in range(num_layers + 1):
                h = hidden_states[layer_idx]
                lens_idx = max(0, layer_idx - 1) if layer_idx > 0 else 0

                tuned_log_probs = torch.log_softmax(tuned_lens(h, idx=lens_idx), dim=-1)
                tuned_kl_sum[layer_idx] += compute_kl_divergence(final_log_probs, tuned_log_probs).sum()

                if logit_lens is not None:
                    logit_log_probs = torch.log_softmax(logit_lens(h, idx=lens_idx), dim=-1)
                    logit_kl_sum[layer_idx] += compute_kl_divergence(final_log_probs, logit_log_probs).sum()

    tuned_kl = (tuned_kl_sum / total_tokens).cpu().numpy() * NATS_TO_BITS

    if logit_lens is not None:
        logit_kl = (logit_kl_sum / total_tokens).cpu().numpy() * NATS_TO_BITS
        return tuned_kl, logit_kl

    return tuned_kl


def load_model_and_tokenizer(model_name, dtype=torch.float16):
    """Load a HuggingFace model and tokenizer with standard settings."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    return model, tokenizer
