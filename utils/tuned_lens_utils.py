"""Shared utilities for tuned lens analysis."""

import torch
import numpy as np
from tqdm import tqdm
from typing import Optional

NATS_TO_BITS = 1.0 / np.log(2)

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This is a simple sentence to test the language model's ability to predict tokens.",
    "In the field of machine learning, neural networks have become increasingly important for a wide variety of tasks including natural language processing.",
    "The development of large language models has revolutionized how we interact with artificial intelligence systems.",
    "Scientists have discovered new evidence that suggests the universe may be expanding at a faster rate than previously thought.",
    "The economic impact of climate change continues to be a major concern for policymakers around the world.",
    "Recent advances in quantum computing have opened up new possibilities for solving complex computational problems.",
    "The history of human civilization is marked by periods of great innovation and technological advancement.",
    "Music has been an integral part of human culture for thousands of years, serving both social and artistic purposes.",
    "The study of philosophy helps us understand fundamental questions about existence, knowledge, and ethics.",
    "Modern medicine has made remarkable progress in treating diseases that were once considered incurable.",
]


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
