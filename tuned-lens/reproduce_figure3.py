"""
Reproduce Figure 3 from "Eliciting Latent Predictions from Transformers with the Tuned Lens"
(https://arxiv.org/abs/2303.08112)

Figure 3 shows: Bias of logit lens and tuned lens outputs relative to the final layer output 
for GPT-Neo-2.7B. The bias is measured as KL divergence (in bits) at each layer.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens, LogitLens


def compute_kl_divergence(log_p: torch.Tensor, log_q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute KL(P || Q) where log_p and log_q are log probabilities."""
    p = log_p.exp()
    return torch.sum(p * (log_p - log_q), dim=dim)


def compute_bias_per_layer(
    model,
    tokenizer,
    tuned_lens,
    logit_lens,
    texts: list[str],
    device: str = "cuda",
    max_length: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the average KL divergence (bias) at each layer for both lenses.
    
    Returns:
        tuned_kl: KL divergence at each layer for tuned lens
        logit_kl: KL divergence at each layer for logit lens
    """
    num_layers = model.config.num_hidden_layers
    
    # Accumulators for KL divergence at each layer
    tuned_kl_sum = torch.zeros(num_layers + 1, device=device)  # +1 for input embeddings
    logit_kl_sum = torch.zeros(num_layers + 1, device=device)
    total_tokens = 0
    
    model.eval()
    tuned_lens.eval()
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Processing texts"):
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            ).to(device)
            
            # Get model outputs with hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Final layer log probabilities (ground truth for bias calculation)
            final_logits = outputs.logits
            final_log_probs = torch.log_softmax(final_logits, dim=-1)
            
            # hidden_states[0] = embeddings, hidden_states[i] = output of layer i-1
            hidden_states = outputs.hidden_states
            
            num_tokens = inputs.input_ids.shape[1]
            total_tokens += num_tokens
            
            # Compute KL for each layer
            for layer_idx in range(num_layers + 1):
                h = hidden_states[layer_idx]
                
                # For input embeddings (layer 0), use idx=0 for the lens
                lens_idx = max(0, layer_idx - 1) if layer_idx > 0 else 0
                
                # Tuned lens prediction
                tuned_logits = tuned_lens(h, idx=lens_idx)
                tuned_log_probs = torch.log_softmax(tuned_logits, dim=-1)
                
                # Logit lens prediction (just unembed directly)
                logit_logits = logit_lens(h, idx=lens_idx)
                logit_log_probs = torch.log_softmax(logit_logits, dim=-1)
                
                # KL(final || lens) - how different is the lens from the final output
                tuned_kl = compute_kl_divergence(final_log_probs, tuned_log_probs)
                logit_kl = compute_kl_divergence(final_log_probs, logit_log_probs)
                
                tuned_kl_sum[layer_idx] += tuned_kl.sum()
                logit_kl_sum[layer_idx] += logit_kl.sum()
    
    # Average and convert from nats to bits
    nats_to_bits = 1.0 / np.log(2)
    tuned_kl_avg = (tuned_kl_sum / total_tokens).cpu().numpy() * nats_to_bits
    logit_kl_avg = (logit_kl_sum / total_tokens).cpu().numpy() * nats_to_bits
    
    return tuned_kl_avg, logit_kl_avg


def plot_figure3(tuned_kl: np.ndarray, logit_kl: np.ndarray, model_name: str, save_path: str = None):
    """
    Plot Figure 3: KL divergence (bias) vs layer for both lenses.
    """
    num_layers = len(tuned_kl)
    
    # Create x-axis: "input" for layer 0, then layer numbers
    x = np.arange(num_layers)
    
    plt.figure(figsize=(10, 6))
    
    # Plot logit lens (red with squares)
    plt.plot(x, logit_kl, 'rs-', label='Logit lens', markersize=6, linewidth=1.5)
    
    # Plot tuned lens (blue with circles)
    plt.plot(x, tuned_kl, 'bo-', label='Tuned lens', markersize=6, linewidth=1.5)
    
    # Customize x-axis
    tick_labels = ['input'] + [str(i) for i in range(1, num_layers)]
    tick_positions = list(range(num_layers))
    
    # Show fewer ticks for readability
    step = max(1, num_layers // 10)
    plt.xticks(tick_positions[::step], tick_labels[::step])
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('KL (bits)', fontsize=12)
    plt.title(f'Bias of logit lens and tuned lens outputs\nrelative to final layer output for {model_name}')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def main():
    # Configuration
    # Note: The paper uses GPT-Neo-2.7B but no pretrained lens is available for it.
    # Available options with pretrained lenses include:
    # - EleutherAI/pythia-2.8b-deduped (similar size, recommended)
    # - gpt2-xl (1.5B parameters)
    # - EleutherAI/gpt-neox-20b (requires significant VRAM)
    # 
    # You can still use GPT-Neo-2.7B, but you'd need to train the lens first.
    # For demonstration, we use pythia-2.8b which has a pretrained lens.
    
    # model_name = "EleutherAI/pythia-2.8b-deduped"  # Has pretrained lens available
    # model_name = "EleutherAI/gpt-neo-2.7B"  # Original paper model (no pretrained lens)
    # model_name = "gpt2-xl"  # 1.5B params, 48 layers, has pretrained lens
    # model_name = "meta-llama/Meta-Llama-3-8B"  # 8B params, 32 layers
    # model_name = "gpt2-large"  # 774M params, 36 layers
    model_name = "allenai/OLMo-1B-0724-hf"  # 1B params, custom trained lens
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"Model loaded with {model.config.num_hidden_layers} layers")
    
    # Load lenses
    print("Loading tuned lens...")
    # Check for local trained lens first
    local_lens_path = str(Path(__file__).parent / "trained_lenses" / model_name)
    try:
        from pathlib import Path
        if Path(local_lens_path).exists():
            print(f"Loading local trained lens from {local_lens_path}")
            tuned_lens = TunedLens.from_model_and_pretrained(
                model, 
                lens_resource_id=local_lens_path,
                weights_only=True
            )
        else:
            tuned_lens = TunedLens.from_model_and_pretrained(
                model, 
                lens_resource_id=model_name,
                weights_only=True  # For security with torch.load
            )
        print("Loaded pretrained tuned lens!")
    except Exception as e:
        print(f"Could not load pretrained tuned lens: {e}")
        print("Creating untrained tuned lens (results will differ from paper)")
        tuned_lens = TunedLens.from_model(model)
    
    tuned_lens = tuned_lens.to(device)
    
    print("Creating logit lens...")
    logit_lens = LogitLens.from_model(model)
    logit_lens = logit_lens.to(device)
    
    # Sample texts for evaluation (similar to paper's use of The Pile)
    sample_texts = [
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
    
    print(f"\nComputing bias for {len(sample_texts)} sample texts...")
    
    tuned_kl, logit_kl = compute_bias_per_layer(
        model, tokenizer, tuned_lens, logit_lens, sample_texts, device
    )
    
    print("\n=== Results ===")
    print(f"Layer\tTuned Lens KL (bits)\tLogit Lens KL (bits)")
    print("-" * 60)
    for i, (t, l) in enumerate(zip(tuned_kl, logit_kl)):
        layer_name = "input" if i == 0 else str(i)
        print(f"{layer_name}\t{t:.4f}\t\t\t{l:.4f}")
    
    # Plot and save
    safe_name = model_name.replace("/", "_")
    plot_figure3(tuned_kl, logit_kl, model_name, save_path=f"figure3_{safe_name}.png")


if __name__ == "__main__":
    main()

