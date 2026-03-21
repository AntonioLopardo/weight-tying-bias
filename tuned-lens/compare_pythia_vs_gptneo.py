"""
Compare tuned lens KL divergence between EleutherAI/pythia-2.8b and EleutherAI/gpt-neo-2.7B.
Uses pretrained lenses from HuggingFace.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens, LogitLens


# Models to compare
# Using locally trained lenses
MODELS = [
    {
        "name": "EleutherAI/pythia-2.8b",
        "label": "Pythia-2.8B",
        "lens_path": str(Path(__file__).parent / "trained_lenses" / "EleutherAI" / "pythia-2.8b"),
    },
    {
        "name": "EleutherAI/gpt-neo-2.7B",
        "label": "GPT-Neo-2.7B",
        "lens_path": str(Path(__file__).parent / "trained_lenses" / "EleutherAI" / "gpt-neo-2.7B"),
    },
]


def compute_kl_divergence(log_p: torch.Tensor, log_q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute KL(P || Q) where log_p and log_q are log probabilities."""
    p = log_p.exp()
    return torch.sum(p * (log_p - log_q), dim=dim)


def check_tied_embeddings(model) -> bool:
    """Check if the model has tied input/output embeddings."""
    try:
        embed_tokens = model.get_input_embeddings()
        lm_head = model.get_output_embeddings()
        if embed_tokens is None or lm_head is None:
            return False
        return embed_tokens.weight.data_ptr() == lm_head.weight.data_ptr()
    except Exception:
        return False


def compute_bias_per_layer(
    model,
    tokenizer,
    tuned_lens,
    logit_lens,
    texts: list[str],
    device: str = "cuda",
    max_length: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the average KL divergence (bias) at each layer for both lenses."""
    num_layers = model.config.num_hidden_layers
    
    tuned_kl_sum = torch.zeros(num_layers + 1, device=device)
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
            
            outputs = model(**inputs, output_hidden_states=True)
            
            final_logits = outputs.logits
            final_log_probs = torch.log_softmax(final_logits, dim=-1)
            hidden_states = outputs.hidden_states
            
            num_tokens = inputs.input_ids.shape[1]
            total_tokens += num_tokens
            
            for layer_idx in range(num_layers + 1):
                h = hidden_states[layer_idx]
                lens_idx = max(0, layer_idx - 1) if layer_idx > 0 else 0
                
                tuned_logits = tuned_lens(h, idx=lens_idx)
                tuned_log_probs = torch.log_softmax(tuned_logits, dim=-1)
                
                logit_logits = logit_lens(h, idx=lens_idx)
                logit_log_probs = torch.log_softmax(logit_logits, dim=-1)
                
                tuned_kl = compute_kl_divergence(final_log_probs, tuned_log_probs)
                logit_kl = compute_kl_divergence(final_log_probs, logit_log_probs)
                
                tuned_kl_sum[layer_idx] += tuned_kl.sum()
                logit_kl_sum[layer_idx] += logit_kl.sum()
    
    nats_to_bits = 1.0 / np.log(2)
    tuned_kl_avg = (tuned_kl_sum / total_tokens).cpu().numpy() * nats_to_bits
    logit_kl_avg = (logit_kl_sum / total_tokens).cpu().numpy() * nats_to_bits
    
    return tuned_kl_avg, logit_kl_avg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Sample texts for evaluation
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
    
    results = {}
    
    for model_config in MODELS:
        model_name = model_config["name"]
        label = model_config["label"]
        
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print('='*60)
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Check tied embeddings
        tied = check_tied_embeddings(model)
        print(f"Tied embeddings: {tied}")
        print(f"Number of layers: {model.config.num_hidden_layers}")
        
        # Load locally trained tuned lens
        lens_path = model_config.get("lens_path", model_name)
        try:
            print(f"Loading tuned lens from {lens_path}...")
            tuned_lens = TunedLens.from_model_and_pretrained(
                model, 
                lens_resource_id=lens_path,
                weights_only=True
            )
            tuned_lens = tuned_lens.to(device)
        except Exception as e:
            print(f"Could not load lens: {e}")
            print("Skipping this model...")
            del model, tokenizer
            torch.cuda.empty_cache()
            continue
        
        logit_lens = LogitLens.from_model(model)
        logit_lens = logit_lens.to(device)
        
        # Compute KL divergence
        tuned_kl, logit_kl = compute_bias_per_layer(
            model, tokenizer, tuned_lens, logit_lens, sample_texts, device
        )
        
        results[label] = {
            "tuned_kl": tuned_kl,
            "logit_kl": logit_kl,
            "tied_embeddings": tied,
            "num_layers": model.config.num_hidden_layers,
        }
        
        # Free memory
        del model, tuned_lens, logit_lens, tokenizer
        torch.cuda.empty_cache()
    
    if len(results) < 2:
        print("\nNot enough models were processed for comparison.")
        return
    
    # Color palette - distinct colors for the two models
    colors = ['#E63946', '#457B9D']  # Red for Pythia, Blue for GPT-Neo
    markers = ['o', 's']
    
    # Get max layers for x-axis
    max_layers = max(data["num_layers"] for data in results.values())
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Tuned Lens comparison
    ax1 = axes[0]
    for i, (label, data) in enumerate(results.items()):
        kl_values = data["tuned_kl"]
        n_layers = len(kl_values)
        # Normalize x-axis to [0, 1] for comparison across different layer counts
        x = np.linspace(0, 1, n_layers)
        
        ax1.plot(
            x, kl_values,
            marker=markers[i],
            color=colors[i],
            label=f'{label} ({data["num_layers"]} layers)',
            markersize=6,
            linewidth=2,
            markevery=max(1, n_layers // 10)
        )
    
    ax1.set_xlabel('Relative Depth (0=input, 1=output)', fontsize=12)
    ax1.set_ylabel('KL Divergence (bits)', fontsize=12)
    ax1.set_title('Tuned Lens: Pythia-2.8B vs GPT-Neo-2.7B', fontsize=13)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Logit Lens comparison
    ax2 = axes[1]
    for i, (label, data) in enumerate(results.items()):
        kl_values = data["logit_kl"]
        n_layers = len(kl_values)
        x = np.linspace(0, 1, n_layers)
        
        ax2.plot(
            x, kl_values,
            marker=markers[i],
            color=colors[i],
            label=f'{label} ({data["num_layers"]} layers)',
            markersize=6,
            linewidth=2,
            markevery=max(1, n_layers // 10)
        )
    
    ax2.set_xlabel('Relative Depth (0=input, 1=output)', fontsize=12)
    ax2.set_ylabel('KL Divergence (bits)', fontsize=12)
    ax2.set_title('Logit Lens: Pythia-2.8B vs GPT-Neo-2.7B', fontsize=13)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('compare_pythia_vs_gptneo.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to compare_pythia_vs_gptneo.png")
    
    # Create improvement plot (Logit KL - Tuned KL)
    fig2, ax3 = plt.subplots(figsize=(12, 7))
    
    for i, (label, data) in enumerate(results.items()):
        tuned_kl = data["tuned_kl"]
        logit_kl = data["logit_kl"]
        n_layers = len(tuned_kl)
        
        improvement = logit_kl - tuned_kl
        x = np.linspace(0, 1, n_layers)
        
        ax3.plot(
            x, improvement,
            marker=markers[i],
            color=colors[i],
            label=f'{label} ({data["num_layers"]} layers)',
            markersize=6,
            linewidth=2,
            markevery=max(1, n_layers // 10)
        )
    
    ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax3.set_xlabel('Relative Depth (0=input, 1=output)', fontsize=12)
    ax3.set_ylabel('KL Improvement (bits)', fontsize=12)
    ax3.set_title('Tuned Lens Improvement: Pythia-2.8B vs GPT-Neo-2.7B\n(positive = tuned lens better than logit lens)', fontsize=13)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('compare_pythia_vs_gptneo_improvement.png', dpi=150, bbox_inches='tight')
    print(f"Figure saved to compare_pythia_vs_gptneo_improvement.png")
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE: Pythia-2.8B vs GPT-Neo-2.7B")
    print("="*80)
    print(f"{'Model':<20} {'Layers':<8} {'Tied Emb':<10} {'Avg Tuned KL':<15} {'Avg Logit KL':<15}")
    print("-"*80)
    
    for label, data in results.items():
        tied_str = "Yes" if data["tied_embeddings"] else "No"
        avg_tuned = np.mean(data["tuned_kl"])
        avg_logit = np.mean(data["logit_kl"])
        print(f"{label:<20} {data['num_layers']:<8} {tied_str:<10} {avg_tuned:<15.4f} {avg_logit:<15.4f}")
    
    plt.show()


if __name__ == "__main__":
    main()

