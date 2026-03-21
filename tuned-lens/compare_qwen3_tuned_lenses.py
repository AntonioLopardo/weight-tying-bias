"""
Compare tuned lens KL divergence across Qwen3 model family.

Models compared:
    - Qwen/Qwen3-0.6B  (28 layers, tied embeddings)
    - Qwen/Qwen3-1.7B  (28 layers, tied embeddings)
    - Qwen/Qwen3-4B    (36 layers, tied embeddings)
    - Qwen/Qwen3-8B    (36 layers, untied embeddings)
    - Qwen/Qwen3-14B   (40 layers, untied embeddings)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens, LogitLens
from pathlib import Path


# Qwen3 model configurations with tied embedding info
QWEN3_MODELS = [
    {
        "name": "Qwen/Qwen3-0.6B",
        "label": "Qwen3-0.6B",
        "tied_embeddings": True,
        "layers": 28,
    },
    {
        "name": "Qwen/Qwen3-1.7B",
        "label": "Qwen3-1.7B",
        "tied_embeddings": True,
        "layers": 28,
    },
    {
        "name": "Qwen/Qwen3-4B",
        "label": "Qwen3-4B",
        "tied_embeddings": True,
        "layers": 36,
    },
    {
        "name": "Qwen/Qwen3-8B",
        "label": "Qwen3-8B",
        "tied_embeddings": False,
        "layers": 36,
    },
]


def compute_kl_divergence(log_p: torch.Tensor, log_q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute KL(P || Q) where log_p and log_q are log probabilities."""
    p = log_p.exp()
    return torch.sum(p * (log_p - log_q), dim=dim)


def check_tied_embeddings(model) -> bool:
    """Check if the model has tied input/output embeddings."""
    # Try to detect tied embeddings by comparing weight tensors
    try:
        embed_tokens = model.get_input_embeddings()
        lm_head = model.get_output_embeddings()
        
        if embed_tokens is None or lm_head is None:
            return False
        
        # Check if they share the same weight tensor
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
    """
    Compute the average KL divergence (bias) at each layer for both lenses.
    
    Returns:
        tuned_kl: KL divergence at each layer for tuned lens
        logit_kl: KL divergence at each layer for logit lens
    """
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
    
    for model_config in QWEN3_MODELS:
        model_name = model_config["name"]
        label = model_config["label"]
        expected_tied = model_config["tied_embeddings"]
        
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
        
        # Verify tied embeddings
        actual_tied = check_tied_embeddings(model)
        print(f"Tied embeddings: {actual_tied} (expected: {expected_tied})")
        
        # Load tuned lens
        local_lens_path = str(Path(__file__).parent / "trained_lenses" / model_name)
        
        if not Path(local_lens_path).exists():
            print(f"WARNING: Tuned lens not found at {local_lens_path}")
            print("Please run train_qwen3_lenses.sh first")
            del model, tokenizer
            torch.cuda.empty_cache()
            continue
        
        print(f"Loading tuned lens from {local_lens_path}")
        tuned_lens = TunedLens.from_model_and_pretrained(
            model, 
            lens_resource_id=local_lens_path,
            weights_only=True
        )
        tuned_lens = tuned_lens.to(device)
        
        logit_lens = LogitLens.from_model(model)
        logit_lens = logit_lens.to(device)
        
        # Compute KL divergence
        tuned_kl, logit_kl = compute_bias_per_layer(
            model, tokenizer, tuned_lens, logit_lens, sample_texts, device
        )
        
        results[label] = {
            "tuned_kl": tuned_kl,
            "logit_kl": logit_kl,
            "tied_embeddings": actual_tied,
            "num_layers": model.config.num_hidden_layers,
        }
        
        # Free memory
        del model, tuned_lens, logit_lens, tokenizer
        torch.cuda.empty_cache()
    
    if not results:
        print("\nNo models were processed. Please train the lenses first.")
        return
    
    # Group models by number of layers
    layer_groups = {}
    for label, data in results.items():
        num_layers = data["num_layers"]
        if num_layers not in layer_groups:
            layer_groups[num_layers] = {}
        layer_groups[num_layers][label] = data
    
    # Filter to only groups with 2+ models for comparison
    comparable_groups = {k: v for k, v in layer_groups.items() if len(v) >= 2}
    
    if not comparable_groups:
        print("\nNo groups with multiple models to compare.")
        print("Layer counts found:", {label: data["num_layers"] for label, data in results.items()})
        return
    
    # Color palette and markers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    def get_linestyle(tied):
        return '-' if tied else '--'
    
    # Create a figure for each layer group
    for num_layers, group_results in sorted(comparable_groups.items()):
        print(f"\n{'='*60}")
        print(f"Comparing models with {num_layers} layers:")
        for label in group_results.keys():
            tied_str = "tied" if group_results[label]["tied_embeddings"] else "untied"
            print(f"  - {label} ({tied_str} embeddings)")
        print('='*60)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Tuned Lens comparison
        ax1 = axes[0]
        for i, (label, data) in enumerate(group_results.items()):
            kl_values = data["tuned_kl"]
            tied = data["tied_embeddings"]
            n_layers = len(kl_values)
            
            x = np.arange(n_layers)
            
            tied_str = "tied" if tied else "untied"
            ax1.plot(
                x, kl_values,
                marker=markers[i % len(markers)],
                linestyle=get_linestyle(tied),
                color=colors[i % len(colors)],
                label=f'{label} ({tied_str})',
                markersize=5,
                linewidth=2,
                markevery=max(1, n_layers // 10)
            )
        
        # X-axis labels
        tick_labels = ['input'] + [str(i) for i in range(1, num_layers + 1)]
        tick_positions = list(range(num_layers + 1))
        step = max(1, (num_layers + 1) // 10)
        ax1.set_xticks(tick_positions[::step])
        ax1.set_xticklabels(tick_labels[::step])
        
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('KL Divergence (bits)', fontsize=12)
        ax1.set_title(f'Tuned Lens: {num_layers}-Layer Models\n(solid = tied embeddings, dashed = untied)', fontsize=13)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Plot 2: Logit Lens comparison
        ax2 = axes[1]
        for i, (label, data) in enumerate(group_results.items()):
            kl_values = data["logit_kl"]
            tied = data["tied_embeddings"]
            n_layers = len(kl_values)
            
            x = np.arange(n_layers)
            
            tied_str = "tied" if tied else "untied"
            ax2.plot(
                x, kl_values,
                marker=markers[i % len(markers)],
                linestyle=get_linestyle(tied),
                color=colors[i % len(colors)],
                label=f'{label} ({tied_str})',
                markersize=5,
                linewidth=2,
                markevery=max(1, n_layers // 10)
            )
        
        ax2.set_xticks(tick_positions[::step])
        ax2.set_xticklabels(tick_labels[::step])
        
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('KL Divergence (bits)', fontsize=12)
        ax2.set_title(f'Logit Lens: {num_layers}-Layer Models\n(solid = tied embeddings, dashed = untied)', fontsize=13)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        filename = f'compare_qwen3_{num_layers}layers.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {filename}")
        
        # Create improvement plot for this group
        fig2, ax3 = plt.subplots(figsize=(12, 7))
        
        for i, (label, data) in enumerate(group_results.items()):
            tuned_kl = data["tuned_kl"]
            logit_kl = data["logit_kl"]
            tied = data["tied_embeddings"]
            n_layers = len(tuned_kl)
            
            improvement = logit_kl - tuned_kl
            x = np.arange(n_layers)
            
            tied_str = "tied" if tied else "untied"
            ax3.plot(
                x, improvement,
                marker=markers[i % len(markers)],
                linestyle=get_linestyle(tied),
                color=colors[i % len(colors)],
                label=f'{label} ({tied_str})',
                markersize=5,
                linewidth=2,
                markevery=max(1, n_layers // 10)
            )
        
        ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        ax3.set_xticks(tick_positions[::step])
        ax3.set_xticklabels(tick_labels[::step])
        ax3.set_xlabel('Layer', fontsize=12)
        ax3.set_ylabel('KL Improvement (bits)', fontsize=12)
        ax3.set_title(f'Tuned Lens Improvement: {num_layers}-Layer Models\n(positive = tuned lens better, solid = tied, dashed = untied)', fontsize=13)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'compare_qwen3_{num_layers}layers_improvement.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {filename}")
    
    # Print comparison table
    print("\n" + "="*90)
    print("COMPARISON TABLE: Qwen3 Model Family")
    print("="*90)
    print(f"{'Model':<20} {'Layers':<8} {'Tied Emb':<10} {'Avg Tuned KL':<15} {'Avg Logit KL':<15}")
    print("-"*90)
    
    for label, data in results.items():
        tied_str = "Yes" if data["tied_embeddings"] else "No"
        avg_tuned = np.mean(data["tuned_kl"])
        avg_logit = np.mean(data["logit_kl"])
        print(f"{label:<20} {data['num_layers']:<8} {tied_str:<10} {avg_tuned:<15.4f} {avg_logit:<15.4f}")
    
    # Print layer-by-layer tables for each comparable group
    for num_layers, group_results in sorted(comparable_groups.items()):
        print("\n" + "="*70)
        print(f"LAYER-BY-LAYER TUNED LENS KL (bits) - {num_layers} Layer Models")
        print("="*70)
        
        # Print header
        print(f"{'Layer':<10}", end="")
        for label in group_results.keys():
            print(f"{label:<20}", end="")
        print()
        print("-"*70)
        
        for i in range(num_layers + 1):
            layer_name = "input" if i == 0 else str(i)
            print(f"{layer_name:<10}", end="")
            for label, data in group_results.items():
                print(f"{data['tuned_kl'][i]:<20.4f}", end="")
            print()
    
    plt.show()


if __name__ == "__main__":
    main()

