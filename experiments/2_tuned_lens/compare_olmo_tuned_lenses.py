"""
Compare tuned lens KL divergence between two OLMo models.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens, LogitLens
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINED_LENSES_DIR = SCRIPT_DIR / "trained_lenses"


def compute_kl_divergence(log_p: torch.Tensor, log_q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute KL(P || Q) where log_p and log_q are log probabilities."""
    p = log_p.exp()
    return torch.sum(p * (log_p - log_q), dim=dim)


def compute_bias_per_layer(model, tokenizer, tuned_lens, texts, device="cuda", max_length=512):
    """Compute the average KL divergence at each layer for the tuned lens."""
    num_layers = model.config.num_hidden_layers
    tuned_kl_sum = torch.zeros(num_layers + 1, device=device)
    total_tokens = 0
    
    model.eval()
    tuned_lens.eval()
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Processing"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
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
                tuned_kl = compute_kl_divergence(final_log_probs, tuned_log_probs)
                tuned_kl_sum[layer_idx] += tuned_kl.sum()
    
    nats_to_bits = 1.0 / np.log(2)
    tuned_kl_avg = (tuned_kl_sum / total_tokens).cpu().numpy() * nats_to_bits
    return tuned_kl_avg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    models_to_compare = [
        ("allenai/OLMo-1B-0724-hf", "OLMo-1B-0724"),
        ("allenai/OLMo-1B-hf", "OLMo-1B"),
    ]
    
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
    
    for model_name, label in models_to_compare:
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
        
        # Load tuned lens
        local_lens_path = str(TRAINED_LENSES_DIR / model_name)
        tuned_lens = TunedLens.from_model_and_pretrained(
            model, 
            lens_resource_id=local_lens_path,
            weights_only=True
        )
        tuned_lens = tuned_lens.to(device)
        
        # Compute KL divergence
        tuned_kl = compute_bias_per_layer(model, tokenizer, tuned_lens, sample_texts, device)
        results[label] = tuned_kl
        
        # Free memory
        del model, tuned_lens, tokenizer
        torch.cuda.empty_cache()
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    colors = ['#2E86AB', '#A23B72']  # Blue and magenta
    markers = ['o', 's']
    
    for i, (label, kl_values) in enumerate(results.items()):
        num_layers = len(kl_values)
        x = np.arange(num_layers)
        plt.plot(x, kl_values, f'{markers[i]}-', color=colors[i], 
                 label=f'{label} (Tuned Lens)', markersize=6, linewidth=2)
    
    # Customize x-axis
    num_layers = len(list(results.values())[0])
    tick_labels = ['input'] + [str(i) for i in range(1, num_layers)]
    tick_positions = list(range(num_layers))
    step = max(1, num_layers // 10)
    plt.xticks(tick_positions[::step], tick_labels[::step])
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('KL Divergence (bits)', fontsize=12)
    plt.title('Tuned Lens Comparison: OLMo-1B vs OLMo-1B-0724\n(Both trained for 100 steps)', fontsize=13)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('compare_olmo_tuned_lenses.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to compare_olmo_tuned_lenses.png")
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE: Tuned Lens KL Divergence (bits)")
    print("="*70)
    print(f"{'Layer':<10}", end="")
    for label in results.keys():
        print(f"{label:<20}", end="")
    print()
    print("-"*70)
    
    num_layers = len(list(results.values())[0])
    for i in range(num_layers):
        layer_name = "input" if i == 0 else str(i)
        print(f"{layer_name:<10}", end="")
        for label, kl_values in results.items():
            print(f"{kl_values[i]:<20.4f}", end="")
        print()
    
    plt.show()


if __name__ == "__main__":
    main()

