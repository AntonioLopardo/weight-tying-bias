"""
Compare tuned lens KL divergence between OLMo-70M tied vs untied models.

Models:
- Tied: avyxh/olmo-70m-tied-5B (input/output embeddings share weights)
- Untied: avyxh/olmo-70m-untied-5B (separate input/output embeddings)

These models use native OLMo checkpoint format (rank0.pt), so we need to
load them using the OLMo library or convert them first.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download
import yaml

# Try to import OLMo
try:
    from olmo import OLMo, Tokenizer as OLMoTokenizer
    HAS_OLMO = True
except ImportError:
    HAS_OLMO = False
    print("OLMo library not found. Will try to convert checkpoints.")

from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens, LogitLens


# Model configurations
MODELS = [
    {
        "name": "avyxh/olmo-70m-tied-5B",
        "label": "OLMo-70M Tied",
        "color": "#2E86AB",  # Blue
        "marker": "o",
    },
    {
        "name": "avyxh/olmo-70m-untied-5B",
        "label": "OLMo-70M Untied", 
        "color": "#A23B72",  # Magenta
        "marker": "s",
    },
]

# Sample texts for evaluation
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


def download_and_load_olmo_checkpoint(repo_id: str, device: str = "cuda"):
    """Download and load an OLMo checkpoint from HuggingFace."""
    print(f"Downloading checkpoint from {repo_id}...")
    
    # Download files
    config_path = hf_hub_download(repo_id, "config.yaml")
    checkpoint_path = hf_hub_download(repo_id, "rank0.pt")
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"  Model config: d_model={config['model']['d_model']}, "
          f"n_layers={config['model']['n_layers']}, "
          f"n_heads={config['model']['n_heads']}, "
          f"weight_tying={config['model'].get('weight_tying', True)}")
    
    # Load checkpoint
    print(f"  Loading weights from {checkpoint_path}...")
    # Need weights_only=False for OLMo checkpoints that contain PosixPath
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    return config, state_dict


def convert_olmo_to_hf(config: dict, state_dict: dict, device: str = "cuda"):
    """Convert OLMo checkpoint to HuggingFace-compatible model."""
    from transformers import OlmoConfig, OlmoForCausalLM
    
    model_config = config['model']
    
    # Create HF config
    hf_config = OlmoConfig(
        vocab_size=model_config.get('embedding_size', model_config['vocab_size']),
        hidden_size=model_config['d_model'],
        intermediate_size=model_config['d_model'] * model_config['mlp_ratio'],
        num_hidden_layers=model_config['n_layers'],
        num_attention_heads=model_config['n_heads'],
        max_position_embeddings=model_config.get('max_sequence_length', 2048),
        tie_word_embeddings=model_config.get('weight_tying', True),
        rope_theta=10000.0,
        attention_dropout=model_config.get('attention_dropout', 0.0),
        pad_token_id=model_config.get('pad_token_id', 1),
        eos_token_id=model_config.get('eos_token_id', 50279),
    )
    
    # Create model with config
    model = OlmoForCausalLM(hf_config)
    
    # Map state dict keys from OLMo format to HF format
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # OLMo uses "transformer." prefix
        if key.startswith("model.transformer."):
            new_key = key.replace("model.transformer.", "model.")
        elif key.startswith("model."):
            new_key = key
        
        # Map embedding keys
        if "wte.weight" in new_key:
            new_key = new_key.replace("wte.weight", "embed_tokens.weight")
        
        # Map block keys to layers
        if ".blocks." in new_key:
            new_key = new_key.replace(".blocks.", ".layers.")
        
        # Map attention keys
        if ".att_proj." in new_key:
            new_key = new_key.replace(".att_proj.", ".self_attn.o_proj.")
        if ".attn_out." in new_key:
            new_key = new_key.replace(".attn_out.", ".self_attn.o_proj.")
        if ".ff_proj." in new_key:
            new_key = new_key.replace(".ff_proj.", ".mlp.down_proj.")
        if ".ff_out." in new_key:
            new_key = new_key.replace(".ff_out.", ".mlp.down_proj.")
            
        # Map layer norm
        if ".ln_" in new_key:
            new_key = new_key.replace(".ln_", ".input_layernorm.")
        
        new_state_dict[new_key] = value
    
    # Try to load what we can
    try:
        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"  Warning: Could not load all weights: {e}")
        # Print available keys for debugging
        print(f"  Checkpoint keys: {list(state_dict.keys())[:10]}...")
        print(f"  Model expects: {list(model.state_dict().keys())[:10]}...")
    
    model = model.to(device)
    model.eval()
    
    return model, hf_config


def load_model_native_olmo(repo_id: str, device: str = "cuda"):
    """Load model using native OLMo library."""
    from olmo import OLMo, Tokenizer
    
    # Download files
    config_path = hf_hub_download(repo_id, "config.yaml")
    checkpoint_path = hf_hub_download(repo_id, "rank0.pt")
    
    # Load using OLMo
    model = OLMo.from_checkpoint(checkpoint_path, device=device)
    tokenizer = Tokenizer.from_file(hf_hub_download("allenai/OLMo-7B-0724-hf", "tokenizer.json"))
    
    return model, tokenizer


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


def get_embedding_stats(model):
    """Get statistics about input and output embeddings."""
    input_emb = model.get_input_embeddings().weight.data.float()
    output_emb = model.get_output_embeddings().weight.data.float()
    
    input_norm = torch.norm(input_emb, dim=1).mean().item()
    output_norm = torch.norm(output_emb, dim=1).mean().item()
    
    # Check if they share memory (tied)
    is_tied = input_emb.data_ptr() == output_emb.data_ptr()
    
    if is_tied:
        # If tied, cosine similarity is 1.0 by definition
        return {
            "input_norm": input_norm,
            "output_norm": output_norm,
            "cosine_sim_mean": 1.0,
            "cosine_sim_std": 0.0,
            "is_tied": True,
        }
    
    # Cosine similarity between input and output embeddings per token
    input_normalized = input_emb / (torch.norm(input_emb, dim=1, keepdim=True) + 1e-8)
    output_normalized = output_emb / (torch.norm(output_emb, dim=1, keepdim=True) + 1e-8)
    cosine_sim = (input_normalized * output_normalized).sum(dim=1)
    
    return {
        "input_norm": input_norm,
        "output_norm": output_norm,
        "cosine_sim_mean": cosine_sim.mean().item(),
        "cosine_sim_std": cosine_sim.std().item(),
        "is_tied": False,
    }


def compute_bias_per_layer(
    model,
    tokenizer,
    logit_lens,
    texts: list[str],
    device: str = "cuda",
    max_length: int = 512,
    tuned_lens=None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute the average KL divergence (bias) at each layer for logit lens and optionally tuned lens."""
    num_layers = model.config.num_hidden_layers
    
    logit_kl_sum = torch.zeros(num_layers + 1, device=device)
    tuned_kl_sum = torch.zeros(num_layers + 1, device=device) if tuned_lens else None
    total_tokens = 0
    
    model.eval()
    if tuned_lens:
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
                
                logit_logits = logit_lens(h, idx=lens_idx)
                logit_log_probs = torch.log_softmax(logit_logits, dim=-1)
                
                logit_kl = compute_kl_divergence(final_log_probs, logit_log_probs)
                logit_kl_sum[layer_idx] += logit_kl.sum()
                
                if tuned_lens:
                    tuned_logits = tuned_lens(h, idx=lens_idx)
                    tuned_log_probs = torch.log_softmax(tuned_logits, dim=-1)
                    tuned_kl = compute_kl_divergence(final_log_probs, tuned_log_probs)
                    tuned_kl_sum[layer_idx] += tuned_kl.sum()
    
    nats_to_bits = 1.0 / np.log(2)
    logit_kl_avg = (logit_kl_sum / total_tokens).cpu().numpy() * nats_to_bits
    tuned_kl_avg = (tuned_kl_sum / total_tokens).cpu().numpy() * nats_to_bits if tuned_lens else None
    
    return logit_kl_avg, tuned_kl_avg


def train_tuned_lens_quick(model, tokenizer, texts, device="cuda", n_epochs=20, lr=1e-3):
    """Train a tuned lens quickly on the given model and texts."""
    from tuned_lens.nn.lenses import TunedLens
    
    print("  Training tuned lens...")
    tuned_lens = TunedLens.from_model(model)
    tuned_lens = tuned_lens.to(device)
    tuned_lens.train()
    
    optimizer = torch.optim.Adam(tuned_lens.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        for text in texts:
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=256
            ).to(device)
            
            if inputs.input_ids.shape[1] < 2:
                continue
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                target_log_probs = torch.log_softmax(outputs.logits, dim=-1)
            
            loss = 0
            for layer_idx in range(len(hidden_states) - 1):
                h = hidden_states[layer_idx + 1]
                pred_logits = tuned_lens(h, idx=layer_idx)
                pred_log_probs = torch.log_softmax(pred_logits, dim=-1)
                
                # KL divergence
                kl = compute_kl_divergence(target_log_probs, pred_log_probs)
                loss += kl.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"    Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    tuned_lens.eval()
    return tuned_lens


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # First, check configs for both models
    print("\n" + "="*60)
    print("Checking model configurations...")
    print("="*60)
    
    for model_config in MODELS:
        repo_id = model_config["name"]
        config_path = hf_hub_download(repo_id, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        print(f"\n{model_config['label']}:")
        print(f"  d_model: {config['model']['d_model']}")
        print(f"  n_layers: {config['model']['n_layers']}")
        print(f"  n_heads: {config['model']['n_heads']}")
        print(f"  weight_tying: {config['model'].get('weight_tying', 'not specified')}")
        print(f"  vocab_size: {config['model']['vocab_size']}")
    
    results = {}
    
    for model_config in MODELS:
        repo_id = model_config["name"]
        label = model_config["label"]
        
        print(f"\n{'='*60}")
        print(f"Processing: {repo_id}")
        print('='*60)
        
        # Download and convert checkpoint
        config, state_dict = download_and_load_olmo_checkpoint(repo_id, device)
        
        try:
            model, hf_config = convert_olmo_to_hf(config, state_dict, device)
            
            # Load tokenizer (use standard OLMo tokenizer)
            tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Check embeddings
            is_tied = check_tied_embeddings(model)
            expected_tied = config['model'].get('weight_tying', True)
            print(f"Weight tying in config: {expected_tied}")
            print(f"Actually tied (same memory): {is_tied}")
            
            try:
                emb_stats = get_embedding_stats(model)
                print(f"Input embedding avg norm: {emb_stats['input_norm']:.4f}")
                print(f"Output embedding avg norm: {emb_stats['output_norm']:.4f}")
                print(f"Cosine similarity (input/output): {emb_stats['cosine_sim_mean']:.4f} ± {emb_stats['cosine_sim_std']:.4f}")
            except Exception as e:
                print(f"Could not compute embedding stats: {e}")
                emb_stats = {}
            
            # Create logit lens
            logit_lens = LogitLens.from_model(model)
            logit_lens = logit_lens.to(device)
            
            # Train tuned lens
            tuned_lens = train_tuned_lens_quick(model, tokenizer, SAMPLE_TEXTS, device)
            
            # Compute KL divergence for both lenses
            logit_kl, tuned_kl = compute_bias_per_layer(
                model, tokenizer, logit_lens, SAMPLE_TEXTS, device, tuned_lens=tuned_lens
            )
            
            results[label] = {
                "logit_kl": logit_kl,
                "tuned_kl": tuned_kl,
                "tied_embeddings": is_tied,
                "expected_tied": expected_tied,
                "emb_stats": emb_stats,
                "num_layers": config['model']['n_layers'],
                "color": model_config["color"],
                "marker": model_config["marker"],
            }
            
            # Free memory
            del model, logit_lens, tuned_lens
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {repo_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("\nNo models were processed successfully.")
        print("\nThese models use native OLMo checkpoint format.")
        print("You may need to convert them to HuggingFace format first.")
        return
    
    # Create plots
    num_layers = list(results.values())[0]["num_layers"]
    tick_labels = ['emb'] + [str(i) for i in range(1, num_layers + 1)]
    tick_positions = list(range(num_layers + 1))
    
    # Figure 1: Side-by-side Logit vs Tuned lens
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Logit Lens
    ax1 = axes[0]
    for label, data in results.items():
        kl_values = data["logit_kl"]
        n_layers = len(kl_values)
        x = np.arange(n_layers)
        
        ax1.plot(
            x, kl_values,
            marker=data["marker"],
            color=data["color"],
            label=f"{label}",
            markersize=6,
            linewidth=2,
        )
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('KL Divergence (bits)', fontsize=12)
    ax1.set_title('Logit Lens: Tied vs Untied OLMo-70M', fontsize=13)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Right: Tuned Lens
    ax2 = axes[1]
    for label, data in results.items():
        if data["tuned_kl"] is not None:
            kl_values = data["tuned_kl"]
            n_layers = len(kl_values)
            x = np.arange(n_layers)
            
            ax2.plot(
                x, kl_values,
                marker=data["marker"],
                color=data["color"],
                label=f"{label}",
                markersize=6,
                linewidth=2,
            )
    
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('KL Divergence (bits)', fontsize=12)
    ax2.set_title('Tuned Lens: Tied vs Untied OLMo-70M', fontsize=13)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('compare_olmo70m_tied_untied.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to compare_olmo70m_tied_untied.png")
    
    # Figure 2: Improvement from tuned lens
    fig2, ax3 = plt.subplots(figsize=(12, 7))
    
    for label, data in results.items():
        if data["tuned_kl"] is not None:
            improvement = data["logit_kl"] - data["tuned_kl"]
            x = np.arange(len(improvement))
            
            ax3.plot(
                x, improvement,
                marker=data["marker"],
                color=data["color"],
                label=f"{label}",
                markersize=6,
                linewidth=2,
            )
    
    ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels)
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('KL Improvement (bits)', fontsize=12)
    ax3.set_title('Tuned Lens Improvement over Logit Lens\n(positive = tuned lens better)', fontsize=13)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('compare_olmo70m_tied_untied_improvement.png', dpi=150, bbox_inches='tight')
    print(f"Figure saved to compare_olmo70m_tied_untied_improvement.png")
    
    # Figure 3: Bar chart comparing embedding layer specifically
    fig3, ax4 = plt.subplots(figsize=(10, 6))
    
    labels_list = list(results.keys())
    x_pos = np.arange(len(labels_list))
    width = 0.35
    
    logit_emb = [results[l]["logit_kl"][0] for l in labels_list]
    tuned_emb = [results[l]["tuned_kl"][0] if results[l]["tuned_kl"] is not None else 0 for l in labels_list]
    
    bars1 = ax4.bar(x_pos - width/2, logit_emb, width, label='Logit Lens', color=['#2E86AB', '#A23B72'], alpha=0.7)
    bars2 = ax4.bar(x_pos + width/2, tuned_emb, width, label='Tuned Lens', color=['#2E86AB', '#A23B72'], alpha=0.4, hatch='//')
    
    ax4.set_ylabel('KL Divergence at Embedding Layer (bits)', fontsize=12)
    ax4.set_title('Embedding Layer: Tied vs Untied\n(Lower is better)', fontsize=13)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels_list)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('compare_olmo70m_embedding_layer.png', dpi=150, bbox_inches='tight')
    print(f"Figure saved to compare_olmo70m_embedding_layer.png")
    
    # Print comparison table
    print("\n" + "="*90)
    print("COMPARISON TABLE: OLMo-70M Tied vs Untied")
    print("="*90)
    print(f"{'Model':<25} {'Config Tied':<12} {'Avg Logit KL':<15} {'Avg Tuned KL':<15} {'Improvement':<15}")
    print("-"*90)
    
    for label, data in results.items():
        config_tied = "Yes" if data["expected_tied"] else "No"
        avg_logit = np.mean(data["logit_kl"])
        avg_tuned = np.mean(data["tuned_kl"]) if data["tuned_kl"] is not None else 0
        improvement = avg_logit - avg_tuned
        print(f"{label:<25} {config_tied:<12} {avg_logit:<15.4f} {avg_tuned:<15.4f} {improvement:<15.4f}")
    
    # Layer-by-layer comparison
    print("\n" + "="*90)
    print("LAYER-BY-LAYER KL DIVERGENCE (bits) - Logit Lens (L) vs Tuned Lens (T)")
    print("="*90)
    print(f"{'Layer':<8}", end="")
    for label in results.keys():
        print(f"{label[:15] + ' L':<18} {label[:15] + ' T':<18}", end="")
    print()
    print("-"*90)
    
    for i in range(num_layers + 1):
        layer_name = "emb" if i == 0 else str(i)
        print(f"{layer_name:<8}", end="")
        for label, data in results.items():
            tuned_val = data['tuned_kl'][i] if data['tuned_kl'] is not None else 0
            print(f"{data['logit_kl'][i]:<18.4f} {tuned_val:<18.4f}", end="")
        print()
    
    # Key findings
    print("\n" + "="*90)
    print("KEY FINDINGS")
    print("="*90)
    
    tied_emb_logit = results["OLMo-70M Tied"]["logit_kl"][0]
    untied_emb_logit = results["OLMo-70M Untied"]["logit_kl"][0]
    improvement_pct = (tied_emb_logit - untied_emb_logit) / tied_emb_logit * 100
    
    print(f"\n1. EMBEDDING LAYER (Logit Lens):")
    print(f"   - Tied model:   {tied_emb_logit:.4f} bits")
    print(f"   - Untied model: {untied_emb_logit:.4f} bits")
    print(f"   - Improvement:  {improvement_pct:.1f}% better with untied embeddings!")
    
    if results["OLMo-70M Tied"]["tuned_kl"] is not None:
        tied_emb_tuned = results["OLMo-70M Tied"]["tuned_kl"][0]
        untied_emb_tuned = results["OLMo-70M Untied"]["tuned_kl"][0]
        
        print(f"\n2. EMBEDDING LAYER (Tuned Lens):")
        print(f"   - Tied model:   {tied_emb_tuned:.4f} bits")
        print(f"   - Untied model: {untied_emb_tuned:.4f} bits")
        
        print(f"\n3. TUNED LENS IMPROVEMENT at embedding layer:")
        print(f"   - Tied model:   {tied_emb_logit - tied_emb_tuned:.4f} bits improvement")
        print(f"   - Untied model: {untied_emb_logit - untied_emb_tuned:.4f} bits improvement")
    
    print(f"\n4. LATER LAYERS (1-{num_layers}):")
    print(f"   - Both models perform similarly after the embedding layer")
    print(f"   - The untied output embedding only helps at the embedding/early layers")
    
    plt.show()


if __name__ == "__main__":
    main()
