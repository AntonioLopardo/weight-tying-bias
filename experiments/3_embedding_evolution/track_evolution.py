"""
Embedding Evolution Tracking Script

Reproduces Figures 3, 8, 9 from the paper by tracking how input and output
embedding matrices evolve during training in untied models.

Metrics:
1. Cumulative drift: cosine similarity to step 0 embeddings
2. Per-step change rate: cosine similarity between consecutive checkpoints

Usage:
    python track_evolution.py --config configs/evolution_olmo_1b.json
    python track_evolution.py --config configs/evolution_pythia_1b.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(config_path: str) -> dict:
    """Load model checkpoint configuration."""
    with open(config_path) as f:
        return json.load(f)


def load_embeddings(model_id: str, revision: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Load input and output embeddings from a HuggingFace model.
    
    Returns:
        input_emb: Input embedding matrix
        output_emb: Output embedding matrix  
        weight_tying: Whether weights are tied
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    
    print(f"  Loading {model_id} @ {revision or 'main'}...")
    
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    state_dict = model.state_dict()
    weight_tying = getattr(config, 'tie_word_embeddings', True)
    
    # Find input embedding
    input_emb = None
    for key in state_dict.keys():
        if any(p in key.lower() for p in ["embed_tokens.weight", "wte.weight", "embed_in.weight"]):
            input_emb = state_dict[key].clone()
            break
    
    if input_emb is None:
        emb_keys = [k for k in state_dict.keys() if "embed" in k.lower() and "weight" in k]
        if emb_keys:
            input_emb = state_dict[emb_keys[0]].clone()
    
    # Find output embedding
    if weight_tying:
        output_emb = input_emb
    else:
        output_emb = None
        for key in state_dict.keys():
            if any(p in key.lower() for p in ["lm_head.weight", "ff_out.weight", "embed_out.weight"]):
                output_emb = state_dict[key].clone()
                break
    
    # Clean up
    del model, state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return input_emb, output_emb, weight_tying


def compute_mean_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute mean per-token cosine similarity between two embedding matrices."""
    emb1_norm = emb1 / (emb1.norm(dim=1, keepdim=True) + 1e-8)
    emb2_norm = emb2 / (emb2.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = (emb1_norm * emb2_norm).sum(dim=1)
    return cos_sim.mean().item()


def extract_step_from_name(name: str) -> int:
    """Extract step number from checkpoint name."""
    match = re.search(r'step(\d+)', name)
    if match:
        return int(match.group(1))
    return 0


def track_embedding_evolution(config: dict, output_dir: Path) -> dict:
    """Track embedding evolution across checkpoints from config."""
    # Sort checkpoints by step number
    checkpoints = []
    for name, spec in config.items():
        if spec.get("class") != "huggingface":
            continue
        step = extract_step_from_name(name)
        checkpoints.append((step, name, spec))
    
    checkpoints.sort(key=lambda x: x[0])
    
    if not checkpoints:
        print("ERROR: No valid checkpoints found in config")
        return {}
    
    # Get model info from first checkpoint
    first_spec = checkpoints[0][2]
    model_id = first_spec["path"]
    model_name = model_id.split("/")[-1]
    
    print(f"\nModel: {model_id}")
    print(f"Checkpoints: {len(checkpoints)}")
    
    results = {
        "model_id": model_id,
        "steps": [],
        "input_vs_init": [],
        "output_vs_init": [],
        "input_consecutive": [],
        "output_consecutive": [],
    }
    
    # Load initial checkpoint
    init_step, init_name, init_spec = checkpoints[0]
    init_input, init_output, weight_tying = load_embeddings(
        init_spec["path"], 
        init_spec.get("revision")
    )
    
    if weight_tying:
        print("\nWARNING: Model uses weight tying. Input and output are identical.")
        print("This script is designed for untied models (Figures 3, 8, 9).")
        return results
    
    print(f"Input shape: {init_input.shape}, Output shape: {init_output.shape}")
    
    prev_input, prev_output = init_input, init_output
    
    for step, name, spec in tqdm(checkpoints, desc="Processing checkpoints"):
        results["steps"].append(step)
        
        if step == checkpoints[0][0]:
            # Initial checkpoint - similarity is 1.0
            results["input_vs_init"].append(1.0)
            results["output_vs_init"].append(1.0)
            results["input_consecutive"].append(1.0)
            results["output_consecutive"].append(1.0)
        else:
            try:
                curr_input, curr_output, _ = load_embeddings(
                    spec["path"], 
                    spec.get("revision")
                )
                
                # Cumulative drift from initialization
                results["input_vs_init"].append(
                    compute_mean_cosine_similarity(curr_input, init_input)
                )
                results["output_vs_init"].append(
                    compute_mean_cosine_similarity(curr_output, init_output)
                )
                
                # Per-step change (vs previous checkpoint)
                results["input_consecutive"].append(
                    compute_mean_cosine_similarity(curr_input, prev_input)
                )
                results["output_consecutive"].append(
                    compute_mean_cosine_similarity(curr_output, prev_output)
                )
                
                prev_input, prev_output = curr_input, curr_output
                
            except Exception as e:
                print(f"\nError loading {name}: {e}")
                results["input_vs_init"].append(float('nan'))
                results["output_vs_init"].append(float('nan'))
                results["input_consecutive"].append(float('nan'))
                results["output_consecutive"].append(float('nan'))
    
    # Save results
    output_file = output_dir / f"evolution_{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results, model_name


def get_display_name(model_name: str) -> str:
    """Get a clean display name for the model."""
    if "OLMo-1B-0724" in model_name:
        return "OLMo-1B-0724"
    elif "OLMo-7B" in model_name:
        return "OLMo-7B-0424"
    elif "pythia-1b" in model_name.lower() or "Pythia" in model_name:
        return "Pythia-1B"
    return model_name


def plot_evolution(results: dict, output_dir: Path, model_name: str):
    """Plot embedding evolution matching the paper style (Figures 3, 8, 9).
    
    Top panel: line plot of cosine similarity to initial embeddings.
    Bottom panel: bar chart + dashed line overlay of consecutive cosine similarity.
    """
    steps = results["steps"]
    display_name = get_display_name(model_name)

    # Exclude step 0 from the plot (it's the reference point)
    plot_steps = steps[1:]
    input_vs_init = results["input_vs_init"][1:]
    output_vs_init = results["output_vs_init"][1:]
    input_consec = results["input_consecutive"][1:]
    output_consec = results["output_consecutive"][1:]

    # Colors matching the paper (IBM palette)
    color_input = '#648FFF'
    color_output = '#FFB000'

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # --- Top panel: Cumulative drift from initialization ---
    ax1 = axes[0]
    ax1.plot(plot_steps, input_vs_init, '-o', color=color_input, markersize=6,
             linewidth=2, label=f'{display_name} Input Embedding')
    ax1.plot(plot_steps, output_vs_init, '-s', color=color_output, markersize=6,
             linewidth=2, label=f'{display_name} Output Projection')
    ax1.set_ylabel('Cos. Sim. - Step 0', fontsize=18)
    ax1.legend(fontsize=14, loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=14)

    # --- Bottom panel: Bar chart + dashed line for consecutive similarity ---
    ax2 = axes[1]

    # Set y-axis limits to zoom into the relevant range
    all_consec = input_consec + output_consec
    y_min = min(all_consec) - 0.01
    # Round down to nearest 0.01
    y_min = max(0, int(y_min * 100) / 100)
    ax2.set_ylim(y_min, 1.0)

    # Compute bar width based on step interval
    step_interval = plot_steps[1] - plot_steps[0] if len(plot_steps) > 1 else 1000
    bar_width = step_interval * 0.35

    # Bars hang from the top (y=1.0) down to the value
    y_top = 1.0
    ax2.bar(
        [s - bar_width / 2 for s in plot_steps],
        [y_top - v for v in input_consec],
        bottom=input_consec,
        width=bar_width, color=color_input, alpha=0.5, edgecolor=color_input
    )
    ax2.bar(
        [s + bar_width / 2 for s in plot_steps],
        [y_top - v for v in output_consec],
        bottom=output_consec,
        width=bar_width, color=color_output, alpha=0.5, edgecolor=color_output
    )

    ax2.plot(plot_steps, input_consec, '--o', color=color_input, markersize=6,
             linewidth=2, label=f'{display_name} Input Embedding')
    ax2.plot(plot_steps, output_consec, '--s', color=color_output, markersize=6,
             linewidth=2, label=f'{display_name} Output Projection')

    ax2.set_xlabel('Training Step', fontsize=18)
    ax2.set_ylabel('Cos. Sim. - Prev. Step', fontsize=18)
    ax2.legend(fontsize=14, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=14)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / f"figure_evolution_{model_name.replace('-', '_')}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Track embedding evolution during training")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to config JSON with checkpoint revisions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: script directory)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    
    # Track evolution
    result = track_embedding_evolution(config, output_dir)
    
    if result:
        results, model_name = result
        if results.get("steps"):
            plot_evolution(results, output_dir, model_name)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
