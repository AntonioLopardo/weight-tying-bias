"""
Compare tuned lens KL divergence between two OLMo models.
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from tuned_lens.nn.lenses import TunedLens
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils.tuned_lens_utils import (
    compute_bias_per_layer, load_model_and_tokenizer, SAMPLE_TEXTS,
)

TRAINED_LENSES_DIR = SCRIPT_DIR / "trained_lenses"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_to_compare = [
        ("allenai/OLMo-1B-0724-hf", "OLMo-1B-0724"),
        ("allenai/OLMo-1B-hf", "OLMo-1B"),
    ]

    results = {}

    for model_name, label in models_to_compare:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print('='*60)

        model, tokenizer = load_model_and_tokenizer(model_name)

        local_lens_path = str(TRAINED_LENSES_DIR / model_name)
        tuned_lens = TunedLens.from_model_and_pretrained(
            model, lens_resource_id=local_lens_path, weights_only=True
        ).to(device)

        tuned_kl = compute_bias_per_layer(model, tokenizer, tuned_lens, SAMPLE_TEXTS, device)
        results[label] = tuned_kl

        del model, tuned_lens, tokenizer
        torch.cuda.empty_cache()

    # Plot comparison
    plt.figure(figsize=(10, 6))

    colors = ['#2E86AB', '#A23B72']
    markers = ['o', 's']

    for i, (label, kl_values) in enumerate(results.items()):
        num_layers = len(kl_values)
        x = np.arange(num_layers)
        plt.plot(x, kl_values, f'{markers[i]}-', color=colors[i],
                 label=f'{label} (Tuned Lens)', markersize=6, linewidth=2)

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


if __name__ == "__main__":
    main()
