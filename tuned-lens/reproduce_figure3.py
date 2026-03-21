"""
Reproduce Figure 3 from "Eliciting Latent Predictions from Transformers with the Tuned Lens"
(https://arxiv.org/abs/2303.08112)

Figure 3 shows: Bias of logit lens and tuned lens outputs relative to the final layer output
for GPT-Neo-2.7B. The bias is measured as KL divergence (in bits) at each layer.
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tuned_lens.nn.lenses import TunedLens, LogitLens

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils.tuned_lens_utils import (
    compute_bias_per_layer, load_model_and_tokenizer, SAMPLE_TEXTS,
)


def plot_figure3(tuned_kl: np.ndarray, logit_kl: np.ndarray, model_name: str, save_path: str = None):
    num_layers = len(tuned_kl)
    x = np.arange(num_layers)

    plt.figure(figsize=(10, 6))
    plt.plot(x, logit_kl, 'rs-', label='Logit lens', markersize=6, linewidth=1.5)
    plt.plot(x, tuned_kl, 'bo-', label='Tuned lens', markersize=6, linewidth=1.5)

    tick_labels = ['input'] + [str(i) for i in range(1, num_layers)]
    tick_positions = list(range(num_layers))
    step = max(1, num_layers // 10)
    plt.xticks(tick_positions[::step], tick_labels[::step])

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('KL (bits)', fontsize=12)
    plt.title(f'Bias of logit lens and tuned lens outputs\nrelative to final layer output for {model_name}')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def main():
    model_name = "allenai/OLMo-1B-0724-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")

    model, tokenizer = load_model_and_tokenizer(model_name)
    print(f"Model loaded with {model.config.num_hidden_layers} layers")

    # Load tuned lens
    print("Loading tuned lens...")
    local_lens_path = str(SCRIPT_DIR / "trained_lenses" / model_name)
    try:
        if Path(local_lens_path).exists():
            print(f"Loading local trained lens from {local_lens_path}")
            tuned_lens = TunedLens.from_model_and_pretrained(
                model, lens_resource_id=local_lens_path, weights_only=True
            )
        else:
            tuned_lens = TunedLens.from_model_and_pretrained(
                model, lens_resource_id=model_name, weights_only=True
            )
        print("Loaded pretrained tuned lens!")
    except Exception as e:
        print(f"Could not load pretrained tuned lens: {e}")
        print("Creating untrained tuned lens (results will differ from paper)")
        tuned_lens = TunedLens.from_model(model)

    tuned_lens = tuned_lens.to(device)

    logit_lens = LogitLens.from_model(model).to(device)

    print(f"\nComputing bias for {len(SAMPLE_TEXTS)} sample texts...")
    tuned_kl, logit_kl = compute_bias_per_layer(
        model, tokenizer, tuned_lens, SAMPLE_TEXTS, device, logit_lens=logit_lens
    )

    print("\n=== Results ===")
    print(f"Layer\tTuned Lens KL (bits)\tLogit Lens KL (bits)")
    print("-" * 60)
    for i, (t, l) in enumerate(zip(tuned_kl, logit_kl)):
        layer_name = "input" if i == 0 else str(i)
        print(f"{layer_name}\t{t:.4f}\t\t\t{l:.4f}")

    safe_name = model_name.replace("/", "_")
    plot_figure3(tuned_kl, logit_kl, model_name, save_path=f"figure3_{safe_name}.png")


if __name__ == "__main__":
    main()
