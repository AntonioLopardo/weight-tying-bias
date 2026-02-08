"""Plotting utility for Figure 5: tied vs untied embedding norm-frequency comparison."""

import os
from typing import Dict

import matplotlib.pyplot as plt
import torch


def plot_figure5_comparison(
    untied_input_norms: torch.Tensor,
    untied_output_norms: torch.Tensor,
    tied_norms: torch.Tensor,
    freq_counts: Dict[int, int],
    out_path: str,
    untied_name: str = "OLMo-1B-untied",
    tied_name: str = "OLMo-1B-tied",
    steps: str = "10k steps",
    ymin: float = 0.8,
    ymax: float = 2.0,
    xmin: float = 0.0,
    xmax: float = 7.0,
) -> str:
    """Create Figure 5: 1x3 comparison of untied input, untied output, and tied embeddings.

    Parameters
    ----------
    untied_input_norms : torch.Tensor
        L2 norms of untied model input embeddings (vocab_size,)
    untied_output_norms : torch.Tensor
        L2 norms of untied model output embeddings (vocab_size,)
    tied_norms : torch.Tensor
        L2 norms of tied model embeddings (vocab_size,) - same for input/output
    freq_counts : Dict[int, int]
        Token ID -> frequency count mapping
    out_path : str
        Output path for the PNG file
    untied_name : str
        Display name for the untied model
    tied_name : str
        Display name for the tied model
    steps : str
        Training step info for title
    ymin, ymax : float
        Y-axis limits (L2 norm)
    xmin, xmax : float
        X-axis limits (log10(freq + 1))

    Returns
    -------
    str
        Path to the saved figure
    """
    # Get common vocab size
    vocab_size = min(
        untied_input_norms.shape[0],
        untied_output_norms.shape[0],
        tied_norms.shape[0]
    )

    # Build frequency tensor
    freq = torch.zeros(vocab_size, dtype=torch.long)
    for tid, cnt in freq_counts.items():
        tid_i = int(tid)
        if 0 <= tid_i < vocab_size:
            freq[tid_i] = int(cnt)

    mask = freq > 0
    if int(mask.sum().item()) == 0:
        raise ValueError("No tokens with frequency > 0")

    # Prepare data
    x_log = torch.log10(freq[mask].float() + 1.0).cpu().numpy()
    y_untied_in = untied_input_norms[:vocab_size][mask].detach().cpu().float().numpy()
    y_untied_out = untied_output_norms[:vocab_size][mask].detach().cpu().float().numpy()
    y_tied = tied_norms[:vocab_size][mask].detach().cpu().float().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # IBM color scheme
    color_input = '#648FFF'   # Blue
    color_output = '#FFB000'  # Yellow
    color_tied = '#DC267F'    # Magenta

    # Panel 1: Untied Input (blue)
    ax = axes[0]
    ax.scatter(x_log, y_untied_in, s=4, alpha=1.0, color=color_input, edgecolors='none')
    ax.set_xlabel('log10(freq + 1)', fontsize=11)
    ax.set_ylabel('L2 norm', fontsize=11)
    ax.set_title(f'{untied_name} @{steps} (Input)', fontsize=12)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.2)

    # Panel 2: Untied Output (orange)
    ax = axes[1]
    ax.scatter(x_log, y_untied_out, s=4, alpha=1.0, color=color_output, edgecolors='none')
    ax.set_xlabel('log10(freq + 1)', fontsize=11)
    ax.set_ylabel('L2 norm', fontsize=11)
    ax.set_title(f'{untied_name} @{steps} (Output)', fontsize=12)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.2)

    # Panel 3: Tied (magenta)
    ax = axes[2]
    ax.scatter(x_log, y_tied, s=4, alpha=1.0, color=color_tied, edgecolors='none')
    ax.set_xlabel('log10(freq + 1)', fontsize=11)
    ax.set_ylabel('L2 norm', fontsize=11)
    ax.set_title(f'{tied_name} @{steps}', fontsize=12)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.2)

    # Save
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[plot_figure5_comparison] Saved to {out_path}")
    return out_path
