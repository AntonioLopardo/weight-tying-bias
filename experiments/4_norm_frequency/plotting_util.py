import os
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import matplotlib.pyplot as plt
import torch

"""Utilities for plotting relationships between token frequencies and embedding norms.

Sections
--------
- Single-model plots
- Helper utilities
- Multi-model plots
- Diagnostic combined figure (single model)
"""


############################################
# Single-model plots
############################################

def plot_logfreq_vs_l2(first_name: str, model_config: Dict, emb_norms_dict: Dict[str, torch.Tensor], token_freqs_dict: Dict[str, Dict[int, int]], out_dir: str) -> None:
    """Scatter of log10(token frequency + 1) vs L2 norm for a single model.

    Parameters
    - first_name: Model name key to index `emb_norms_dict` and `token_freqs_dict`.
    - model_config: Optional configuration dict (kept for API symmetry; not used here).
    - emb_norms_dict: Mapping model name -> tensor of per-token embedding L2 norms.
    - token_freqs_dict: Mapping model name -> dict of token_id -> raw frequency.
    - out_dir: Directory to save the resulting PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    spec = model_config.get(first_name, {}) if isinstance(model_config, dict) else {}
    norms_t = emb_norms_dict[first_name].detach().cpu().float()
    freq_counts = token_freqs_dict[first_name]
    mask, x, y, _freq = _build_masked_logfreq_and_vals(norms_t, freq_counts)

    plt.figure(figsize=(7, 5))
    plt.scatter(x.numpy(), y.numpy(), s=2, alpha=0.25)
    plt.xlabel('log10(token frequency + 1)')
    plt.ylabel('L2 norm')
    plt.title(f'{first_name}: log frequency vs L2 norm (n={int(mask.sum().item())})')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{first_name}_logfreq_vs_l2.png")
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_binned_l2_vs_logfreq(name: str, norms_t: torch.Tensor, freq_counts: Dict[int, int], num_bins: int, out_dir: str) -> None:
    """Bucketed mean L2 norm vs log10(token frequency + 1) for a single model.

    Computes equal-width bins in log-frequency space and plots the mean and
    standard deviation of L2 norms in each bin.

    Parameters
    - name: Model identifier used for plot title and output filename.
    - norms_t: Tensor of per-token L2 norms (shape: vocab_size).
    - freq_counts: Dict of token_id -> raw frequency counts.
    - num_bins: Number of equal-width bins over the covered log-frequency range.
    - out_dir: Directory to save the resulting PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    mask, logf, vals, _freq = _build_masked_logfreq_and_vals(norms_t, freq_counts)
    if mask.sum().item() == 0:
        return

    min_b, max_b = float(logf.min().item()), float(logf.max().item())
    if max_b == min_b:
        max_b = min_b + 1.0
    bins = torch.linspace(min_b, max_b, steps=num_bins + 1)
    boundaries = bins[1:-1]
    bin_idx = torch.bucketize(logf, boundaries)

    sum_by_bin = defaultdict(float)
    ssq_by_bin = defaultdict(float)
    count_by_bin = defaultdict(int)
    for i in range(len(bin_idx)):
        b = int(bin_idx[i].item())
        v = float(vals[i].item())
        sum_by_bin[b] += v
        ssq_by_bin[b] += v * v
        count_by_bin[b] += 1

    bin_centers = ((bins[:-1] + bins[1:]) / 2.0).tolist()
    mean_l2 = []
    std_l2 = []
    for b in range(num_bins):
        c = count_by_bin.get(b, 0)
        if c == 0:
            mean_l2.append(float('nan'))
            std_l2.append(float('nan'))
        else:
            s = sum_by_bin[b]
            m = s / c
            var = max(ssq_by_bin[b] / c - m * m, 0.0)
            mean_l2.append(m)
            std_l2.append(var ** 0.5)

    import matplotlib.pyplot as plt  # local import to keep top clean

    plt.figure(figsize=(7.5, 5))
    plt.errorbar(bin_centers, mean_l2, yerr=std_l2, fmt='-o', markersize=3, capsize=2, alpha=0.8)
    plt.xlabel('log10(token frequency + 1) bin center')
    plt.ylabel('Mean L2 norm (±1 std)')
    plt.title(f'{name}: L2 norm vs log-frequency buckets (bins={num_bins})')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{name}_l2_vs_logfreq_bins{num_bins}.png")
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()


############################################
# Helper utilities
############################################

def _resolve_dataset_path_for_model(model_id: str, freq_dataset_dir: Optional[str], fallback_data_path: str) -> str:
    """Given a `freq_dataset_dir`, try to pick a dataset file that matches `model_id`.

    Heuristics:
    - Prefer files whose filename contains the last segment of model_id (case-insensitive)
    - Then prefer files containing the full sanitized model_id (slashes -> '__')
    - Fall back to the largest file in the directory
    - If directory is invalid/empty or no files found, return `fallback_data_path`
    """
    try:
        if not freq_dataset_dir or not os.path.isdir(freq_dataset_dir):
            return fallback_data_path
        files = []
        for name in os.listdir(freq_dataset_dir):
            full = os.path.join(freq_dataset_dir, name)
            if os.path.isfile(full):
                files.append(full)
        if not files:
            return fallback_data_path
        tail = model_id.split("/")[-1].lower()
        safe_model = model_id.replace("/", "__").lower()
        # First pass: filenames containing tail
        cand1 = [p for p in files if tail in os.path.basename(p).lower()]
        if cand1:
            # If multiple, choose largest
            cand1.sort(key=lambda p: os.path.getsize(p), reverse=True)
            return cand1[0]
        # Second pass: filenames containing sanitized model id
        cand2 = [p for p in files if safe_model in os.path.basename(p).lower()]
        if cand2:
            cand2.sort(key=lambda p: os.path.getsize(p), reverse=True)
            return cand2[0]
        # Fallback: largest file
        files.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return files[0]
    except Exception:
        return fallback_data_path

def _build_freq_tensor(vocab_size: int, freq_counts: Dict[int, int]) -> torch.Tensor:
    """Construct a dense frequency tensor from a sparse token_id->count mapping."""
    freq = torch.zeros(vocab_size, dtype=torch.long)
    for tid, cnt in freq_counts.items():
        tid_i = int(tid)
        if 0 <= tid_i < vocab_size:
            freq[tid_i] = int(cnt)
    return freq


def _build_masked_logfreq_and_vals(norms_t: torch.Tensor, freq_counts: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (mask, log10(freq+1)[mask], norms_t[mask], freq_tensor).

    The returned tensors are on CPU and float where appropriate to facilitate plotting.
    """
    vocab_size = int(norms_t.shape[0])
    freq = torch.zeros(vocab_size, dtype=torch.long)
    for tid, cnt in freq_counts.items():
        tid_i = int(tid)
        if 0 <= tid_i < vocab_size:
            freq[tid_i] = int(cnt)
    mask = freq > 0
    logf = torch.log10(freq[mask].float() + 1.0)
    vals = norms_t[mask].float()
    return mask, logf, vals, freq


def _compute_ranks_from_freq_and_norms(freq_vals: torch.Tensor, norm_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute frequency and norm ranks (1=highest) for aligned 1D tensors."""
    order_f = torch.argsort(freq_vals, descending=True)
    freq_rank = torch.empty_like(order_f, dtype=torch.long)
    freq_rank.scatter_(0, order_f, torch.arange(1, order_f.numel() + 1, device=order_f.device))

    order_n = torch.argsort(norm_vals, descending=True)
    norm_rank = torch.empty_like(order_n, dtype=torch.long)
    norm_rank.scatter_(0, order_n, torch.arange(1, order_n.numel() + 1, device=order_n.device))
    return freq_rank.cpu(), norm_rank.cpu()


############################################
# Multi-model plots
############################################

def plot_rank_vs_l2_all_models(model_to_freq_counts: Dict[str, Dict[int, int]], model_to_norms: Dict[str, torch.Tensor], out_svg: str, out_png: str, num_bins: int = 50, log_bins: bool = False, x_log_scale: bool = False) -> None:
    """Mean embedding-norm rank vs token-frequency-rank intervals across models.

    Parameters
    - model_to_freq_counts: Mapping name -> token_id->frequency for each model.
    - model_to_norms: Mapping name -> per-token L2 norm tensor for each model.
    - out_svg, out_png: Output paths (PNG is used; SVG kept for API symmetry).
    - num_bins: Number of rank intervals.
    - log_bins: If True, use logarithmically spaced rank intervals.
    - x_log_scale: If True, render x-axis on a logarithmic scale.
    """
    model_to_ranks: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    global_max_rank = 1

    for name, norms_t in model_to_norms.items():
        if name not in model_to_freq_counts:
            continue
        vocab_size = int(norms_t.shape[0])
        freq = _build_freq_tensor(vocab_size, model_to_freq_counts[name])
        mask = freq > 0
        if int(mask.sum().item()) == 0:
            continue
        fr, nr = _compute_ranks_from_freq_and_norms(freq[mask].float(), norms_t[mask].float())
        model_to_ranks[name] = (fr, nr)
        rmax = int(fr.max().item())
        if rmax > global_max_rank:
            global_max_rank = rmax

    if not model_to_ranks:
        return

    if log_bins:
        edges_t = torch.logspace(torch.log10(torch.tensor(1.0)), torch.log10(torch.tensor(float(global_max_rank))), steps=num_bins + 1)
        bin_centers = torch.sqrt(edges_t[:-1] * edges_t[1:])
    else:
        edges_t = torch.linspace(1.0, float(global_max_rank), steps=num_bins + 1)
        bin_centers = (edges_t[:-1] + edges_t[1:]) / 2.0

    plt.figure(figsize=(9, 6))
    for name, (fr, nr) in model_to_ranks.items():
        means = []
        fr_f = fr.float()
        nr_f = nr.float()
        bin_idx = torch.bucketize(fr_f, edges_t[1:-1])
        for b in range(num_bins):
            m = bin_idx == b
            c = int(m.sum().item())
            if c == 0:
                means.append(float('nan'))
            else:
                vals = nr_f[m]
                means.append(float(vals.mean().item()))
        plt.plot(bin_centers.numpy(), means, label=name, linewidth=1.5, alpha=0.9)

    if x_log_scale:
        plt.xscale('log')
    # Place rank 1 (most frequent) on the right for consistency with single-model plots
    plt.gca().invert_xaxis()
    plt.xlabel('Token frequency rank interval center')
    plt.ylabel('Mean L2 norm rank')
    plt.title('Mean L2 norm rank vs token frequency-rank intervals — all models')
    plt.grid(alpha=0.25, which='both' if x_log_scale else 'major')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_freq_rank_vs_mean_l2_all_models(model_to_freq_counts: Dict[str, Dict[int, int]], model_to_norms: Dict[str, torch.Tensor], out_svg: str, out_png: str, num_bins: int = 50, log_bins: bool = False, x_log_scale: bool = False) -> None:
    """Mean raw L2 norm vs token-frequency-rank intervals across models.

    Parameters
    - model_to_freq_counts: Mapping name -> token_id->frequency for each model.
    - model_to_norms: Mapping name -> per-token L2 norm tensor for each model.
    - out_svg, out_png: Output paths (PNG is used; SVG kept for API symmetry).
    - num_bins: Number of rank intervals.
    - log_bins: If True, use logarithmically spaced rank intervals.
    - x_log_scale: If True, render x-axis on a logarithmic scale.
    """
    model_series: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    global_max_rank = 1

    for name, norms_t in model_to_norms.items():
        if name not in model_to_freq_counts:
            continue
        vocab_size = int(norms_t.shape[0])
        freq = _build_freq_tensor(vocab_size, model_to_freq_counts[name])
        mask = freq > 0
        if int(mask.sum().item()) == 0:
            continue

        # Frequency ranks (1 = most frequent)
        fr, _ = _compute_ranks_from_freq_and_norms(freq[mask].float(), norms_t[mask].float())
        vals = norms_t[mask].float()  # raw L2 values

        model_series[name] = (fr, vals)
        rmax = int(fr.max().item())
        if rmax > global_max_rank:
            global_max_rank = rmax

    if not model_series:
        return

    if log_bins:
        edges_t = torch.logspace(torch.log10(torch.tensor(1.0)), torch.log10(torch.tensor(float(global_max_rank))), steps=num_bins + 1)
        bin_centers = torch.sqrt(edges_t[:-1] * edges_t[1:])
    else:
        edges_t = torch.linspace(1.0, float(global_max_rank), steps=num_bins + 1)
        bin_centers = (edges_t[:-1] + edges_t[1:]) / 2.0

    plt.figure(figsize=(9, 6))
    for name, (fr, vals) in sorted(model_series.items()):
        bin_idx = torch.bucketize(fr.float(), edges_t[1:-1])
        means: List[float] = []
        for b in range(num_bins):
            m = bin_idx == b
            c = int(m.sum().item())
            if c == 0:
                means.append(float('nan'))
            else:
                means.append(float(vals[m].mean().item()))
        plt.plot(bin_centers.numpy(), means, label=name, linewidth=1.5, alpha=0.9)

    if x_log_scale:
        plt.xscale('log')
    # Place rank 1 (most frequent) on the right for consistency with single-model plots
    plt.gca().invert_xaxis()
    plt.xlabel('Token frequency rank interval center')
    plt.ylabel('Mean L2 norm')
    plt.title('Mean L2 norm vs token frequency-rank intervals — all models')
    plt.grid(alpha=0.25, which='both' if x_log_scale else 'major')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_l2_zscore_vs_logfreq_all_models(model_to_freq_counts: Dict[str, Dict[int, int]], model_to_norms: Dict[str, torch.Tensor], out_svg: str, out_png: str, num_bins: int = 15) -> None:
    """Mean L2 z-score vs log10(token frequency + 1) across multiple models.

    Bins the shared log-frequency range and plots mean z-scored norms per model.
    """
    model_series = {}
    global_min_logf = float('inf')
    global_max_logf = float('-inf')

    for name, norms_t in model_to_norms.items():
        if name not in model_to_freq_counts:
            continue
        vocab_size = int(norms_t.shape[0])
        freq = _build_freq_tensor(vocab_size, model_to_freq_counts[name])
        mask = freq > 0
        if int(mask.sum().item()) == 0:
            continue
        logf = torch.log10(freq[mask].float() + 1.0)
        vals = norms_t[mask].float()
        mu = vals.mean()
        sigma = vals.std(unbiased=False).clamp(min=1e-8)
        norm_z = (vals - mu) / sigma
        model_series[name] = (logf, norm_z)
        fmin = float(logf.min().item())
        fmax = float(logf.max().item())
        global_min_logf = min(global_min_logf, fmin)
        global_max_logf = max(global_max_logf, fmax)

    if not model_series:
        return

    if global_max_logf <= global_min_logf:
        global_max_logf = global_min_logf + 1.0
    edges_t = torch.linspace(global_min_logf, global_max_logf, steps=num_bins + 1)
    centers = ((edges_t[:-1] + edges_t[1:]) / 2.0).cpu().numpy()

    plt.figure(figsize=(9, 6))
    for name, (logf, norm_vals) in sorted(model_series.items()):
        bin_idx = torch.bucketize(logf, edges_t[1:-1])
        means = []
        for b in range(num_bins):
            m = bin_idx == b
            c = int(m.sum().item())
            if c == 0:
                means.append(float('nan'))
            else:
                vals_b = norm_vals[m].float()
                means.append(float(vals_b.mean().item()))
        plt.plot(centers, means, label=name, linewidth=1.6, alpha=0.95)

    plt.xlabel('log10(token frequency + 1)')
    plt.ylabel('Mean L2 norm z-score')
    plt.title(f'Mean L2 norm z-score vs log10 token frequency — all models (bins={num_bins})')
    plt.ylim(-5, 5)
    plt.axhline(0.0, color='black', linewidth=0.8, alpha=0.5, linestyle='--')
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_l2_vs_logfreq_all_models(model_to_freq_counts: Dict[str, Dict[int, int]], model_to_norms: Dict[str, torch.Tensor], out_svg: str, out_png: str, num_bins: int = 15) -> None:
    """Mean L2 norm vs log10(token frequency + 1) across multiple models.

    Bins the shared log-frequency range and plots mean raw L2 norms per model.
    """
    model_series = {}
    global_min_logf = float('inf')
    global_max_logf = float('-inf')

    for name, norms_t in model_to_norms.items():
        if name not in model_to_freq_counts:
            continue
        vocab_size = int(norms_t.shape[0])
        freq = _build_freq_tensor(vocab_size, model_to_freq_counts[name])
        mask = freq > 0
        if int(mask.sum().item()) == 0:
            continue
        logf = torch.log10(freq[mask].float() + 1.0)
        vals = norms_t[mask].float()
        model_series[name] = (logf, vals)
        fmin = float(logf.min().item())
        fmax = float(logf.max().item())
        global_min_logf = min(global_min_logf, fmin)
        global_max_logf = max(global_max_logf, fmax)

    if not model_series:
        return

    if global_max_logf <= global_min_logf:
        global_max_logf = global_min_logf + 1.0
    edges_t = torch.linspace(global_min_logf, global_max_logf, steps=num_bins + 1)
    centers = ((edges_t[:-1] + edges_t[1:]) / 2.0).cpu().numpy()

    plt.figure(figsize=(9, 6))
    for name, (logf, norm_vals) in sorted(model_series.items()):
        bin_idx = torch.bucketize(logf, edges_t[1:-1])
        means = []
        for b in range(num_bins):
            m = bin_idx == b
            c = int(m.sum().item())
            if c == 0:
                means.append(float('nan'))
            else:
                vals_b = norm_vals[m].float()
                means.append(float(vals_b.mean().item()))
        plt.plot(centers, means, label=name, linewidth=1.6, alpha=0.95)

    plt.xlabel('log10(token frequency + 1)')
    plt.ylabel('Mean L2 norm')
    plt.title(f'Mean L2 norm vs log10 token frequency — all models (bins={num_bins})')
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_l2z_vs_freq_percentile_overlay(model_to_freq_counts: Dict[str, Dict[int, int]], model_to_norms: Dict[str, torch.Tensor], out_svg: str, out_png: str, num_quantiles: int = 100) -> None:
    """Mean L2 z-score vs token frequency percentile across multiple models.

    Higher percentiles correspond to more frequent tokens. Curves are overlaid.
    """
    edges_p = torch.linspace(0.0, 100.0, steps=num_quantiles + 1)
    centers_p = (edges_p[:-1] + edges_p[1:]) / 2.0

    plt.figure(figsize=(18, 12))
    for name, norms_t in sorted(model_to_norms.items()):
        if name not in model_to_freq_counts:
            continue
        vocab_size = int(norms_t.shape[0])
        freq = _build_freq_tensor(vocab_size, model_to_freq_counts[name])
        mask = freq > 0
        if int(mask.sum().item()) == 0:
            continue
        freq_vals = freq[mask].float()
        vals = norms_t[mask].float()
        mu = vals.mean()
        sigma = vals.std(unbiased=False).clamp(min=1e-8)
        vals_z = (vals - mu) / sigma

        n = freq_vals.numel()
        order_f = torch.argsort(freq_vals, descending=True)
        freq_pct = torch.empty_like(freq_vals)
        freq_pct[order_f] = (torch.arange(1, n + 1, device=order_f.device, dtype=torch.float32) / n) * 100.0

        bin_idx = torch.bucketize(freq_pct, edges_p[1:-1])
        means = []
        for b in range(num_quantiles):
            m = bin_idx == b
            if int(m.sum().item()) == 0:
                means.append(float('nan'))
            else:
                vals_b = vals_z[m].float()
                means.append(float(vals_b.mean().item()))

        plt.plot(centers_p.numpy(), means, label=name, linewidth=1.6, alpha=0.95)

    plt.xlabel('Token frequency quantile (left: least frequent, right: most frequent)')
    plt.ylabel('Mean L2 norm z-score')
    plt.title(f'Mean L2 norm z-score vs token frequency quantile — all models (Q={num_quantiles})')
    plt.xlim(100, 0)
    plt.ylim(-2, 2)
    plt.yticks([-2, -1, 0, 1, 2])
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()


############################################
# Diagnostic combined figure (single model)
############################################

def plot_combined_single_model(name: str, norms_t: torch.Tensor, freq_counts: Dict[int, int], out_png_base: str, include_root: bool = True, ymin: Optional[float] = None, ymax: Optional[float] = None) -> None:
    """Generate a 2x(3 or 4) panel figure summarizing frequency-vs-norm views.

    Panels include raw frequency, optional 1/8-root frequency, log10(freq+1),
    and rank views with both power-of-ten and decile reference lines.

    Parameters
    - name: Model identifier for titles.
    - norms_t: Per-token L2 norms tensor.
    - freq_counts: Dict of token_id -> raw frequency.
    - out_png_base: File prefix for output image.
    - include_root: If True, include the 1/8-root frequency columns.
    """
    import numpy as np
    import math
    from matplotlib.lines import Line2D

    vocab_size = int(norms_t.shape[0])
    freq = _build_freq_tensor(vocab_size, freq_counts)
    mask = freq > 0
    num_points = int(mask.sum().item())
    if num_points == 0:
        return

    x_logfreq = torch.log10(freq[mask].float() + 1.0).cpu().numpy()
    x_8throotfreq = torch.pow(freq[mask].float(), 1.0/8.0).cpu().numpy()
    y_norms = norms_t[mask].cpu().numpy()

    freq_vals = freq[mask].float()
    order_f = torch.argsort(freq_vals, descending=True)
    freq_rank = torch.empty_like(order_f, dtype=torch.long)
    freq_rank.scatter_(0, order_f, torch.arange(1, order_f.numel() + 1, device=order_f.device))
    x_rank = freq_rank.cpu().numpy()
    x_freq_raw = freq[mask].cpu().numpy()

    raw_vals = x_freq_raw
    if raw_vals.size > 0:
        min_raw = max(1.0, float(raw_vals.min()))
        max_raw = max(min_raw, float(raw_vals.max()))
        min_pow = int(math.floor(math.log10(min_raw)))
        max_pow = int(math.ceil(math.log10(max_raw)))
        pow10_raw = [10.0 ** p for p in range(min_pow, max_pow + 1)]
        deciles_raw = np.quantile(raw_vals, np.linspace(0.1, 0.9, 9))
    else:
        pow10_raw = []
        deciles_raw = np.array([])

    if include_root:
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0][0]
    ax.scatter(x_freq_raw, y_norms, s=2, alpha=0.25)
    for v in pow10_raw:
        ax.axvline(v, color='red', linewidth=0.6, alpha=0.9)
    ax.set_xlabel('Token frequency (raw count)')
    ax.set_ylabel('L2 norm')
    ax.set_title(f'{name}: raw freq (pow10)')
    ax.grid(alpha=0.2)

    col_idx = 1
    if include_root:
        ax = axes[0][1]
        ax.scatter(x_8throotfreq, y_norms, s=2, alpha=0.25)
        for v in pow10_raw:
            ax.axvline(v ** (1.0/8.0), color='red', linewidth=0.6, alpha=0.9)
        ax.set_xlabel('token frequency^(1/8)')
        ax.set_ylabel('L2 norm')
        ax.set_title(f'{name}: 1/8-root freq (pow10)')
        ax.grid(alpha=0.2)
        col_idx = 2

    ax = axes[0][col_idx]
    ax.scatter(x_logfreq, y_norms, s=2, alpha=0.25)
    for v in pow10_raw:
        ax.axvline(math.log10(v + 1.0), color='red', linewidth=0.6, alpha=0.9)
    ax.set_xlabel('log10(token frequency + 1)')
    ax.set_ylabel('L2 norm')
    ax.set_title(f'{name}: log10 freq (pow10)')
    ax.grid(alpha=0.2)

    ax = axes[0][col_idx + 1]
    ax.scatter(x_rank, y_norms, s=2, alpha=0.25)
    if raw_vals.size > 0 and len(pow10_raw) > 0:
        order_desc = raw_vals.argsort()[::-1]
        sorted_raw = raw_vals[order_desc]
        n = sorted_raw.size
        for v in pow10_raw:
            idx = (sorted_raw >= v).sum()
            if 1 <= idx <= n:
                ax.axvline(idx, color='red', linewidth=0.6, alpha=0.9)
    ax.set_xlabel('Token frequency rank (1 = most frequent)')
    ax.set_ylabel('L2 norm')
    ax.set_title(f'{name}: rank (pow10)')
    ax.grid(alpha=0.2)
    ax.invert_xaxis()

    ax = axes[1][0]
    ax.scatter(x_freq_raw, y_norms, s=2, alpha=0.25)
    for q in deciles_raw:
        ax.axvline(float(q), color='tab:orange', linewidth=0.9, alpha=0.9)
    ax.set_xlabel('Token frequency (raw count)')
    ax.set_ylabel('L2 norm')
    ax.set_title(f'{name}: raw freq (deciles)')
    ax.grid(alpha=0.2)

    if include_root:
        ax = axes[1][1]
        ax.scatter(x_8throotfreq, y_norms, s=2, alpha=0.25)
        for q in deciles_raw:
            ax.axvline(float(q) ** (1.0/8.0), color='tab:orange', linewidth=0.9, alpha=0.9)
        ax.set_xlabel('token frequency^(1/8)')
        ax.set_ylabel('L2 norm')
        ax.set_title(f'{name}: 1/8-root freq (deciles)')
        ax.grid(alpha=0.2)
        col_idx2 = 2
    else:
        col_idx2 = 1

    ax = axes[1][col_idx2]
    ax.scatter(x_logfreq, y_norms, s=2, alpha=0.25)
    for q in deciles_raw:
        ax.axvline(math.log10(float(q) + 1.0), color='tab:orange', linewidth=0.9, alpha=0.9)
    ax.set_xlabel('log10(token frequency + 1)')
    ax.set_ylabel('L2 norm')
    ax.set_title(f'{name}: log10 freq (deciles)')
    ax.grid(alpha=0.2)

    ax = axes[1][col_idx2 + 1]
    ax.scatter(x_rank, y_norms, s=2, alpha=0.25)
    if deciles_raw.size > 0:
        order_desc2 = raw_vals.argsort()[::-1]
        sorted_raw2 = raw_vals[order_desc2]
        n2 = sorted_raw2.size
        for q in deciles_raw:
            idx = (sorted_raw2 >= float(q)).sum()
            if 1 <= idx <= n2:
                ax.axvline(idx, color='tab:orange', linewidth=0.9, alpha=0.9)
    ax.set_xlabel('Token frequency rank (1 = most frequent)')
    ax.set_ylabel('L2 norm')
    ax.set_title(f'{name}: rank (deciles)')
    ax.grid(alpha=0.2)
    ax.invert_xaxis()

    # Apply fixed y-limits across all subplots if provided
    if ymin is not None or ymax is not None:
        try:
            y_lo = float(ymin) if ymin is not None else float(y_norms.min())
            y_hi = float(ymax) if ymax is not None else float(y_norms.max())
            if y_hi <= y_lo:
                y_hi = y_lo + 1e-6
            for ax_all in axes.flatten():
                ax_all.set_ylim(y_lo, y_hi)
        except Exception:
            pass

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='red', lw=1.2, label='Powers of ten (raw freq)'),
        Line2D([0], [0], color='tab:orange', lw=1.2, label='Deciles (raw freq)')
    ]
    fig.legend(handles=legend_elems, loc='upper center', ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = f"{out_png_base}.png"
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()



############################################
# Next-token probability and rank-rank plots
############################################

def plot_next_token_prob_vs_norm_for_model(
    target_name: str,
    model_id: str,
    *,
    data_path: str,
    cache_dir: str,
    freq_type: str,
    revision: Optional[str],
    seeds: int,
    seq_length: int,
    noise_scale: float,
    seed: int,
    noise_mode: str,
    rank_center: int,
    rank_width: int,
    out_dir: str,
    avg_probs_file: Optional[str] = None,
    freq_dataset_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Compute white-noise next-token probabilities and plot vs input L2 norm.

    Returns (out_png, out_svg).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizer_util import build_parallel_freqs_for_model, build_distinct_bigram_counts_for_model

    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with torch.no_grad():
        embedding_layer = model.get_input_embeddings()
        embedding_dim = int(model.config.hidden_size)
        embedding_norms = torch.norm(embedding_layer.weight, dim=1)
        avg_norm = float(torch.mean(embedding_norms).item())
        token_vec_norms = embedding_norms.detach().cpu().numpy()

    try:
        vocab_size = int(model.get_output_embeddings().weight.shape[0])
    except Exception:
        vocab_size = int(tokenizer.vocab_size)

    if avg_probs_file and os.path.exists(avg_probs_file):
        avg_probs_np = np.load(avg_probs_file)
        avg_probs = torch.from_numpy(avg_probs_np).to(torch.float32)
    else:
        accumulated_probs = torch.zeros(vocab_size, dtype=torch.float32)
        _mode = (noise_mode or "white").lower()
        if _mode == "white":
            for i in range(int(seeds)):
                seed_i = int(seed) + i
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                with torch.no_grad():
                    noise_embedding = torch.randn(int(seq_length), embedding_dim, device=device)
                    noise_embedding = avg_norm * (noise_embedding / (torch.norm(noise_embedding, dim=1, keepdim=True) + 1e-12))
                    noise_embedding = noise_embedding * float(noise_scale)
                    outputs = model(inputs_embeds=noise_embedding.unsqueeze(0))
                    logits = outputs.logits
                    last_logits = logits[:, -1, :]
                    probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                if accumulated_probs.shape[0] != probs.shape[0]:
                    new_accum = torch.zeros_like(probs)
                    n_common = min(accumulated_probs.shape[0], probs.shape[0])
                    new_accum[:n_common] = accumulated_probs[:n_common]
                    accumulated_probs = new_accum
                    vocab_size = int(probs.shape[0])
                accumulated_probs += probs
        elif _mode == "data":
            import random as _random
            rng = _random.Random(int(seed))
            samples_collected = 0
            # Stream from file and extract token windows of length seq_length
            while samples_collected < int(seeds):
                try:
                    with open(data_path, "r", encoding="utf-8", errors="ignore") as fh:
                        while samples_collected < int(seeds):
                            skip_lines = rng.randint(0, 1000)
                            for _ in range(skip_lines):
                                if fh.readline() == "":
                                    fh.seek(0)
                                    break
                            # Accumulate text until we have enough tokens
                            buf_lines = []
                            token_ids = []
                            max_lines = 200
                            for _ in range(max_lines):
                                line = fh.readline()
                                if line == "":
                                    fh.seek(0)
                                    line = fh.readline()
                                    if line == "":
                                        break
                                buf_lines.append(line.strip("\n"))
                                text = "\n".join(buf_lines)
                                token_ids = tokenizer.encode(text, add_special_tokens=False)
                                if len(token_ids) >= int(seq_length):
                                    break
                            if len(token_ids) < int(seq_length):
                                break
                            window_ids = token_ids[-int(seq_length):]
                            input_ids = torch.tensor(window_ids, dtype=torch.long, device=device).unsqueeze(0)
                            with torch.no_grad():
                                outputs = model(input_ids=input_ids)
                                logits = outputs.logits
                                last_logits = logits[:, -1, :]
                                probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                            if accumulated_probs.shape[0] != probs.shape[0]:
                                new_accum = torch.zeros_like(probs)
                                n_common = min(accumulated_probs.shape[0], probs.shape[0])
                                new_accum[:n_common] = accumulated_probs[:n_common]
                                accumulated_probs = new_accum
                                vocab_size = int(probs.shape[0])
                            accumulated_probs += probs
                            samples_collected += 1
                except Exception:
                    break

        elif _mode == "ancestral_sampling":
            # Build dataset-driven probabilities: normalize token frequencies to sum to 1
            from tokenizer_util import build_parallel_freqs_for_model
            ds_path = _resolve_dataset_path_for_model(model_id, freq_dataset_dir, data_path)
            freqs = build_parallel_freqs_for_model(model_id, data_path=ds_path, use_cache=True, cache_dir=cache_dir, revision=revision)
            probs = torch.zeros(vocab_size, dtype=torch.float32)
            total = 0.0
            for tok_id, cnt in freqs.items():
                if 0 <= tok_id < probs.shape[0]:
                    c = float(cnt)
                    probs[tok_id] = c
                    total += c
            if total > 0.0:
                probs /= float(total)
            avg_probs = probs.to(torch.float32)
        else:
            # Default back to white if unknown mode
            for i in range(int(seeds)):
                seed_i = int(seed) + i
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                with torch.no_grad():
                    noise_embedding = torch.randn(int(seq_length), embedding_dim, device=device)
                    noise_embedding = avg_norm * (noise_embedding / (torch.norm(noise_embedding, dim=1, keepdim=True) + 1e-12))
                    noise_embedding = noise_embedding * float(noise_scale)
                    outputs = model(inputs_embeds=noise_embedding.unsqueeze(0))
                    logits = outputs.logits
                    last_logits = logits[:, -1, :]
                    probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                if accumulated_probs.shape[0] != probs.shape[0]:
                    new_accum = torch.zeros_like(probs)
                    n_common = min(accumulated_probs.shape[0], probs.shape[0])
                    new_accum[:n_common] = accumulated_probs[:n_common]
                    accumulated_probs = new_accum
                    vocab_size = int(probs.shape[0])
                accumulated_probs += probs
            avg_probs = (accumulated_probs / float(seeds)).to(torch.float32)
        if _mode in ("white", "data"):
            avg_probs = (accumulated_probs / float(seeds)).to(torch.float32)

    # Build frequency counts according to freq_type
    ft = (freq_type or "unigram").lower()
    if ft == "unigram":
        token_freqs = build_parallel_freqs_for_model(model_id, data_path=data_path, use_cache=True, cache_dir=cache_dir, revision=revision)
    elif ft == "bigram":
        token_freqs = build_distinct_bigram_counts_for_model(model_id, data_path=data_path, use_cache=True, cache_dir=cache_dir, revision=revision)
    else:
        raise ValueError(f"Unsupported freq_type: {freq_type}")

    probs_np = avg_probs.numpy()
    freq_arr = np.zeros_like(probs_np)
    for tok_id, count in token_freqs.items():
        if 0 <= tok_id < freq_arr.shape[0]:
            freq_arr[tok_id] = float(count)

    order_desc = np.argsort(-freq_arr, kind="mergesort")
    token_rank = np.empty_like(order_desc)
    token_rank[order_desc] = np.arange(1, order_desc.shape[0] + 1)

    center = int(rank_center)
    width = max(1, int(rank_width))
    n_tokens = token_rank.shape[0]
    start_rank = max(1, center - width // 2)
    end_rank = min(n_tokens, start_rank + width - 1)
    sel_mask = (token_rank >= start_rank) & (token_rank <= end_rank)

    if token_vec_norms.shape[0] != probs_np.shape[0]:
        norms_adj = np.zeros_like(probs_np)
        n_common = min(token_vec_norms.shape[0], probs_np.shape[0])
        norms_adj[:n_common] = token_vec_norms[:n_common]
    else:
        norms_adj = token_vec_norms

    freq_sel = freq_arr[sel_mask]
    if freq_sel.size > 0:
        fmin = float(freq_sel.min())
        fmax = float(freq_sel.max())
        fden = (fmax - fmin) if (fmax - fmin) > 1e-12 else 1.0
        freq_scaled = (freq_sel - fmin) / fden
        marker_sizes = 12.0 + 108.0 * (np.sqrt(freq_scaled))
    else:
        marker_sizes = 12.0

    fig, ax = plt.subplots(figsize=(10, 6))
    y_vals = probs_np[sel_mask]
    x_vals = norms_adj[sel_mask]
    ax.scatter(x_vals, y_vals, s=marker_sizes, alpha=0.6, color="#1f77b4", edgecolors="none")

    if x_vals.size >= 2:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#d62728", linewidth=2.0, label=f"linear fit slope={slope:.3e}")
    else:
        slope, intercept = float("nan"), float("nan")

    ax.set_xlabel("Input embedding L2 norm")
    ax.set_ylabel("Avg next-token probability")
    ax.set_title(f"{target_name}: Next-token prob vs norm (rank r∈[{start_rank},{end_rank}], dot size∝freq)\nlinear fit slope={slope:.3e}")
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in target_name)
    out_png = os.path.join(out_dir, f"{safe_name}_norm_vs_next_token_prob_rank{center}_w{width}.png")
    out_svg = os.path.join(out_dir, f"{safe_name}_norm_vs_next_token_prob_rank{center}_w{width}.svg")
    try:
        fig.savefig(out_png, dpi=150)
        # SVG saving disabled
    finally:
        plt.close(fig)
    return out_png, out_svg


def plot_next_token_prob_vs_freq_for_model(
    target_name: str,
    model_id: str,
    *,
    data_path: str,
    cache_dir: str,
    freq_type: str,
    revision: Optional[str],
    seeds: int,
    seq_length: int,
    noise_scale: float,
    seed: int,
    noise_mode: str,
    rank_center: int,
    rank_width: int,
    out_dir: str,
    avg_probs_file: Optional[str] = None,
    freq_dataset_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Compute next-token probabilities and plot vs token frequency (x-axis).

    Selection window is by token frequency rank, as in the norm-based variant.
    Returns (out_png, out_svg).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizer_util import build_parallel_freqs_for_model, build_distinct_bigram_counts_for_model

    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with torch.no_grad():
        embedding_layer = model.get_input_embeddings()
        embedding_dim = int(model.config.hidden_size)
        embedding_norms = torch.norm(embedding_layer.weight, dim=1)
        avg_norm = float(torch.mean(embedding_norms).item())

    try:
        vocab_size = int(model.get_output_embeddings().weight.shape[0])
    except Exception:
        vocab_size = int(tokenizer.vocab_size)

    if avg_probs_file and os.path.exists(avg_probs_file):
        avg_probs_np = np.load(avg_probs_file)
        avg_probs = torch.from_numpy(avg_probs_np).to(torch.float32)
    else:
        accumulated_probs = torch.zeros(vocab_size, dtype=torch.float32)
        _mode = (noise_mode or "white").lower()
        if _mode == "white":
            for i in range(int(seeds)):
                seed_i = int(seed) + i
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                with torch.no_grad():
                    noise_embedding = torch.randn(int(seq_length), embedding_dim, device=device)
                    noise_embedding = avg_norm * (noise_embedding / (torch.norm(noise_embedding, dim=1, keepdim=True) + 1e-12))
                    noise_embedding = noise_embedding * float(noise_scale)
                    outputs = model(inputs_embeds=noise_embedding.unsqueeze(0))
                    logits = outputs.logits
                    last_logits = logits[:, -1, :]
                    probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                if accumulated_probs.shape[0] != probs.shape[0]:
                    new_accum = torch.zeros_like(probs)
                    n_common = min(accumulated_probs.shape[0], probs.shape[0])
                    new_accum[:n_common] = accumulated_probs[:n_common]
                    accumulated_probs = new_accum
                    vocab_size = int(probs.shape[0])
                accumulated_probs += probs
        elif _mode == "data":
            import random as _random
            rng = _random.Random(int(seed))
            samples_collected = 0
            while samples_collected < int(seeds):
                try:
                    with open(data_path, "r", encoding="utf-8", errors="ignore") as fh:
                        while samples_collected < int(seeds):
                            skip_lines = rng.randint(0, 1000)
                            for _ in range(skip_lines):
                                if fh.readline() == "":
                                    fh.seek(0)
                                    break
                            buf_lines = []
                            token_ids = []
                            max_lines = 200
                            for _ in range(max_lines):
                                line = fh.readline()
                                if line == "":
                                    fh.seek(0)
                                    line = fh.readline()
                                    if line == "":
                                        break
                                buf_lines.append(line.strip("\n"))
                                text = "\n".join(buf_lines)
                                token_ids = tokenizer.encode(text, add_special_tokens=False)
                                if len(token_ids) >= int(seq_length):
                                    break
                            if len(token_ids) < int(seq_length):
                                break
                            window_ids = token_ids[-int(seq_length):]
                            input_ids = torch.tensor(window_ids, dtype=torch.long, device=device).unsqueeze(0)
                            with torch.no_grad():
                                outputs = model(input_ids=input_ids)
                                logits = outputs.logits
                                last_logits = logits[:, -1, :]
                                probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                            if accumulated_probs.shape[0] != probs.shape[0]:
                                new_accum = torch.zeros_like(probs)
                                n_common = min(accumulated_probs.shape[0], probs.shape[0])
                                new_accum[:n_common] = accumulated_probs[:n_common]
                                accumulated_probs = new_accum
                                vocab_size = int(probs.shape[0])
                            accumulated_probs += probs
                            samples_collected += 1
                except Exception:
                    break

        elif _mode == "ancestral_sampling":
            from tokenizer_util import build_parallel_freqs_for_model
            ds_path = _resolve_dataset_path_for_model(model_id, freq_dataset_dir, data_path)
            freqs = build_parallel_freqs_for_model(model_id, data_path=ds_path, use_cache=True, cache_dir=cache_dir, revision=revision)
            probs = torch.zeros(vocab_size, dtype=torch.float32)
            total = 0.0
            for tok_id, cnt in freqs.items():
                if 0 <= tok_id < probs.shape[0]:
                    c = float(cnt)
                    probs[tok_id] = c
                    total += c
            if total > 0.0:
                probs /= float(total)
            avg_probs = probs.to(torch.float32)
        else:
            for i in range(int(seeds)):
                seed_i = int(seed) + i
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                with torch.no_grad():
                    noise_embedding = torch.randn(int(seq_length), embedding_dim, device=device)
                    noise_embedding = avg_norm * (noise_embedding / (torch.norm(noise_embedding, dim=1, keepdim=True) + 1e-12))
                    noise_embedding = noise_embedding * float(noise_scale)
                    outputs = model(inputs_embeds=noise_embedding.unsqueeze(0))
                    logits = outputs.logits
                    last_logits = logits[:, -1, :]
                    probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                if accumulated_probs.shape[0] != probs.shape[0]:
                    new_accum = torch.zeros_like(probs)
                    n_common = min(accumulated_probs.shape[0], probs.shape[0])
                    new_accum[:n_common] = accumulated_probs[:n_common]
                    accumulated_probs = new_accum
                    vocab_size = int(probs.shape[0])
                accumulated_probs += probs
            avg_probs = (accumulated_probs / float(seeds)).to(torch.float32)
        if _mode in ("white", "data"):
            avg_probs = (accumulated_probs / float(seeds)).to(torch.float32)

    # Build frequency counts according to freq_type
    ft = (freq_type or "unigram").lower()
    if ft == "unigram":
        token_freqs = build_parallel_freqs_for_model(model_id, data_path=data_path, use_cache=True, cache_dir=cache_dir, revision=revision)
    elif ft == "bigram":
        token_freqs = build_distinct_bigram_counts_for_model(model_id, data_path=data_path, use_cache=True, cache_dir=cache_dir, revision=revision)
    else:
        raise ValueError(f"Unsupported freq_type: {freq_type}")

    probs_np = avg_probs.numpy()
    freq_arr = np.zeros_like(probs_np)
    for tok_id, count in token_freqs.items():
        if 0 <= tok_id < freq_arr.shape[0]:
            freq_arr[tok_id] = float(count)

    # Build frequency ranks and select window
    order_desc = np.argsort(-freq_arr, kind="mergesort")
    token_rank = np.empty_like(order_desc)
    token_rank[order_desc] = np.arange(1, order_desc.shape[0] + 1)

    center = int(rank_center)
    width = max(1, int(rank_width))
    n_tokens = token_rank.shape[0]
    start_rank = max(1, center - width // 2)
    end_rank = min(n_tokens, start_rank + width - 1)
    sel_mask = (token_rank >= start_rank) & (token_rank <= end_rank)

    x_vals = np.log10(freq_arr[sel_mask] + 1.0)
    y_vals = np.log10(np.clip(probs_np[sel_mask], 1e-12, None))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_vals, y_vals, s=14.0, alpha=0.6, color="#1f77b4", edgecolors="none")

    if x_vals.size >= 2:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#d62728", linewidth=2.0, label=f"linear fit slope={slope:.3e}")
    else:
        slope, intercept = float("nan"), float("nan")

    ax.set_xlabel("log10(token frequency + 1)")
    ax.set_ylabel("log10(avg next-token probability)")
    ax.set_title(
        f"{target_name}: log10 next-token prob vs log10 token frequency (rank r∈[{start_rank},{end_rank}])\n"
        f"linear fit slope={slope:.3e}"
    )
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in target_name)
    out_png = os.path.join(out_dir, f"{safe_name}_logfreq_vs_logprob_rank{center}_w{width}.png")
    out_svg = os.path.join(out_dir, f"{safe_name}_logfreq_vs_logprob_rank{center}_w{width}.svg")
    try:
        fig.savefig(out_png, dpi=150)
        # SVG saving disabled
    finally:
        plt.close(fig)
    return out_png, out_svg

def plot_rankrank_probs_for_model(
    target_name: str,
    model_id: str,
    *,
    data_path: str,
    cache_dir: str,
    freq_type: str,
    revision: Optional[str],
    seeds: int,
    seq_length: int,
    noise_scale: float,
    seed: int,
    noise_mode: str,
    rank_center: int,
    rank_width: int,
    out_dir: str,
    log_file: Optional[str] = None,
    sweep_start: Optional[int] = None,
    sweep_end: Optional[int] = None,
    sweep_step: int = 100,
    avg_probs_file: Optional[str] = None,
    freq_dataset_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Compute rank(norm) vs rank(avg next-token prob) and save plot for one model.

    Returns (out_png, out_svg).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizer_util import build_parallel_freqs_for_model, build_distinct_bigram_counts_for_model

    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with torch.no_grad():
        embedding_layer = model.get_input_embeddings()
        embedding_dim = int(model.config.hidden_size)
        embedding_norms = torch.norm(embedding_layer.weight, dim=1).to("cpu")
        norms_np_full = embedding_norms.numpy()
        avg_norm = float(torch.mean(embedding_norms).item())

    try:
        vocab_size = int(model.get_output_embeddings().weight.shape[0])
    except Exception:
        vocab_size = int(tokenizer.vocab_size)

    if avg_probs_file and os.path.exists(avg_probs_file):
        avg_probs = np.load(avg_probs_file)
    else:
        accumulated_probs = torch.zeros(vocab_size, dtype=torch.float32)
        _mode = (noise_mode or "white").lower()
        if _mode == "white":
            for i in range(int(seeds)):
                seed_i = int(seed) + i
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                with torch.no_grad():
                    noise_embedding = torch.randn(int(seq_length), embedding_dim, device=device)
                    noise_embedding = avg_norm * (noise_embedding / (torch.norm(noise_embedding, dim=1, keepdim=True) + 1e-12))
                    noise_embedding = noise_embedding * float(noise_scale)
                    outputs = model(inputs_embeds=noise_embedding.unsqueeze(0))
                    logits = outputs.logits
                    last_logits = logits[:, -1, :]
                    probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                if accumulated_probs.shape[0] != probs.shape[0]:
                    new_accum = torch.zeros_like(probs)
                    n_common = min(accumulated_probs.shape[0], probs.shape[0])
                    new_accum[:n_common] = accumulated_probs[:n_common]
                    accumulated_probs = new_accum
                    vocab_size = int(probs.shape[0])
                accumulated_probs += probs
        elif _mode == "data":
            import random as _random
            rng = _random.Random(int(seed))
            samples_collected = 0
            while samples_collected < int(seeds):
                try:
                    with open(data_path, "r", encoding="utf-8", errors="ignore") as fh:
                        while samples_collected < int(seeds):
                            skip_lines = rng.randint(0, 1000)
                            for _ in range(skip_lines):
                                if fh.readline() == "":
                                    fh.seek(0)
                                    break
                            buf_lines = []
                            token_ids = []
                            max_lines = 200
                            for _ in range(max_lines):
                                line = fh.readline()
                                if line == "":
                                    fh.seek(0)
                                    line = fh.readline()
                                    if line == "":
                                        break
                                buf_lines.append(line.strip("\n"))
                                text = "\n".join(buf_lines)
                                token_ids = tokenizer.encode(text, add_special_tokens=False)
                                if len(token_ids) >= int(seq_length):
                                    break
                            if len(token_ids) < int(seq_length):
                                break
                            window_ids = token_ids[-int(seq_length):]
                            input_ids = torch.tensor(window_ids, dtype=torch.long, device=device).unsqueeze(0)
                            with torch.no_grad():
                                outputs = model(input_ids=input_ids)
                                logits = outputs.logits
                                last_logits = logits[:, -1, :]
                                probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                            if accumulated_probs.shape[0] != probs.shape[0]:
                                new_accum = torch.zeros_like(probs)
                                n_common = min(accumulated_probs.shape[0], probs.shape[0])
                                new_accum[:n_common] = accumulated_probs[:n_common]
                                accumulated_probs = new_accum
                                vocab_size = int(probs.shape[0])
                            accumulated_probs += probs
                            samples_collected += 1
                except Exception:
                    break

        elif _mode == "ancestral_sampling":
            from tokenizer_util import build_parallel_freqs_for_model
            ds_path = _resolve_dataset_path_for_model(model_id, freq_dataset_dir, data_path)
            freqs = build_parallel_freqs_for_model(model_id, data_path=ds_path, use_cache=True, cache_dir=cache_dir, revision=revision)
            probs = torch.zeros(vocab_size, dtype=torch.float32)
            total = 0.0
            for tok_id, cnt in freqs.items():
                if 0 <= tok_id < probs.shape[0]:
                    c = float(cnt)
                    probs[tok_id] = c
                    total += c
            if total > 0.0:
                probs /= float(total)
            avg_probs = probs.to(torch.float32).numpy()
        else:
            for i in range(int(seeds)):
                seed_i = int(seed) + i
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                with torch.no_grad():
                    noise_embedding = torch.randn(int(seq_length), embedding_dim, device=device)
                    noise_embedding = avg_norm * (noise_embedding / (torch.norm(noise_embedding, dim=1, keepdim=True) + 1e-12))
                    noise_embedding = noise_embedding * float(noise_scale)
                    outputs = model(inputs_embeds=noise_embedding.unsqueeze(0))
                    logits = outputs.logits
                    last_logits = logits[:, -1, :]
                    probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                if accumulated_probs.shape[0] != probs.shape[0]:
                    new_accum = torch.zeros_like(probs)
                    n_common = min(accumulated_probs.shape[0], probs.shape[0])
                    new_accum[:n_common] = accumulated_probs[:n_common]
                    accumulated_probs = new_accum
                    vocab_size = int(probs.shape[0])
                accumulated_probs += probs
            avg_probs = (accumulated_probs / float(seeds)).to(torch.float32).numpy()
        if _mode in ("white", "data"):
            avg_probs = (accumulated_probs / float(seeds)).to(torch.float32).numpy()

    # Build frequency counts according to freq_type
    ft = (freq_type or "unigram").lower()
    if ft == "unigram":
        token_freqs = build_parallel_freqs_for_model(model_id, data_path=data_path, use_cache=True, cache_dir=cache_dir, revision=revision)
    elif ft == "bigram":
        token_freqs = build_distinct_bigram_counts_for_model(model_id, data_path=data_path, use_cache=True, cache_dir=cache_dir, revision=revision)
    else:
        raise ValueError(f"Unsupported freq_type: {freq_type}")

    n_common = min(norms_np_full.shape[0], avg_probs.shape[0])
    norms_np = norms_np_full[:n_common]
    probs_np = avg_probs[:n_common]
    freq_arr = np.zeros(n_common, dtype=np.float32)
    for tok_id, count in token_freqs.items():
        if 0 <= tok_id < n_common:
            freq_arr[tok_id] = float(count)

    order_freq_desc = np.argsort(-freq_arr, kind="mergesort")
    rank_freq = np.empty_like(order_freq_desc)
    rank_freq[order_freq_desc] = np.arange(1, n_common + 1)

    order_norm_desc = np.argsort(-norms_np, kind="mergesort")
    rank_norm = np.empty_like(order_norm_desc)
    rank_norm[order_norm_desc] = np.arange(1, n_common + 1)

    order_prob_desc = np.argsort(-probs_np, kind="mergesort")
    rank_prob = np.empty_like(order_prob_desc)
    rank_prob[order_prob_desc] = np.arange(1, n_common + 1)

    center = int(rank_center)
    width = max(1, int(rank_width))
    start_rank = max(1, center - width // 2)
    end_rank = min(n_common, start_rank + width - 1)
    sel_mask = (rank_freq >= start_rank) & (rank_freq <= end_rank)

    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in target_name)

    if sweep_start is not None and sweep_end is not None:
        W = int(width)
        sum_y = np.zeros(W, dtype=np.float64)
        cnt_y = np.zeros(W, dtype=np.int64)
        total_windows = 0
        for c in range(int(sweep_start), int(sweep_end) + 1, int(sweep_step)):
            s = max(1, int(c) - W // 2)
            e = min(n_common, s + W - 1)
            m = (rank_freq >= s) & (rank_freq <= e)
            if not np.any(m):
                continue
            xg = rank_norm[m].astype(np.int64)
            yg = rank_prob[m].astype(np.int64)
            ord_x = np.argsort(xg, kind="mergesort")
            rx = np.empty_like(xg)
            rx[ord_x] = np.arange(1, xg.shape[0] + 1, dtype=np.int64)
            ord_y = np.argsort(yg, kind="mergesort")
            ry = np.empty_like(yg)
            ry[ord_y] = np.arange(1, yg.shape[0] + 1, dtype=np.int64)
            for i in range(rx.shape[0]):
                pos = int(rx[i]) - 1
                if 0 <= pos < W:
                    sum_y[pos] += float(ry[i])
                    cnt_y[pos] += 1
            total_windows += 1

        valid = cnt_y > 0
        xv = (np.arange(1, W + 1, dtype=np.float64))[valid]
        yv = (sum_y[valid] / cnt_y[valid]).astype(np.float64)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(xv, yv, color="#ff7f0e", linewidth=2.5, alpha=0.95, label="sweep mean")
        ax.scatter(xv, yv, s=26.0, color="#ff7f0e", alpha=0.95, edgecolors="none")

        if xv.size >= 2:
            slope, intercept = np.polyfit(xv, yv, 1)
            x_line = np.linspace(xv.min(), xv.max(), 200, dtype=np.float64)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="#d62728", linewidth=2.0, label=f"linear fit slope={slope:.3f}")
        else:
            slope, intercept = float("nan"), float("nan")

        ax.set_xlabel("Windowed rank of input embedding L2 norm (1 = highest)")
        ax.set_ylabel("Windowed rank of avg next-token probability (1 = highest)")
        ax.set_title(
            f"{target_name}: Sweep-averaged rank curve (centers {int(sweep_start)}..{int(sweep_end)}, step={int(sweep_step)}, W={int(width)})\n"
            f"linear fit slope={slope:.3f}"
        )
        ax.legend(loc="lower right")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
        fig.tight_layout()

        out_png = os.path.join(
            out_dir,
            f"{safe_name}_rank_norm_vs_rank_next_token_prob_sweep_s{int(sweep_start)}_e{int(sweep_end)}_step{int(sweep_step)}_w{int(width)}.png",
        )
        out_svg = os.path.join(
            out_dir,
            f"{safe_name}_rank_norm_vs_rank_next_token_prob_sweep_s{int(sweep_start)}_e{int(sweep_end)}_step{int(sweep_step)}_w{int(width)}.svg",
        )
        try:
            fig.savefig(out_png, dpi=150)
            # SVG saving disabled
            if log_file:
                try:
                    with open(log_file, "a", encoding="utf-8") as _lf:
                        _lf.write(
                            f"model={target_name}\tmode=sweep\tstart={int(sweep_start)}\tend={int(sweep_end)}\tstep={int(sweep_step)}\t"
                            f"width={int(width)}\tslope={slope:.6f}\tintercept={intercept:.6f}\twindows={total_windows}\n"
                        )
                except Exception:
                    pass
        finally:
            plt.close(fig)
        return out_png, out_svg

def plot_rankrank_slopes_vs_center_for_model(
    target_name: str,
    model_id: str,
    *,
    data_path: str,
    cache_dir: str,
    freq_type: str,
    revision: Optional[str],
    seeds: int,
    seq_length: int,
    noise_scale: float,
    seed: int,
    noise_mode: str,
    sweep_start: int,
    sweep_end: int,
    sweep_step: int,
    rank_width: int,
    out_dir: str,
    log_file: Optional[str] = None,
    avg_probs_file: Optional[str] = None,
    freq_dataset_dir: Optional[str] = None,
    fit_on_values: bool = False,
    log_prob: bool = False,
) -> Tuple[str, str]:
    """
    Compute the linear-fit slope of rank(norm) vs rank(avg next-token probability)
    within a moving frequency-rank window centered at multiple ranks, and plot
    slope vs center rank.
    Returns (out_png, out_tsv).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizer_util import build_parallel_freqs_for_model, build_distinct_bigram_counts_for_model

    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with torch.no_grad():
        embedding_layer = model.get_input_embeddings()
        embedding_dim = int(model.config.hidden_size)
        embedding_norms = torch.norm(embedding_layer.weight, dim=1).to("cpu")
        norms_np_full = embedding_norms.numpy()
        avg_norm = float(torch.mean(embedding_norms).item())

    try:
        vocab_size = int(model.get_output_embeddings().weight.shape[0])
    except Exception:
        vocab_size = int(tokenizer.vocab_size)

    # Average next-token probabilities (reuse if provided)
    if avg_probs_file and os.path.exists(avg_probs_file):
        avg_probs = np.load(avg_probs_file)
    else:
        accumulated_probs = torch.zeros(vocab_size, dtype=torch.float32)
        _mode = (noise_mode or "white").lower()
        if _mode == "white":
            for i in range(int(seeds)):
                seed_i = int(seed) + i
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                with torch.no_grad():
                    noise_embedding = torch.randn(int(seq_length), embedding_dim, device=device)
                    noise_embedding = avg_norm * (noise_embedding / (torch.norm(noise_embedding, dim=1, keepdim=True) + 1e-12))
                    noise_embedding = noise_embedding * float(noise_scale)
                    outputs = model(inputs_embeds=noise_embedding.unsqueeze(0))
                    logits = outputs.logits
                    last_logits = logits[:, -1, :]
                    probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                if accumulated_probs.shape[0] != probs.shape[0]:
                    new_accum = torch.zeros_like(probs)
                    n_common = min(accumulated_probs.shape[0], probs.shape[0])
                    new_accum[:n_common] = accumulated_probs[:n_common]
                    accumulated_probs = new_accum
                    vocab_size = int(probs.shape[0])
                accumulated_probs += probs
            avg_probs = (accumulated_probs / float(seeds)).to(torch.float32).numpy()
        elif _mode == "data":
            import random as _random
            rng = _random.Random(int(seed))
            samples_collected = 0
            while samples_collected < int(seeds):
                try:
                    with open(data_path, "r", encoding="utf-8", errors="ignore") as fh:
                        while samples_collected < int(seeds):
                            skip_lines = rng.randint(0, 1000)
                            for _ in range(skip_lines):
                                if fh.readline() == "":
                                    fh.seek(0)
                                    break
                            buf_lines = []
                            token_ids = []
                            max_lines = 200
                            for _ in range(max_lines):
                                line = fh.readline()
                                if line == "":
                                    fh.seek(0)
                                    line = fh.readline()
                                    if line == "":
                                        break
                                buf_lines.append(line.strip("\n"))
                                text = "\n".join(buf_lines)
                                token_ids = tokenizer.encode(text, add_special_tokens=False)
                                if len(token_ids) >= int(seq_length):
                                    break
                            if len(token_ids) < int(seq_length):
                                break
                            window_ids = token_ids[-int(seq_length):]
                            input_ids = torch.tensor(window_ids, dtype=torch.long, device=device).unsqueeze(0)
                            with torch.no_grad():
                                outputs = model(input_ids=input_ids)
                                logits = outputs.logits
                                last_logits = logits[:, -1, :]
                                probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                            if accumulated_probs.shape[0] != probs.shape[0]:
                                new_accum = torch.zeros_like(probs)
                                n_common = min(accumulated_probs.shape[0], probs.shape[0])
                                new_accum[:n_common] = accumulated_probs[:n_common]
                                accumulated_probs = new_accum
                                vocab_size = int(probs.shape[0])
                            accumulated_probs += probs
                            samples_collected += 1
                except Exception:
                    break
            avg_probs = (accumulated_probs / float(max(1, samples_collected))).to(torch.float32).numpy()
        elif _mode == "ancestral_sampling":
            ds_path = _resolve_dataset_path_for_model(model_id, freq_dataset_dir, data_path)
            freqs = build_parallel_freqs_for_model(model_id, data_path=ds_path, use_cache=True, cache_dir=cache_dir, revision=revision)
            probs = torch.zeros(vocab_size, dtype=torch.float32)
            total = 0.0
            for tok_id, cnt in freqs.items():
                if 0 <= tok_id < probs.shape[0]:
                    c = float(cnt)
                    probs[tok_id] = c
                    total += c
            if total > 0.0:
                probs /= float(total)
            avg_probs = probs.to(torch.float32).numpy()
        else:
            for i in range(int(seeds)):
                seed_i = int(seed) + i
                torch.manual_seed(seed_i)
                np.random.seed(seed_i)
                with torch.no_grad():
                    noise_embedding = torch.randn(int(seq_length), embedding_dim, device=device)
                    noise_embedding = avg_norm * (noise_embedding / (torch.norm(noise_embedding, dim=1, keepdim=True) + 1e-12))
                    noise_embedding = noise_embedding * float(noise_scale)
                    outputs = model(inputs_embeds=noise_embedding.unsqueeze(0))
                    logits = outputs.logits
                    last_logits = logits[:, -1, :]
                    probs = torch.softmax(last_logits, dim=-1).squeeze(0).to("cpu")
                if accumulated_probs.shape[0] != probs.shape[0]:
                    new_accum = torch.zeros_like(probs)
                    n_common = min(accumulated_probs.shape[0], probs.shape[0])
                    new_accum[:n_common] = accumulated_probs[:n_common]
                    accumulated_probs = new_accum
                    vocab_size = int(probs.shape[0])
                accumulated_probs += probs
            avg_probs = (accumulated_probs / float(seeds)).to(torch.float32).numpy()

    # Build frequency counts
    ft = (freq_type or "unigram").lower()
    if ft == "unigram":
        token_freqs = build_parallel_freqs_for_model(model_id, data_path=data_path, use_cache=True, cache_dir=cache_dir, revision=revision)
    elif ft == "bigram":
        token_freqs = build_distinct_bigram_counts_for_model(model_id, data_path=data_path, use_cache=True, cache_dir=cache_dir, revision=revision)
    else:
        raise ValueError(f"Unsupported freq_type: {freq_type}")

    n_common = min(norms_np_full.shape[0], avg_probs.shape[0])
    norms_np = norms_np_full[:n_common]
    probs_np = avg_probs[:n_common]
    freq_arr = np.zeros(n_common, dtype=np.float32)
    for tok_id, count in token_freqs.items():
        if 0 <= tok_id < n_common:
            freq_arr[tok_id] = float(count)

    # Global ranks
    order_freq_desc = np.argsort(-freq_arr, kind="mergesort")
    rank_freq = np.empty_like(order_freq_desc)
    rank_freq[order_freq_desc] = np.arange(1, n_common + 1)
    order_norm_desc = np.argsort(-norms_np, kind="mergesort")
    rank_norm = np.empty_like(order_norm_desc)
    rank_norm[order_norm_desc] = np.arange(1, n_common + 1)
    order_prob_desc = np.argsort(-probs_np, kind="mergesort")
    rank_prob = np.empty_like(order_prob_desc)
    rank_prob[order_prob_desc] = np.arange(1, n_common + 1)

    width = max(1, int(rank_width))
    centers: list[int] = []
    slopes: list[float] = []
    intercepts: list[float] = []
    counts: list[int] = []

    for c in range(int(sweep_start), int(sweep_end) + 1, int(sweep_step)):
        s = max(1, int(c) - width // 2)
        e = min(n_common, s + width - 1)
        m = (rank_freq >= s) & (rank_freq <= e)
        if not np.any(m):
            continue
        if not fit_on_values:
            xg = rank_norm[m].astype(np.int64)
            yg = rank_prob[m].astype(np.int64)
            if xg.shape[0] < 2:
                continue
            ord_x = np.argsort(xg, kind="mergesort")
            rx = np.empty_like(xg)
            rx[ord_x] = np.arange(1, xg.shape[0] + 1, dtype=np.int64)
            ord_y = np.argsort(yg, kind="mergesort")
            ry = np.empty_like(yg)
            ry[ord_y] = np.arange(1, yg.shape[0] + 1, dtype=np.int64)
            xv = rx.astype(np.float64)
            yv = ry.astype(np.float64)
        else:
            xv = norms_np[m].astype(np.float64)
            if log_prob:
                # Use log10(probability) for stability; clamp to avoid log(0)
                yv = np.log10(np.clip(probs_np[m].astype(np.float64), 1e-12, None))
            else:
                yv = probs_np[m].astype(np.float64)
            if xv.size < 2:
                continue
        if xv.size >= 2:
            slope, intercept = np.polyfit(xv, yv, 1)
        else:
            slope, intercept = float("nan"), float("nan")
        centers.append(int(c))
        slopes.append(float(slope))
        intercepts.append(float(intercept))
        counts.append(int(xv.size))

    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in target_name)

    # Save TSV of results
    mode_suffix = "_values" if fit_on_values else ""
    if fit_on_values and log_prob:
        mode_suffix = "_values_logprob"
    out_tsv = os.path.join(
        out_dir,
        f"{safe_name}_rankrank_slope_vs_center_s{int(sweep_start)}_e{int(sweep_end)}_step{int(sweep_step)}_w{int(width)}{mode_suffix}.tsv",
    )
    try:
        with open(out_tsv, "w", encoding="utf-8") as fh:
            fh.write("center\tslope\tintercept\tn\n")
            for c, s, b, n in zip(centers, slopes, intercepts, counts):
                fh.write(f"{int(c)}\t{float(s):.6f}\t{float(b):.6f}\t{int(n)}\n")
    except Exception:
        pass

    # Optional logging
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as _lf:
                for c, s, b, n in zip(centers, slopes, intercepts, counts):
                    _lf.write(
                        f"model={target_name}\tmode=per_center\tcenter={int(c)}\twidth={int(width)}\t"
                        f"slope={float(s):.6f}\tintercept={float(b):.6f}\tn={int(n)}\n"
                    )
        except Exception:
            pass

    # Plot slopes vs centers
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, slopes, color="#1f77b4", linewidth=2.0)
    ax.scatter(centers, slopes, s=14.0, color="#1f77b4", alpha=0.9, edgecolors="none")
    ax.set_xlabel("Center frequency rank")
    if fit_on_values:
        ax.set_ylabel("Linear-fit slope (values: norm vs log10 prob)" if log_prob else "Linear-fit slope (values)")
    else:
        ax.set_ylabel("Linear-fit slope (windowed ranks)")
    ax.set_title(
        f"{target_name}: slope vs center (s={int(sweep_start)}..{int(sweep_end)}, step={int(sweep_step)}, W={int(width)}, "
        f"fit={'values(logprob)' if (fit_on_values and log_prob) else ('values' if fit_on_values else 'ranks')})"
    )
    # Show highest (largest) ranks first on the left
    ax.invert_xaxis()
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    out_png = os.path.join(
        out_dir,
        f"{safe_name}_rankrank_slope_vs_center_s{int(sweep_start)}_e{int(sweep_end)}_step{int(sweep_step)}_w{int(width)}{mode_suffix}.png",
    )
    try:
        fig.savefig(out_png, dpi=150)
    finally:
        plt.close(fig)

    return out_png, out_tsv


############################################
# Figure 5: Tied vs Untied comparison (1x3)
############################################

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
    import numpy as np
    import math
    
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
    
    # Colors matching the reference
    color_input = '#1f77b4'   # Blue
    color_output = '#ff7f0e'  # Orange
    color_tied = '#d62728'    # Magenta/Red-pink
    
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
