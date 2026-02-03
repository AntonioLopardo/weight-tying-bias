#!/usr/bin/env python3
"""
Plot gradient provenance metrics from training CSV.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_gradient_provenance(csv_path: str, output_path: str = None, window: int = 20):
    """Plot gradient provenance metrics with rolling average smoothing."""
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Colors matching the new color scheme
    color_input = "#648FFF"   # Blue for Input
    color_output = "#FFB000"  # Gold for Output
    
    # Apply rolling average for smoothing
    emb_l2_smooth = df['embedding_grad_l2_norm'].rolling(window=window, min_periods=1).mean()
    out_l2_smooth = df['output_proj_grad_l2_norm'].rolling(window=window, min_periods=1).mean()
    
    # Create figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
    plt.subplots_adjust(wspace=0.2, bottom=0.15, left=0.07, right=0.98)
    
    # Plot 1: Absolute L2 Norms (smoothed)
    ax1 = axes[0]
    # Raw data (transparent, behind)
    ax1.semilogy(df['step'], df['embedding_grad_l2_norm'], 
                 alpha=0.2, linewidth=1, color=color_input)
    ax1.semilogy(df['step'], df['output_proj_grad_l2_norm'], 
                 alpha=0.2, linewidth=1, color=color_output)
    # Smoothed data (solid, on top)
    ax1.semilogy(df['step'], emb_l2_smooth, label='Input Embedding', 
                 alpha=0.9, linewidth=2.5, color=color_input)
    ax1.semilogy(df['step'], out_l2_smooth, label='Output Projection', 
                 alpha=0.9, linewidth=2.5, color=color_output)
    ax1.set_xlabel('Step', fontsize=26)
    ax1.set_ylabel('L2 Norm (log scale)', fontsize=26)
    ax1.tick_params(axis='both', labelsize=22)
    ax1.legend(fontsize=22)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative balance (percentage of total, smoothed)
    ax2 = axes[1]
    emb_l2 = emb_l2_smooth.values
    out_l2 = out_l2_smooth.values
    total = emb_l2 + out_l2 + 1e-10
    emb_pct = 100.0 * emb_l2 / total
    out_pct = 100.0 * out_l2 / total
    
    ax2.fill_between(df['step'], 0, emb_pct, alpha=0.5, color=color_input, label='Input Embedding')
    ax2.fill_between(df['step'], emb_pct, 100, alpha=0.5, color=color_output, label='Output Projection')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Step', fontsize=26)
    ax2.set_ylabel('% of Total L2 Norm', fontsize=26)
    ax2.tick_params(axis='both', labelsize=22)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=22)
    ax2.grid(True, alpha=0.3)
    
    # Determine output path
    if output_path is None:
        output_path = str(Path(csv_path).parent / 'gradient_provenance_plot.png')
    
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    
    # Also show some statistics
    print("\n=== Gradient Provenance Statistics ===")
    print(f"Steps: {df['step'].min()} - {df['step'].max()}")
    print(f"\nEmbedding Grad L2 Norm:")
    print(f"  Mean: {df['embedding_grad_l2_norm'].mean():.4f}")
    print(f"  Std:  {df['embedding_grad_l2_norm'].std():.4f}")
    print(f"  Min:  {df['embedding_grad_l2_norm'].min():.4f}")
    print(f"  Max:  {df['embedding_grad_l2_norm'].max():.4f}")
    print(f"\nOutput Proj Grad L2 Norm:")
    print(f"  Mean: {df['output_proj_grad_l2_norm'].mean():.4f}")
    print(f"  Std:  {df['output_proj_grad_l2_norm'].std():.4f}")
    print(f"  Min:  {df['output_proj_grad_l2_norm'].min():.4f}")
    print(f"  Max:  {df['output_proj_grad_l2_norm'].max():.4f}")
    
    # Compute ratio for stats
    ratio = df['output_proj_grad_l2_norm'] / (df['embedding_grad_l2_norm'] + 1e-10)
    print(f"\nRatio (Output/Embedding):")
    print(f"  Mean: {ratio.mean():.2f}x")
    print(f"  Median: {ratio.median():.2f}x")
    
    # Relative balance stats
    print(f"\nRelative Balance (avg %):")
    print(f"  Input Embedding: {emb_pct.mean():.1f}%")
    print(f"  Output Projection: {out_pct.mean():.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot gradient provenance metrics')
    parser.add_argument('csv_path', help='Path to gradient_provenance.csv')
    parser.add_argument('--output', '-o', help='Output path for plot (default: same dir as CSV)')
    args = parser.parse_args()
    
    plot_gradient_provenance(args.csv_path, args.output)

