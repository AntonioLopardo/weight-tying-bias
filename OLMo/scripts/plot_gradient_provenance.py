#!/usr/bin/env python3
"""
Plot gradient provenance metrics from training CSV.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_gradient_provenance(csv_path: str, output_path: str = None):
    """Plot gradient provenance metrics."""
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gradient Provenance Tracking - Tied Embeddings', fontsize=14, fontweight='bold')
    
    # Plot 1: L2 Norms comparison
    ax1 = axes[0, 0]
    ax1.semilogy(df['step'], df['embedding_grad_l2_norm'], label='Input Embedding', alpha=0.8, linewidth=1)
    ax1.semilogy(df['step'], df['output_proj_grad_l2_norm'], label='Output Projection (pre-scale)', alpha=0.8, linewidth=1)
    if 'output_proj_grad_l2_norm_post_clip' in df.columns:
        # Plot scaled/clipped gradients
        post_clip_data = df[df['output_proj_grad_l2_norm_post_clip'].notna()]
        if len(post_clip_data) > 0:
            ax1.semilogy(post_clip_data['step'], post_clip_data['output_proj_grad_l2_norm_post_clip'], 
                        label='Output Projection (post-scale)', alpha=0.8, linewidth=2, color='orange')
    if 'output_proj_clip_threshold' in df.columns:
        # Filter out NaN/empty values
        clip_data = df[df['output_proj_clip_threshold'].notna()]
        if len(clip_data) > 0:
            ax1.semilogy(clip_data['step'], clip_data['output_proj_clip_threshold'], 
                        label='Clip Threshold', alpha=0.8, linewidth=2, linestyle='--', color='red')
    if 'embedding_grad_rolling_avg' in df.columns:
        # Show rolling average
        roll_data = df[df['embedding_grad_rolling_avg'].notna()]
        if len(roll_data) > 0:
            ax1.semilogy(roll_data['step'], roll_data['embedding_grad_rolling_avg'], 
                        label='Embedding Rolling Avg', alpha=0.6, linewidth=1, linestyle=':', color='green')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('L2 Norm (log scale)')
    ax1.set_title('Gradient L2 Norms')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: L1 Norms comparison
    ax2 = axes[0, 1]
    ax2.semilogy(df['step'], df['embedding_grad_l1_norm'], label='Input Embedding', alpha=0.8, linewidth=1)
    ax2.semilogy(df['step'], df['output_proj_grad_l1_norm'], label='Output Projection', alpha=0.8, linewidth=1)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('L1 Norm (log scale)')
    ax2.set_title('Gradient L1 Norms')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ratio of output_proj to embedding gradient norms
    ax3 = axes[1, 0]
    ratio = df['output_proj_grad_l2_norm'] / (df['embedding_grad_l2_norm'] + 1e-10)
    ax3.semilogy(df['step'], ratio, label='Output/Embedding Ratio', alpha=0.8, linewidth=1, color='purple')
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal contribution')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Ratio (log scale)')
    ax3.set_title('Output Proj / Embedding Gradient Ratio (L2)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss curve
    ax4 = axes[1, 1]
    if 'loss' in df.columns and df['loss'].sum() > 0:
        ax4.plot(df['step'], df['loss'], label='Training Loss', alpha=0.8, linewidth=1, color='green')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # If no loss data, show abs mean comparison
        ax4.semilogy(df['step'], df['embedding_grad_abs_mean'], label='Embedding |grad| mean', alpha=0.8, linewidth=1)
        ax4.semilogy(df['step'], df['output_proj_grad_abs_mean'], label='Output Proj |grad| mean', alpha=0.8, linewidth=1)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Absolute Mean (log scale)')
        ax4.set_title('Gradient Absolute Means')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Determine output path
    if output_path is None:
        output_path = str(Path(csv_path).parent / 'gradient_provenance_plot.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    print(f"\nRatio (Output/Embedding):")
    print(f"  Mean: {ratio.mean():.2f}x")
    print(f"  Median: {ratio.median():.2f}x")
    
    if 'output_proj_grad_l2_norm_post_clip' in df.columns:
        post_clip = df['output_proj_grad_l2_norm_post_clip'].dropna()
        if len(post_clip) > 0:
            print(f"\nOutput Proj Grad L2 Norm (POST-SCALE):")
            print(f"  Mean: {post_clip.mean():.4f}")
            print(f"  Std:  {post_clip.std():.4f}")
            print(f"  Min:  {post_clip.min():.4f}")
            print(f"  Max:  {post_clip.max():.4f}")
            scale_ratio = df['output_proj_grad_l2_norm'].mean() / post_clip.mean()
            print(f"\nEffective Scale Factor: {1/scale_ratio:.4f} (pre/post ratio: {scale_ratio:.1f}x)")
    
    if 'output_proj_clip_threshold' in df.columns:
        clip_data = df[df['output_proj_clip_threshold'].notna()]
        if len(clip_data) > 0:
            clipped = (df['output_proj_grad_l2_norm'] > df['output_proj_clip_threshold']).sum()
            print(f"\nClipping Statistics:")
            print(f"  Steps where clipping applied: {clipped} / {len(df)} ({100*clipped/len(df):.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot gradient provenance metrics')
    parser.add_argument('csv_path', help='Path to gradient_provenance.csv')
    parser.add_argument('--output', '-o', help='Output path for plot (default: same dir as CSV)')
    args = parser.parse_args()
    
    plot_gradient_provenance(args.csv_path, args.output)

