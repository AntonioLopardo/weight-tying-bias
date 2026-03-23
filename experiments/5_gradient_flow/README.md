# Experiment 5: Gradient Flow Analysis (Section 5.3)

**Paper Outputs:** Figure 4

## Overview

Measures gradient contributions from input vs output layers to the shared embedding matrix during training of a tied model. Uses PyTorch backward hooks to observe (but not modify) gradient flows.

## Key Finding

> Output-layer gradients dominate the total signal throughout training, accounting for ~70% of the gradient norm over 10,000 steps.

- Output gradients consistently exceed input gradients
- This explains why tied embeddings develop output-like structure

## Scripts

| Script | Description |
|--------|-------------|
| `plot_gradient_provenance.py` | Generates Figure 4 from logged CSV data |
| `gradient_provenance_tracking.md` | Documentation for OLMo integration |

## Reproducing Figure 4

### Prerequisites

1. **Gradient provenance CSV** in this directory:
   - `OLMo-1B-tied-no-scale-10000/gradient_provenance.csv` - Per-step gradient norms from 10,000 steps of training
   - `OLMo-1B-tied-no-scale-10000/config.yaml` - Training config used to produce the CSV
   - `OLMo-1B-tied-no-scale-10000/model.pt` - Model weights at step 10,000 (downloaded via `./download_artifacts.sh 5` from the repo root)

2. **Training data** (shared across experiments):
   - `../text_data/dolma_v1_7/dolma_v1_7_30B.npy` — see `../text_data/README.md`

### Generate Figure 4

```bash
# From the repository root
source .venv/bin/activate

# Generate Figure 4 from existing CSV data (first 1000 steps, matching paper)
python plot_gradient_provenance.py \
    OLMo-1B-tied-no-scale-10000/gradient_provenance.csv \
    --max-steps 1000 \
    --output results/figures/figure4_gradient_provenance.png

# Output: results/figures/figure4_gradient_provenance.png
```

The script applies a rolling average (window=20) for smoothing and produces a two-panel figure:
- **Left panel**: Raw + smoothed L2 norms on log scale
- **Right panel**: Percentage stacked area chart (input vs output contribution)

### Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `csv_path` | Required | Path to `gradient_provenance.csv` |
| `--output` | Same directory as CSV | Output path for the figure |
| `--max-steps` | None (all steps) | Only plot up to this many steps |

---

## Model Training Reproduction

The model used for Figure 4 was trained using the **bundled OLMo fork** (`../../OLMo/`) with gradient provenance tracking hooks. See [`../../OLMo/PROVENANCE.md`](../../OLMo/PROVENANCE.md) for details on the modifications.

### Training Configuration

| Setting | Value |
|---------|-------|
| Architecture | OLMo-1B (1.17B parameters) |
| Layers | 16 |
| Hidden Size | 2048 |
| Attention Heads | 16 |
| Activation | SwiGLU |
| Position Encoding | RoPE |
| Sequence Length | 4,096 |
| Vocab Size | 50,280 |
| Weight Tying | **true** |
| Gradient Tracking | `track_embedding_gradient_provenance: true` |
| Data | Dolma v1.7 (30B tokens subset) |
| Batch Size | 512 (microbatch=8, grad_accum=8) |
| Learning Rate | 3e-4 |
| Warmup Steps | 1,000 |
| Scheduler | Cosine with warmup |
| Max Steps | 10,000 |

### Setup

```bash
# From the repository root
pip install -e './OLMo[all]'

# Training data (shared across experiments)
# Data should be at: experiments/text_data/dolma_v1_7/dolma_v1_7_30B.npy
# See experiments/text_data/README.md for details
```

### Training Config

The training config used is stored alongside the data:

- **Tied model with tracking**: `OLMo-1B-tied-no-scale-10000/config.yaml` (sets `weight_tying: true`, `track_embedding_gradient_provenance: true`)

The critical config settings are:

```yaml
model:
  weight_tying: true
  track_embedding_gradient_provenance: true
```

### Train Model

```bash
# From the repository root
torchrun --nproc_per_node=1 OLMo/scripts/train.py \
    experiments/5_gradient_flow/OLMo-1B-tied-no-scale-10000/config.yaml
```

Checkpoints and `gradient_provenance.csv` will be saved to the `save_folder` specified in the config at every training step.
