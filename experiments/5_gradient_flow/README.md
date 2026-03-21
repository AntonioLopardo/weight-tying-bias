# Experiment 5: Gradient Flow Analysis (Section 5.3)

**Paper Outputs:** Figure 4

## Overview

Measures gradient contributions from input vs output layers to the shared embedding matrix during training of a tied model. Uses PyTorch backward hooks to observe (but not modify) gradient flows.

## Key Finding

> Output-layer gradients account for 80-90% of the total signal in early training, gradually decreasing to around 75% by step 1000.

- Output gradients exceed input gradients by factor of 5-10x throughout first 1000 steps
- This explains why tied embeddings develop output-like structure

## Scripts

| Script | Description |
|--------|-------------|
| `plot_gradient_provenance.py` | Generates Figure 4 from logged CSV data |
| `gradient_provenance_tracking.md` | Documentation for OLMo integration |

## Reproducing Figure 4

### Prerequisites

1. **Gradient provenance CSV** in this directory:
   - `OLMo-1B-tied-grad-provenance/gradient_provenance.csv` - Per-step gradient norms from 1000 steps of training
   - `OLMo-1B-tied-grad-provenance/config.yaml` - Training config used to produce the CSV
   - `OLMo-1B-tied-grad-provenance/model.pt` - Model weights at step 1000

2. **Training data** (shared across experiments):
   - `../text_data/dolma_v1_7/dolma_v1_7_30B.npy` — see `../text_data/README.md`

### Generate Figure 4

```bash
# Activate virtual environment
source .venv/bin/activate

# Generate Figure 4 from existing CSV data
python plot_gradient_provenance.py \
    OLMo-1B-tied-grad-provenance/gradient_provenance.csv \
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

---

## Model Training Reproduction

The model used for Figure 4 was trained using the **bundled OLMo fork** (`../../OLMo/`) with gradient provenance tracking hooks. See [`../../OLMo/PROVENANCE.md`](../../OLMo/PROVENANCE.md) for details on the modifications.

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

- **Tied model with tracking**: `OLMo-1B-tied-grad-provenance/config.yaml` (sets `weight_tying: true`, `track_embedding_gradient_provenance: true`)

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
    experiments/5_gradient_flow/OLMo-1B-tied-grad-provenance/config.yaml
```

Checkpoints and `gradient_provenance.csv` will be saved to the `save_folder` specified in the config at every training step.

