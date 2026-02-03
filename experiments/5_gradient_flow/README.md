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
| `verify_gradient_provenance.py` | Verifies gradient tracking doesn't affect training |
| `gradient_provenance_tracking.md` | Documentation for OLMo integration |

## Pre-generated Figures

Available in `results/figures/`:
- `figure4_gradient_provenance_100steps.png` - First 100 training steps
- `figure4_gradient_provenance_1000steps.png` - First 1000 training steps

## Usage

### Plotting from Existing Data

```bash
python plot_gradient_provenance.py path/to/gradient_provenance.csv
```

### Generating New Data

Requires training OLMo with gradient provenance tracking enabled:

```yaml
# In OLMo config YAML
model:
  weight_tying: true
  track_embedding_gradient_provenance: true
```

Then run training:

```bash
cd /path/to/OLMo
torchrun --nproc_per_node=1 scripts/train.py configs/custom/OLMo-1B-grad-provenance.yaml
```

The training logs gradient metrics to CSV which can be plotted with `plot_gradient_provenance.py`.

## Implementation Details

From `gradient_provenance_tracking.md`:

1. **Embedding gradients**: A `register_full_backward_hook` on `transformer.wte` captures gradients from embedding lookup
2. **Output projection gradients**: Hook on logits tensor computes gradient w.r.t. `wte.weight` as `grad_output.T @ cached_input`

### Mathematical Guarantee

When only tracking (not clipping), the implementation only **observes** gradients without modifying them:
- Losses are identical with/without tracking
- Gradients are identical with/without tracking
- Final weights after optimization are identical

## Available Checkpoints

Checkpoints with gradient provenance data:
- `OLMo/checkpoints/OLMo-1B-grad-provenance_100/` (100 steps)
- `OLMo/checkpoints/OLMo-1B-grad-provenance-1000/` (1000 steps)

## Expected Results (Figure 4)

Left panel (log scale):
- Output gradient norms: ~10x higher than input throughout
- Gap narrows slightly over 1000 steps

Right panel (percentage):
- Step 0: Output ~85%, Input ~15%
- Step 1000: Output ~75%, Input ~25%
