# OLMo Fork Provenance

This directory contains a modified copy of [OLMo](https://github.com/allenai/OLMo) used for experiments 5 and 6 of the weight-tying bias paper.

## Upstream Base

- **Repository:** https://github.com/allenai/OLMo
- **Version:** v0.6.0 (between releases v0.6.0 and v0.6.1)
- **Date of fork:** January 2025

## Modifications

All changes are gated behind config flags and have no effect on vanilla OLMo training when left at their defaults.

### `olmo/config.py`

Added six config parameters (~50 lines):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `weight_tying` | `True` | Explicit toggle for weight tying |
| `track_embedding_gradient_provenance` | `False` | Enable gradient flow measurement hooks |
| `clip_output_proj_to_embedding_grad_norm` | `False` | Enable gradient clipping on output projection |
| `output_proj_grad_clipping_window_size` | `5` | Rolling window for clipping threshold |
| `embedding_grad_scale_factor` | `None` | Scale input embedding gradients by this factor |
| `output_proj_grad_scale_factor` | `None` | Scale output projection gradients by this factor |

### `olmo/model.py`

Added ~350 lines implementing gradient tracking and scaling:

- **Gradient provenance hooks** — backward hooks on the embedding and output projection layers that record per-step L2/L1 norms and means of gradients flowing through each path. Used in Experiment 5 (Figure 4).
- **Gradient scaling hooks** — multiply input or output gradients by a configurable factor during backpropagation. Used in Experiment 6 (Table 2, Table 6).

### `olmo/train.py`

Added ~200 lines for logging:

- Collects gradient provenance metrics after each backward pass and logs them to WandB.
- Optionally writes per-step gradient norms to `gradient_provenance.csv` for offline analysis.

### `configs/custom/`

24 training configuration files for the paper's experiments (gradient provenance, gradient scaling at various factors, ablations).

## How This Relates to Experiments

| Experiment | Paper Output | OLMo Feature Used |
|---|---|---|
| 5 — Gradient Flow | Figure 4 | `track_embedding_gradient_provenance: true` |
| 6 — Gradient Scaling | Table 2, Table 6 | `embedding_grad_scale_factor: 5.0` (and other values) |

Experiments 1–4 use only pre-trained checkpoints from HuggingFace and do not require this fork.
