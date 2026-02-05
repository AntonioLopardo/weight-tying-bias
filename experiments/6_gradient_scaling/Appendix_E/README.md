# Appendix E: Gradient Scaling at Step 1K (Table 6)

**Paper Output:** Table 6

## Overview

Repeats the gradient scaling ablation from Section 6 (Table 2) at an earlier training point (step 1K instead of step 10K), with additional scaling factors (input x2 and input x10).

## Key Finding

> Even at 1K steps, higher input gradient scaling reduces output alignment, consistent with the causal effect observed at 10K steps — though the differences are smaller at this early stage.

## Scripts

| Script | Description | Paper Output |
|--------|-------------|--------------|
| `reproduce_table6.py` | Computes Procrustes alignment for 3 scaling conditions at step 1K | Table 6 |

## Reproduce Table 6

### Prerequisites

1. **Model checkpoints** in this directory:
   - `OLMo-1B-tied-no-scale-1000/model.pt` — Tied baseline (no scaling) at step 1K
   - `OLMo-1B-tied-emb2-1000/model.pt` — Tied with 2x input gradient scaling at step 1K
   - `OLMo-1B-tied-emb10-1000/model.pt` — Tied with 10x input gradient scaling at step 1K

2. **Untied reference** (downloaded automatically from HuggingFace):
   - `allenai/OLMo-1B-0724-hf` at revision `step1000-tokens2B`

3. **Python environment** with `torch`, `transformers`:

```bash
source /home/vec_norm/.venv/bin/activate
```

### Compute Alignment (Table 6)

```bash
python reproduce_table6.py

# Output:
# Tied (no scaling)    vs Untied Input: 0.172   vs Untied Output: 0.197
# Tied (input x2)     vs Untied Input: 0.173   vs Untied Output: 0.197
# Tied (input x10)    vs Untied Input: 0.173   vs Untied Output: 0.190
```

### Expected Results (Table 6, Step 1K)

| Model | vs Untied Input | vs Untied Output |
|-------|-----------------|------------------|
| Tied (no scaling) | 0.172 | 0.197 |
| Tied (input x2) | 0.173 | 0.197 |
| Tied (input x10) | 0.173 | **0.190** |

Compare with Table 2 (step 10K) in the parent directory — the effect is weaker at 1K but follows the same direction.

---

## Model Training Reproduction

The models were trained using a custom fork of [OLMo](https://github.com/allenai/OLMo) with gradient scaling hooks in `olmo/model.py`. The key parameter is `embedding_grad_scale_factor`, which scales the input embedding gradients during backpropagation.

### Training Configuration

Same as Table 2 (see `../README.md`) except:

| Setting | Value |
|---------|-------|
| Max Steps | **1,000** (vs 10,000 for Table 2) |
| Batch Size | 8 (reduced from 512 for tractability) |

### Train Models

```bash
cd /home/vec_norm/OLMo

# Baseline (no scaling)
torchrun --nproc_per_node=1 scripts/train.py \
    weight-tying-bias/experiments/6_gradient_scaling/Appendix_E/OLMo-1B-tied-no-scale-1000/config.yaml

# 2x input gradient scaling
torchrun --nproc_per_node=1 scripts/train.py \
    weight-tying-bias/experiments/6_gradient_scaling/Appendix_E/OLMo-1B-tied-emb2-1000/config.yaml \
    --model.embedding_grad_scale_factor=2.0

# 10x input gradient scaling
torchrun --nproc_per_node=1 scripts/train.py \
    weight-tying-bias/experiments/6_gradient_scaling/Appendix_E/OLMo-1B-tied-emb10-1000/config.yaml \
    --model.embedding_grad_scale_factor=10.0
```

**Note:** The scaling factor was applied via command-line override (`--model.embedding_grad_scale_factor`) rather than in the config YAML. The base configs are identical except for run names.
