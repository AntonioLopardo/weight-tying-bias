# Experiment 6: Gradient Scaling Ablation (Section 6)

**Paper Outputs:** Table 2, Table 6 (Appendix E)

## Overview

Causal intervention where input-layer gradients are artificially scaled during training of tied models. Tests whether gradient balance causally determines embedding structure.

## Key Finding

> Scaling input gradients shifts embedding structure predictably towards untied input embeddings and away from untied output embeddings.

With 5× input gradient scaling at step 10K:
- Input alignment: 0.216 → 0.222 (increases)
- Output alignment: 0.384 → 0.369 (decreases)

## Scripts

| Script | Description |
|--------|-------------|
| `reproduce_table2.py` | Reproduces Table 2 from local checkpoints + official HF reference |
| [`Appendix_E/reproduce_table6.py`](Appendix_E/) | Reproduces Table 6 (step 1K, scaling factors x2 and x10) |

## Reproducing Table 2

### Prerequisites

1. **Model checkpoints** in this directory:
   - `OLMo-1B-tied-no-scale-10000/model.pt` - Tied baseline (no scaling) at step 10K
   - `OLMo-1B-tied-emb5-10000/model.pt` - Tied with 5× input gradient scaling at step 10K

2. **Untied reference** (downloaded automatically from HuggingFace):
   - `allenai/OLMo-1B-0724-hf` at revision `step10000-tokens20B`

3. **Training data** (shared across experiments, only needed if retraining):
   - `../text_data/dolma_v1_7/dolma_v1_7_30B.npy` — see `../text_data/README.md`

### Compute Alignment (Table 2)

```bash
# Activate virtual environment
source /home/vec_norm/.venv/bin/activate

# Reproduce Table 2
python reproduce_table2.py

# Output:
# Tied (no scaling)    vs Untied Input: 0.216   vs Untied Output: 0.384
# Tied (input x5)      vs Untied Input: 0.222   vs Untied Output: 0.369
```

### Expected Results (Table 2, Step 10K)

| Model | vs Untied Input | vs Untied Output |
|-------|-----------------|------------------|
| Tied (no scaling) | 0.216 | 0.384 |
| Tied (input ×5) | **0.222** | 0.369 |

### Expected Results (Table 6, Appendix E, Step 1K)

| Model | vs Untied Input | vs Untied Output |
|-------|-----------------|------------------|
| Tied (no scaling) | 0.172 | 0.197 |
| Tied (input ×2) | 0.173 | 0.197 |
| Tied (input ×10) | 0.173 | **0.190** |

---

## Model Training Reproduction

The models were trained using a custom fork of [OLMo](https://github.com/allenai/OLMo) with gradient scaling hooks added to `olmo/model.py`. The key config difference between baseline and scaled models is `embedding_grad_scale_factor`.

### Training Configuration

| Setting | Value |
|---------|-------|
| Architecture | OLMo-1B (1.17B parameters) |
| Layers | 16 |
| Hidden Size | 2048 |
| Attention Heads | 16 |
| Activation | SwiGLU |
| Position Encoding | RoPE |
| Sequence Length | 4096 |
| Vocab Size | 50,280 |
| Weight Tying | **true** |
| Data | Dolma v1.7 (30B tokens subset) |
| Batch Size | 512 |
| Learning Rate | 3e-4 |
| Warmup Steps | 2500 |
| Scheduler | Cosine with warmup |
| Max Steps | 10,000 |

### Setup

```bash
# Clone custom OLMo fork with gradient scaling hooks
cd /home/vec_norm/OLMo

pip install -e '.[all]'

# Training data (shared across experiments)
# Data should be at: experiments/text_data/dolma_v1_7/dolma_v1_7_30B.npy
# See experiments/text_data/README.md for details
```

### Training Configs

The training configs are stored alongside the model checkpoints:

- **Baseline (no scaling)**: `OLMo-1B-tied-no-scale-10000/config.yaml` (sets `embedding_grad_scale_factor: null`)
- **5× input scaling**: `OLMo-1B-tied-emb5-10000/config.yaml` (sets `embedding_grad_scale_factor: 5.0`)
- **Untied reference**: `OLMo-1B-untied-10000/config.yaml` (sets `weight_tying: false`)

### Train Models

```bash
cd /home/vec_norm/OLMo

# Train tied baseline (no scaling)
torchrun --nproc_per_node=8 scripts/train.py \
    weight-tying-bias/experiments/6_gradient_scaling/OLMo-1B-tied-no-scale-10000/config.yaml

# Train tied with 5× input gradient scaling
torchrun --nproc_per_node=8 scripts/train.py \
    weight-tying-bias/experiments/6_gradient_scaling/OLMo-1B-tied-emb5-10000/config.yaml

# Train untied reference
torchrun --nproc_per_node=8 scripts/train.py \
    weight-tying-bias/experiments/6_gradient_scaling/OLMo-1B-untied-10000/config.yaml
```

Checkpoints will be saved to each config's `save_folder`.
