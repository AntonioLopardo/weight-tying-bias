# Appendix E: Gradient Scaling (Tables 6, 7)

**Paper Output:** Table 6, Table 7

## Overview

Extends the gradient scaling ablation from Section 6:
- **Table 6**: Procrustes alignment at step 1K with scaling factors 1×, 2×, and 10×
- **Table 7**: Downstream evaluation (perplexity + benchmarks) of tied vs tied-emb5 at step 10K

## Scripts

| Script | Description | Paper Output |
|--------|-------------|--------------|
| `reproduce_table6.py` | Procrustes alignment for 3 scaling conditions at step 1K | Table 6 |
| `reproduce_table7.py` | Downstream evaluation: tied vs tied-emb5 via lm-evaluation-harness | Table 7 |

## Reproduce Table 6

### Prerequisites

1. **Model checkpoints** in this directory:
   - `OLMo-1B-tied-no-scale-1000/model.pt` — Tied baseline (no scaling) at step 1K
   - `OLMo-1B-tied-emb2-1000/model.pt` — Tied with 2× input gradient scaling at step 1K
   - `OLMo-1B-tied-emb10-1000/model.pt` — Tied with 10× input gradient scaling at step 1K

2. **Untied reference** (downloaded automatically from HuggingFace):
   - `allenai/OLMo-1B-0724-hf` at revision `step1000-tokens2B`

3. **Python environment** with `torch`, `transformers`

### Run

```bash
python reproduce_table6.py
```

### Expected Results (Table 6, Step 1K)

| Model | vs Untied Input | vs Untied Output |
|-------|-----------------|------------------|
| Tied (no scaling) | 0.172 | 0.197 |
| Tied (input ×2) | 0.173 | 0.197 |
| Tied (input ×10) | 0.173 | **0.190** |

---

## Reproduce Table 7

### Prerequisites

1. **Model checkpoints**:
   - `../OLMo-1B-tied-no-scale-10000/model.pt` (tied baseline, parent directory)
   - `../../4_norm_frequency/OLMo-1B-tied/model.pt` (tied baseline)
   - `../OLMo-1B-tied-emb5-10000/model.pt` (tied-emb5, parent directory)

2. **OLMo tokenizer**: `OLMo/olmo_data/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json` (ships with the OLMo submodule)

3. **Python environment** with `torch`, `transformers`, `lm_eval`, `tokenizers`, `pandas`

### Run

```bash
# From the repository root
python experiments/6_gradient_scaling/Appendix_E/reproduce_table7.py

# Custom batch size / device
python experiments/6_gradient_scaling/Appendix_E/reproduce_table7.py --batch-size 16 --device cuda
```

### Expected Results (Table 7, Step 10K)

| Benchmark | Tied | Tied (input ×5) |
|-----------|------|-----------------|
| WikiText-2 PPL ↓ | **35.71** | 36.64 |
| PIQA | 0.620 | **0.631** |
| HellaSwag | **0.329** | 0.326 |
| Winogrande | **0.514** | 0.509 |
| ARC-Easy | 0.396 | **0.402** |
| ARC-Challenge | 0.233 | **0.241** |
| BoolQ | **0.576** | 0.548 |
| OpenBookQA | 0.256 | **0.278** |
| BLiMP | 0.755 | **0.757** |

---

## Model Training Reproduction

The models were trained using the **bundled OLMo fork** (`../../../OLMo/`) with gradient scaling hooks. See [`../../../OLMo/PROVENANCE.md`](../../../OLMo/PROVENANCE.md) for details. The key parameter is `embedding_grad_scale_factor`.

### Training Configuration

Same as Table 2 (see `../README.md`) except:

| Setting | Value |
|---------|-------|
| Max Steps | **1,000** (vs 10,000 for Table 2) |
| Batch Size | 8 (reduced from 512 for tractability) |

### Train Models

```bash
# From the repository root

# Baseline (no scaling)
torchrun --nproc_per_node=1 OLMo/scripts/train.py \
    experiments/6_gradient_scaling/Appendix_E/OLMo-1B-tied-no-scale-1000/config.yaml

# 2x input gradient scaling
torchrun --nproc_per_node=1 OLMo/scripts/train.py \
    experiments/6_gradient_scaling/Appendix_E/OLMo-1B-tied-emb2-1000/config.yaml \
    --model.embedding_grad_scale_factor=2.0

# 10x input gradient scaling
torchrun --nproc_per_node=1 OLMo/scripts/train.py \
    experiments/6_gradient_scaling/Appendix_E/OLMo-1B-tied-emb10-1000/config.yaml \
    --model.embedding_grad_scale_factor=10.0
```

**Note:** The scaling factor was applied via command-line override (`--model.embedding_grad_scale_factor`) rather than in the config YAML. The base configs are identical except for run names.
