# Gradient Scaling Checkpoints

## Paper Context

These checkpoints support **Section 6 (Causal Ablation: Selective Gradient Scaling)**, **Table 2**, **Table 6**, and **Table 7** in the paper.

The experiment tests whether gradient imbalance *causes* the unembedding bias by artificially amplifying input-layer gradients during training.

## Contents

Checkpoints are stored alongside their training configs in the parent directory:

### Table 2 / Table 7 (Step 10K, 20B tokens)

| Directory | Input Gradient Scale | Description |
|-----------|---------------------|-------------|
| `../OLMo-1B-tied-no-scale-10000/` | 1× (baseline) | Tied baseline |
| `../OLMo-1B-tied-emb5-10000/` | 5× | Tied with input gradient scaling |

### Table 6 (Appendix E, Step 1K)

| Directory | Input Gradient Scale | Description |
|-----------|---------------------|-------------|
| `../Appendix_E/OLMo-1B-tied-no-scale-1000/` | 1× (baseline) | Tied baseline at step 1K |
| `../Appendix_E/OLMo-1B-tied-emb2-1000/` | 2× | Tied with 2× input scaling |
| `../Appendix_E/OLMo-1B-tied-emb10-1000/` | 10× | Tied with 10× input scaling |

Each checkpoint directory contains:
- `config.yaml` — OLMo training configuration
- `model.pt` — Model weights (gitignored, download via `./download_artifacts.sh 6`)

## Training Your Own

The models were trained using the **bundled OLMo fork** (`../../OLMo/`) with gradient scaling hooks. See [`../../OLMo/PROVENANCE.md`](../../OLMo/PROVENANCE.md) for details on the modifications. The key config parameter is `embedding_grad_scale_factor`.

```bash
# From the repository root
pip install -e './OLMo[all]'

# Train tied baseline (no scaling)
torchrun --nproc_per_node=8 OLMo/scripts/train.py \
    experiments/6_gradient_scaling/OLMo-1B-tied-no-scale-10000/config.yaml

# Train tied with 5× input gradient scaling
torchrun --nproc_per_node=8 OLMo/scripts/train.py \
    experiments/6_gradient_scaling/OLMo-1B-tied-emb5-10000/config.yaml
```

**Resource Requirements:**
- 8× A100 GPUs (or equivalent)
- ~10-20 hours for 10K steps
- ~200GB disk space for checkpoints
