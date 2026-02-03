# Gradient Scaling Checkpoints

We are working on adding the model checkpoints in the future.

## Paper Context

These checkpoints support **Section 6 (Causal Ablation: Selective Gradient Scaling)** and **Table 2 / Table 6** in the paper.

The experiment tests whether gradient imbalance *causes* the unembedding bias by artificially amplifying input-layer gradients during training.

## Expected Contents

Once available, this directory will contain OLMo-1B checkpoints trained from scratch with various gradient scaling configurations:

### Table 2: Gradient Scaling at Step 10K (20B tokens)

| Checkpoint | Input Gradient Scale | vs Untied Input | vs Untied Output |
|-----------|---------------------|-----------------|------------------|
| `step10000-baseline/` | 1× (baseline) | 0.216 | 0.384 |
| `step10000-input5x/` | 5× | 0.222 | 0.369 |

### Table 6 (Appendix E): Gradient Scaling at Step 1K

| Checkpoint | Input Gradient Scale | vs Untied Input | vs Untied Output |
|-----------|---------------------|-----------------|------------------|
| `step1000-baseline/` | 1× | 0.172 | 0.197 |
| `step1000-input2x/` | 2× | 0.173 | 0.197 |
| `step1000-input10x/` | 10× | 0.173 | 0.190 |

### Gradient Provenance Analysis (Figure 4)

| Directory | Steps Logged | Purpose |
|-----------|-------------|---------|
| `grad-provenance-1000/` | 0-1000 | Main figure: gradient flow over first 1000 steps |

**Contents per checkpoint:**
- `gradient_log.csv` - Per-step L2 norms of input vs output gradients
- `model.pt` - Model weights at checkpoint

## Training Your Own

These checkpoints require training OLMo from scratch with modified gradient hooks.

### Step 1: Set up OLMo
```bash
git clone https://github.com/allenai/OLMo.git
cd OLMo
pip install -e .
```

### Step 2: Modify for gradient scaling

Add gradient hooks to scale input gradients (see `experiments/5_gradient_flow/gradient_provenance_tracking.md`):

```python
# In the model's embedding layer
def scale_input_grad(grad):
    return grad * SCALE_FACTOR  # e.g., 5.0

model.transformer.wte.weight.register_hook(scale_input_grad)
```

### Step 3: Train with standard OLMo config
```bash
torchrun --nproc_per_node=8 scripts/train.py configs/official/OLMo-1B.yaml
```

**Resource Requirements:**
- 8× A100 GPUs (or equivalent)
- ~10-20 hours for 10K steps
- ~200GB disk space for checkpoints
