# Output Projection Gradient Clipping for Tied Embeddings

## Summary

This document summarizes the implementation of output projection gradient clipping, which limits the gradient contribution from the output projection path when using tied embeddings (`weight_tying=True`).

## Problem

When using weight tying, the same `wte.weight` matrix is used for both:
1. **Input Embedding**: Token lookup via `wte(input_ids)`
2. **Output Projection**: Logits computation via `F.linear(hidden_states, wte.weight)`

During backpropagation, gradients from both paths accumulate into `wte.weight.grad`. Empirically, the output projection gradients are often **5-10x larger** than the embedding gradients, potentially dominating the weight updates.

## Solution

Clip the output projection gradients so their norm does not exceed a threshold based on the input embedding gradient norm.

### Implementation Details

#### Version 1 (Simple)
- Use the **previous step's** embedding gradient L2 norm as the clip threshold
- If output proj gradient norm > threshold, scale it down

#### Version 2 (Rolling Average with Scale Factor)
- Maintain a **rolling average** of the last N embedding gradient norms (default: 5)
- Clip threshold = `rolling_average × scale_factor` (default: 0.1)
- More stable and allows tuning the relative contribution

### What's Being Clipped

We register a hook on the **logits tensor** that modifies the gradient before it flows back:

```python
# In forward pass:
logits = F.linear(x, wte.weight)

# Hook on logits:
def output_proj_grad_hook(grad):  # grad = d_loss/d_logits
    rolling_avg = mean(last_5_embedding_grad_norms)
    clip_threshold = rolling_avg * 0.1  # 1/10 of rolling avg
    
    grad_norm = norm(grad)
    if grad_norm > clip_threshold:
        return grad * (clip_threshold / grad_norm)
    return None  # No modification needed
```

### Gradient Flow Diagram

```
Forward:
  input_ids → [wte.weight] → embeddings → transformer → hidden → [wte.weight] → logits → loss
                   ↑                                                    ↑
              (embedding)                                        (output proj)

Backward:
  loss → d_logits ──[CLIPPED HERE]──→ d_wte (from output proj)
                                              ↓
                                         wte.weight.grad
                                              ↑
  loss → ... → d_embeddings → d_wte (from embedding) [NOT CLIPPED]
```

## Configuration

### Config Options

```yaml
model:
  weight_tying: true
  track_embedding_gradient_provenance: true
  clip_output_proj_to_embedding_grad_norm: true
```

### Programmatic API

```python
# Enable with default settings (window=5, scale=0.1)
model.enable_output_proj_gradient_clipping()

# Or customize
model.enable_output_proj_gradient_clipping(window_size=5, scale_factor=0.1)

# Disable
model.disable_output_proj_gradient_clipping()
```

## Experiments

### Experiment 1: No Clipping (Baseline)
- **Run**: `OLMo-1B-grad-provenance-1000`
- **Config**: `track_embedding_gradient_provenance=True`, no clipping
- **Results**: Output proj gradients ~6x larger than embedding gradients

### Experiment 2: Simple Clipping (v1)
- **Run**: `OLMo-1B-grad-clip-1000`
- **Config**: Clip to previous step's embedding grad norm
- **Results**: Still significant ratio, but limited

### Experiment 3: Rolling Average + Scale (v2)
- **Run**: `OLMo-1B-grad-clip-v2-1000`
- **Config**: 
  - `window_size=5` (rolling average of last 5 steps)
  - `scale_factor=0.1` (clip to 1/10 of rolling average)
- **Results**: Aggressive clipping, output proj contribution significantly reduced

## Metrics Logged

| Metric | Description |
|--------|-------------|
| `embedding_grad_l2_norm` | L2 norm of gradients from input embeddings |
| `output_proj_grad_l2_norm` | L2 norm of gradients from output projection (pre-clipping) |
| `embedding_grad_rolling_avg` | Rolling average of embedding grad norms (last N steps) |
| `output_proj_clip_threshold` | Actual clip threshold used (rolling_avg × scale_factor) |

## Files Modified

- `olmo/config.py`: Added `clip_output_proj_to_embedding_grad_norm` config option
- `olmo/model.py`: 
  - Added `enable_output_proj_gradient_clipping(window_size, scale_factor)` method
  - Added `disable_output_proj_gradient_clipping()` method
  - Implemented gradient hook with rolling average logic
- `olmo/train.py`: Integrated clipping enablement and CSV logging
- `scripts/plot_gradient_provenance.py`: Visualization script

## Key Observations

1. **Output projection gradients dominate**: Without clipping, output proj gradients are typically 5-10x larger than embedding gradients

2. **Timing constraint**: The output projection gradient is computed BEFORE the embedding gradient in the backward pass, so we must use previous step(s) as reference

3. **Rolling average is more stable**: Using a window of 5 steps smooths out step-to-step variance

4. **Scale factor controls aggressiveness**: 
   - `scale_factor=1.0`: Clip to match embedding gradient
   - `scale_factor=0.1`: Clip to 1/10 of embedding gradient (aggressive)

## Usage Example

```bash
# Run training with clipping
cd /home/vec_norm/OLMo
source /home/vec_norm/.venv/bin/activate
nohup torchrun --nproc_per_node=1 scripts/train.py \
    configs/custom/OLMo-1B-grad-clip-v2-1000.yaml \
    > checkpoints/OLMo-1B-grad-clip-v2-1000/train.log 2>&1 &

# Generate plots
python scripts/plot_gradient_provenance.py \
    checkpoints/OLMo-1B-grad-clip-v2-1000/gradient_provenance.csv
```

## References

- Related: `docs/gradient_provenance_tracking.md` - Tracking without clipping
- W&B Project: https://wandb.ai/REDACTED/VecNorm

---

*Created: December 22, 2025*

