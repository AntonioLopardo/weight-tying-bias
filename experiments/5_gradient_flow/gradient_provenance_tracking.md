# Gradient Provenance Tracking for Tied Embeddings

When using tied/shared embedding matrices (`weight_tying=True`), gradients from both the input embedding lookup and the output projection flow into the same `wte.weight` parameter. This feature allows you to track the separate contributions from each source.

## Overview

With gradient provenance tracking enabled, OLMo records scalar metrics (norms, means) for gradients flowing from:
1. **Input embeddings**: Gradients from the embedding lookup (`wte(input_ids)`)
2. **Output projection**: Gradients from the logits computation (`F.linear(x, wte.weight)`)

The tracking is implemented using PyTorch backward hooks that observe gradient contributions **without modifying them**, ensuring training behavior remains mathematically identical.

## Output Projection Gradient Clipping

In addition to tracking, you can optionally **clip** the output projection gradients so their norm does not exceed the input embedding gradient norm. This can help balance the gradient contributions from both sources.

**Important**: Since the output projection gradient is computed before the input embedding gradient during backpropagation, the clipping uses the embedding gradient norm from the **previous training step** as the threshold.

## Configuration

Enable gradient provenance tracking in your config YAML:

```yaml
model:
  weight_tying: true  # Must be enabled for provenance tracking to be meaningful
  track_embedding_gradient_provenance: true
```

To also enable output projection gradient clipping:

```yaml
model:
  weight_tying: true
  track_embedding_gradient_provenance: true
  clip_output_proj_to_embedding_grad_norm: true
```

Or programmatically:

```python
from olmo.config import ModelConfig

config = ModelConfig(
    weight_tying=True,
    track_embedding_gradient_provenance=True,
    clip_output_proj_to_embedding_grad_norm=True,  # Optional: clip output proj gradients
    # ... other config options
)
```

## Metrics

When enabled, the following metrics are logged each training step under the `grad_provenance/` prefix:

| Metric | Description |
|--------|-------------|
| `embedding_grad_norm` | L2 norm of gradients from input embeddings |
| `embedding_grad_mean` | Mean of gradients from input embeddings |
| `embedding_grad_abs_mean` | Mean of absolute gradients from input embeddings |
| `output_proj_grad_norm` | L2 norm of gradients from output projection |
| `output_proj_grad_mean` | Mean of gradients from output projection |
| `output_proj_grad_abs_mean` | Mean of absolute gradients from output projection |
| `output_proj_clip_threshold` | The embedding grad norm used as clip threshold (when clipping enabled) |

These metrics are automatically logged to W&B (if configured) and included in console output.

## Programmatic Usage

You can also use the gradient provenance tracking API directly:

```python
from olmo.model import OLMo

# Create model with weight tying
model = OLMo(config)

# Enable tracking
model.enable_gradient_provenance_tracking()

# Optionally enable gradient clipping
model.enable_output_proj_gradient_clipping()

# Run forward-backward pass
output = model(input_ids)
loss = compute_loss(output.logits, labels)
loss.backward()

# Get metrics
metrics = model.get_gradient_provenance_metrics()
print(f"Embedding grad norm: {metrics['embedding_grad_l2_norm']}")
print(f"Output proj grad norm: {metrics['output_proj_grad_l2_norm']}")
# When clipping is enabled:
print(f"Clip threshold: {metrics.get('output_proj_clip_threshold', 'N/A')}")

# Clear metrics for next step
model.clear_gradient_provenance_metrics()

# Disable tracking when done
model.disable_gradient_provenance_tracking()
model.disable_output_proj_gradient_clipping()
```

## Implementation Details

### How It Works

1. **Embedding gradients**: A `register_full_backward_hook` is attached to `transformer.wte` to capture gradients flowing back through the embedding layer.

2. **Output projection gradients**: During the forward pass, the input to `F.linear(x, wte.weight)` is cached. A hook is registered on the logits tensor to compute the gradient w.r.t. `wte.weight` as `grad_output.T @ cached_input`.

### Mathematical Guarantee

When only tracking is enabled (without clipping), the implementation only **observes** gradients without modifying them. The actual gradient accumulation into `wte.weight.grad` happens normally through PyTorch's autograd. This ensures:

- Losses are identical with/without tracking
- Gradients are identical with/without tracking  
- Final weights after optimization are identical

### Gradient Clipping Behavior

When `clip_output_proj_to_embedding_grad_norm=True`:

1. The output projection gradient (flowing back through `F.linear(x, wte.weight)`) is scaled down if its L2 norm exceeds the threshold
2. The threshold is the **previous step's** input embedding gradient L2 norm
3. On the first step, no clipping occurs (no previous reference available)
4. This modifies training behavior intentionally - use with care and monitor metrics

### Verification

Run the verification script to confirm correct behavior:

```bash
source .venv/bin/activate  # or your virtual environment
python scripts/verify_gradient_provenance.py
```

This script tests:
1. Weight equivalence between tracked and untracked training
2. Correct metric capture
3. Enable/disable functionality

## Performance Considerations

- **Memory**: The output projection input tensor is cached during forward pass (size: `batch_size × seq_len × d_model`)
- **Compute**: Computing the full output projection gradient requires a matrix multiplication of size `(vocab_size × batch*seq) @ (batch*seq × d_model)`
- **Overhead**: Minimal when disabled; hooks are only registered when tracking is enabled

For large models, consider enabling tracking only for specific analysis runs rather than full training.

## Files Modified

- `olmo/config.py`: Added `track_embedding_gradient_provenance` and `clip_output_proj_to_embedding_grad_norm` config options
- `olmo/model.py`: Added tracking/clipping methods and hooks to `OLMo` class
- `olmo/train.py`: Integrated metrics logging in `Trainer`
- `scripts/verify_gradient_provenance.py`: Verification test script

