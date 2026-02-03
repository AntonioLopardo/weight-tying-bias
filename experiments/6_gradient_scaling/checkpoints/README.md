# Gradient Scaling Checkpoints

We are working on adding the model checkpoints in the future.

## Expected Contents

Once available, this directory will contain OLMo-1B checkpoints trained with various gradient scaling configurations:

### Gradient Scaling Experiments
- `OLMo-1B-tied-no-scale-*` - Baseline (no scaling)
- `OLMo-1B-tied-emb2-*` - Input gradients scaled by 2×
- `OLMo-1B-tied-emb5-*` - Input gradients scaled by 5×
- `OLMo-1B-tied-emb10-*` - Input gradients scaled by 10×
- `OLMo-1B-tied-out0.1-*` - Output gradients scaled by 0.1×
- `OLMo-1B-tied-out0.5-*` - Output gradients scaled by 0.5×

### Gradient Provenance Analysis
- `OLMo-1B-grad-provenance_100/` - Gradient logging at step 100
- `OLMo-1B-grad-provenance-1000/` - Gradient logging at step 1000

## Training Your Own

These checkpoints were generated using a modified version of OLMo with gradient hooks. See `experiments/5_gradient_flow/gradient_provenance_tracking.md` for implementation details.

The training configurations are available in the OLMo repository under `configs/custom/`.
