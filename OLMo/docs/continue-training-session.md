# OLMo 1B Continued Training Session

**Date:** December 12, 2025

## Objective

Continue training the OLMo-1B model from an existing checkpoint with a new (larger) dataset, while preserving optimizer state and learning rate schedule.

## Source Files

- **Checkpoint:** `/home/vec_norm/OLMo/checkpoints/OLMo-1B-0724-reproduce/step2000-unsharded/`
- **New Dataset:** `/home/vec_norm/OLMo/data/dolma_v1_7/dolma_v1_7_30B.npy` (44GB, ~30B tokens)
- **Virtual Environment:** `/home/vec_norm/.venv`

## Key Configuration Decisions

### 1. Preserving Optimizer State
```yaml
reset_optimizer_state: false  # Keep Adam momentum and second moments
reset_trainer_state: false    # Keep trainer state
```

### 2. Learning Rate Schedule
- Original training had 2500-step warmup, checkpoint at step 2000
- With `restore_dataloader: false`, step counter resets to 0
- Set `t_warmup: 500` to complete the remaining warmup

### 3. Batch Size Calculation
**Goal:** 1000 steps ≈ 2B tokens

```
tokens_per_step = global_train_batch_size × max_sequence_length
global_train_batch_size = 2B / (4096 × 1000) ≈ 488
```

Chose `global_train_batch_size: 512` for GPU efficiency:
- 1000 steps = 512 × 4096 × 1000 = **2.1B tokens**

### 4. Microbatch Size
Increased to 8 for B200 GPUs with 183GB memory:
```yaml
device_train_microbatch_size: 8
```

## Final Configuration

Created: `/home/vec_norm/OLMo/configs/custom/OLMo-1B-0724-continue.yaml`

```yaml
run_name: OLMo-1B-0724-continue
seed: 6198

model:
  d_model: 2048
  n_heads: 16
  n_layers: 16
  max_sequence_length: 4096
  # ... (same as original OLMo-1B-0724)

optimizer:
  name: adamw
  learning_rate: 3.0e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]

scheduler:
  name: cosine_with_warmup
  t_warmup: 500      # Remaining warmup from original 2500
  alpha_f: 0.1

load_path: /home/vec_norm/OLMo/checkpoints/OLMo-1B-0724-reproduce/step2000-unsharded
reset_optimizer_state: false
reset_trainer_state: false
restore_dataloader: false  # Start fresh with new dataset

global_train_batch_size: 512  # 1000 steps ≈ 2.1B tokens
device_train_microbatch_size: 8
max_duration: 15000

data:
  paths:
    - /home/vec_norm/OLMo/data/dolma_v1_7/dolma_v1_7_30B.npy
```

## Training Command

```bash
cd /home/vec_norm/OLMo && \
source /home/vec_norm/.venv/bin/activate && \
torchrun --nproc_per_node=8 scripts/train.py configs/custom/OLMo-1B-0724-continue.yaml
```

## Performance Metrics

- **GPUs:** 8x NVIDIA B200 (183GB each)
- **Time per step:** ~3.8 seconds
- **Throughput:** ~68K tokens/sec per GPU
- **1000 steps:** ~63 minutes (~2.1B tokens)
- **Full 15,000 steps:** ~16 hours (~31.5B tokens)

## Key Learnings

1. **`restore_dataloader: false`** resets `global_step` to 0 - adjust warmup accordingly
2. **`restore_dataloader: true`** with different batch sizes can cause issues with data position
3. **DDP** replicates full model on each GPU - may need smaller microbatch than FSDP
4. **Optimizer state** (Adam moments) is the most important thing to preserve for stable continued training

## Checkpoints

New checkpoints saved to: `/home/vec_norm/OLMo/checkpoints/OLMo-1B-0724-continue/`
- Unsharded checkpoints every 500 steps
- Sharded checkpoints every 1000 steps



