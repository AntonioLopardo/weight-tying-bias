# OLMo-300M Extended Training Run

**Date**: December 9-10, 2025  
**Status**: ✅ Completed

## Overview

| Parameter | Value |
|-----------|-------|
| Model | OLMo-300M (371M params) |
| Total Steps | 2,500 |
| Total Tokens | ~10.5B |
| Effective Batch Size | 1024 |
| Sequence Length | 4096 |
| Tokens per Step | 4,194,304 (~4.2M) |
| Total Training Time | ~12 hours |

## Training Phases

### Phase 1: Initial Training (1 GPU)
| Parameter | Value |
|-----------|-------|
| Hardware | 1x NVIDIA H200 (140GB VRAM) |
| Steps | 0 → 1000 |
| Duration | ~10 hours |
| Throughput | ~110,000 tok/sec |
| Log File | `train_300M_phase1_1gpu.log` |

### Phase 2: Continued Training (8 GPUs)
| Parameter | Value |
|-----------|-------|
| Hardware | 8x NVIDIA H200 (140GB VRAM each) |
| Steps | 1000 → 2500 |
| Duration | ~2 hours |
| Throughput | ~870,000 tok/sec |
| Log File | `train_300M_phase2_8gpu.log` |

## Final Results

| Metric | Initial (Step 1) | Step 1000 | Final (Step 2500) |
|--------|------------------|-----------|-------------------|
| Loss | 11.01 | 4.98 | 3.80 |
| Perplexity | 60,487 | 145 | 44.6 |

## Training Commands

### Phase 1: Single GPU Training
```bash
cd /home/vec_norm/OLMo && source /home/vec_norm/.venv/bin/activate

nohup torchrun --nproc_per_node=1 scripts/train.py configs/tiny/OLMo-300M.yaml \
  --run_name=OLMo-300M-2B \
  --max_duration=477 \
  --save_folder=/home/vec_norm/OLMo/checkpoints/OLMo-300M-2B \
  --remote_save_folder=null \
  --save_interval_unsharded=100 \
  --save_num_unsharded_checkpoints_to_keep=5 \
  --eval_interval=100 \
  --global_train_batch_size=1024 \
  --device_train_microbatch_size=16 \
  --scheduler.t_warmup=5000 \
  --wandb=null \
  --evaluators='[]' \
  --save_overwrite \
  --data.paths='[/home/vec_norm/OLMo/data/dolma/part-001-00000.npy]' \
  > train_300M.log 2>&1 &
```

### Phase 2: Multi-GPU Training (Resume)
```bash
cd /home/vec_norm/OLMo && source /home/vec_norm/.venv/bin/activate

LATEST_CKPT=$(ls -td /home/vec_norm/OLMo/checkpoints/OLMo-300M-2B/step*-unsharded 2>/dev/null | head -1)

nohup torchrun --nproc_per_node=8 scripts/train.py configs/tiny/OLMo-300M.yaml \
  --run_name=OLMo-300M-2B \
  --max_duration=2000 \
  --save_folder=/home/vec_norm/OLMo/checkpoints/OLMo-300M-2B \
  --remote_save_folder=null \
  --save_interval_unsharded=100 \
  --save_num_unsharded_checkpoints_to_keep=5 \
  --eval_interval=100 \
  --global_train_batch_size=1024 \
  --device_train_microbatch_size=16 \
  --scheduler.t_warmup=5000 \
  --wandb=null \
  --evaluators='[]' \
  --save_overwrite \
  --load_path="$LATEST_CKPT" \
  --data.paths='[/home/vec_norm/OLMo/data/dolma/part-001-00000.npy]' \
  > train_300M_8gpu.log 2>&1 &
```

## Key Configuration Details

- **Base Config**: `configs/tiny/OLMo-300M.yaml`
- **Gradient Accumulation**: 64 steps (1 GPU) / 8 steps (8 GPUs)
- **Warmup**: 5000 steps
- **Precision**: amp_bf16
- **Optimizer**: AdamW (lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95))
- **Scheduler**: cosine_with_warmup
- **Distributed Strategy**: DDP

## Checkpoint & Log Locations

```
/home/vec_norm/OLMo/checkpoints/OLMo-300M-2B/
├── config.yaml
├── train_300M_phase1_1gpu.log          # Steps 0-1000 (1 GPU)
├── train_300M_phase2_8gpu.log          # Steps 1000-2500 (8 GPUs)
├── train_300M_phase2_8gpu_failed.log   # Failed attempt (missing --save_overwrite)
├── step2100-unsharded/
├── step2200-unsharded/
├── step2300-unsharded/
├── step2400-unsharded/
├── step2500-unsharded/                 # Latest checkpoint
└── latest-unsharded -> step2500-unsharded
```

## Model Architecture

```
OLMo-300M (371,491,840 parameters)
├── d_model: 1024
├── n_heads: 16
├── n_layers: 16
├── mlp_ratio: 8
├── max_sequence_length: 4096
├── vocab_size: 50280
├── activation: swiglu
├── layer_norm: rms
├── position_encoding: rope
└── weight_tying: false
```

## Data Source

- **File**: `/home/vec_norm/OLMo/data/dolma/part-001-00000.npy`
- **Size**: ~24GB (~12B tokens)
- **Format**: uint16 memmap
- **Tokenizer**: `tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json`

## Training Progress Log

| Step | Loss | Perplexity | Notes |
|------|------|------------|-------|
| 1 | 11.01 | 60,487 | Initial (random weights) |
| 16 | 10.76 | 46,871 | Early improvement |
| 1000 | 4.98 | 145 | End of Phase 1 |
| 1100 | 4.85 | 128 | Phase 2 begins (8 GPUs) |
| 1500 | 4.50 | 90 | Continued improvement |
| 1780 | 4.16 | 64 | |
| 1900 | 4.09 | 59.5 | |
| 2500 | 3.80 | 44.6 | Final |

## Speed Benchmark Results

| Configuration | Throughput | Time per 1B tokens |
|---------------|------------|-------------------|
| 1x H200 | ~110,000 tok/sec | ~2.5 hours |
| 8x H200 | ~870,000 tok/sec | ~19 minutes |

## Usage

### Load the trained model

```python
from olmo import OLMo, Tokenizer

model = OLMo.from_checkpoint("/home/vec_norm/OLMo/checkpoints/OLMo-300M-2B/step2500-unsharded")
tokenizer = Tokenizer.from_checkpoint("/home/vec_norm/OLMo/checkpoints/OLMo-300M-2B/step2500-unsharded")

# Generate text
inputs = tokenizer.encode("The meaning of life is")
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs))
```

### Convert to HuggingFace format

```bash
python scripts/convert_olmo_to_hf.py \
  --checkpoint /home/vec_norm/OLMo/checkpoints/OLMo-300M-2B/step2500-unsharded \
  --output /home/vec_norm/OLMo/hf-models/OLMo-300M-2B
```

## Troubleshooting

### S3 Credentials Error
If you see `NoCredentialsError`, make sure `--remote_save_folder=null` is set.

### OOM Error
Reduce `--device_train_microbatch_size` from 16 to 8.

### Data Directory Exists Error
Add `--save_overwrite` flag when resuming training.

### RNG State Warning (when changing GPU count)
This warning is expected when resuming from an unsharded checkpoint with a different world size. It can be safely ignored.

### Monitoring Commands
```bash
# Watch live progress
tail -f /path/to/logfile.log

# Check if still running
ps aux | grep train.py

# Check GPU usage
nvidia-smi

# Kill training
pkill -f "train.py"
```

---

*Last updated: December 10, 2025*
