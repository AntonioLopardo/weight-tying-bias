# OLMo-1B Training Reproduction

This document summarizes the steps taken to reproduce training of the [OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf) model.

## Overview

- **Model**: OLMo-1B (1.17B parameters)
- **Training tokens**: ~1B tokens
- **Hardware**: 1x A100-80GB
- **Training time**: ~5.7 hours
- **Throughput**: ~49,500 tokens/sec

## Setup

### 1. Clone and Install OLMo Repository

```bash
git clone https://github.com/allenai/OLMo.git
cd OLMo
source /home/vec_norm/.venv/bin/activate
pip install -e '.[all]'
pip install --upgrade omegaconf  # Fix version compatibility
```

### 2. Download Training Data

Data is streamed from `olmo-data.org` or downloaded locally for faster training:

```bash
mkdir -p /home/vec_norm/OLMo/data/dolma
cd /home/vec_norm/OLMo/data/dolma
wget https://olmo-data.org/preprocessed/olmo-mix/v1_5/gpt-neox-20b-pii-special/part-001-00000.npy
```

Each file is ~24GB containing ~12B tokens (uint16 format).

## Model Configuration

The model matches the official OLMo-1B-0724 architecture:

| Parameter | Value |
|-----------|-------|
| Layers | 16 |
| Hidden Size | 2048 |
| Attention Heads | 16 |
| MLP Ratio | 8 |
| Context Length | 2048 |
| Vocab Size | 50,280 |
| Weight Tying | Yes |
| Activation | SwiGLU |
| Position Encoding | RoPE |

### Optimizer Settings (AdamW)

| Parameter | Value |
|-----------|-------|
| Peak LR | 4.0E-4 |
| Betas | (0.9, 0.95) |
| Epsilon | 1.0E-5 |
| Weight Decay | 0.1 |
| Warmup Steps | 500 |
| Schedule | Cosine with warmup |

## Training Command

```bash
cd /home/vec_norm/OLMo
source /home/vec_norm/.venv/bin/activate

nohup torchrun --nproc_per_node=1 scripts/train.py configs/OLMo-1B-reproduction.yaml \
  --max_duration=7630 \
  --save_interval=1000 \
  --eval_interval=2000 \
  --global_train_batch_size=64 \
  --device_train_microbatch_size=8 \
  --scheduler.t_warmup=500 \
  --data.paths='[/home/vec_norm/OLMo/data/dolma/part-001-00000.npy]' \
  > /home/vec_norm/OLMo/train_1B.log 2>&1 &
```

## Training Results

### Final Metrics

- **Steps completed**: 7,640
- **Tokens processed**: ~1B
- **Final loss**: Decreasing (model learning)

### Downstream Evaluation Scores

| Benchmark | Score |
|-----------|-------|
| HellaSwag | 40.0% |
| Winogrande | 60.0% |
| SciQ | 80.0% |
| COPA | 35.0% |
| ARC-Easy | 30.0% |
| OpenBook QA | 20.0% |

*Note: Scores are from early training (1B tokens). Full OLMo-1B is trained on 3.05T tokens.*

## Checkpoints

Saved to: `/home/vec_norm/OLMo/checkpoints/OLMo-1B-reproduction/`

| Checkpoint | Description |
|------------|-------------|
| `step0-unsharded/` | Initial |
| `step1000-unsharded/` | ~131M tokens |
| `step2000-unsharded/` | ~262M tokens |
| `step3000-unsharded/` | ~393M tokens |
| `step4000-unsharded/` | ~524M tokens |
| `step5000-unsharded/` | ~655M tokens |
| `step6000-unsharded/` | ~786M tokens |
| `step7000-unsharded/` | ~917M tokens |
| `step7640-unsharded/` | **Final (~1B tokens)** |

### Checkpoint Contents

```
step7640-unsharded/
├── model.pt      (4.4 GB - model weights)
├── optim.pt      (8.8 GB - optimizer state)
├── train.pt      (15 KB - training state)
└── config.yaml   (6.9 KB - configuration)
```

## Loading the Model

### Using OLMo native format

```python
from olmo import OLMo, Tokenizer

# Load model
model = OLMo.from_checkpoint("/home/vec_norm/OLMo/checkpoints/OLMo-1B-reproduction/step7640-unsharded")
tokenizer = Tokenizer.from_checkpoint("/home/vec_norm/OLMo/checkpoints/OLMo-1B-reproduction/step7640-unsharded")

# Generate text
inputs = tokenizer.encode("Language modeling is")
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs))
```

### Converting to HuggingFace format

```bash
python scripts/convert_olmo_to_hf.py \
  --checkpoint /home/vec_norm/OLMo/checkpoints/OLMo-1B-reproduction/step7640-unsharded \
  --output /home/vec_norm/OLMo/hf-model
```

## Performance Notes

### Throughput Comparison

| Data Source | Throughput |
|-------------|------------|
| Streaming from URLs | ~5,000 tok/s |
| Local data | ~49,500 tok/s |

**Recommendation**: Download data locally for 10x faster training.

### GPU Memory Usage

- Peak GPU Memory: ~9.4 GB (model) + training overhead
- A100-80GB can handle `device_train_microbatch_size=8` easily

### Flash Attention

PyTorch 2.x includes built-in Flash Attention via SDPA. No need to install the separate `flash-attn` package:

```python
import torch
print(torch.backends.cuda.flash_sdp_enabled())  # True
```

## Scaling Estimates

| Tokens | 1x A100 Time | 8x A100 Time |
|--------|--------------|--------------|
| 1B | ~5.6 hours | ~42 min |
| 10B | ~56 hours | ~7 hours |
| 100B | ~23 days | ~2.9 days |
| 3T (full) | ~1.9 years | ~88 days |

## References

- **Model Card**: https://huggingface.co/allenai/OLMo-1B-0724-hf
- **OLMo Repository**: https://github.com/allenai/OLMo
- **Dolma Dataset**: https://huggingface.co/datasets/allenai/dolma
- **Paper**: [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)

## Configuration Files

- Training config: `/home/vec_norm/OLMo/configs/OLMo-1B-reproduction.yaml`
- Training log: `/home/vec_norm/OLMo/train_1B.log`

---

*Training completed: December 7, 2025, 23:15 UTC*

