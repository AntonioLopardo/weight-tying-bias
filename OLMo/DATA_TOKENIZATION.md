# Dolma v1.7 Dataset Tokenization Guide

This document describes how to download and tokenize the [emozilla/dolma-v1_7-30B](https://huggingface.co/datasets/emozilla/dolma-v1_7-30B) dataset from HuggingFace for continued OLMo training.

## Dataset Overview

| Property | Value |
|----------|-------|
| Source | [emozilla/dolma-v1_7-30B](https://huggingface.co/datasets/emozilla/dolma-v1_7-30B) |
| Description | 1% sample of Dolma v1.7 |
| Examples | 34,469,834 |
| Tokens | ~30B |
| License | ODC-BY |
| Format | Parquet (32 files, ~60GB) |

## Tokenizer

OLMo uses the GPT-NeoX tokenizer with PII special tokens:

```
/home/vec_norm/OLMo/olmo_data/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json
```

| Property | Value |
|----------|-------|
| Vocab Size | 50,280 |
| EOS Token ID | 50,279 |
| Output Format | uint16 numpy memmap |

## Tokenization Scripts

Three scripts were developed with increasing performance:

### 1. Basic Script (`tokenize_dolma_hf.py`)
- Location: `/home/vec_norm/scripts/tokenize_dolma_hf.py`
- Method: Non-batched, loads full dataset
- Speed: ~3,500 examples/s (4 workers)
- Time: ~2.7 hours

### 2. Batched Script (`tokenize_dolma_fast.py`)
- Location: `/home/vec_norm/scripts/tokenize_dolma_fast.py`
- Method: Batched tokenization with `dataset.map()`
- Speed: ~6,000-10,000 examples/s (16 workers)
- Time: ~1-1.5 hours

### 3. Streaming Script (`tokenize_dolma_stream.py`) ⭐ Recommended
- Location: `/home/vec_norm/scripts/tokenize_dolma_stream.py`
- Method: Streaming + parallel workers + direct disk write
- Speed: ~9,000+ examples/s (16 workers)
- Time: ~63 minutes

## Usage

### Quick Start (Streaming - Recommended)

```bash
cd /home/vec_norm
source .venv/bin/activate

python scripts/tokenize_dolma_stream.py \
    --output /home/vec_norm/OLMo/data/dolma_v1_7/dolma_v1_7_30B.npy \
    --num-workers 16 \
    --batch-size 1000
```

### Batched Approach (Better with more cores)

```bash
python scripts/tokenize_dolma_fast.py \
    --output /home/vec_norm/OLMo/data/dolma_v1_7/dolma_v1_7_30B.npy \
    --num-proc 16 \
    --batch-size 5000
```

## Performance Benchmarks

Tested on AMD EPYC 9655 (16 vCPUs allocated):

| Approach | Workers | Speed | Est. Time | Memory |
|----------|---------|-------|-----------|--------|
| Basic (non-batched) | 4 | 3,500/s | 2.7 hrs | High |
| Batched (`dataset.map`) | 16 | 4,300-10,000/s | 1-1.5 hrs | High |
| **Streaming** | 16 | 9,000+/s | ~63 min | Low |

### Scaling with More Cores

| Cores | Est. Speed | Est. Time |
|-------|------------|-----------|
| 16 | ~9,000/s | ~63 min |
| 32 | ~15,000-18,000/s | ~35 min |
| 64 | ~25,000-30,000/s | ~20 min |
| 96 | ~30,000-35,000/s | ~17 min |

**Note:** The streaming approach is bottlenecked by sequential iteration. With more cores, the batched approach (`tokenize_dolma_fast.py`) scales better since it can shard data across workers.

## Output Configuration

After tokenization, update your OLMo config to use the new data:

```yaml
data:
  paths:
    - /home/vec_norm/OLMo/data/dolma_v1_7/dolma_v1_7_30B.npy
  memmap_dtype: uint16
```

## Continuing Training

To continue training from the existing checkpoint with the new data:

```bash
cd /home/vec_norm/OLMo
source /home/vec_norm/.venv/bin/activate

torchrun --nproc_per_node=1 scripts/train.py configs/OLMo-1B-reproduction.yaml \
    --load_path=/home/vec_norm/OLMo/checkpoints/OLMo-1B-0724-reproduce/step2000-unsharded \
    --data.paths='[/home/vec_norm/OLMo/data/dolma_v1_7/dolma_v1_7_30B.npy]' \
    --restore_dataloader=false \
    --reset_trainer_state=true
```

Key flags:
- `--load_path`: Path to checkpoint to resume from
- `--restore_dataloader=false`: Don't try to restore dataloader state (new data)
- `--reset_trainer_state=true`: Reset step counter, scheduler, etc.

## File Sizes

| File | Size |
|------|------|
| Raw Parquet (HuggingFace) | ~60 GB |
| Tokenized NPY (~30B tokens) | ~56 GB (uint16) |

## Troubleshooting

### Out of Memory
- Use the streaming script instead of batched
- Reduce `--batch-size`
- Reduce `--num-workers`

### Slow Tokenization
- Increase `--num-workers` (up to CPU count)
- Increase `--batch-size` for batched approach
- Use batched approach if you have many cores (32+)

### Disk Space
- Ensure ~70GB free for the output file
- The streaming script pre-allocates 65GB then truncates

