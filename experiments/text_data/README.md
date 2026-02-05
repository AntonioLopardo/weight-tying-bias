# Training Data

Shared training data used across experiments. Multiple experiment configs reference this directory so the data is stored once rather than duplicated.

## Dolma v1.7

`dolma_v1_7/dolma_v1_7_30B.npy` — A 30B-token subset of the [Dolma v1.7](https://huggingface.co/datasets/allenai/dolma) dataset, pre-tokenized with the GPT-NeoX tokenizer and stored as a memory-mapped NumPy array (`uint16`).

Used by:
- **Experiment 4** (Norm-Frequency): training tied and untied OLMo-1B for 10k steps
- **Experiment 5** (Gradient Flow): training tied OLMo-1B with gradient provenance tracking for 1k steps

### Preparing the data

To recreate this file, tokenize Dolma v1.7 into a flat NumPy memmap:

```bash
# See the OLMo repository for tokenization scripts:
# https://github.com/allenai/OLMo
python scripts/prepare_memmap.py \
    --input <dolma_v1_7_shards> \
    --output text_data/dolma_v1_7/dolma_v1_7_30B.npy \
    --tokenizer allenai/eleuther-ai-gpt-neox-20b-pii-special \
    --dtype uint16
```
