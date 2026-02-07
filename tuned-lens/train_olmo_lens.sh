#!/bin/bash
# Train a tuned lens for allenai/OLMo-1B-0724-hf
#
# The training uses KL divergence loss to match the final layer's predictions
# at each intermediate layer.

# Activate the virtual environment
source /home/vec_norm/.venv/bin/activate
cd /home/vec_norm/tuned-lens

# Option 1: Using wikitext dataset (small, good for testing)
python -m tuned_lens train \
    --model.name allenai/OLMo-1B-0724-hf \
    --data.name wikitext wikitext-103-raw-v1 \
    --data.split validation \
    --data.text_column text \
    --per_gpu_batch_size 2 \
    --num_steps 500 \
    --tokens_per_step 131072 \
    --output /home/vec_norm/tuned-lens/trained_lenses/allenai/OLMo-1B-0724-hf

# Option 2: Using The Pile (recommended for best results, but requires downloading)
# First download: wget https://the-eye.eu/public/AI/pile/val.jsonl.zst && unzstd val.jsonl.zst
# python -m tuned_lens train \
#     --model.name allenai/OLMo-1B-0724-hf \
#     --data.name val.jsonl \
#     --per_gpu_batch_size 2 \
#     --num_steps 500 \
#     --tokens_per_step 131072 \
#     --output /home/vec_norm/tuned-lens/trained_lenses/allenai/OLMo-1B-0724-hf

