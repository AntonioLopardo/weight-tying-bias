#!/bin/bash
# Train tuned lenses for Pythia-2.8B and GPT-Neo-2.7B
#
# Models to train:
#   - EleutherAI/pythia-2.8b (32 layers, untied embeddings)
#   - EleutherAI/gpt-neo-2.7B (32 layers, tied embeddings)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Training parameters
NUM_STEPS=100
TOKENS_PER_STEP=65536

# Memory allocation fix
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "Training tuned lens for: EleutherAI/gpt-neo-2.7B"
echo "========================================"

python -m tuned_lens train \
    --model.name "EleutherAI/gpt-neo-2.7B" \
    --data.name wikitext wikitext-103-raw-v1 \
    --split train \
    --text_column text \
    --per_gpu_batch_size 4 \
    --num_steps "$NUM_STEPS" \
    --tokens_per_step "$TOKENS_PER_STEP" \
    --output "$SCRIPT_DIR/trained_lenses/EleutherAI/gpt-neo-2.7B"

echo "Completed training for GPT-Neo-2.7B"
echo ""

echo "========================================"
echo "Training tuned lens for: EleutherAI/pythia-2.8b"
echo "========================================"

python -m tuned_lens train \
    --model.name "EleutherAI/pythia-2.8b" \
    --data.name wikitext wikitext-103-raw-v1 \
    --split train \
    --text_column text \
    --per_gpu_batch_size 4 \
    --num_steps "$NUM_STEPS" \
    --tokens_per_step "$TOKENS_PER_STEP" \
    --output "$SCRIPT_DIR/trained_lenses/EleutherAI/pythia-2.8b"

echo "Completed training for Pythia-2.8B"
echo ""

echo "========================================"
echo "All tuned lenses trained!"
echo "========================================"

