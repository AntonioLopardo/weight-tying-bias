#!/bin/bash
# Train tuned lenses for OLMo-1B models (tied and untied)
#
# Models:
#   - allenai/OLMo-1B-hf      (16 layers, tied embeddings)   → Figure 2
#   - allenai/OLMo-1B-0724-hf (16 layers, untied embeddings) → Figure 2
#
# The training uses KL divergence loss to match the final layer's predictions
# at each intermediate layer.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Training parameters
NUM_STEPS=500
TOKENS_PER_STEP=131072

# Memory allocation fix
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "Training tuned lens for: allenai/OLMo-1B-hf"
echo "========================================"

python -m tuned_lens train \
    --model.name allenai/OLMo-1B-hf \
    --data.name wikitext wikitext-103-raw-v1 \
    --data.split validation \
    --data.text_column text \
    --per_gpu_batch_size 2 \
    --num_steps "$NUM_STEPS" \
    --tokens_per_step "$TOKENS_PER_STEP" \
    --output "$SCRIPT_DIR/trained_lenses/allenai/OLMo-1B-hf"

echo "Completed training for OLMo-1B-hf"
echo ""

echo "========================================"
echo "Training tuned lens for: allenai/OLMo-1B-0724-hf"
echo "========================================"

python -m tuned_lens train \
    --model.name allenai/OLMo-1B-0724-hf \
    --data.name wikitext wikitext-103-raw-v1 \
    --data.split validation \
    --data.text_column text \
    --per_gpu_batch_size 2 \
    --num_steps "$NUM_STEPS" \
    --tokens_per_step "$TOKENS_PER_STEP" \
    --output "$SCRIPT_DIR/trained_lenses/allenai/OLMo-1B-0724-hf"

echo "Completed training for OLMo-1B-0724-hf"
echo ""

echo "========================================"
echo "All OLMo tuned lenses trained!"
echo "========================================"

