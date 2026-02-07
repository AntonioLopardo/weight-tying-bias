#!/bin/bash
# Train tuned lenses for Qwen3 model family
#
# Models to train:
#   - Qwen/Qwen3-0.6B  (28 layers, tied embeddings)
#   - Qwen/Qwen3-1.7B  (28 layers, tied embeddings)
#   - Qwen/Qwen3-4B    (36 layers, tied embeddings)
#   - Qwen/Qwen3-8B    (36 layers, untied embeddings)
#   - Qwen/Qwen3-14B   (40 layers, untied embeddings)
#
# The training uses KL divergence loss to match the final layer's predictions
# at each intermediate layer.

set -e

# Activate the virtual environment
source /home/vec_norm/.venv/bin/activate
cd /home/vec_norm/tuned-lens

# Training parameters
NUM_STEPS=50
TOKENS_PER_STEP=32768

# Define models to train
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
)

# Batch sizes per model
declare -A BATCH_SIZES
BATCH_SIZES["Qwen/Qwen3-0.6B"]=16
BATCH_SIZES["Qwen/Qwen3-1.7B"]=16
BATCH_SIZES["Qwen/Qwen3-4B"]=16
BATCH_SIZES["Qwen/Qwen3-8B"]=12

# Memory allocation fix
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train each model
for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Training tuned lens for: $MODEL"
    echo "========================================"
    
    BATCH_SIZE=${BATCH_SIZES[$MODEL]}
    OUTPUT_DIR="/home/vec_norm/tuned-lens/trained_lenses/$MODEL"
    
    # Skip if already trained
    if [ -f "$OUTPUT_DIR/params.pt" ]; then
        echo "Lens already exists at $OUTPUT_DIR, skipping..."
        continue
    fi
    
    python -m tuned_lens train \
        --model.name "$MODEL" \
        --data.name wikitext wikitext-103-raw-v1 \
        --split train \
        --text_column text \
        --per_gpu_batch_size "$BATCH_SIZE" \
        --num_steps "$NUM_STEPS" \
        --tokens_per_step "$TOKENS_PER_STEP" \
        --output "$OUTPUT_DIR"
    
    echo "Completed training for $MODEL"
    echo ""
done

echo "========================================"
echo "All Qwen3 tuned lenses trained!"
echo "========================================"

