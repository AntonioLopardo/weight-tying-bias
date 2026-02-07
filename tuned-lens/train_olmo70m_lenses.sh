#!/bin/bash
# Train tuned lenses for OLMo-70M tied and untied models
#
# Models:
#   - avyxh/olmo-70m-tied-5B (tied embeddings)
#   - avyxh/olmo-70m-untied-5B (untied embeddings)

# Activate the virtual environment
source /home/vec_norm/.venv/bin/activate
cd /home/vec_norm/tuned-lens

echo "=========================================="
echo "Training tuned lens for olmo-70m-tied-5B"
echo "=========================================="

python -m tuned_lens train \
    --model.name avyxh/olmo-70m-tied-5B \
    --data.name wikitext wikitext-103-raw-v1 \
    --split train \
    --text_column text \
    --per_gpu_batch_size 8 \
    --num_steps 100 \
    --tokens_per_step 32768 \
    --output /home/vec_norm/tuned-lens/trained_lenses/avyxh/olmo-70m-tied-5B

echo ""
echo "=========================================="
echo "Training tuned lens for olmo-70m-untied-5B"
echo "=========================================="

python -m tuned_lens train \
    --model.name avyxh/olmo-70m-untied-5B \
    --data.name wikitext wikitext-103-raw-v1 \
    --split train \
    --text_column text \
    --per_gpu_batch_size 8 \
    --num_steps 100 \
    --tokens_per_step 32768 \
    --output /home/vec_norm/tuned-lens/trained_lenses/avyxh/olmo-70m-untied-5B

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo "Lenses saved to:"
echo "  - /home/vec_norm/tuned-lens/trained_lenses/avyxh/olmo-70m-tied-5B"
echo "  - /home/vec_norm/tuned-lens/trained_lenses/avyxh/olmo-70m-untied-5B"

