#!/bin/bash
# Download model checkpoints and artifacts from HuggingFace Hub
# Usage:
#   ./download_artifacts.sh              # Download everything
#   ./download_artifacts.sh 4            # Download only experiment 4
#   ./download_artifacts.sh 2 4 6        # Download experiments 2, 4, and 6
#
# Requires: pip install huggingface_hub

set -e

HF_REPO="AntonioLopardo/weight-tying-bias-artifacts"
DEST_DIR="$(cd "$(dirname "$0")" && pwd)"

download_experiment() {
    local exp="$1"
    case "$exp" in
        2)
            echo "Downloading trained lenses (Experiment 2: Figures 2, 6, 7)..."
            huggingface-cli download "$HF_REPO" \
                --include "experiments/2_tuned_lens/trained_lenses/*" \
                --local-dir "$DEST_DIR"
            ;;
        4)
            echo "Downloading norm-frequency checkpoints (Experiment 4: Figure 5)..."
            huggingface-cli download "$HF_REPO" \
                --include "experiments/4_norm_frequency/OLMo-1B-tied/model.pt" \
                --include "experiments/4_norm_frequency/OLMo-1B-untied/model.pt" \
                --local-dir "$DEST_DIR"
            ;;
        5)
            echo "Downloading gradient flow checkpoint (Experiment 5: Figure 4)..."
            huggingface-cli download "$HF_REPO" \
                --include "experiments/5_gradient_flow/OLMo-1B-tied-grad-provenance/model.pt" \
                --local-dir "$DEST_DIR"
            ;;
        6)
            echo "Downloading gradient scaling checkpoints (Experiment 6: Tables 2, 6)..."
            huggingface-cli download "$HF_REPO" \
                --include "experiments/6_gradient_scaling/OLMo-1B-tied-no-scale-10000/model.pt" \
                --include "experiments/6_gradient_scaling/OLMo-1B-tied-emb5-10000/model.pt" \
                --include "experiments/6_gradient_scaling/OLMo-1B-untied-10000/model.pt" \
                --include "experiments/6_gradient_scaling/Appendix_E/*/model.pt" \
                --local-dir "$DEST_DIR"
            ;;
        *)
            echo "Unknown experiment: $exp"
            echo "Available: 2 (tuned lenses), 4 (norm-frequency), 5 (gradient flow), 6 (gradient scaling)"
            return 1
            ;;
    esac
}

# Check huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found. Install with: pip install huggingface_hub"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Downloading all artifacts from $HF_REPO..."
    huggingface-cli download "$HF_REPO" --local-dir "$DEST_DIR"
else
    for exp in "$@"; do
        download_experiment "$exp"
    done
fi

echo "Done."
