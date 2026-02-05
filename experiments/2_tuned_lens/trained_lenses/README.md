# Trained Tuned Lenses

Pre-trained tuned lens probes for reproducing Figures 2, 6, and 7. Each lens consists of one affine translator per layer, trained to minimize KL divergence between the translator's output distribution and the final layer's output distribution.

## Contents

### Figure 2 — OLMo-1B Tied vs Untied

| Model | HuggingFace ID | Weight Tying | Layers | d_model |
|-------|---------------|--------------|--------|---------|
| OLMo-1B | `allenai/OLMo-1B-hf` | **Tied** | 16 | 2048 |
| OLMo-1B-0724 | `allenai/OLMo-1B-0724-hf` | Untied | 16 | 2048 |

### Figure 6 — Pythia-2.8B vs GPT-Neo-2.7B

| Model | HuggingFace ID | Weight Tying | Layers | d_model |
|-------|---------------|--------------|--------|---------|
| GPT-Neo-2.7B | `EleutherAI/gpt-neo-2.7B` | **Tied** | 32 | 2560 |
| Pythia-2.8B | `EleutherAI/pythia-2.8b` | Untied | 32 | 2560 |

### Figure 7 — Qwen3 Family

| Model | HuggingFace ID | Weight Tying | Layers |
|-------|---------------|--------------|--------|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | **Tied** | 28 |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | **Tied** | 28 |
| Qwen3-4B | `Qwen/Qwen3-4B` | **Tied** | 36 |
| Qwen3-8B | `Qwen/Qwen3-8B` | Untied | 36 |

## Directory Structure

```
trained_lenses/
├── allenai/
│   ├── OLMo-1B-hf/
│   │   ├── config.json
│   │   └── params.pt
│   └── OLMo-1B-0724-hf/
│       ├── config.json
│       └── params.pt
├── EleutherAI/
│   ├── gpt-neo-2.7B/
│   │   ├── config.json
│   │   └── params.pt
│   └── pythia-2.8b/
│       ├── config.json
│       └── params.pt
└── Qwen/
    ├── Qwen3-0.6B/
    ├── Qwen3-1.7B/
    ├── Qwen3-4B/
    └── Qwen3-8B/
    (each with config.json + params.pt)
```

## File Format

- **`config.json`** — Metadata: model name, hidden size, number of layers, unembedding hash
- **`params.pt`** — PyTorch state dict with trained translator weights (one `Linear(d_model, d_model)` per layer)

## Retraining

See `train_olmo_lens.sh`, `train_pythia_gptneo_lenses.sh`, and `train_qwen3_lenses.sh` in the parent directory for the exact training commands used. All lenses were trained with the [`tuned-lens`](https://github.com/AlignmentResearch/tuned-lens) package on WikiText-103.
