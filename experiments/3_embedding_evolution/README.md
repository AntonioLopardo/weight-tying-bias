# Experiment 3: Embedding Evolution Dynamics (Section 5.1, Appendix D)

**Paper Outputs:** Figure 3, Figure 8 (Appendix D), Figure 9 (Appendix D)

## Overview

Tracks how input and output embedding matrices evolve during training in untied models. Uses two complementary metrics:
1. **Cumulative drift**: Mean cosine similarity between each checkpoint's embeddings and the initial (step 0) embeddings
2. **Per-step change rate**: Mean cosine similarity between each token's embedding at consecutive checkpoints

## Key Finding

> Output embedding matrices undergo substantially larger updates than input embedding matrices, especially early in training.

- OLMo-1B-0724 (untied): By step 20K, output similarity to init ≈ 0.56, input ≈ 0.87
- Pythia-1B: Output drops to 0.79 similarity in first checkpoint, input stays at 0.92

## Scripts

| Script | Description |
|--------|-------------|
| `track_evolution.py` | Loads HuggingFace checkpoints, computes cosine similarity metrics, saves JSON results and generates plots |

## Reproducing Figures 3, 8, 9

### Prerequisites

1. **Python environment** with `torch`, `transformers`, `matplotlib`, `tqdm`:

```bash
source /home/vec_norm/.venv/bin/activate
```

2. **HuggingFace access** — model checkpoints are downloaded automatically from the Hub

### Generate Figures

```bash
# Figure 3: OLMo-1B-0724 (untied) — 11 checkpoints, steps 0–20K at 2K intervals
python track_evolution.py --config configs/evolution_olmo_1b.json

# Figure 8: OLMo-7B-0424 (untied) — 6 checkpoints, steps 0–50K at 10K intervals
python track_evolution.py --config configs/evolution_olmo_7b.json

# Figure 9: Pythia-1B (untied) — 16 checkpoints, steps 0–15K at 1K intervals
python track_evolution.py --config configs/evolution_pythia_1b.json

# Custom output directory
python track_evolution.py --config configs/evolution_olmo_1b.json --output-dir ./results
```

### Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | Required | Path to config JSON with checkpoint revisions |
| `--output-dir` | `.` (current directory) | Directory to save results and figures |

### Configuration

Configs specify HuggingFace model checkpoints with verified revision names:

```json
{
  "olmo-1b-0724-step0": {
    "class": "huggingface",
    "path": "allenai/OLMo-1B-0724-hf",
    "revision": "step0-tokens0B"
  },
  "olmo-1b-0724-step2000": {
    "class": "huggingface",
    "path": "allenai/OLMo-1B-0724-hf",
    "revision": "step2000-tokens4B"
  }
}
```

---

## Models Used

All models are loaded directly from HuggingFace Hub — no local checkpoints required.

| Config | Model | HuggingFace ID | Weight Tying | Checkpoints | Paper Figure |
|--------|-------|----------------|--------------|-------------|--------------|
| `evolution_olmo_1b.json` | OLMo-1B-0724 | `allenai/OLMo-1B-0724-hf` | Untied | 11 (steps 0–20K, 2K intervals) | Figure 3 |
| `evolution_olmo_7b.json` | OLMo-7B-0424 | `allenai/OLMo-7B-0424-hf` | Untied | 6 (steps 0–50K, 10K intervals) | Figure 8 |
| `evolution_pythia_1b.json` | Pythia-1B | `EleutherAI/pythia-1b` | Untied | 16 (steps 0–15K, 1K intervals) | Figure 9 |

**Notes:**
- OLMo-7B uses a larger batch size (4M tokens/step vs 2M for 1B), so token counts per step differ
- All revision tags have been verified against the HuggingFace Hub API

## Output

The script generates:
1. **JSON results**: `evolution_{model_name}.json` with all computed metrics (steps, input/output vs init, input/output consecutive)
2. **PNG figure**: `figure_evolution_{model_name}.png` with two panels:
   - **Top**: Cosine similarity to initial embeddings (cumulative drift)
   - **Bottom**: Cosine similarity between consecutive checkpoints (change rate)

## Expected Results (Figure 3)

For OLMo-1B-0724 (untied) over first 20K steps:
- Output matrix drifts from 1.0 to ~0.56 similarity with initialization
- Input matrix drifts from 1.0 to ~0.87 similarity with initialization
- Per-step: Output shows ~0.96 consecutive similarity (large changes)
- Per-step: Input remains near 1.0 (minimal changes)
