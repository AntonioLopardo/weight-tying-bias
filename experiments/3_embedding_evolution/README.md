# Experiment 3: Embedding Evolution Dynamics (Section 5.1)

**Paper Outputs:** Figure 3, Figure 8 (Appendix D), Figure 9 (Appendix D)

## Overview

Tracks how input and output embedding matrices evolve during training in untied models. Uses two metrics:
1. **Cumulative drift**: Cosine similarity to initial (step 0) embeddings
2. **Per-step change rate**: Cosine similarity between consecutive checkpoints

## Key Finding

> Output embedding matrices undergo substantially larger updates than input embedding matrices, especially early in training.

- OLMo-1B-0724: By step 20K, output similarity to init = 0.56, input = 0.87
- Pythia-1B: Output drops to 0.79 similarity in first checkpoint, input stays at 0.92

## Usage

```bash
# Figure 3: OLMo-1B-0724
python track_evolution.py --config configs/evolution_olmo_1b.json

# Figure 8: OLMo-7B-0724
python track_evolution.py --config configs/evolution_olmo_7b.json

# Figure 9: Pythia-1B  
python track_evolution.py --config configs/evolution_pythia_1b.json

# Custom output directory
python track_evolution.py --config configs/evolution_olmo_1b.json --output-dir ./results
```

## Config Format

Configs specify HuggingFace model checkpoints with verified revision names:

```json
{
  "olmo-1b-0724-step0": {
    "class": "huggingface",
    "path": "allenai/OLMo-1B-0724-hf",
    "revision": "step0-tokens0B"
  },
  "olmo-1b-0724-step1000": {
    "class": "huggingface", 
    "path": "allenai/OLMo-1B-0724-hf",
    "revision": "step1000-tokens2B"
  }
}
```

## Available Configs

| Config | Model | Steps | Paper Figure |
|--------|-------|-------|--------------|
| `configs/evolution_olmo_1b.json` | OLMo-1B-0724 (untied) | 0-100K | Figure 3 |
| `configs/evolution_pythia_1b.json` | Pythia-1B (untied) | 0-14K | Figure 9 |
| `configs/evolution_olmo_7b.json` | OLMo-7B-0724 (untied) | 0-50K | Figure 8 |

**Note:** OLMo-7B has a larger batch size (4M tokens/step vs 2M), so token counts differ from 1B.

## Output

The script generates:
1. **JSON results file**: `evolution_{model_name}.json` with all metrics
2. **PNG figure**: `figure_evolution_{model_name}.png` matching paper Figures 3/8/9

## Expected Results (Figure 3)

For OLMo-1B-0724 (untied) over first 20K steps:
- Output matrix drifts from 1.0 to ~0.56 similarity with initialization
- Input matrix drifts from 1.0 to ~0.87 similarity with initialization
- Per-step: Output shows ~0.96 consecutive similarity (large changes)
- Per-step: Input remains near 1.0 (minimal changes)
