# Appendix D: Additional Embedding Evolution Results (Figures 8–9)

**Paper Outputs:** Figure 8, Figure 9

## Overview

Extends the embedding evolution analysis from Section 5.1 (Figure 3, OLMo-1B) to two additional models:

- **Figure 8**: OLMo-7B-0424 (untied) — 6 checkpoints, steps 0–50K at 10K intervals
- **Figure 9**: Pythia-1B (untied) — 16 checkpoints, steps 0–15K at 1K intervals

Uses the same `track_evolution.py` script and metrics (cumulative drift + per-step change rate) as the main Figure 3.

## Key Findings

> The output-biased update pattern holds across model scales and architectures.

- **OLMo-7B**: By step 50K, output similarity to init drops to ~0.38, input stays at ~0.75
- **Pythia-1B**: Output drops to ~0.79 similarity in first checkpoint; input stays at ~0.92

## Reproduce Figures

### Prerequisites

1. **Python environment** with `torch`, `transformers`, `matplotlib`, `tqdm`:

```bash
source /home/vec_norm/.venv/bin/activate
```

2. **HuggingFace access** — checkpoints downloaded automatically

### Generate Figure 8 (OLMo-7B)

```bash
python ../track_evolution.py --config ../configs/evolution_olmo_7b.json --output-dir .
# Output: figure_evolution_olmo-7b-0424.png
```

### Generate Figure 9 (Pythia-1B)

```bash
python ../track_evolution.py --config ../configs/evolution_pythia_1b.json --output-dir .
# Output: figure_evolution_pythia-1b.png
```

## Models Used

| Config | Model | HuggingFace ID | Checkpoints | Figure |
|--------|-------|----------------|-------------|--------|
| `../configs/evolution_olmo_7b.json` | OLMo-7B-0424 | `allenai/OLMo-7B-0424-hf` | 6 (steps 0–50K, 10K intervals) | Fig 8 |
| `../configs/evolution_pythia_1b.json` | Pythia-1B | `EleutherAI/pythia-1b` | 16 (steps 0–15K, 1K intervals) | Fig 9 |

All checkpoints are loaded directly from HuggingFace Hub — no local model files required.

## Output

Each run generates:
1. **JSON results**: `evolution_{model_name}.json` with computed metrics
2. **PNG figure**: Two-panel plot (cumulative drift + per-step change rate)

## Notes

- OLMo-7B uses a larger batch size (4M tokens/step vs 2M for 1B), so token counts per step differ
- Pythia-1B uses uniform 1K-step intervals throughout the range
- All HuggingFace revision tags have been verified against the Hub API
