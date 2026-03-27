# Appendix C: Additional Tuned Lens Results (Figures 6–7)

**Paper Outputs:** Figure 6, Figure 7

## Overview

Extends the tuned lens analysis from Section 4.2 (Figure 2, OLMo) to two additional model families:

- **Figure 6**: Pythia-2.8B (untied) vs GPT-Neo-2.7B (tied) — both 32 layers
- **Figure 7**: Qwen3-4B (tied) vs Qwen3-8B (untied) — both 36 layers

## Key Findings

> The tied-embedding bias pattern generalises across model families: tied models consistently show higher early-layer KL divergence.

- GPT-Neo-2.7B (tied) starts at ~17 bits vs ~7 bits for Pythia-2.8B (untied)
- Qwen3-4B (tied) starts at ~25 bits vs ~12 bits for Qwen3-8B (untied)

## Scripts

| Script | Description | Paper Output |
|--------|-------------|--------------|
| `reproduce_figure6.py` | Pythia-2.8B vs GPT-Neo-2.7B | Figure 6 |
| `reproduce_figure7.py` | Qwen3-4B vs Qwen3-8B | Figure 7 |

## Reproduce Figures

### Prerequisites

1. **Python environment** with `torch`, `transformers`, `tuned-lens`, `matplotlib` (see root README for setup)

2. **Pre-trained tuned lenses** in `../trained_lenses/` (included in parent experiment directory):
   - `EleutherAI/pythia-2.8b` and `EleutherAI/gpt-neo-2.7B` → Figure 6
   - `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B` → Figure 7

   If missing, train them first:
   ```bash
   cd ..
   bash train_pythia_gptneo_lenses.sh   # Figure 6
   bash train_qwen3_lenses.sh           # Figure 7
   ```

3. **GPU recommended** — models are loaded in float16 with `device_map="auto"`

### Generate Figure 6

```bash
python reproduce_figure6.py
# Output: figure6_pythia_vs_gptneo.png
```

### Generate Figure 7

```bash
python reproduce_figure7.py
# Output: figure7_qwen3_comparison.png
```

## Models Used

| Model | HuggingFace ID | Weight Tying | Layers | Figure |
|-------|---------------|--------------|--------|--------|
| Pythia-2.8B | `EleutherAI/pythia-2.8b` | Untied | 32 | Fig 6 |
| GPT-Neo-2.7B | `EleutherAI/gpt-neo-2.7B` | **Tied** | 32 | Fig 6 |
| Qwen3-4B | `Qwen/Qwen3-4B` | **Tied** | 36 | Fig 7 |
| Qwen3-8B | `Qwen/Qwen3-8B` | Untied | 36 | Fig 7 |

All models are loaded from HuggingFace at runtime.
