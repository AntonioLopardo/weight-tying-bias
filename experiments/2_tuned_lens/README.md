# Experiment 2: Tuned Lens Analysis (Section 4.2)

**Paper Outputs:** Figure 2, Figure 6 (Appendix C), Figure 7 (Appendix C)

## Overview

Uses the tuned lens (Belrose et al., 2023) to measure how well each layer's representations align with the output space. Higher KL divergence indicates less effective contribution to the residual stream.

## Key Finding

> Tied models exhibit systematically higher KL divergence compared to untied counterparts, especially in early and middle layers.

- OLMo-1B: Tied starts at ~9.2 bits vs ~7.0 bits for untied (2+ bit gap)
- GPT-Neo vs Pythia: 17 bits vs 7 bits (10 bit gap)
- Qwen3-4B vs Qwen3-8B: 25 bits vs 12 bits (13 bit gap)

## Pre-trained Lenses

Available in `trained_lenses/` for:
- `allenai/OLMo-1B-hf` (tied)
- `allenai/OLMo-1B-0724-hf` (untied)
- `EleutherAI/pythia-2.8b` (untied)
- `EleutherAI/gpt-neo-2.7B` (tied)
- `Qwen/Qwen3-4B` (tied)
- `Qwen/Qwen3-8B` (untied)

## Scripts

| Script | Description | Paper Output |
|--------|-------------|--------------|
| `compare_olmo_tuned_lenses.py` | OLMo-1B tied vs untied | Figure 2 |
| `compare_pythia_vs_gptneo.py` | Pythia-2.8B vs GPT-Neo-2.7B | Figure 6 |
| `compare_qwen3_tuned_lenses.py` | Qwen3-4B vs Qwen3-8B | Figure 7 |

## Usage

```bash
# Figure 2: OLMo comparison
python compare_olmo_tuned_lenses.py

# Figure 6: Pythia vs GPT-Neo
python compare_pythia_vs_gptneo.py

# Figure 7: Qwen3 comparison
python compare_qwen3_tuned_lenses.py
```

## Training New Lenses

```bash
# Using tuned-lens CLI
tuned-lens train --model allenai/OLMo-1B-hf --output trained_lenses/allenai/OLMo-1B-hf
```
