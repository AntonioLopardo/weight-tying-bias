# Experiment 2: Tuned Lens Analysis (Section 4.2)

**Paper Outputs:** Figure 2, Figure 6 (Appendix C), Figure 7 (Appendix C)

## Overview

Uses the tuned lens (Belrose et al., 2023) to measure how well each layer's representations align with the output space. Higher KL divergence indicates less effective contribution to the residual stream.

## Key Finding

> Tied models exhibit systematically higher KL divergence compared to untied counterparts, especially in early and middle layers.

- OLMo-1B: Tied starts at ~9.2 bits vs ~7.0 bits for untied (2+ bit gap)
- GPT-Neo vs Pythia: 17 bits vs 7 bits (10 bit gap)
- Qwen3-4B vs Qwen3-8B: 25 bits vs 12 bits (13 bit gap)

## Scripts

| Script | Description | Paper Output |
|--------|-------------|--------------|
| `compare_olmo_tuned_lenses.py` | OLMo-1B tied vs untied | Figure 2 |
| `compare_pythia_vs_gptneo.py` | Pythia-2.8B vs GPT-Neo-2.7B | Figure 6 |
| `compare_qwen3_tuned_lenses.py` | Qwen3 family comparison | Figure 7 |
| [`Appendix_C/reproduce_figure6.py`](Appendix_C/) | Standalone Figure 6 reproduction | Figure 6 |
| [`Appendix_C/reproduce_figure7.py`](Appendix_C/) | Standalone Figure 7 reproduction | Figure 7 |
| `reproduce_figure3.py` | Reproduce Fig 3 from Belrose et al. | — |
| `compare_olmo70m_tied_untied.py` | OLMo-70M tied vs untied | — |
| `compare_pythia_tuned_lenses.py` | Pythia model family comparison | — |

## Reproducing Figures 2, 6, 7

### Prerequisites

1. **Pre-trained tuned lenses** in `trained_lenses/` (included):
   - `allenai/OLMo-1B-hf` (tied) and `allenai/OLMo-1B-0724-hf` (untied) → Figure 2
   - `EleutherAI/gpt-neo-2.7B` (tied) and `EleutherAI/pythia-2.8b` (untied) → Figure 6
   - `Qwen/Qwen3-4B` (tied) and `Qwen/Qwen3-8B` (untied) → Figure 7

2. **Python packages**: `torch`, `transformers`, `tuned-lens`, `matplotlib`, `numpy`, `tqdm`

### Generate Figure 2 (OLMo)

```bash
python compare_olmo_tuned_lenses.py
# Output: compare_olmo_tuned_lenses.png
```

### Generate Figure 6 (Pythia vs GPT-Neo)

```bash
python compare_pythia_vs_gptneo.py
# Output: compare_pythia_vs_gptneo.png
```

### Generate Figure 7 (Qwen3)

```bash
python compare_qwen3_tuned_lenses.py
# Output: compare_qwen3_36layers.png
```

### Models Used

| Model | HuggingFace ID | Weight Tying | Layers | Figure |
|-------|---------------|--------------|--------|--------|
| OLMo-1B | `allenai/OLMo-1B-hf` | **Tied** | 16 | Fig 2 |
| OLMo-1B-0724 | `allenai/OLMo-1B-0724-hf` | Untied | 16 | Fig 2 |
| GPT-Neo-2.7B | `EleutherAI/gpt-neo-2.7B` | **Tied** | 32 | Fig 6 |
| Pythia-2.8B | `EleutherAI/pythia-2.8b` | Untied | 32 | Fig 6 |
| Qwen3-4B | `Qwen/Qwen3-4B` | **Tied** | 36 | Fig 7 |
| Qwen3-8B | `Qwen/Qwen3-8B` | Untied | 36 | Fig 7 |

All models are loaded from HuggingFace at runtime. The tuned lens probes are stored locally in `trained_lenses/`.

---

## Tuned Lens Training Reproduction

The pre-trained lenses included in `trained_lenses/` were trained using the [`tuned-lens`](https://github.com/AlignmentResearch/tuned-lens) package. Each lens is an affine translator per layer, trained to minimize KL divergence between its predictions and the final layer output.

### Setup

```bash
pip install tuned-lens
```

### Training Scripts

| Script | Models | Steps |
|--------|--------|-------|
| `train_olmo_lens.sh` | OLMo-1B-hf, OLMo-1B-0724-hf | 500 |
| `train_pythia_gptneo_lenses.sh` | Pythia-2.8B, GPT-Neo-2.7B | 100 |
| `train_qwen3_lenses.sh` | Qwen3-0.6B, 1.7B, 4B, 8B | 50 |

### Train Lenses

```bash
# Train OLMo lenses (Figure 2)
bash train_olmo_lens.sh

# Train Pythia & GPT-Neo lenses (Figure 6)
bash train_pythia_gptneo_lenses.sh

# Train Qwen3 lenses (Figure 7)
bash train_qwen3_lenses.sh
```

### Training Configuration

| Setting | Value |
|---------|-------|
| Loss | KL divergence |
| Dataset | WikiText-103 |
| Lens type | Linear (affine translator per layer) |
| Initialization | Identity (zero weights + zero bias) |

Trained lenses are saved as `config.json` + `params.pt` per model in `trained_lenses/`.
