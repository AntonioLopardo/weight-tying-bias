# Experiment 1: Embedding Space Alignment (Section 4.1)

**Paper Outputs:** Table 1, Figure 1, Table 5 (Appendix B)

## Overview

Compares tied embedding matrices to untied input/output matrices using:
- **Identity**: No transformation (raw alignment)
- **Orthogonal**: Procrustes analysis (rotation-preserving)
- **Linear**: Unconstrained least squares

## Key Finding

> Tied embedding matrices align more closely with output (unembedding) matrices than with input embedding matrices.

- OLMo: Output→Tied = 0.719 vs Input→Tied = 0.525 (Linear)
- GPT-Neo/Pythia: Output→Tied = 0.637 vs Input→Tied = 0.507 (Linear)

## Scripts

| Script | Description | Paper Output |
|--------|-------------|--------------|
| `reproduce_figure1.py` | Token-level alignment histograms | **Figure 1** |
| `compare_embeddings.py` | Single model analysis | Metrics in Table 1 |
| `compare_cross_model.py` | OLMo tied vs untied | Table 1 |
| `compare_pythia_gptneo.py` | Pythia-2.8B vs GPT-Neo-2.7B | Table 1 |
| [`Appendix_B/reproduce_table5.py`](Appendix_B/) | KNN@10 overlap (all 3 families) | Table 5 |
| [`Appendix_B/reproduce_spectral_distance.py`](Appendix_B/) | Omnibus embedding spectral distance | Appendix B |
| [`Appendix_B/compare_qwen.py`](Appendix_B/) | Qwen3-4B vs Qwen3-8B (different dims) | Exploratory |
| [`Appendix_B/nn_k1.py`](Appendix_B/) | KNN@1 overlap (Qwen only) | Exploratory |

## Reproducing Figure 1

### Prerequisites

- **Python environment** with `torch`, `transformers`, `numpy`, `matplotlib`
- Models are downloaded automatically from HuggingFace:
  - `allenai/OLMo-1B-hf` (tied)
  - `allenai/OLMo-1B-0724-hf` (untied)

### Generate Figure

```bash
python reproduce_figure1.py
```

Output: `token_level_alignment.png`

### Expected Results

| Alignment | Mean Cosine Similarity |
|-----------|----------------------|
| Input → Tied | 0.525 |
| Output → Tied | 0.719 |

---

## Reproducing Table 1

```bash
# OLMo alignment
python compare_embeddings.py --hf-model allenai/OLMo-1B-hf        # Tied
python compare_embeddings.py --hf-model allenai/OLMo-1B-0724-hf   # Untied
python compare_cross_model.py                                      # Cross-model

# GPT-Neo/Pythia alignment
python compare_embeddings.py --hf-model EleutherAI/gpt-neo-2.7B   # Tied
python compare_embeddings.py --hf-model EleutherAI/pythia-2.8b    # Untied
python compare_pythia_gptneo.py                                    # Cross-model

# Table 5: KNN@10 overlap
python Appendix_B/reproduce_table5.py
```

## Expected Results (Table 1)

| Comparison | Identity | Orthogonal | Linear |
|------------|----------|------------|--------|
| **OLMo-1B (tied) vs OLMo-1B-0724 (untied)** |
| Input (U) → Output (U) | 0.012 | 0.440 | 0.574 |
| Output (U) → Tied | 0.014 | 0.669 | **0.719** |
| Input (U) → Tied | 0.001 | 0.420 | 0.525 |
| **GPT-Neo-2.7B (tied) vs Pythia-2.8B (untied)** |
| Input (U) → Output (U) | -0.001 | 0.456 | 0.518 |
| Output (U) → Tied | 0.001 | 0.507 | **0.637** |
| Input (U) → Tied | 0.000 | 0.376 | 0.507 |
