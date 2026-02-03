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
| `compare_embeddings.py` | Single model analysis | Metrics in Table 1 |
| `compare_cross_model.py` | OLMo tied vs untied | Table 1, Figure 1 |
| `compare_pythia_gptneo.py` | Pythia-2.8B vs GPT-Neo-2.7B | Table 1 |
| `compare_qwen.py` | Qwen3-4B vs Qwen3-8B | (different dims, see KNN) |
| `nn_k1.py` | KNN@10 overlap analysis | Table 5 |

## Usage

```bash
# Table 1: OLMo alignment
python compare_embeddings.py --hf-model allenai/OLMo-1B-hf        # Tied
python compare_embeddings.py --hf-model allenai/OLMo-1B-0724-hf   # Untied
python compare_cross_model.py                                      # Cross-model

# Table 1: GPT-Neo/Pythia alignment
python compare_embeddings.py --hf-model EleutherAI/gpt-neo-2.7B   # Tied
python compare_embeddings.py --hf-model EleutherAI/pythia-2.8b    # Untied
python compare_pythia_gptneo.py                                    # Cross-model

# Table 5: KNN@10 overlap
python nn_k1.py
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
