# Appendix B: KNN Overlap and Spectral Distance Analysis

**Paper Outputs:** Table 5, Spectral Distance

## Scripts

| Script | Description | Output |
|--------|-------------|--------|
| `reproduce_table5.py` | KNN@10 overlap (all 3 model families) | Table 5 |
| `reproduce_spectral_distance.py` | Omnibus embedding spectral distance (all 3 model families) | Appendix B |
| `compare_qwen.py` | Qwen3-4B vs Qwen3-8B embedding alignment | Exploratory |
| `nn_k1.py` | KNN@1 overlap (Qwen only) | Exploratory |

---

# KNN@10 Overlap (Table 5)

**Paper Output:** Table 5

## Overview

Measures KNN@10 overlap between embedding matrices of tied and untied models. For each token, computes its 10 nearest neighbors (by cosine similarity) in two matrices independently, then reports the fraction of shared neighbors, averaged over all tokens.

This metric complements Procrustes alignment (Table 1) by capturing local neighborhood structure rather than global alignment.

## Key Finding

> Tied embeddings preserve more nearest-neighbor structure with untied **output** matrices than with untied **input** matrices, consistent with the output-bias hypothesis.

## Reproduce Table 5

### Prerequisites

1. **Python environment** with `torch`, `transformers` (see root README for setup)

2. **HuggingFace access** — all models downloaded automatically

### Generate Table 5

```bash
python reproduce_table5.py
```

### Models Used

| Tied Model | Untied Model | Vocab Alignment |
|------------|-------------|-----------------|
| `allenai/OLMo-1B-hf` | `allenai/OLMo-1B-0724-hf` | Same tokenizer (50,304 tokens) |
| `Qwen/Qwen3-4B` (d=2560) | `Qwen/Qwen3-8B` (d=4096) | Same tokenizer (151,936 tokens) |
| `EleutherAI/gpt-neo-2.7B` | `EleutherAI/pythia-2.8b` | Aligned vocabulary (~36,938 common tokens) |

**Note:** KNN@10 overlap works across different hidden dimensions (Qwen3-4B vs 8B) because neighbors are identified by token index, not vector distance.

### Expected Results (Table 5)

| Comparison | KNN@10 |
|------------|--------|
| **OLMo-1B (tied) vs OLMo-1B-0724 (untied)** | |
| Tied vs Untied Input | 0.496 |
| Tied vs Untied Output | **0.733** |
| Untied Input vs Untied Output | 0.455 |
| **Qwen3-4B (tied) vs Qwen3-8B (untied)** | |
| Tied vs Untied Input | 0.366 |
| Tied vs Untied Output | **0.710** |
| Untied Input vs Untied Output | 0.366 |
| **GPT-Neo-2.7B (tied) vs Pythia-2.8B (untied)** | |
| Tied vs Untied Input | 0.372 |
| Tied vs Untied Output | 0.408 |
| Untied Input vs Untied Output | **0.611** |

**Note:** GPT-Neo/Pythia uses aligned vocabulary (~36,938 common tokens).

---

# Spectral Distance

Measures spectral distance between embedding spaces using the omnibus embedding approach from ["Comparing Foundation Models using Data Kernels"](https://arxiv.org/abs/2305.05126) (arxiv 2305.05126).

Algorithm:
1. Build k-NN adjacency matrices for each embedding space (symmetric, hollow)
2. Construct omnibus matrix: `M = [[A_a, (A_a+A_b)/2], [(A_a+A_b)/2, A_b]]`
3. Adjacency Spectral Embedding (ASE): top eigenpairs of M via `scipy.sparse.linalg.eigsh`
4. Latent positions `Z = U * sqrt(|S|)`, split into `Z_a`, `Z_b`
5. Spectral distance = `||Z_a - Z_b||_2 / min(||Z_a||_2, ||Z_b||_2)`

## Reproduce

```bash
python reproduce_spectral_distance.py
```

Optional flags:
```
--k INT            Nearest neighbors for graph construction (default: 64)
--n-components INT ASE embedding dimension (default: 128)
```

### Expected Results

| Comparison | Spectral Distance |
|---|---|
| **OLMo-1B (tied) vs OLMo-1B-0724 (untied)** | |
| Tied vs Untied Input | 1.372574 |
| Tied vs Untied Output | **0.587576** |
| Untied Input vs Untied Output | 1.349170 |
| **Qwen3-4B (tied) vs Qwen3-8B (untied)** | |
| Tied vs Untied Input | 1.440570 |
| Tied vs Untied Output | **0.809454** |
| Untied Input vs Untied Output | 1.421029 |
| **GPT-Neo-2.7B (tied) vs Pythia-2.8B (untied)** | |
| Tied vs Untied Input | 1.462022 |
| Tied vs Untied Output | 1.034541 |
| Untied Input vs Untied Output | **0.948573** |

Lower = more similar. For OLMo and Qwen3, the tied matrix is closest to the untied output. For GPT-Neo/Pythia, the two untied matrices from different families are most similar to each other, but the tied matrix is still closer to output than to input (1.035 vs 1.462), consistent with the output-bias hypothesis.
