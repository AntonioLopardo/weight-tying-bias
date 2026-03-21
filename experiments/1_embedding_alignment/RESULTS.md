# Embedding Similarity Analysis Results

Inspired by ["Emerging Cross-lingual Structure in Pretrained Language Models" (ACL 2020)](https://aclanthology.org/2020.acl-main.536.pdf)

This document contains the results of comparing input and output embedding matrices across various language models.

---

## Table of Contents
1. [Overview](#overview)
2. [OLMo Models](#olmo-models)
3. [Qwen3 Models](#qwen3-models)
4. [EleutherAI Models (Pythia & GPT-Neo)](#eleutherai-models)
5. [Cross-Model Comparisons](#cross-model-comparisons)
6. [Key Findings](#key-findings)
7. [Commands Reference](#commands-reference)

---

## Overview

We analyze the relationship between input embeddings (token → hidden) and output embeddings (hidden → logits) across models with different weight tying configurations.

### Models Analyzed

| Model | Weight Tying | Vocab Size | Hidden Dim |
|-------|--------------|------------|------------|
| OLMo-1B-0724-reproduce (step 2000) | ❌ No | 50,304 | 2048 |
| [OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf) | ❌ No | 50,304 | 2048 |
| [OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf) | ✅ Yes | 50,304 | 2048 |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | ✅ Yes | 151,936 | 2560 |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | ❌ No | 151,936 | 4096 |
| [Pythia-2.8B](https://huggingface.co/EleutherAI/pythia-2.8b) | ❌ No | 50,304 | 2560 |
| [GPT-Neo-2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B) | ✅ Yes | 50,257 | 2560 |

---

## OLMo Models

### OLMo-1B-0724-reproduce (Step 2000 - Early Training)

```bash
source .venv/bin/activate && python emb_similarity/compare_embeddings.py \
    --checkpoint checkpoints/OLMo-1B-0724-reproduce/step2000-unsharded
```

**Results:**
```
Input embedding shape: torch.Size([50304, 2048])
Output embedding shape: torch.Size([50304, 2048])
Weight tying: False

1. Per-Token Cosine Similarity:
   Mean:   0.000122
   Std:    0.022130
   Median: 0.000207

2. Global Similarity Metrics:
   Pearson correlation:      0.000126
   Frobenius diff norm:      311.6978
   Relative Frobenius diff:  1.414743
   Input Frobenius norm:     213.1632
   Output Frobenius norm:    227.4791

3. Singular Value Analysis (top-50):
   SV spectrum correlation:  0.969060
   Top-1 SV ratio (in/out):  0.355905
   Top-5 SVs (input):  ['7.81', '7.35', '6.71', '6.39', '6.21']
   Top-5 SVs (output): ['21.95', '16.17', '15.28', '14.05', '12.44']

4. Procrustes Alignment (Orthogonal):
   Alignment error:              281.7986
   Cos sim after alignment (μ):  0.181825

5. Subspace Overlap (k=100):
   Mean canonical correlation:   0.189836
```

### OLMo-1B-0724-hf (Fully Trained - 3.05T tokens)

```bash
source .venv/bin/activate && python emb_similarity/compare_embeddings.py \
    --hf-model allenai/OLMo-1B-0724-hf
```

**Results:**
```
Input embedding shape: torch.Size([50304, 2048])
Output embedding shape: torch.Size([50304, 2048])
Weight tying: False

1. Per-Token Cosine Similarity:
   Mean:   0.008321
   Std:    0.026566
   Median: 0.008523

2. Global Similarity Metrics:
   Pearson correlation:      0.011167
   Frobenius diff norm:      207.9938
   Relative Frobenius diff:  1.670650
   Input Frobenius norm:     45.5171
   Output Frobenius norm:    203.4804

3. Singular Value Analysis (top-50):
   SV spectrum correlation:  0.944843
   Top-1 SV ratio (in/out):  0.295819
   Top-5 SVs (input):  ['9.38', '4.23', '3.82', '3.46', '3.11']
   Top-5 SVs (output): ['31.70', '21.80', '20.30', '17.19', '16.12']

4. Procrustes Alignment (Orthogonal):
   Cos sim after alignment (μ):  0.439968

5. Subspace Overlap (k=100):
   Mean canonical correlation:   0.179548
```

### OLMo-1B-hf (Original - Tied Embeddings)

```bash
source .venv/bin/activate && python emb_similarity/compare_embeddings.py \
    --hf-model allenai/OLMo-1B-hf
```

**Results:**
```
Weight tying: True
Note: Input and output embeddings share the same memory (weight tying).
All similarity metrics will be perfect (1.0).

Input/Output Frobenius norm: 531.7175
Top-5 SVs: ['99.17', '52.84', '47.40', '41.25', '38.97']
```

### OLMo Cross-Model Comparison

```bash
source .venv/bin/activate && python emb_similarity/compare_cross_model.py
```

| Comparison | Raw Cos (μ) | Pearson | Procrustes |
|------------|-------------|---------|------------|
| Untied-Input (0724) vs Tied (original) | 0.0000 | 0.0009 | 0.4196 |
| Untied-Output (0724) vs Tied (original) | 0.0162 | 0.0138 | **0.6694** |
| Untied-Input (0724) vs Untied-Output (0724) | 0.0083 | 0.0112 | 0.4400 |

**Key finding:** Output embedding aligns better with tied embedding (0.67 vs 0.42).

---

## Qwen3 Models

### Qwen3-4B (Tied)

```bash
source .venv/bin/activate && python emb_similarity/compare_embeddings.py \
    --hf-model Qwen/Qwen3-4B
```

**Results:**
```
Weight tying: True
Input embedding shape: torch.Size([151936, 2560])
Embedding Frobenius norm: 402.0878
Top-5 SVs: ['126.67', '24.21', '20.91', '20.53', '18.54']
```

### Qwen3-8B (Untied)

```bash
source .venv/bin/activate && python emb_similarity/compare_embeddings.py \
    --hf-model Qwen/Qwen3-8B
```

**Results:**
```
Weight tying: False
Input embedding shape: torch.Size([151936, 4096])
Output embedding shape: torch.Size([151936, 4096])

1. Per-Token Cosine Similarity:
   Mean:   0.004137
   Std:    0.016045

2. Global Similarity Metrics:
   Pearson correlation:      0.004337
   Input Frobenius norm:     489.4881
   Output Frobenius norm:    572.2631

3. Singular Value Analysis:
   SV spectrum correlation:  0.928017
   Top-1 SV ratio (in/out):  0.461140

4. Procrustes Alignment:
   Cos sim after alignment (μ):  0.256734
```

### Qwen3 Cross-Model Comparison

```bash
source .venv/bin/activate && python emb_similarity/compare_qwen.py
```

| Metric | Qwen3-4B (tied) | Qwen3-8B (untied) |
|--------|-----------------|-------------------|
| Hidden dim | 2560 | 4096 |
| Embedding norm | 402 | in: 489 / out: 572 |

**Within-model (8B):**
- Raw cosine: 0.004
- Procrustes: 0.257

**Per-token norm correlations:**
| Comparison | Correlation |
|------------|-------------|
| Qwen3-4B (tied) vs Qwen3-8B (input) | 0.6898 |
| Qwen3-4B (tied) vs Qwen3-8B (output) | **0.9732** |
| Qwen3-8B input vs output | 0.7323 |

**Key finding:** Output embedding has 97% norm correlation with tied embedding.

---

## EleutherAI Models

### Pythia-2.8B (Untied)

```bash
source .venv/bin/activate && python emb_similarity/compare_embeddings.py \
    --hf-model EleutherAI/pythia-2.8b
```

**Results:**
```
Weight tying: False
Input embedding shape: torch.Size([50304, 2560])
Output embedding shape: torch.Size([50304, 2560])

1. Per-Token Cosine Similarity:
   Mean:   -0.001339  (slightly negative!)
   Std:    0.019957

2. Global Similarity Metrics:
   Pearson correlation:      -0.001346
   Input Frobenius norm:     219.3107
   Output Frobenius norm:    219.8994

3. Singular Value Analysis:
   SV spectrum correlation:  0.694145  (lower than other models)
   Top-1 SV ratio (in/out):  0.337804
   Top-5 SVs (input):  ['20.51', '20.21', '15.91', '15.06', '14.26']
   Top-5 SVs (output): ['60.71', '17.25', '10.17', '9.34', '9.01']

4. Procrustes Alignment:
   Cos sim after alignment (μ):  0.417858
```

### GPT-Neo-2.7B (Tied)

```bash
source .venv/bin/activate && python emb_similarity/compare_embeddings.py \
    --hf-model EleutherAI/gpt-neo-2.7B
```

**Results:**
```
Weight tying: True
Input embedding shape: torch.Size([50257, 2560])
Embedding Frobenius norm: 984.9922  (much larger!)
Top-5 SVs: ['897.00', '71.54', '62.86', '48.99', '43.08']
```

### Pythia vs GPT-Neo Cross-Model Comparison

```bash
source .venv/bin/activate && python emb_similarity/compare_pythia_gptneo.py
```

| Comparison | Raw Cos (μ) | Pearson | Procrustes |
|------------|-------------|---------|------------|
| Pythia Input vs GPT-Neo (tied) | -0.0005 | -0.0005 | 0.166 |
| Pythia Output vs GPT-Neo (tied) | 0.0013 | -0.0004 | 0.178 |
| Pythia Input vs Pythia Output | -0.0013 | -0.0013 | **0.418** |

**Per-token norm correlations:**
| Comparison | Correlation |
|------------|-------------|
| Pythia Input vs GPT-Neo | -0.0413 |
| Pythia Output vs GPT-Neo | -0.0017 |
| Pythia Input vs Output | 0.2291 |

**Key finding:** Cross-model alignment is very low (~0.17) despite similar architecture and training data (Pile).

---

## Cross-Model Comparisons

### Summary: Within-Model (Untied) Input vs Output

| Model | Raw Cosine | Procrustes | SV Correlation |
|-------|------------|------------|----------------|
| OLMo-1B-0724 (step 2000) | 0.000 | 0.182 | 0.969 |
| OLMo-1B-0724-hf (final) | 0.008 | 0.440 | 0.945 |
| Qwen3-8B | 0.004 | 0.257 | 0.928 |
| Pythia-2.8B | -0.001 | 0.418 | 0.694 |

### Summary: Untied Output vs Tied Embedding (Same Model Family)

| Comparison | Procrustes | Norm Correlation |
|------------|------------|------------------|
| OLMo-1B-0724 Output vs OLMo-1B Tied | 0.669 | N/A |
| Qwen3-8B Output vs Qwen3-4B Tied | N/A (diff dims) | 0.973 |

---

## Key Findings

1. **Untied embeddings are essentially uncorrelated**: Raw cosine similarity between input/output is near zero (~0.00-0.01) across all untied models.

2. **Procrustes alignment reveals structure**: After orthogonal alignment, cosine similarity improves significantly (0.18 → 0.44 for trained models), supporting the ACL 2020 paper's finding about latent symmetries.

3. **Training improves alignability**: OLMo at step 2000 has 0.18 Procrustes alignment vs 0.44 after full training.

4. **Output embeddings are more "traditional"**: Untied output embeddings align better with tied embeddings (0.67-0.97 correlation) than input embeddings do (0.42-0.69).

5. **Cross-model alignment is poor**: Even models with same architecture and training data (Pythia vs GPT-Neo) have very low alignment (~0.17), confirming that embedding alignment requires shared training.

6. **Tied embeddings have larger norms**: GPT-Neo (tied) has 4.5x larger norm than Pythia (untied), likely due to accumulated gradient updates from both directions.

7. **SV spectrum correlation varies**: Most models show 0.93+ SV correlation between input/output, but Pythia is notably lower at 0.69.

---

## Commands Reference

### Single Model Analysis

```bash
# Local checkpoint
source .venv/bin/activate && python emb_similarity/compare_embeddings.py \
    --checkpoint /path/to/checkpoint

# HuggingFace model
source .venv/bin/activate && python emb_similarity/compare_embeddings.py \
    --hf-model model_id

# With specific revision
source .venv/bin/activate && python emb_similarity/compare_embeddings.py \
    --hf-model allenai/OLMo-1B-0724-hf --revision step1000-tokens4B

# Save results to JSON
source .venv/bin/activate && python emb_similarity/compare_embeddings.py \
    --hf-model model_id --output results.json
```

### Cross-Model Comparisons

```bash
# OLMo comparison
source .venv/bin/activate && python emb_similarity/compare_cross_model.py

# Qwen3 comparison
source .venv/bin/activate && python emb_similarity/compare_qwen.py

# Pythia vs GPT-Neo
source .venv/bin/activate && python emb_similarity/compare_pythia_gptneo.py
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `compare_embeddings.py` | Main analysis script for single models |
| `compare_cross_model.py` | OLMo tied vs untied comparison |
| `compare_qwen.py` | Qwen3-4B vs Qwen3-8B comparison |
| `compare_pythia_gptneo.py` | Pythia-2.8B vs GPT-Neo-2.7B comparison |

---

## References

- Conneau, A., et al. (2020). ["Emerging Cross-lingual Structure in Pretrained Language Models"](https://aclanthology.org/2020.acl-main.536.pdf). ACL 2020.
- [OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf) - Allen AI
- [OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf) - Allen AI
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) - Alibaba
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) - Alibaba
- [Pythia-2.8B](https://huggingface.co/EleutherAI/pythia-2.8b) - EleutherAI
- [GPT-Neo-2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B) - EleutherAI

