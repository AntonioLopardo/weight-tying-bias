# Weight Tying Biases Token Embeddings Towards the Unembedding Space

> **Work in Progress**: Scripts may not work as intended and documentation is incomplete.

Reproduction repository for the ACL submission analyzing how weight tying affects the relationship between input (embedding) and output (unembedding) matrices in language models.

## Status

This codebase consolidates scripts and artifacts from multiple sources. While the core analysis scripts exist, they have not been fully tested end-to-end in a clean environment. Known limitations:

- **Dependency versions** are approximate and may need adjustment
- **Paths** in some scripts may reference original development locations
- **Some experiments** require large model downloads or training infrastructure
- **Documentation** for individual scripts varies in completeness

We are working to improve reproducibility. Please open an issue if you encounter problems.

## Abstract

Weight tying—sharing parameters between input and output embedding matrices—is common in LM design, yet its impact on the learned embedding space remains poorly understood. We show that:

1. **Tied embeddings align with output, not input** (Table 1, Figure 1)
2. **Early layers suffer** — elevated KL divergence in tuned lens (Figure 2)
3. **Output gradients dominate** — 80-90% of learning signal early in training (Figure 4)
4. **Gradient scaling confirms causality** — amplifying input gradients shifts structure (Table 2)

## Paper → Code Mapping

| Section | Experiment | Outputs | Status |
|---------|-----------|---------|--------|
| 4.1 | Embedding Alignment | Table 1, Figure 1 | Scripts exist, needs testing |
| 4.2 | Tuned Lens Analysis | Figure 2 | Scripts + pre-trained lenses |
| 5.1 | Embedding Evolution | Figure 3 | Script exists, needs testing |
| 5.2 | Norm-Frequency | Figure 5 | Scripts exist, needs testing |
| 5.3 | Gradient Flow | Figure 4 | Plotting script + pre-generated figures |
| 6 | Gradient Scaling | Table 2 | Checkpoints available |
| App. B | KNN Overlap | Table 5 | Script exists |
| App. C | Tuned Lens (extended) | Figures 6, 7 | Scripts + pre-generated figures |
| App. D | Evolution (extended) | Figures 8, 9 | Script exists |
| App. E | Scaling (extended) | Table 6 | Checkpoints available |

## Repository Structure

```
weight-tying-bias/
├── experiments/
│   ├── 1_embedding_alignment/    # Section 4.1: Table 1, Figure 1, Table 5
│   ├── 2_tuned_lens/             # Section 4.2: Figures 2, 6, 7
│   ├── 3_embedding_evolution/    # Section 5.1: Figures 3, 8, 9
│   ├── 4_norm_frequency/         # Section 5.2: Figure 5
│   ├── 5_gradient_flow/          # Section 5.3: Figure 4
│   └── 6_gradient_scaling/       # Section 6: Tables 2, 6
│
├── results/
│   ├── tables/
│   └── figures/
│
├── README.md
└── requirements.txt
```

## Quick Start

### Experiment 1: Embedding Alignment (Table 1)

```bash
cd experiments/1_embedding_alignment

# OLMo comparison
python compare_cross_model.py

# GPT-Neo/Pythia comparison
python compare_pythia_gptneo.py

# KNN overlap (Table 5)
python nn_k1.py
```

### Experiment 2: Tuned Lens (Figure 2)

```bash
cd experiments/2_tuned_lens

# OLMo tied vs untied
python compare_olmo_tuned_lenses.py

# Pythia vs GPT-Neo (Figure 6)
python compare_pythia_vs_gptneo.py

# Qwen3 (Figure 7)
python compare_qwen3_tuned_lenses.py
```

### Experiment 3: Embedding Evolution (Figure 3)

```bash
cd experiments/3_embedding_evolution

# OLMo-1B-0724 (Figure 3)
python track_evolution.py --config configs/evolution_olmo_1b.json

# Pythia-1B (Figure 9)
python track_evolution.py --config configs/evolution_pythia_1b.json
```

### Experiment 4: Norm-Frequency (Figure 5)

```bash
cd experiments/4_norm_frequency
python main.py plot-logfreq-vs-l2 --config configs/tok_config_olmo_1b_both.json
```

## Models

See **[MODELS.md](MODELS.md)** for a comprehensive list of all models, their HuggingFace IDs, checkpoint formats, and paper references.

| Model | HuggingFace ID | Weight Tying | Paper Role |
|-------|---------------|--------------|------------|
| OLMo-1B | `allenai/OLMo-1B-hf` | **Tied** | Main comparison |
| OLMo-1B-0724 | `allenai/OLMo-1B-0724-hf` | Untied | Main comparison |
| GPT-Neo-2.7B | `EleutherAI/gpt-neo-2.7B` | **Tied** | Cross-family validation |
| Pythia-2.8B | `EleutherAI/pythia-2.8b` | Untied | Cross-family validation |
| Qwen3-4B | `Qwen/Qwen3-4B` | **Tied** | Scale-dependent validation |
| Qwen3-8B | `Qwen/Qwen3-8B` | Untied | Scale-dependent validation |

### Loading Models with Specific Checkpoints

```python
from transformers import AutoModelForCausalLM

# OLMo at step 10000 (for Figure 3)
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-1B-0724-hf",
    revision="step10000-tokens20B"
)

# Pythia at step 1000 (for Figure 9)
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-1b",
    revision="step1000"
)
```

## Key Results

### Table 1: Embedding Alignment (Linear transformation)

| Comparison | OLMo | GPT-Neo/Pythia |
|------------|------|----------------|
| Output(U) → Tied | **0.719** | **0.637** |
| Input(U) → Tied | 0.525 | 0.507 |

### Figure 4: Gradient Flow (First 1000 steps)

- Output gradients: 80-90% of total signal
- Input gradients: 10-20% of total signal

### Table 2: Gradient Scaling Effect (Step 10K)

| Model | vs Untied Input | vs Untied Output |
|-------|-----------------|------------------|
| Baseline | 0.216 | 0.384 |
| Input ×5 | 0.222 (+0.006) | 0.369 (-0.015) |

## Known Issues & TODOs

- [ ] Scripts need path updates for standalone use (currently may reference `/home/vec_norm/`)
- [ ] `track_evolution.py` untested with all checkpoint revisions
- [ ] Some HuggingFace checkpoint revisions may not exist or have different naming
- [ ] Tuned lens training scripts not yet documented
- [ ] Missing end-to-end test script to verify all experiments run
- [ ] `requirements.txt` versions are approximate

## Requirements

```bash
pip install -r requirements.txt
```

**Note**: Some experiments require significant compute resources (GPU recommended) and may download large models (1-8GB each).

## Artifacts

Code and artifacts also available at: https://osf.io/cn54y/
