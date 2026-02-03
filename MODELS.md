# Models Used in the Paper

This document lists all models used in "Weight Tying Biases Token Embeddings Towards the Unembedding Space" with their HuggingFace identifiers, weight tying status, and paper references.

## Quick Reference

| Model | HuggingFace ID | Weight Tying | Paper Reference |
|-------|---------------|--------------|-----------------|
| OLMo-1B (tied) | `allenai/OLMo-1B-hf` | **Tied** | Table 1, Fig 1, 2, 5 |
| OLMo-1B-0724 (untied) | `allenai/OLMo-1B-0724-hf` | Untied | Table 1, 2, Fig 1, 3, 5 |
| GPT-Neo-2.7B | `EleutherAI/gpt-neo-2.7B` | **Tied** | Table 1, 5; Fig 6 |
| Pythia-2.8B | `EleutherAI/pythia-2.8b` | Untied | Table 1, 5; Fig 6 |
| Pythia-1B | `EleutherAI/pythia-1b` | Untied | Fig 9 (Appendix D) |
| OLMo-7B-0724 | `allenai/OLMo-7B-0724-hf` | Untied | Fig 8 (Appendix D) |
| Qwen3-4B | `Qwen/Qwen3-4B` | **Tied** | Table 5; Fig 7 |
| Qwen3-8B | `Qwen/Qwen3-8B` | Untied | Table 5; Fig 7 |

---

## Detailed Model Information

### OLMo Family (Allen AI)

#### OLMo-1B (Tied) - February 2024
- **HuggingFace ID:** `allenai/OLMo-1B-hf`
- **Weight Tying:** Yes (tied embeddings)
- **Parameters:** ~1.2B
- **Training Data:** Dolma v1
- **Paper Usage:**
  - Table 1: Embedding alignment comparison
  - Figure 1: Main result showing tied alignment
  - Figure 2: Tuned lens KL divergence
  - Figure 5: Norm-frequency relationship (right panel)
  - Figure 4: Gradient flow analysis (trained from scratch)

#### OLMo-1B-0724 (Untied) - July 2024
- **HuggingFace ID:** `allenai/OLMo-1B-0724-hf`
- **Weight Tying:** No (untied embeddings)
- **Parameters:** ~1.2B
- **Training Data:** Dolma v1.7
- **Checkpoints Available:** Every 1000 steps
- **Checkpoint Format:** `step{N}-tokens{M}B` (e.g., `step10000-tokens20B`)
- **Paper Usage:**
  - Table 1: Baseline for alignment comparison
  - Table 2: Reference for gradient scaling experiments
  - Figure 1: Main result (untied comparison)
  - Figure 2: Tuned lens comparison
  - Figure 3: Embedding evolution dynamics
  - Figure 5: Norm-frequency (left panel: input/output)

**Key Checkpoints:**
```
step0-tokens0B         # Initialization
step1000-tokens2B      # Early training
step5000-tokens10B     # Early training
step10000-tokens20B    # Used for Table 2
step20000-tokens41B    # Figure 3 analysis
step50000-tokens104B   # Mid training
step100000-tokens209B  # Later training
```

#### OLMo-7B-0724 (Untied)
- **HuggingFace ID:** `allenai/OLMo-7B-0724-hf`
- **Weight Tying:** No
- **Paper Usage:**
  - Figure 8 (Appendix D): Embedding evolution at scale

---

### Pythia/GPT-Neo Family (EleutherAI)

These models share the same codebase and training infrastructure, trained on The Pile dataset.

#### GPT-Neo-2.7B (Tied)
- **HuggingFace ID:** `EleutherAI/gpt-neo-2.7B`
- **Weight Tying:** Yes
- **Parameters:** ~2.7B
- **Architecture:** Similar to GPT-3
- **Paper Usage:**
  - Table 1: Cross-model alignment with Pythia
  - Table 5 (Appendix B): KNN overlap analysis
  - Figure 6 (Appendix C): Tuned lens comparison

#### Pythia-2.8B (Untied)
- **HuggingFace ID:** `EleutherAI/pythia-2.8b`
- **Weight Tying:** No
- **Parameters:** ~2.8B
- **Checkpoints:** 154 checkpoints across training
- **Paper Usage:**
  - Table 1: Comparison with GPT-Neo
  - Table 5 (Appendix B): KNN overlap
  - Figure 6 (Appendix C): Tuned lens

#### Pythia-1B (Untied)
- **HuggingFace ID:** `EleutherAI/pythia-1b`
- **Weight Tying:** No
- **Checkpoints:** 154 across training, format: `step{N}`
- **Paper Usage:**
  - Figure 9 (Appendix D): Embedding evolution dynamics

**Key Checkpoints:**
```
step0      # Initialization
step1, step2, step4, step8, step16, step32, step64, step128, step256, step512  # Very early
step1000, step2000, ..., step14000  # Regular intervals
```

---

### Qwen3 Family (Alibaba)

The Qwen3 family uses scale-dependent weight tying: smaller models tie, larger models untie.

#### Qwen3-4B (Tied)
- **HuggingFace ID:** `Qwen/Qwen3-4B`
- **Weight Tying:** Yes
- **Layers:** 36
- **Paper Usage:**
  - Table 5 (Appendix B): KNN overlap
  - Figure 7 (Appendix C): Tuned lens

#### Qwen3-8B (Untied)
- **HuggingFace ID:** `Qwen/Qwen3-8B`
- **Weight Tying:** No
- **Layers:** 36
- **Paper Usage:**
  - Table 5 (Appendix B): KNN overlap
  - Figure 7 (Appendix C): Tuned lens

#### Additional Qwen3 Models (Available for Extended Analysis)
- `Qwen/Qwen3-0.6B` (tied, 28 layers)
- `Qwen/Qwen3-1.7B` (tied, 28 layers)
- `Qwen/Qwen3-14B` (untied, 40 layers)

---

## Custom-Trained Models (Section 6: Gradient Scaling)

These models were trained from scratch using the OLMo training framework with modified gradient flows.

| Model | Input Gradient Scale | Training Steps | Purpose |
|-------|---------------------|----------------|---------|
| OLMo-1B (baseline) | 1× | 10,000 | Control |
| OLMo-1B (scaled) | 2× | 1,000 | Early intervention |
| OLMo-1B (scaled) | 5× | 10,000 | Main result (Table 2) |
| OLMo-1B (scaled) | 10× | 1,000 | Strong intervention |

**Training Configuration:**
- Base config: OLMo-1B standard configuration
- Modifications: Gradient hooks to scale input-layer gradients
- Framework: [OLMo training code](https://github.com/allenai/OLMo)

---

## Loading Models

### Basic Loading (Final Checkpoint)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-0724-hf")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
```

### Loading Specific Checkpoints (OLMo)
```python
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-1B-0724-hf",
    revision="step10000-tokens20B"
)
```

### Loading Specific Checkpoints (Pythia)
```python
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-1b",
    revision="step1000"
)
```

### Checking Weight Tying
```python
def check_weight_tying(model):
    """Check if model uses tied embeddings."""
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    return input_emb.weight.data_ptr() == output_emb.weight.data_ptr()
```

---

## Figure → Model Mapping

| Figure | Models Used |
|--------|-------------|
| Figure 1 | OLMo-1B (tied) vs OLMo-1B-0724 (untied) |
| Figure 2 | OLMo-1B (tied) vs OLMo-1B-0724 (untied) |
| Figure 3 | OLMo-1B-0724 (untied) checkpoints |
| Figure 4 | OLMo-1B (tied, trained from scratch) |
| Figure 5 | OLMo-1B (tied) and OLMo-1B-0724 (untied) at step 10k |
| Figure 6 | GPT-Neo-2.7B (tied) vs Pythia-2.8B (untied) |
| Figure 7 | Qwen3-4B (tied) vs Qwen3-8B (untied) |
| Figure 8 | OLMo-7B-0724 (untied) checkpoints |
| Figure 9 | Pythia-1B (untied) checkpoints |

| Table | Models Used |
|-------|-------------|
| Table 1 | OLMo-1B, OLMo-1B-0724, GPT-Neo-2.7B, Pythia-2.8B |
| Table 2 | OLMo-1B variants with gradient scaling |
| Table 3-4 | Pythia family (parameter counts) |
| Table 5 | OLMo-1B pair, Qwen3-4B/8B, GPT-Neo/Pythia |
| Table 6 | OLMo-1B gradient scaling variants |
| Table 7 | mBERT (from Chung et al., 2020) |

---

## License Information

| Model Family | License |
|--------------|---------|
| OLMo | Apache 2.0 |
| Pythia | Apache 2.0 |
| GPT-Neo | MIT |
| Qwen3 | Apache 2.0 |

All licenses permit use for scientific research.
