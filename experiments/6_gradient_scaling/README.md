# Experiment 6: Gradient Scaling Ablation (Section 6)

**Paper Outputs:** Table 2, Table 6 (Appendix E)

## Overview

Causal intervention where input-layer gradients are artificially scaled during training of tied models. Tests whether gradient balance causally determines embedding structure.

## Key Finding

> Scaling input gradients shifts embedding structure predictably towards untied input embeddings and away from untied output embeddings.

With 5× input gradient scaling at step 10K:
- Input alignment: 0.216 → 0.222 (increases)
- Output alignment: 0.384 → 0.369 (decreases)

## Available Checkpoints

Linked from `OLMo/checkpoints/`:

### Gradient Scaling Experiments
| Checkpoint | Input Scale | Steps |
|------------|-------------|-------|
| `OLMo-1B-tied-no-scale-*` | 1× (baseline) | 1K, 10K |
| `OLMo-1B-tied-emb2-1000` | 2× | 1K |
| `OLMo-1B-tied-emb5-*` | 5× | 1K, 10K |
| `OLMo-1B-tied-emb10-1000` | 10× | 1K |
| `OLMo-1B-tied-out0.1-1000` | 1× (output 0.1×) | 1K |
| `OLMo-1B-tied-out0.5-*` | 1× (output 0.5×) | 1K |

### Baseline Comparisons
| Checkpoint | Description |
|------------|-------------|
| `OLMo-1B-tied` | Standard tied training |
| `OLMo-1B-untied` | Standard untied training |
| `OLMo-1B-0724-reproduce` | OLMo-1B-0724 reproduction |

## Usage

```python
import torch
from transformers import AutoModelForCausalLM

# Load scaled checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/OLMo-1B-tied-emb5-1000/step1000-unsharded"
)

# Compare to untied reference
ref_model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-1B-0724-hf",
    revision="step1000-tokens4B"
)

# Compute alignment after Procrustes
# ...
```

## Expected Results

**Table 2 (Step 10K):**

| Model | vs Untied Input | vs Untied Output |
|-------|-----------------|------------------|
| Tied (no scaling) | 0.216 | 0.384 |
| Tied (input ×5) | **0.222** | 0.369 |

**Table 6 (Step 1K):**

| Model | vs Untied Input | vs Untied Output |
|-------|-----------------|------------------|
| Tied (no scaling) | 0.172 | 0.197 |
| Tied (input ×2) | 0.173 | 0.197 |
| Tied (input ×10) | 0.173 | **0.190** |
