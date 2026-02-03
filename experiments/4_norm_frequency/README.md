# Experiment 4: Norm-Frequency Relationship (Section 5.2)

**Paper Outputs:** Figure 5

## Overview

Plots L2 norm of each token's embedding against its log-frequency in a corpus. Compares:
- Untied input embeddings
- Untied output embeddings  
- Tied embeddings

## Key Finding

> The tied embedding matrix visually resembles the untied output matrix, not the input matrix.

- Untied input: Norms clustered tightly around 0.8-1.0 (near initialization)
- Untied output & Tied: Both show characteristic pattern where norms increase with frequency up to ~10^4 occurrences, then slope downward for very high-frequency tokens

## Scripts

| Script | Description |
|--------|-------------|
| `main.py` | CLI with `plot-logfreq-vs-l2` command |
| `tokenizer_util.py` | Token frequency computation |
| `plotting_util.py` | Visualization utilities |

## Usage

```bash
# Plot norm vs frequency for a model
python main.py plot-logfreq-vs-l2 --config configs/tok_config_olmo_1b_both.json

# Generate Figure 5 (requires OLMo checkpoints at step 10K)
python main.py plot-logfreq-vs-l2 \
    --config configs/tok_config_olmo_1b_early_10k.json \
    --output results/figures/figure5_norm_frequency.png
```

## Configuration

Example config for OLMo-1B at step 10K:

```json
{
    "olmo_1b_0724_step10k": {
        "class": "huggingface",
        "model": "allenai/OLMo-1B-0724-hf",
        "revision": "step10000-tokens40B"
    },
    "olmo_1b_tied": {
        "class": "huggingface", 
        "model": "allenai/OLMo-1B-hf"
    }
}
```
