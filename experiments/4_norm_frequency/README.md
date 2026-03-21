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
| `main.py` | CLI with `plot-figure5` and other plotting commands |
| `tokenizer_util.py` | Token frequency computation, embedding loading |
| `plotting_util.py` | Visualization utilities including `plot_figure5_comparison` |

## Reproducing Figure 5

### Prerequisites

1. **Model checkpoints** in this directory:
   - `OLMo-1B-tied/model.pt` - Tied OLMo-1B at ~10k steps
   - `OLMo-1B-untied/model.pt` - Untied OLMo-1B at ~10k steps

2. **Text data** for frequency computation:
   - `../text_data/eng_latn_300mb.txt`

3. **Training data** (shared across experiments):
   - `../text_data/dolma_v1_7/dolma_v1_7_30B.npy` — see `../text_data/README.md`

### Generate Figure 5

```bash
# Activate virtual environment
source .venv/bin/activate

# Generate Figure 5 with default settings
python main.py plot-figure5 \
    --config configs/tok_config_figure5_local.json \
    --output results/figures/figure5_norm_frequency.png \
    --steps "10k steps" \
    --ymin 0.7 \
    --ymax 2.0

# Output: results/figures/figure5_norm_frequency.png
```

### Configuration

The config file `configs/tok_config_figure5_local.json` specifies local model paths:

```json
{
  "OLMo-1B-tied-10k": {
    "class": "olmo_local",
    "path": "experiments/4_norm_frequency/OLMo-1B-tied",
    "tokenizer": "EleutherAI/gpt-neox-20b",
    "family": "OLMo-1B-tied"
  },
  "OLMo-1B-untied-10k": {
    "class": "olmo_local",
    "path": "experiments/4_norm_frequency/OLMo-1B-untied",
    "tokenizer": "EleutherAI/gpt-neox-20b",
    "family": "OLMo-1B-untied"
  }
}
```

### Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | Required | Path to model config JSON |
| `--output` | `FIG_DIR/figure5_norm_frequency.png` | Output path for the figure |
| `--steps` | `"10k steps"` | Training steps label for titles |
| `--ymin` | `0.8` | Lower y-axis limit (L2 norm) |
| `--ymax` | `2.0` | Upper y-axis limit (L2 norm) |
| `--xmin` | `0.0` | Lower x-axis limit (log10 freq) |
| `--xmax` | `7.0` | Upper x-axis limit (log10 freq) |

---

## Model Training Reproduction

The models used for Figure 5 were trained using the [OLMo](https://github.com/allenai/OLMo) framework with matching configurations (except `weight_tying`).

### Training Configuration

| Setting | Value |
|---------|-------|
| Architecture | OLMo-1B (1.17B parameters) |
| Layers | 16 |
| Hidden Size | 2048 |
| Attention Heads | 16 |
| Activation | SwiGLU |
| Position Encoding | RoPE |
| Sequence Length | 4096 |
| Vocab Size | 50,280 |
| Data | Dolma v1.7 (30B tokens subset) |
| Batch Size | 512 |
| Learning Rate | 3e-4 |
| Warmup Steps | 2500 |
| Scheduler | Cosine with warmup |
| Max Steps | 10,000 |

### Setup

```bash
# Clone OLMo repository
git clone https://github.com/allenai/OLMo.git
cd OLMo

pip install -e '.[all]'

# Training data (shared across experiments)
# Data should be at: experiments/text_data/dolma_v1_7/dolma_v1_7_30B.npy
# See experiments/text_data/README.md for details
```

### Training Configs

The training configs used are stored alongside the model checkpoints:

- **Tied model**: `OLMo-1B-tied/config.yaml` (sets `weight_tying: true`)
- **Untied model**: `OLMo-1B-untied/config.yaml` (sets `weight_tying: false`)

### Train Models

```bash
# From the repository root
# Train tied model
torchrun --nproc_per_node=8 OLMo/scripts/train.py \
    experiments/4_norm_frequency/OLMo-1B-tied/config.yaml

# Train untied model
torchrun --nproc_per_node=8 OLMo/scripts/train.py \
    experiments/4_norm_frequency/OLMo-1B-untied/config.yaml
```

Checkpoints will be saved to the `save_folder` specified in each config.
