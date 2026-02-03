# Trained Tuned Lenses

We are working on adding the pre-trained tuned lens models in the future.

## Expected Contents

Once available, this directory will contain trained tuned lens probes for the following models:

### Paper Figures 2, 6, 7 (Tuned Lens KL Divergence)

| Model | HuggingFace ID | Weight Tying | Figure |
|-------|---------------|--------------|--------|
| OLMo-1B | `allenai/OLMo-1B-hf` | **Tied** | Fig 2 |
| OLMo-1B-0724 | `allenai/OLMo-1B-0724-hf` | Untied | Fig 2 |
| GPT-Neo-2.7B | `EleutherAI/gpt-neo-2.7B` | **Tied** | Fig 6 |
| Pythia-2.8B | `EleutherAI/pythia-2.8b` | Untied | Fig 6 |
| Qwen3-4B | `Qwen/Qwen3-4B` | **Tied** | Fig 7 |
| Qwen3-8B | `Qwen/Qwen3-8B` | Untied | Fig 7 |

### Directory Structure

```
trained_lenses/
├── allenai/
│   ├── OLMo-1B-hf/
│   │   ├── config.json
│   │   └── params.pt
│   └── OLMo-1B-0724-hf/
│       └── ...
├── EleutherAI/
│   ├── gpt-neo-2.7B/
│   └── pythia-2.8b/
└── Qwen/
    ├── Qwen3-4B/
    └── Qwen3-8B/
```

## Training Your Own

You can train tuned lenses using the `tuned-lens` package:

```bash
pip install tuned-lens

# Example: Train lens for OLMo-1B (tied)
tuned-lens train --model allenai/OLMo-1B-hf \
    --output trained_lenses/allenai/OLMo-1B-hf

# Example: Train lens for Pythia-2.8B (untied)
tuned-lens train --model EleutherAI/pythia-2.8b \
    --output trained_lenses/EleutherAI/pythia-2.8b
```

**Training Tips:**
- GPU recommended (training takes several hours per model)
- Default settings work well for most models
- For larger models (8B+), consider using `--batch-size 4` and `--gradient-accumulation-steps 4`

See https://github.com/AlignmentResearch/tuned-lens for full documentation.
