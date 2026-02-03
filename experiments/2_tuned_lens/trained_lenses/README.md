# Trained Tuned Lenses

We are working on adding the pre-trained tuned lens models in the future.

## Expected Contents

Once available, this directory will contain trained tuned lens probes for:

- `allenai/OLMo-1B-hf` (tied)
- `allenai/OLMo-1B-0724-hf` (untied)
- `EleutherAI/pythia-2.8b` (untied)
- `EleutherAI/gpt-neo-2.7B` (tied)
- `Qwen/Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B` (tied)
- `Qwen/Qwen3-8B` (untied)

## Training Your Own

You can train tuned lenses using the `tuned-lens` package:

```bash
pip install tuned-lens

tuned-lens train --model allenai/OLMo-1B-hf --output trained_lenses/allenai/OLMo-1B-hf
```

See https://github.com/AlignmentResearch/tuned-lens for documentation.
