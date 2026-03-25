# Research Ideas: Weight-Tying Bias × Layer Stacking (RYS)

## Background

### Weight-Tying Bias (this paper)
- Tied embedding/unembedding matrices are pulled toward the *output* space due to gradient dominance (70% output vs 30% input early in training)
- This creates elevated KL divergence in early/middle layers (tuned lens analysis): the model must "unlearn" the output-space bias across depth
- Causal evidence: scaling input gradients by 5× shifts embedding toward input space

### RYS / Layer Stacking ([blog post](https://dnhkng.github.io/posts/rys/))
- Duplicating a contiguous block of middle layers (e.g., layers 45–51 of Qwen2-72B) without retraining yields +2.61% avg benchmark improvement
- Single-layer duplication fails; multi-layer circuits (the "reasoning cortex") must run as a unit
- Optimal layers found by empirical grid search; no principled selection criterion proposed

**Core insight connecting both:** Weight tying creates a spatial bias that manifests as a *depth inefficiency* — early layers waste capacity compensating for the output-space attractor. Layer stacking adds effective depth at inference time. These are two sides of the same coin: one identifies *where* the model struggles across depth, the other provides a mechanism to add depth cheaply.

---

## Idea 1: KL Divergence Profile as a Principled Layer-Selection Signal

**Hypothesis:** The tuned lens KL divergence curve (already computed in this paper) encodes exactly where the model is doing the heaviest representational work. The RYS blog found optimal layers by brute-force grid search. Combining them: use the KL profile to predict which layer block to stack.

**Specific predictions:**
- The optimal block to stack sits at the *steepest KL decline* — layers actively compressing representations have the most reusable structure
- For weight-tied models, this steep region occurs *later* than for untied models (because early layers are stuck compensating for output-space bias)
- Implication: tied models should have their optimal stacking window pushed toward middle/late layers

**Experiment (low compute, high novelty):**
1. Take OLMo-1B (tied) and OLMo-1B-0724 (untied) — already used in the paper
2. Use the tuned lens KL profiles from `experiments/2_tuned_lens/`
3. Apply layer stacking at (a) high-KL zones, (b) steep-decline zones, (c) low-KL plateau
4. Measure benchmark performance deltas for each zone
5. Check whether the optimal zone differs between tied and untied models

**Why it's interesting:** Provides the first *mechanistic* account of why certain layers benefit from stacking, rather than post-hoc rationalization.

---

## Idea 2: Early-Layer Stacking as a Targeted Fix for Weight-Tying Pathology

**Hypothesis:** For tied models, the early-layer KL spike is a specific bottleneck: the model receives an output-biased embedding and must spend multiple layers escaping it. Stacking these early high-KL layers gives them more forward-pass iterations to do this escape work.

**Contrast with RYS:** RYS found that stacking *middle* layers helps — but RYS was applied to untied models. For tied models, the pathology is in *early* layers. This is a falsifiable architectural difference.

**Experiment:**
1. OLMo-1B (tied) vs OLMo-1B-0724 (untied)
2. Stack layers 0–4 (high-KL early zone)
3. Stack layers 6–12 (transition zone)
4. Stack layers 12–16 (low-KL late zone)
5. Compare performance gain magnitudes across tied vs untied

**Prediction:** Tied model benefits more from early-layer stacking; untied model benefits more from middle-layer stacking.

**Significance:** If confirmed: the optimal stacking strategy is architecture-dependent and diagnosable from KL profiles — practical guidance for deploying layer stacking on any pretrained model.

---

## Idea 3: Layer Stacking as an Inference-Time Analog of Gradient Scaling

**Most speculative, most theoretically interesting.**

The paper showed that scaling input gradients by 5× during *training* shifts the embedding toward input space (counteracting output-space bias). The RYS blog showed that layer stacking at *inference time* improves performance — no retraining needed.

**Hypothesis:** Layer stacking partially undoes the output-space attractor effect at inference time. Each pass through a stacked block lets the representation drift further from the output-space manifold that the embedding forced it toward.

**Measurement:**
1. Take OLMo-1B (tied)
2. Apply early-layer stacking (following Idea 2)
3. Run the embedding alignment analysis (Experiment 1) on *intermediate representations* at each layer
4. Check whether the layer-stacked model's intermediate representations shift toward *input* space vs *output* space compared to baseline

**Prediction:** Stacking early layers for tied models should show intermediate representations that more closely resemble those of untied models — as if stacking partially compensates for the gradient imbalance.

**Why it matters:** If true, this connects two seemingly unrelated papers through a single mechanism and suggests that layer stacking is not merely "adding compute" but specifically counteracting known representational biases.

---

## Idea 4: Principled Architecture — Compensatory Stacking for Weight-Tied Models

**Novel architecture proposal:**

Weight tying saves parameters (~`vocab_size × hidden_dim`) at the cost of representational efficiency in early layers. What if those saved parameters were reinvested as extra early-layer depth through layer sharing (ALBERT-style)?

```
Tokens → [Tied Embedding — output-space biased]
       → [Layer 1] × K  (stacked early layers — cheap due to weight sharing, no new params)
       → [Layers 2..N]  (normal)
       → [Tied Unembedding]
```

- Weight tying: saves V × D parameters (huge for large vocab)
- Early layer stacking: costs only K extra forward passes per token (no new parameters)
- Combined: same parameter count as untied baseline, but with compensatory early depth

**Analogy:** This is to ALBERT what RYS is to vanilla transformers — using layer repetition to get depth for free, but here motivated by the specific pathology introduced by weight tying.

**Training consideration:** Layers applied K times receive K gradient contributions in the backward pass. This might naturally amplify input-side gradients for the shared layer, partially self-correcting the output-space bias without any explicit gradient scaling intervention.

---

## Idea 5: Fixed-Point Interpretation — Unifying Both Papers

**Theoretical synthesis:**

Both papers implicitly describe the same phenomenon from different angles:

- **Weight tying** creates an explicit *fixed-point attractor* at the output distribution. The embedding space is pulled toward the manifold where `embed(x) ≈ unembed(x)`. The model must converge *from* this attractor during the forward pass.
- **Layer stacking** empirically shows that re-running the same layers improves performance — iterative refinement, which is what a system does when converging toward a fixed point.

**Unified framing:** The forward pass of a weight-tied LM is an attempt to converge from the output-space attractor (the embedding) to a contextual representation and back. Layer stacking allows more iterations of this convergence process.

Connects to:
- Universal Transformers (Dehghani et al., 2019) — learned early-exit based on convergence
- DEQ (deep equilibrium models) — explicit fixed-point formulation
- Implicit layers

**Research direction:** Design a weight-tied transformer where the number of stacking iterations is *adaptive* — layers repeat until the intermediate representation stabilizes relative to the output-space manifold. The tied embedding provides both the initialization (output-biased) and the convergence criterion (how close are we to the output manifold?).

---

## Recommended Starting Point

**Idea 1 + Idea 2 combined** are the cleanest entry point:
- KL profiles already computed in `experiments/2_tuned_lens/`
- Paired tied/untied OLMo-1B models already used throughout
- Layer stacking is a pure inference trick — implement as a wrapper around HuggingFace `generate()`, no training required

**Target deliverable:** A figure showing stacking performance delta as a function of which layer block is stacked, overlaid on the tuned lens KL curve — for both tied and untied models. If the optimal stacking zone aligns with the steep-KL-decline zone, and if this differs between tied/untied models, it's a publishable finding.

---

## Existing Infrastructure

All ideas can be tested without new training — experimental infrastructure already exists:
- Tuned lens KL curves: `experiments/2_tuned_lens/`
- Paired tied/untied OLMo-1B models: already used throughout the paper
- Benchmark evaluation: `lm-evaluation-harness` (or standalone)
- Layer stacking: pure inference trick, implement as a wrapper around HuggingFace `generate()`
