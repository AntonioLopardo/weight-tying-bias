"""CLI entrypoints for computing norms, token frequencies, and plots.

Sections
--------
- Configuration and imports
- Helper utilities
- Command implementations
- CLI wiring (argparse)
"""

import os
import json
import argparse
from collections import Counter
import gc

from tokenizer_util import (
    ensure,
    load_model_config,
    load_input_embeddings_from_repo,
    load_input_embeddings_from_olmo,
    load_input_embeddings_from_olmo_native,
    load_input_embeddings_from_olmo_local,
    load_output_embeddings_from_olmo_local,
    compute_l2_norms,
    build_parallel_freqs_for_model,
    build_distinct_bigram_counts_for_model,
    RESULTS_DIR,
    FIG_DIR,
    CONFIG_PATH,
    DATA_PATH,
    TOKEN_CACHE_DIR,
    EMBEDDING_CACHE_DIR,
    get_effective_block_size,
    hidden_gain_estimate_center_vector,
    hidden_gain_aggregate_per_token,
)



import torch
from plotting_util import (
    plot_logfreq_vs_l2,
    plot_binned_l2_vs_logfreq,
    plot_rank_vs_l2_all_models,
    plot_freq_rank_vs_mean_l2_all_models,
    plot_mean_l2_zscore_vs_logfreq_all_models,
    plot_mean_l2_vs_logfreq_all_models,
    plot_mean_l2z_vs_freq_percentile_overlay,
    plot_combined_single_model,
    plot_next_token_prob_vs_norm_for_model,
    plot_next_token_prob_vs_freq_for_model,
    plot_rankrank_probs_for_model,
    plot_rankrank_slopes_vs_center_for_model,
)




# ----------------------------------------
# Helper utilities
# ----------------------------------------
def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist (no-op if it does)."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
def _get_model_id(name: str, spec: dict) -> str:
    """Resolve a model identifier from the config spec.

    Prefers `path`, then `model`, then falls back to the name key.
    """
    return spec.get("path") or spec.get("model") or name


def _get_revision_from_spec(spec: dict) -> str | None:
    """Resolve an optional Hugging Face revision from the config spec.

    Supports either `branch` (preferred) or `revision`.
    """
    return spec.get("branch") or spec.get("revision")


def _get_tokenizer_from_spec(spec: dict) -> str | None:
    """Get optional tokenizer override from the config spec."""
    return spec.get("tokenizer")


def _iter_models(cfg: dict):
    """Yield (name, spec, model_id) for supported model classes."""
    for name, spec in cfg.items():
        if spec.get("class") not in ("huggingface", "olmo", "olmo_native", "olmo_local"):
            continue
        yield name, spec, _get_model_id(name, spec)


def _load_embeddings_for_spec(spec: dict, model_id: str):
    revision = _get_revision_from_spec(spec)
    model_class = spec.get("class", "huggingface")
    if model_class == "olmo":
        return load_input_embeddings_from_olmo(model_id, revision=revision)
    elif model_class == "olmo_native":
        return load_input_embeddings_from_olmo_native(model_id, revision=revision)
    elif model_class == "olmo_local":
        local_path = spec.get("path", model_id)
        return load_input_embeddings_from_olmo_local(local_path)
    return load_input_embeddings_from_repo(model_id, revision=revision)


def _load_norms_for_model_id(model_id: str) -> torch.Tensor:
    """Load embeddings for a repo model and return per-token L2 norms tensor."""
    emb, _ = load_input_embeddings_from_repo(model_id)
    return compute_l2_norms(emb.detach().cpu().float())


def _compute_centered_squared_norms(emb: torch.Tensor) -> torch.Tensor:
    """Compute centered squared norms: ||v - mean(V)||^2 for each embedding vector.

    Args:
        emb: Embedding matrix of shape (vocab_size, hidden_dim)

    Returns:
        Tensor of shape (vocab_size,) with centered squared norms
    """
    emb_float = emb.detach().cpu().float()
    center = emb_float.mean(dim=0, keepdim=True)  # (1, hidden_dim)
    centered = emb_float - center  # (vocab_size, hidden_dim)
    squared_norms = (centered ** 2).sum(dim=1)  # (vocab_size,)
    return squared_norms

def _load_output_embeddings_for_spec(spec: dict, model_id: str):
    """Load output (lm_head) embeddings for a model via transformers or native OLMo."""
    revision = _get_revision_from_spec(spec)
    model_class = spec.get("class", "huggingface")

    if model_class == "olmo_native":
        # Load output embeddings from native OLMo checkpoint
        from huggingface_hub import hf_hub_download  # type: ignore
        try:
            ckpt_path = hf_hub_download(repo_id=model_id, filename="rank0.pt", revision=revision)
        except Exception as e:
            raise FileNotFoundError(f"Could not download rank0.pt from {model_id}: {e}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle nested state dict
        if "model" in state and isinstance(state["model"], dict):
            inner_state = state["model"]
        else:
            inner_state = state

        # Output embedding candidates for OLMo
        emb = None
        out_candidates = [
            "transformer.ff_out.weight",  # OLMo unembedding
            "lm_head.weight",
            "model.lm_head.weight",
            "ff_out.weight",
        ]
        for key in out_candidates:
            if key in inner_state:
                emb = inner_state[key]
                print(f"[_load_output_embeddings_for_spec] Using output key '{key}' from rank0.pt")
                break

        if emb is None:
            # Fallback: look for any key with 'out' or 'head' in name
            for key in inner_state.keys():
                if ('out' in key.lower() or 'head' in key.lower()) and 'weight' in key.lower():
                    if isinstance(inner_state[key], torch.Tensor) and inner_state[key].ndim == 2:
                        emb = inner_state[key]
                        print(f"[_load_output_embeddings_for_spec] Using fallback output key '{key}' from rank0.pt")
                        break

        if emb is None or emb.ndim != 2:
            available_keys = [k for k in inner_state.keys() if isinstance(inner_state.get(k), torch.Tensor)]
            raise ValueError(f"Could not find output embedding tensor in {model_id}. Available tensor keys: {available_keys[:20]}")

        emb = emb.detach().cpu()
        meta = {
            "model_id": model_id,
            "vocab_size": emb.shape[0],
            "hidden_size": emb.shape[1],
            "revision": revision,
        }
        return emb, meta

    if model_class == "olmo_local":
        # Load output embeddings from local OLMo checkpoint
        local_path = spec.get("path", model_id)
        return load_output_embeddings_from_olmo_local(local_path)

    # Standard HuggingFace path
    try:
        ensure("transformers")
    except Exception:
        pass
    from transformers import AutoModelForCausalLM  # type: ignore
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    try:
        out = model.get_output_embeddings()
        emb = out.weight.detach().cpu()
    except Exception as e:
        raise RuntimeError(f"Failed to load output embeddings for {model_id}: {e}")
    meta = {
        "model_id": model_id,
        "vocab_size": emb.shape[0],
        "hidden_size": emb.shape[1],
        "revision": revision,
    }
    return emb, meta


def _build_freqs_for_model_id(model_id: str, *, data_path: str, use_cache: bool, cache_dir: str, revision: str | None = None, tokenizer_id: str | None = None) -> Counter:
    """Build token frequency Counter for a model using parallel tokenization."""
    return build_parallel_freqs_for_model(
        model_id,
        data_path=data_path,
        use_cache=use_cache,
        cache_dir=cache_dir,
        revision=revision,
        tokenizer_id=tokenizer_id,
    )

def _plot_compare_input_output_two_by_two(name: str, norms_input: torch.Tensor, norms_output: torch.Tensor, freq_counts: Counter, out_png_base: str, ymin=None, ymax=None) -> None:
    """Create a 2x2 figure: top row input, bottom row output; cols: raw freq vs L2, log10(freq+1) vs L2."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        try:
            ensure("matplotlib")
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    import math

    def _dense_freq(vocab_size: int, counter: Counter) -> torch.Tensor:
        f = torch.zeros(vocab_size, dtype=torch.long)
        for tid, cnt in counter.items():
            ti = int(tid)
            if 0 <= ti < vocab_size:
                f[ti] = int(cnt)
        return f

    vocab_in = int(norms_input.shape[0])
    vocab_out = int(norms_output.shape[0])
    vocab_size = min(vocab_in, vocab_out)
    norms_input = norms_input[:vocab_size].detach().cpu().float()
    norms_output = norms_output[:vocab_size].detach().cpu().float()
    freq = _dense_freq(vocab_size, freq_counts)
    mask = freq > 0
    if int(mask.sum().item()) == 0:
        return

    x_raw = freq[mask].cpu().numpy().astype(float)
    x_log = torch.log10(freq[mask].float() + 1.0).cpu().numpy()
    y_in = norms_input[mask].cpu().numpy()
    y_out = norms_output[mask].cpu().numpy()

    # Reference vertical lines for raw powers of ten
    raw_vals = x_raw
    if raw_vals.size > 0:
        min_raw = max(1.0, float(raw_vals.min()))
        max_raw = max(min_raw, float(raw_vals.max()))
        min_pow = int(math.floor(math.log10(min_raw)))
        max_pow = int(math.ceil(math.log10(max_raw)))
        pow10_raw = [10.0 ** p for p in range(min_pow, max_pow + 1)]
    else:
        pow10_raw = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Top-left: input vs raw freq
    ax = axes[0][0]
    ax.scatter(x_raw, y_in, s=2, alpha=0.25)
    for v in pow10_raw:
        ax.axvline(v, color='red', linewidth=0.6, alpha=0.8)
    ax.set_xlabel('Token frequency (raw count)')
    ax.set_ylabel('L2 norm')
    ax.set_title(f'{name}: INPUT — raw freq')
    ax.grid(alpha=0.2)

    # Top-right: input vs log10 freq
    ax = axes[0][1]
    ax.scatter(x_log, y_in, s=2, alpha=0.25)
    for v in pow10_raw:
        ax.axvline(math.log10(v + 1.0), color='red', linewidth=0.6, alpha=0.8)
    ax.set_xlabel('log10(token frequency + 1)')
    ax.set_ylabel('L2 norm')
    ax.set_title(f'{name}: INPUT — log10 freq')
    ax.grid(alpha=0.2)

    # Bottom-left: output vs raw freq
    ax = axes[1][0]
    ax.scatter(x_raw, y_out, s=2, alpha=0.25, color="tab:green")
    for v in pow10_raw:
        ax.axvline(v, color='red', linewidth=0.6, alpha=0.8)
    ax.set_xlabel('Token frequency (raw count)')
    ax.set_ylabel('L2 norm')
    ax.set_title(f'{name}: OUTPUT — raw freq')
    ax.grid(alpha=0.2)

    # Bottom-right: output vs log10 freq
    ax = axes[1][1]
    ax.scatter(x_log, y_out, s=2, alpha=0.25, color="tab:green")
    for v in pow10_raw:
        ax.axvline(math.log10(v + 1.0), color='red', linewidth=0.6, alpha=0.8)
    ax.set_xlabel('log10(token frequency + 1)')
    ax.set_ylabel('L2 norm')
    ax.set_title(f'{name}: OUTPUT — log10 freq')
    ax.grid(alpha=0.2)

    # Optional uniform y-limits
    if ymin is not None or ymax is not None:
        try:
            y_lo = float(ymin) if ymin is not None else float(min(y_in.min(), y_out.min()))
            y_hi = float(ymax) if ymax is not None else float(max(y_in.max(), y_out.max()))
            if y_hi <= y_lo:
                y_hi = y_lo + 1e-6
            for ax_all in axes.flatten():
                ax_all.set_ylim(y_lo, y_hi)
        except Exception:
            pass

    out_png = f"{out_png_base}.png"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def _build_freqs_for_model_id_by_type(model_id: str, *, data_path: str, use_cache: bool, cache_dir: str, freq_type: str, revision: str | None = None, tokenizer_id: str | None = None) -> Counter:
    """Build frequency-like stats based on freq_type: 'unigram' or 'bigram'."""
    ft = (freq_type or "unigram").lower()
    if ft == "unigram":
        return _build_freqs_for_model_id(model_id, data_path=data_path, use_cache=use_cache, cache_dir=cache_dir, revision=revision, tokenizer_id=tokenizer_id)
    elif ft == "bigram":
        return build_distinct_bigram_counts_for_model(
            model_id,
            data_path=data_path,
            use_cache=use_cache,
            cache_dir=cache_dir,
            revision=revision,
            tokenizer_id=tokenizer_id,
        )
    else:
        raise ValueError(f"Unsupported --freq-type: {freq_type}. Use 'unigram' or 'bigram'.")


# ----------------------------------------
# Command implementations
# ----------------------------------------
def cmd_setup(args):
    """Load and report the configured models."""
    cfg = load_model_config(CONFIG_PATH)
    print(f"Loaded {len(cfg)} models from config: {list(cfg.keys())}")


def cmd_embeddings(args):
    """Download and cache input embedding matrices for configured models.

    Returns
    - dict: name -> torch.Tensor of input embeddings
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = load_model_config(CONFIG_PATH)
    emb_mat_dict = {}
    for name, spec, model_id in _iter_models(cfg):
        print(f"\nProcessing: {name} -> {model_id}")
        try:
            emb, meta = _load_embeddings_for_spec(spec, model_id)
            emb_mat_dict[name] = emb
            print(meta)
        except Exception as e:
            print(f"Failed for {name}: {e}")
    return emb_mat_dict


def cmd_norms(args):
    """Compute L2 norms for input embeddings of configured models.

    Returns
    - dict: name -> torch.Tensor of per-token L2 norms
    """
    cfg = load_model_config(CONFIG_PATH)
    emb_norms_dict = {}
    for name, spec, model_id in _iter_models(cfg):
        try:
            emb, _ = _load_embeddings_for_spec(spec, model_id)
            emb_t = emb.detach().cpu().float()
            norms = compute_l2_norms(emb_t)
            emb_norms_dict[name] = norms
            print(
                f"{name}: vocab={emb_t.shape[0]}, hidden={emb_t.shape[1]}, "
                f"mean={norms.mean().item():.4f}, std={norms.std(unbiased=False).item():.4f}, "
                f"min={norms.min().item():.4f}, max={norms.max().item():.4f}"
            )
        except Exception as e:
            print(f"Failed {name}: {e}")
    return emb_norms_dict


def cmd_tokenize(args):
    """Tokenize the dataset and build token frequency Counters per model.

    Returns
    - dict: name -> Counter(token_id -> raw count)
    """
    cfg = load_model_config(CONFIG_PATH)
    token_freqs_dict = {}
    for name, spec, model_id in _iter_models(cfg):
        try:
            print(f"\nParallel tokenizing: {name} -> {model_id}")
            freqs = _build_freqs_for_model_id_by_type(
                model_id,
                data_path=DATA_PATH,
                use_cache=(not args.no_cache),
                cache_dir=TOKEN_CACHE_DIR,
                freq_type=args.freq_type,
                revision=_get_revision_from_spec(spec),
            )
            token_freqs_dict[name] = freqs
            print(f"{name}: {len(freqs)} unique token_ids; top5={freqs.most_common(5)}")
        except Exception as e:
            print(f"Failed {name}: {e}")
    return token_freqs_dict


def cmd_tokenize_bigram(args):
    """Compute distinct bigram neighbor counts per token for each model.

    Uses the union of left and right neighbors.
    """
    cfg = load_model_config(CONFIG_PATH)
    out = {}
    for name, spec, model_id in _iter_models(cfg):
        try:
            print(f"\nComputing distinct bigrams: {name} -> {model_id}")
            c = build_distinct_bigram_counts_for_model(
                model_id,
                data_path=args.data,
                use_cache=(not args.no_cache),
                cache_dir=args.cache_dir,
                revision=_get_revision_from_spec(spec),
            )
            out[name] = c
            print(f"{name}: computed distinct bigram counts for {len(c)} tokens; top5={c.most_common(5)}")
        except Exception as e:
            print(f"Failed {name}: {e}")
    return out


def cmd_plot_first(args):
    """Plot log-frequency vs L2 for the first (or specified) model."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(args.config)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return

    target_name = args.model or next(iter(cfg.keys()))
    if target_name not in cfg:
        raise ValueError(f"Model {target_name} not found in config")
    spec = cfg[target_name]
    model_id = _get_model_id(target_name, spec)

    emb, _ = _load_embeddings_for_spec(spec, model_id)
    norms = compute_l2_norms(emb.detach().cpu().float())
    token_freqs = _build_freqs_for_model_id_by_type(
        model_id,
        data_path=DATA_PATH,
        use_cache=(not args.no_cache),
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
        revision=_get_revision_from_spec(spec),
    )

    emb_norms_dict = {target_name: norms}
    token_freqs_dict = {target_name: token_freqs}
    plot_logfreq_vs_l2(target_name, cfg, emb_norms_dict, token_freqs_dict, FIG_DIR)


def cmd_plot_binned(args):
    """Plot mean L2 vs log-frequency buckets for all configured models."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    num_bins = int(args.bins)
    for name, spec, model_id in _iter_models(cfg):
        try:
            emb, _ = _load_embeddings_for_spec(spec, model_id)
            norms_t = compute_l2_norms(emb.detach().cpu().float())
            freqs = _build_freqs_for_model_id_by_type(
                model_id,
                data_path=DATA_PATH,
                use_cache=(not args.no_cache),
                cache_dir=TOKEN_CACHE_DIR,
                freq_type=args.freq_type,
                revision=_get_revision_from_spec(spec),
            )
            plot_binned_l2_vs_logfreq(name, norms_t, freqs, num_bins, FIG_DIR)
        except Exception as e:
            print(f"Skipping {name}: {e}")


def _load_norms_and_freqs(cfg, data_path: str, use_cache: bool, cache_dir: str, freq_type: str):
    """Batch load norms and frequency Counters for HF models in the config."""
    model_to_norms = {}
    model_to_freqs = {}
    for name, spec, model_id in _iter_models(cfg):
        try:
            emb, _ = _load_embeddings_for_spec(spec, model_id)
            norms_t = compute_l2_norms(emb.detach().cpu().float())
            model_to_norms[name] = norms_t
            model_to_freqs[name] = _build_freqs_for_model_id_by_type(
                model_id,
                data_path=data_path,
                use_cache=use_cache,
                cache_dir=cache_dir,
                freq_type=freq_type,
                revision=_get_revision_from_spec(spec),
            )
        except Exception as e:
            print(f"Skip {name}: {e}")
    return model_to_norms, model_to_freqs


def cmd_plot_rank_vs_l2(args):
    """Plot mean norm rank vs frequency-rank intervals for all models."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    model_to_norms, model_to_freqs = _load_norms_and_freqs(
        cfg,
        data_path=DATA_PATH,
        use_cache=(not args.no_cache),
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
    )
    out_svg = os.path.join(FIG_DIR, "all_models_rank_vs_l2.svg")
    out_png = os.path.join(FIG_DIR, "all_models_rank_vs_l2.png")
    plot_rank_vs_l2_all_models(model_to_freqs, model_to_norms, out_svg, out_png, num_bins=int(args.bins), log_bins=bool(args.log_bins), x_log_scale=bool(args.x_log))


def cmd_plot_rank_vs_l2_raw(args):
    """Plot mean raw L2 norm vs frequency-rank intervals for all models."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    model_to_norms, model_to_freqs = _load_norms_and_freqs(
        cfg,
        data_path=DATA_PATH,
        use_cache=(not args.no_cache),
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
    )
    out_svg = os.path.join(FIG_DIR, "all_models_rank_vs_l2_raw.svg")
    out_png = os.path.join(FIG_DIR, "all_models_rank_vs_l2_raw.png")
    plot_freq_rank_vs_mean_l2_all_models(
        model_to_freqs,
        model_to_norms,
        out_svg,
        out_png,
        num_bins=int(args.bins),
        log_bins=bool(args.log_bins),
        x_log_scale=bool(args.x_log),
    )


def cmd_plot_rank_vs_l2_by_family(args):
    """Plot mean L2 norm rank vs frequency-rank intervals, per family."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    model_to_norms, model_to_freqs = _load_norms_and_freqs(
        cfg,
        data_path=DATA_PATH,
        use_cache=(not args.no_cache),
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
    )

    name_to_family = _load_family_spec()
    if not name_to_family:
        print("No family spec found or all families empty; nothing to plot.")
        return

    family_groups = {}
    for name, family in name_to_family.items():
        if name in model_to_norms and name in model_to_freqs:
            g = family_groups.setdefault(family, {"norms": {}, "freqs": {}})
            g["norms"][name] = model_to_norms[name]
            g["freqs"][name] = model_to_freqs[name]

    if not family_groups:
        print("No overlapping models between family spec and loaded models; nothing to plot.")
        return

    for family, group in sorted(family_groups.items()):
        norms = group["norms"]
        freqs = group["freqs"]
        if not norms or not freqs:
            continue
        safe_family = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in family)
        out_svg = os.path.join(FIG_DIR, f"all_models_rank_vs_l2_family_{safe_family}.svg")
        out_png = os.path.join(FIG_DIR, f"all_models_rank_vs_l2_family_{safe_family}.png")
        print(f"Plotting rank-vs-L2 for family '{family}' with {len(norms)} models -> {os.path.basename(out_png)}")
        try:
            plot_rank_vs_l2_all_models(
                freqs,
                norms,
                out_svg,
                out_png,
                num_bins=int(args.bins),
                log_bins=bool(args.log_bins),
                x_log_scale=bool(args.x_log),
            )
        except Exception as e:
            print(f"Skipping family {family}: {e}")


def cmd_plot_rank_vs_l2_raw_by_family(args):
    """Plot mean raw L2 norm vs frequency-rank intervals, per family."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    model_to_norms, model_to_freqs = _load_norms_and_freqs(
        cfg,
        data_path=DATA_PATH,
        use_cache=(not args.no_cache),
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
    )

    name_to_family = _load_family_spec()
    if not name_to_family:
        print("No family spec found or all families empty; nothing to plot.")
        return

    family_groups = {}
    for name, family in name_to_family.items():
        if name in model_to_norms and name in model_to_freqs:
            g = family_groups.setdefault(family, {"norms": {}, "freqs": {}})
            g["norms"][name] = model_to_norms[name]
            g["freqs"][name] = model_to_freqs[name]

    if not family_groups:
        print("No overlapping models between family spec and loaded models; nothing to plot.")
        return

    for family, group in sorted(family_groups.items()):
        norms = group["norms"]
        freqs = group["freqs"]
        if not norms or not freqs:
            continue
        safe_family = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in family)
        out_svg = os.path.join(FIG_DIR, f"all_models_rank_vs_l2_raw_family_{safe_family}.svg")
        out_png = os.path.join(FIG_DIR, f"all_models_rank_vs_l2_raw_family_{safe_family}.png")
        print(f"Plotting rank-vs-L2 RAW for family '{family}' with {len(norms)} models -> {os.path.basename(out_png)}")
        try:
            plot_freq_rank_vs_mean_l2_all_models(
                freqs,
                norms,
                out_svg,
                out_png,
                num_bins=int(args.bins),
                log_bins=bool(args.log_bins),
                x_log_scale=bool(args.x_log),
            )
        except Exception as e:
            print(f"Skipping family {family}: {e}")

def cmd_plot_zscore_vs_logfreq(args):
    """Plot mean L2 z-score vs log-frequency across models."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    model_to_norms, model_to_freqs = _load_norms_and_freqs(
        cfg,
        data_path=DATA_PATH,
        use_cache=(not args.no_cache),
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
    )
    out_svg = os.path.join(FIG_DIR, "mean_l2_zscore_vs_logfreq.svg")
    out_png = os.path.join(FIG_DIR, "mean_l2_zscore_vs_logfreq.png")
    plot_mean_l2_zscore_vs_logfreq_all_models(model_to_freqs, model_to_norms, out_svg, out_png, num_bins=int(args.bins))


def cmd_plot_l2_vs_logfreq(args):
    """Plot mean L2 norm vs log-frequency across models."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    model_to_norms, model_to_freqs = _load_norms_and_freqs(
        cfg,
        data_path=DATA_PATH,
        use_cache=(not args.no_cache),
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
    )
    out_svg = os.path.join(FIG_DIR, "mean_l2_vs_logfreq.svg")
    out_png = os.path.join(FIG_DIR, "mean_l2_vs_logfreq.png")
    plot_mean_l2_vs_logfreq_all_models(model_to_freqs, model_to_norms, out_svg, out_png, num_bins=int(args.bins))


def _load_family_spec() -> dict:
    """Load name->family mapping from tok_config_specific_families.json.

    Returns a mapping of model name to family string. Missing or empty families are omitted.
    """
    here = os.path.dirname(__file__)
    path = os.path.join(here, "tok_config_specific_families.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    out = {}
    if isinstance(raw, dict):
        for name, spec in raw.items():
            fam = None
            try:
                fam = (spec or {}).get("family")
            except Exception:
                fam = None
            if isinstance(fam, str) and fam.strip():
                out[name] = fam.strip()
    return out


def cmd_plot_l2_vs_logfreq_by_family(args):
    """Plot mean L2 norm vs log-frequency for each family in the family spec."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    model_to_norms, model_to_freqs = _load_norms_and_freqs(
        cfg,
        data_path=DATA_PATH,
        use_cache=(not args.no_cache),
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
    )

    name_to_family = _load_family_spec()
    if not name_to_family:
        print("No family spec found or all families empty; nothing to plot.")
        return

    # Build family -> {name->tensor} groupings intersecting available models
    family_groups = {}
    for name, family in name_to_family.items():
        if name in model_to_norms and name in model_to_freqs:
            g = family_groups.setdefault(family, {"norms": {}, "freqs": {}})
            g["norms"][name] = model_to_norms[name]
            g["freqs"][name] = model_to_freqs[name]

    if not family_groups:
        print("No overlapping models between family spec and loaded models; nothing to plot.")
        return

    # Emit one figure per family
    for family, group in sorted(family_groups.items()):
        norms = group["norms"]
        freqs = group["freqs"]
        if not norms or not freqs:
            continue
        safe_family = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in family)
        out_svg = os.path.join(FIG_DIR, f"mean_l2_vs_logfreq_family_{safe_family}.svg")
        out_png = os.path.join(FIG_DIR, f"mean_l2_vs_logfreq_family_{safe_family}.png")
        print(f"Plotting family '{family}' with {len(norms)} models -> {os.path.basename(out_png)}")
        try:
            plot_mean_l2_vs_logfreq_all_models(freqs, norms, out_svg, out_png, num_bins=int(args.bins))
        except Exception as e:
            print(f"Skipping family {family}: {e}")

def cmd_plot_zscore_vs_percentile(args):
    """Plot mean L2 z-score vs frequency percentile across models."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    model_to_norms, model_to_freqs = _load_norms_and_freqs(
        cfg,
        data_path=DATA_PATH,
        use_cache=(not args.no_cache),
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
    )
    out_svg = os.path.join(FIG_DIR, "all_models_mean_l2z_vs_freq_percentile_overlay.svg")
    out_png = os.path.join(FIG_DIR, "all_models_mean_l2z_vs_freq_percentile_overlay.png")
    plot_mean_l2z_vs_freq_percentile_overlay(model_to_freqs, model_to_norms, out_svg, out_png, num_quantiles=int(args.quantiles))


def cmd_plot_combined_single(args):
    """Generate combined diagnostic plot for a single model."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    name = args.model or next(iter(cfg.keys()))
    spec = cfg[name]
    model_id = _get_model_id(name, spec)
    if getattr(args, "embedding_kind", "input") == "output":
        # Special path: compare INPUT vs OUTPUT in a 2x2; disable cache for frequencies
        emb_in, _ = _load_embeddings_for_spec(spec, model_id)
        emb_out, _ = _load_output_embeddings_for_spec(spec, model_id)
        norms_in = compute_l2_norms(emb_in.detach().cpu().float())
        norms_out = compute_l2_norms(emb_out.detach().cpu().float())
        freqs = _build_freqs_for_model_id_by_type(
            model_id,
            data_path=DATA_PATH,
            use_cache=False,  # ignore cache when comparing output embeddings
            cache_dir=TOKEN_CACHE_DIR,
            freq_type=args.freq_type,
            revision=_get_revision_from_spec(spec),
        )
        base = os.path.join(FIG_DIR, f"{name}_input_vs_output_two_by_two")
        _plot_compare_input_output_two_by_two(
            name,
            norms_in,
            norms_out,
            freqs,
            base,
            ymin=args.ymin,
            ymax=args.ymax,
        )
    else:
        emb, _ = _load_embeddings_for_spec(spec, model_id)
        norms_t = compute_l2_norms(emb.detach().cpu().float())
        freqs = _build_freqs_for_model_id_by_type(
            model_id,
            data_path=DATA_PATH,
            use_cache=(not args.no_cache),
            cache_dir=TOKEN_CACHE_DIR,
            freq_type=args.freq_type,
            revision=_get_revision_from_spec(spec),
        )
        try:
            total_tokens = int(sum(freqs.values()))
        except Exception:
            total_tokens = -1
        print(
            f"Tokenization complete for {name}: {len(freqs)} unique token_ids; "
            f"total_tokens={total_tokens}; top5={freqs.most_common(5)}"
        )
        base = os.path.join(FIG_DIR, f"{name}_combined_pow10_and_deciles" + ("" if args.include_root else "_no_root"))
        plot_combined_single_model(
            name,
            norms_t,
            freqs,
            base,
            include_root=bool(args.include_root),
            ymin=args.ymin,
            ymax=args.ymax,
        )


def cmd_plot_combined_all(args):
    """Generate combined diagnostic plots for all configured models."""
    _ensure_dir(FIG_DIR)
    config_path = args.config or CONFIG_PATH
    cfg = load_model_config(config_path)
    for name, spec, model_id in _iter_models(cfg):
        try:
            tokenizer_id = _get_tokenizer_from_spec(spec)
            if getattr(args, "embedding_kind", "input") == "output":
                # Compare INPUT vs OUTPUT with fresh frequencies (no cache)
                emb_in, _ = _load_embeddings_for_spec(spec, model_id)
                emb_out, _ = _load_output_embeddings_for_spec(spec, model_id)
                norms_in = compute_l2_norms(emb_in.detach().cpu().float())
                norms_out = compute_l2_norms(emb_out.detach().cpu().float())
                freqs = _build_freqs_for_model_id_by_type(
                    model_id,
                    data_path=DATA_PATH,
                    use_cache=False,  # ignore cache when comparing output embeddings
                    cache_dir=TOKEN_CACHE_DIR,
                    freq_type=args.freq_type,
                    revision=_get_revision_from_spec(spec),
                    tokenizer_id=tokenizer_id,
                )
                base = os.path.join(FIG_DIR, f"{name}_input_vs_output_two_by_two")
                _plot_compare_input_output_two_by_two(
                    name,
                    norms_in,
                    norms_out,
                    freqs,
                    base,
                    ymin=args.ymin,
                    ymax=args.ymax,
                )
            else:
                emb, _ = _load_embeddings_for_spec(spec, model_id)
                norms_t = compute_l2_norms(emb.detach().cpu().float())
                freqs = _build_freqs_for_model_id_by_type(
                    model_id,
                    data_path=DATA_PATH,
                    use_cache=(not args.no_cache),
                    cache_dir=TOKEN_CACHE_DIR,
                    freq_type=args.freq_type,
                    tokenizer_id=tokenizer_id,
                )
                base = os.path.join(FIG_DIR, f"{name}_combined_pow10_and_deciles" + ("" if args.include_root else "_no_root"))
                plot_combined_single_model(
                    name,
                    norms_t,
                    freqs,
                    base,
                    include_root=bool(args.include_root),
                    ymin=args.ymin,
                    ymax=args.ymax,
                )
        except Exception as e:
            print(f"Skipping {name}: {e}")


def cmd_plot_combined_all_centered(args):
    """Generate combined diagnostic plots for all configured models using centered squared norms.

    Instead of L2 norms, computes ||v - mean(V)||^2 where mean(V) is the mean embedding vector.
    """
    _ensure_dir(FIG_DIR)
    config_path = args.config or CONFIG_PATH
    cfg = load_model_config(config_path)
    for name, spec, model_id in _iter_models(cfg):
        try:
            if getattr(args, "embedding_kind", "input") == "output":
                # Compare INPUT vs OUTPUT with fresh frequencies (no cache)
                emb_in, _ = _load_embeddings_for_spec(spec, model_id)
                emb_out, _ = _load_output_embeddings_for_spec(spec, model_id)
                norms_in = _compute_centered_squared_norms(emb_in)
                norms_out = _compute_centered_squared_norms(emb_out)
                freqs = _build_freqs_for_model_id_by_type(
                    model_id,
                    data_path=DATA_PATH,
                    use_cache=False,  # ignore cache when comparing output embeddings
                    cache_dir=TOKEN_CACHE_DIR,
                    freq_type=args.freq_type,
                    revision=_get_revision_from_spec(spec),
                )
                base = os.path.join(FIG_DIR, f"{name}_input_vs_output_two_by_two_centered")
                _plot_compare_input_output_two_by_two(
                    name,
                    norms_in,
                    norms_out,
                    freqs,
                    base,
                    ymin=args.ymin,
                    ymax=args.ymax,
                )
            else:
                emb, _ = _load_embeddings_for_spec(spec, model_id)
                norms_t = _compute_centered_squared_norms(emb)
                freqs = _build_freqs_for_model_id_by_type(
                    model_id,
                    data_path=DATA_PATH,
                    use_cache=(not args.no_cache),
                    cache_dir=TOKEN_CACHE_DIR,
                    freq_type=args.freq_type,
                )
                base = os.path.join(FIG_DIR, f"{name}_combined_pow10_and_deciles_centered" + ("" if args.include_root else "_no_root"))
                plot_combined_single_model(
                    name,
                    norms_t,
                    freqs,
                    base,
                    include_root=bool(args.include_root),
                    ymin=args.ymin,
                    ymax=args.ymax,
                )
        except Exception as e:
            print(f"Skipping {name}: {e}")


def cmd_plot_next_probs(args):
    """Thin wrapper: delegate to plotting utility for next-probability plot."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return

    target_name = args.model or next(iter(cfg.keys()))
    if target_name not in cfg:
        raise ValueError(f"Model {target_name} not found in config")
    spec = cfg[target_name]
    model_id = _get_model_id(target_name, spec)

    out_png, out_svg = plot_next_token_prob_vs_norm_for_model(
        target_name,
        model_id,
        data_path=DATA_PATH,
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
        revision=_get_revision_from_spec(spec),
        seeds=int(args.seeds),
        seq_length=int(args.seq_length),
        noise_scale=float(args.noise_scale),
        seed=int(args.seed),
        noise_mode=str(args.noise_mode),
        rank_center=int(args.rank_center),
        rank_width=int(args.rank_width),
        out_dir=FIG_DIR,
        avg_probs_file=args.avg_probs_file,
        freq_dataset_dir=getattr(args, "freq_dataset_dir", None),
    )
    print(f"Saved figure to: {out_png} and {out_svg}")


def cmd_plot_next_probs_all(args):
    """Run next-probability plots for all configured models using the utility."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return

    for target_name, spec, model_id in _iter_models(cfg):
        try:
            print(f"\n[All] Processing model: {target_name} -> {model_id}")
            out_png, out_svg = plot_next_token_prob_vs_norm_for_model(
                target_name,
                model_id,
                data_path=DATA_PATH,
                cache_dir=TOKEN_CACHE_DIR,
                freq_type=args.freq_type,
                revision=_get_revision_from_spec(spec),
                seeds=int(args.seeds),
                seq_length=int(args.seq_length),
                noise_scale=float(args.noise_scale),
                seed=int(args.seed),
                noise_mode=str(args.noise_mode),
                rank_center=int(args.rank_center),
                rank_width=int(args.rank_width),
                out_dir=FIG_DIR,
                avg_probs_file=args.avg_probs_file,
                freq_dataset_dir=getattr(args, "freq_dataset_dir", None),
            )
            print(f"[All] Saved for {target_name}: {out_png} and {out_svg}")
        except Exception as e:
            print(f"[All] Skipping {target_name}: {e}")


def cmd_plot_next_probs_freq(args):
    """Plot next-token probability vs token frequency for a single model."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return

    target_name = args.model or next(iter(cfg.keys()))
    if target_name not in cfg:
        raise ValueError(f"Model {target_name} not found in config")
    spec = cfg[target_name]
    model_id = _get_model_id(target_name, spec)

    out_png, out_svg = plot_next_token_prob_vs_freq_for_model(
        target_name,
        model_id,
        data_path=DATA_PATH,
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
        revision=_get_revision_from_spec(spec),
        seeds=int(args.seeds),
        seq_length=int(args.seq_length),
        noise_scale=float(args.noise_scale),
        seed=int(args.seed),
        noise_mode=str(args.noise_mode),
        rank_center=int(args.rank_center),
        rank_width=int(args.rank_width),
        out_dir=FIG_DIR,
        avg_probs_file=args.avg_probs_file,
        freq_dataset_dir=getattr(args, "freq_dataset_dir", None),
    )
    print(f"Saved figure to: {out_png} and {out_svg}")


def cmd_plot_next_probs_freq_all(args):
    """Run next-probability vs log-frequency plots for all configured models."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return

    for target_name, spec, model_id in _iter_models(cfg):
        try:
            print(f"\n[All] Processing model: {target_name} -> {model_id}")
            out_png, out_svg = plot_next_token_prob_vs_freq_for_model(
                target_name,
                model_id,
                data_path=DATA_PATH,
                cache_dir=TOKEN_CACHE_DIR,
                freq_type=args.freq_type,
                revision=_get_revision_from_spec(spec),
                seeds=int(args.seeds),
                seq_length=int(args.seq_length),
                noise_scale=float(args.noise_scale),
                seed=int(args.seed),
                noise_mode=str(args.noise_mode),
                rank_center=int(args.rank_center),
                rank_width=int(args.rank_width),
                out_dir=FIG_DIR,
                avg_probs_file=args.avg_probs_file,
                freq_dataset_dir=getattr(args, "freq_dataset_dir", None),
            )
            print(f"[All] Saved for {target_name}: {out_png} and {out_svg}")
        except Exception as e:
            print(f"[All] Skipping {target_name}: {e}")


def cmd_report_topfreq(args):
    """Report models where mean L2(top-K frequent tokens) > mean L2(rest)."""
    cfg = load_model_config(args.config)
    topk = int(args.topk)
    qualifying = []

    for name, spec, model_id in _iter_models(cfg):
        try:
            emb, _ = _load_embeddings_for_spec(spec, model_id)
            norms_t = compute_l2_norms(emb.detach().cpu().float())
            freqs = _build_freqs_for_model_id(model_id, data_path=args.data, use_cache=(not args.no_cache), cache_dir=args.cache_dir)

            top_items = freqs.most_common(topk)
            top_ids = [token_id for token_id, _ in top_items if 0 <= token_id < norms_t.shape[0]]

            if len(top_ids) == 0 or len(top_ids) >= norms_t.shape[0]:
                print(f"Skipping {name}: invalid top-{topk} selection relative to vocab size {norms_t.shape[0]}")
                continue

            top_index_tensor = torch.tensor(top_ids, dtype=torch.long)
            top_mean = norms_t[top_index_tensor].mean().item()

            vocab_size = norms_t.shape[0]
            rest_mask = torch.ones(vocab_size, dtype=torch.bool)
            rest_mask[top_index_tensor] = False
            rest_vals = norms_t[rest_mask]
            if rest_vals.numel() == 0:
                print(f"Skipping {name}: no remaining tokens after excluding top-{topk}")
                continue
            rest_mean = rest_vals.mean().item()

            if top_mean > rest_mean:
                qualifying.append((name, top_mean, rest_mean))
        except Exception as e:
            print(f"Skipping {name}: {e}")

    if len(qualifying) == 0:
        print(f"No models satisfy: mean(norms of top-{topk} most frequent) > mean(norms of rest)")
        return

    print(f"Models with mean L2(top-{topk}) > mean L2(rest):")
    for name, top_mean, rest_mean in qualifying:
        print(f"{name}: top{topk}_mean={top_mean:.4f}, rest_mean={rest_mean:.4f}")

def cmd_plot_rankrank_probs(args):
    """Thin wrapper: delegate to plotting utility for rank-rank probability plot."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return

    target_name = args.model or next(iter(cfg.keys()))
    if target_name not in cfg:
        raise ValueError(f"Model {target_name} not found in config")
    spec = cfg[target_name]
    model_id = _get_model_id(target_name, spec)

    log_path = args.log_file or os.path.join(FIG_DIR, "rankrank_slopes.log")
    out_png, out_svg = plot_rankrank_probs_for_model(
        target_name,
        model_id,
        data_path=DATA_PATH,
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
        revision=_get_revision_from_spec(spec),
        seeds=int(args.seeds),
        seq_length=int(args.seq_length),
        noise_scale=float(args.noise_scale),
        seed=int(args.seed),
        noise_mode=str(args.noise_mode),
        rank_center=int(args.rank_center),
        rank_width=int(args.rank_width),
        out_dir=FIG_DIR,
        log_file=log_path,
        sweep_start=args.rank_sweep_start,
        sweep_end=args.rank_sweep_end,
        sweep_step=int(args.rank_sweep_step),
        avg_probs_file=args.avg_probs_file,
        freq_dataset_dir=getattr(args, "freq_dataset_dir", None),
    )
    print(f"Saved figure to: {out_png} and {out_svg}")

def cmd_plot_rankrank_probs_all(args):
    """Run rank-rank probability plots for all configured models using the utility."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return

    for target_name, spec, model_id in _iter_models(cfg):
        try:
            print(f"\n[All] Processing model: {target_name} -> {model_id}")
            log_path = args.log_file or os.path.join(FIG_DIR, "rankrank_slopes.log")
            out_png, out_svg = plot_rankrank_probs_for_model(
                target_name,
                model_id,
                data_path=DATA_PATH,
                cache_dir=TOKEN_CACHE_DIR,
                freq_type=args.freq_type,
                revision=_get_revision_from_spec(spec),
                seeds=int(args.seeds),
                seq_length=int(args.seq_length),
                noise_scale=float(args.noise_scale),
                seed=int(args.seed),
                noise_mode=str(args.noise_mode),
                rank_center=int(args.rank_center),
                rank_width=int(args.rank_width),
                out_dir=FIG_DIR,
                log_file=log_path,
                sweep_start=args.rank_sweep_start,
                sweep_end=args.rank_sweep_end,
                sweep_step=int(args.rank_sweep_step),
                avg_probs_file=args.avg_probs_file,
                freq_dataset_dir=getattr(args, "freq_dataset_dir", None),
            )
            print(f"[All] Saved for {target_name}: {out_png} and {out_svg}")
        except Exception as e:
            print(f"[All] Skipping {target_name}: {e}")

def cmd_plot_rankrank_slopes(args):
    """Compute slope vs center plot for a single model over a range of centers."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return
    if args.rank_sweep_start is None or args.rank_sweep_end is None:
        raise ValueError("--rank-sweep-start and --rank-sweep-end are required for plot-rankrank-slopes")

    target_name = args.model or next(iter(cfg.keys()))
    if target_name not in cfg:
        raise ValueError(f"Model {target_name} not found in config")
    spec = cfg[target_name]
    model_id = _get_model_id(target_name, spec)

    log_path = args.log_file or os.path.join(FIG_DIR, "rankrank_slopes.log")
    out_png, out_tsv = plot_rankrank_slopes_vs_center_for_model(
        target_name,
        model_id,
        data_path=DATA_PATH,
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
        revision=_get_revision_from_spec(spec),
        seeds=int(args.seeds),
        seq_length=int(args.seq_length),
        noise_scale=float(args.noise_scale),
        seed=int(args.seed),
        noise_mode=str(args.noise_mode),
        sweep_start=int(args.rank_sweep_start),
        sweep_end=int(args.rank_sweep_end),
        sweep_step=int(args.rank_sweep_step),
        rank_width=int(args.rank_width),
        out_dir=FIG_DIR,
        log_file=log_path,
        avg_probs_file=args.avg_probs_file,
        freq_dataset_dir=getattr(args, "freq_dataset_dir", None),
        fit_on_values=bool(getattr(args, "fit_on_values", False)),
    )
    print(f"Saved slope-vs-center figure to: {out_png}")
    print(f"Wrote slopes to TSV: {out_tsv}")

def cmd_plot_rankrank_slopes_all(args):
    """Compute slope vs center plots for all configured models (rank vs rank by default)."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return
    if args.rank_sweep_start is None or args.rank_sweep_end is None:
        raise ValueError("--rank-sweep-start and --rank-sweep-end are required for plot-rankrank-slopes-all")

    for target_name, spec, model_id in _iter_models(cfg):
        try:
            print(f"\n[All] Processing model: {target_name} -> {model_id}")
            log_path = args.log_file or os.path.join(FIG_DIR, "rankrank_slopes.log")
            out_png, out_tsv = plot_rankrank_slopes_vs_center_for_model(
                target_name,
                model_id,
                data_path=DATA_PATH,
                cache_dir=TOKEN_CACHE_DIR,
                freq_type=args.freq_type,
                revision=_get_revision_from_spec(spec),
                seeds=int(args.seeds),
                seq_length=int(args.seq_length),
                noise_scale=float(args.noise_scale),
                seed=int(args.seed),
                noise_mode=str(args.noise_mode),
                sweep_start=int(args.rank_sweep_start),
                sweep_end=int(args.rank_sweep_end),
                sweep_step=int(args.rank_sweep_step),
                rank_width=int(args.rank_width),
                out_dir=FIG_DIR,
                log_file=log_path,
                avg_probs_file=args.avg_probs_file,
                freq_dataset_dir=getattr(args, "freq_dataset_dir", None),
                fit_on_values=bool(getattr(args, "fit_on_values", False)),
            )
            print(f"[All] Saved slopes for {target_name}: {out_png} and {out_tsv}")
        except Exception as e:
            print(f"[All] Skipping {target_name}: {e}")

def cmd_plot_rankrank_slopes_logprob(args):
    """Like plot-rankrank-slopes, but fit on (norm, log10(prob)) within the window."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return
    if args.rank_sweep_start is None or args.rank_sweep_end is None:
        raise ValueError("--rank-sweep-start and --rank-sweep-end are required for plot-rankrank-slopes-logprob")

    target_name = args.model or next(iter(cfg.keys()))
    if target_name not in cfg:
        raise ValueError(f"Model {target_name} not found in config")
    spec = cfg[target_name]
    model_id = _get_model_id(target_name, spec)

    log_path = args.log_file or os.path.join(FIG_DIR, "rankrank_slopes.log")
    out_png, out_tsv = plot_rankrank_slopes_vs_center_for_model(
        target_name,
        model_id,
        data_path=DATA_PATH,
        cache_dir=TOKEN_CACHE_DIR,
        freq_type=args.freq_type,
        revision=_get_revision_from_spec(spec),
        seeds=int(args.seeds),
        seq_length=int(args.seq_length),
        noise_scale=float(args.noise_scale),
        seed=int(args.seed),
        noise_mode=str(args.noise_mode),
        sweep_start=int(args.rank_sweep_start),
        sweep_end=int(args.rank_sweep_end),
        sweep_step=int(args.rank_sweep_step),
        rank_width=int(args.rank_width),
        out_dir=FIG_DIR,
        log_file=log_path,
        avg_probs_file=args.avg_probs_file,
        freq_dataset_dir=getattr(args, "freq_dataset_dir", None),
        fit_on_values=True,
        log_prob=True,
    )
    print(f"Saved slope-vs-center (logprob) figure to: {out_png}")
    print(f"Wrote slopes (logprob) to TSV: {out_tsv}")

def cmd_plot_rankrank_slopes_logprob_all(args):
    """Compute slope vs center plots using (norm, log10(prob)) for all models."""
    _ensure_dir(FIG_DIR)
    cfg = load_model_config(CONFIG_PATH)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return
    if args.rank_sweep_start is None or args.rank_sweep_end is None:
        raise ValueError("--rank-sweep-start and --rank-sweep-end are required for plot-rankrank-slopes-logprob-all")

    for target_name, spec, model_id in _iter_models(cfg):
        try:
            print(f"\n[All] Processing model (logprob): {target_name} -> {model_id}")
            log_path = args.log_file or os.path.join(FIG_DIR, "rankrank_slopes.log")
            out_png, out_tsv = plot_rankrank_slopes_vs_center_for_model(
                target_name,
                model_id,
                data_path=DATA_PATH,
                cache_dir=TOKEN_CACHE_DIR,
                freq_type=args.freq_type,
                revision=_get_revision_from_spec(spec),
                seeds=int(args.seeds),
                seq_length=int(args.seq_length),
                noise_scale=float(args.noise_scale),
                seed=int(args.seed),
                noise_mode=str(args.noise_mode),
                sweep_start=int(args.rank_sweep_start),
                sweep_end=int(args.rank_sweep_end),
                sweep_step=int(args.rank_sweep_step),
                rank_width=int(args.rank_width),
                out_dir=FIG_DIR,
                log_file=log_path,
                avg_probs_file=args.avg_probs_file,
                freq_dataset_dir=getattr(args, "freq_dataset_dir", None),
                fit_on_values=True,
                log_prob=True,
            )
            print(f"[All] Saved slopes (logprob) for {target_name}: {out_png} and {out_tsv}")
        except Exception as e:
            print(f"[All] Skipping {target_name}: {e}")

def cmd_plot_hidden_gain(args):
    """Reproduce the notebook analysis: hidden-state mean squared norm vs information content.

    - Streams a text file, tokenizes into blocks, runs the model to get last-layer hidden states
    - Optionally mean-centers by subtracting a global mean vector estimated from a fraction of tokens
    - Aggregates per-token frequency and mean squared norm
    - Computes information content = ln(N_total / freq)
    - Saves aggregates and produces a two-panel scatter plot:
        [mean squared norm vs info content] and [log10(freq) vs mean squared norm] with linear fits
    """
    # Lazily ensure runtime deps
    try:
        ensure("numpy")
    except Exception:
        pass
    try:
        ensure("pandas")
    except Exception:
        pass
    try:
        ensure("matplotlib")
    except Exception:
        pass
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    _ensure_dir(FIG_DIR)
    config_path = args.config or CONFIG_PATH
    cfg = load_model_config(config_path)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return

    target_name = args.model or next(iter(cfg.keys()))
    if target_name not in cfg:
        raise ValueError(f"Model {target_name} not found in config")
    spec = cfg[target_name]
    model_id = _get_model_id(target_name, spec)
    revision = _get_revision_from_spec(spec)

    data_path = DATA_PATH
    out_dir = os.path.join(FIG_DIR, "hidden_gain")
    _ensure_dir(out_dir)
    safe_model = model_id.replace("/", "_")

    # Hardcoded settings to match notebook
    max_tokens = 1_000_000
    block_size = 4096
    batch_size = 4
    center_fraction = 0.1
    topk = 10000

    print(f"Reproducing hidden-gain analysis for: {target_name} ({model_id})")
    print(f"Data: {data_path}")
    print(f"Max tokens: {max_tokens}, block_size: {block_size}, batch_size: {batch_size}, center_fraction: {center_fraction}")

    model, tok, device = _load_model_and_tokenizer_for_hidden_gain(spec, model_id, revision)
    model.config.output_hidden_states = True
    model.eval()
    if not torch.cuda.is_available():
        model = model.to(device)

    # Use shared utility to respect model max sequence length
    effective_block = get_effective_block_size(model, block_size)
    if effective_block < block_size:
        print(f"Capping block size from {block_size} to model max {effective_block}")
    else:
        print(f"Block size: {effective_block}")

    # Estimate center vector on a subset (fraction of max_tokens)
    center_vec = None
    if max_tokens > 0 and center_fraction > 0:
        target_center_tokens = max(1, int(max_tokens * center_fraction))
        center_vec = hidden_gain_estimate_center_vector(
            model,
            tok,
            data_path,
            effective_block,
            batch_size,
            target_center_tokens,
        )
        if center_vec is not None:
            print(f"Computed center_vec (target {target_center_tokens}). Shape: {tuple(center_vec.shape)}")
        else:
            print("Centering skipped (no tokens counted).")

    # Aggregate per token using shared utility
    agg_df, meta = hidden_gain_aggregate_per_token(
        model,
        tok,
        data_path,
        effective_block,
        batch_size,
        max_tokens,
        center_vec=center_vec,
        topk_for_corr=topk,
    )
    if int(meta.get("total_count", 0)) == 0:
        print("No tokens processed; nothing to plot.")
        return
    print(f"Processed tokens: {int(meta.get('total_count', 0))}; unique token_ids: {int(meta.get('unique_tokens', 0))}")

    # Correlation on top-K by frequency
    agg_sorted = agg_df.sort_values("freq", ascending=False)
    top_k_df = agg_sorted.head(topk).reset_index(drop=True)
    corr = top_k_df[["mean_squared_norm", "info_content"]].corr().iloc[0, 1]
    print(f"Pearson r(mean_squared_norm, info_content) [top {len(top_k_df)} by freq]: {corr:.4f}")

    # Save aggregates
    agg_out_path_parquet = os.path.join(out_dir, f"{safe_model}_token_aggregates_ic.parquet")
    agg_out_path_csv = os.path.join(out_dir, f"{safe_model}_token_aggregates_ic.csv")
    saved_parquet = False
    try:
        agg_df.to_parquet(agg_out_path_parquet, index=False)
        print(f"Wrote aggregates (parquet): {agg_out_path_parquet}")
        saved_parquet = True
    except Exception as e:
        print(f"Parquet save failed ({e}); writing CSV instead.")
    if not saved_parquet:
        agg_df.to_csv(agg_out_path_csv, index=False)
        print(f"Wrote aggregates (csv): {agg_out_path_csv}")

    # Plots
    x_norm = top_k_df["mean_squared_norm"].astype(float).to_numpy()
    y_ic = top_k_df["info_content"].astype(float).to_numpy()
    y_log10 = np.log10(top_k_df["freq"].astype(float).to_numpy())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    try:
        fig.suptitle(f"{target_name} — {model_id}", fontsize=12)
    except Exception:
        pass
    # Norm vs IC
    axes[0].scatter(x_norm, y_ic, s=6, alpha=0.3)
    m_ic, b_ic = np.polyfit(x_norm, y_ic, 1)
    # Pearson correlation (Norm vs IC)
    corr_ic = float(np.corrcoef(x_norm, y_ic)[0, 1]) if x_norm.size and y_ic.size else float("nan")
    x_line = np.linspace(x_norm.min(), x_norm.max(), 200)
    axes[0].plot(x_line, m_ic * x_line + b_ic, color="crimson", linewidth=2, label=f"fit: y={m_ic:.3g}x+{b_ic:.3g}")
    axes[0].set_xlabel("Centered Squared Norms of Last Layer activations")
    axes[0].set_ylabel("Information content (ln frequency)")
    axes[0].set_title(f"Norm vs IC (top {len(top_k_df)} tokens)")
    axes[0].text(
        0.02,
        0.98,
        f"r = {corr_ic:.3f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=(1, 1, 1, 0.8), edgecolor="none"),
    )
    axes[0].legend(frameon=False)
    # log10(freq) vs Norm
    x_log = y_log10
    y_norm = x_norm
    axes[1].scatter(x_log, y_norm, s=6, alpha=0.3, color="tab:blue")
    m_flip, b_flip = np.polyfit(x_log, y_norm, 1)
    # Pearson correlation (log10(freq) vs Norm)
    corr_log = float(np.corrcoef(x_log, y_norm)[0, 1]) if x_log.size and y_norm.size else float("nan")
    x_line2 = np.linspace(x_log.min(), x_log.max(), 200)
    axes[1].plot(x_line2, m_flip * x_line2 + b_flip, color="red", linewidth=2, label=f"fit: y={m_flip:.3g}x+{b_flip:.3g}")
    axes[1].set_xlabel("log10 frequency")
    axes[1].set_ylabel("Centered Squared Norms of Last Layer activations")
    axes[1].set_title(f"log10(freq) vs Norm (top {len(top_k_df)} tokens)")
    axes[1].text(
        0.02,
        0.98,
        f"r = {corr_log:.3f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=(1, 1, 1, 0.8), edgecolor="none"),
    )
    axes[1].legend(frameon=False)

    # Include key params in filename for traceability
    suffix_bits = [
        f"mt{max_tokens}",
        f"eb{effective_block}",
        f"bsz{batch_size}",
        f"cf{int(center_fraction * 100)}",
        f"tk{topk}",
    ]
    if revision:
        safe_rev = str(revision).replace("/", "_").replace(":", "_")
        suffix_bits.append(f"rev{safe_rev}")
    fig_base = os.path.join(out_dir, f"{safe_model}_hidden_gain_scatter_{'_'.join(suffix_bits)}")
    png_path = fig_base + ".png"
    svg_path = fig_base + ".svg"
    fig.savefig(png_path, dpi=200)
    # SVG saving disabled
    plt.close(fig)
    print(f"Saved figure to: {png_path} and {svg_path}")

def _load_model_and_tokenizer_for_hidden_gain(spec: dict, model_id: str, revision: str | None):
    """Load model and tokenizer for hidden-gain analysis, respecting class='olmo', 'olmo_native', or 'huggingface'."""
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

    model_class = spec.get("class", "huggingface")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_class == "olmo_native":
        # Native OLMo checkpoint format (config.yaml + rank0.pt)
        try:
            from olmo import OLMo, Tokenizer as OLMoTokenizer  # type: ignore
        except ImportError:
            ensure("ai2-olmo")
            from olmo import OLMo, Tokenizer as OLMoTokenizer  # type: ignore
        from huggingface_hub import snapshot_download  # type: ignore

        # Download checkpoint to local cache
        local_path = snapshot_download(repo_id=model_id, revision=revision)
        print(f"[olmo_native] Downloaded checkpoint to: {local_path}")

        # Load the OLMo model from checkpoint
        model = OLMo.from_checkpoint(local_path)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        # Load tokenizer - OLMo native tokenizer or fall back to transformers
        try:
            tok = OLMoTokenizer.from_checkpoint(local_path)
        except Exception:
            # Fall back to HF tokenizer with allenai/gpt-neox-olmo-dolma-v1_5 (OLMo's tokenizer)
            tok = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5", trust_remote_code=True)
        if hasattr(tok, 'pad_token') and tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Wrap native OLMo in a HF-compatible interface for hidden states
        class OLMoHiddenStateWrapper:
            """Wrapper to make native OLMo model compatible with hidden-gain analysis."""
            def __init__(self, olmo_model):
                self._olmo = olmo_model
                self.config = type('Config', (), {'output_hidden_states': True})()

            def eval(self):
                self._olmo.eval()
                return self

            def to(self, device):
                self._olmo.to(device)
                return self

            def __call__(self, input_ids, **kwargs):
                # Run OLMo forward pass with output_hidden_states
                out = self._olmo(input_ids, output_hidden_states=True)
                # Return in HF-style format
                class Output:
                    def __init__(self, hidden_states):
                        self.hidden_states = hidden_states
                return Output(out.hidden_states)

        model = OLMoHiddenStateWrapper(model)

    elif model_class == "olmo":
        # Use hf_olmo for HF-format OLMo checkpoints
        try:
            from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast  # type: ignore
        except ImportError:
            ensure("ai2-olmo")
            from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast  # type: ignore

        tok = OLMoTokenizerFast.from_pretrained(model_id, revision=revision) if revision else OLMoTokenizerFast.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if torch.cuda.is_available():
            model = OLMoForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                revision=revision,
            ) if revision else OLMoForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        else:
            model = OLMoForCausalLM.from_pretrained(model_id, revision=revision) if revision else OLMoForCausalLM.from_pretrained(model_id)
    else:
        # Standard huggingface path
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, revision=revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                revision=revision,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)

    return model, tok, device


def cmd_plot_hidden_gain_all(args):
    """Run hidden-gain analysis and plots for all configured models."""
    # Lazily ensure runtime deps
    try:
        ensure("numpy")
    except Exception:
        pass
    try:
        ensure("pandas")
    except Exception:
        pass
    try:
        ensure("matplotlib")
    except Exception:
        pass
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    _ensure_dir(FIG_DIR)
    config_path = args.config or CONFIG_PATH
    cfg = load_model_config(config_path)
    if not isinstance(cfg, dict) or len(cfg) == 0:
        print("Empty config")
        return

    data_path = DATA_PATH
    out_dir = os.path.join(FIG_DIR, "hidden_gain")
    _ensure_dir(out_dir)

    # Hardcoded settings to match notebook / single-model command
    max_tokens = 2_000_000
    block_size = 4096
    batch_size = 4
    center_fraction = 0.1
    topk = 10000

    for target_name, spec, model_id in _iter_models(cfg):
        try:
            revision = _get_revision_from_spec(spec)
            safe_model = model_id.replace("/", "_")

            print(f"\n[All] Hidden-gain for: {target_name} ({model_id})")
            print(f"[All] Data: {data_path}")
            print(f"[All] Max tokens: {max_tokens}, block_size: {block_size}, batch_size: {batch_size}, center_fraction: {center_fraction}")

            model, tok, device = _load_model_and_tokenizer_for_hidden_gain(spec, model_id, revision)
            model.config.output_hidden_states = True
            model.eval()
            if not torch.cuda.is_available():
                model = model.to(device)

            effective_block = get_effective_block_size(model, block_size)
            if effective_block < block_size:
                print(f"[All] Capping block size from {block_size} to model max {effective_block}")
            else:
                print(f"[All] Block size: {effective_block}")

            center_vec = None
            if max_tokens > 0 and center_fraction > 0:
                target_center_tokens = max(1, int(max_tokens * center_fraction))
                center_vec = hidden_gain_estimate_center_vector(
                    model,
                    tok,
                    data_path,
                    effective_block,
                    batch_size,
                    target_center_tokens,
                )
                if center_vec is not None:
                    print(f"[All] Computed center_vec (target {target_center_tokens}). Shape: {tuple(center_vec.shape)}")
                else:
                    print("[All] Centering skipped (no tokens counted).")

            agg_df, meta = hidden_gain_aggregate_per_token(
                model,
                tok,
                data_path,
                effective_block,
                batch_size,
                max_tokens,
                center_vec=center_vec,
                topk_for_corr=topk,
            )
            if int(meta.get("total_count", 0)) == 0:
                print(f"[All] No tokens processed for {target_name}; skipping.")
                continue
            print(f"[All] {target_name}: processed={int(meta.get('total_count', 0))}, unique={int(meta.get('unique_tokens', 0))}, corr_topk={meta.get('corr_topk', None)}")

            # Save aggregates
            agg_out_path_parquet = os.path.join(out_dir, f"{safe_model}_token_aggregates_ic.parquet")
            agg_out_path_csv = os.path.join(out_dir, f"{safe_model}_token_aggregates_ic.csv")
            saved_parquet = False
            try:
                agg_df.to_parquet(agg_out_path_parquet, index=False)
                print(f"[All] Wrote aggregates (parquet): {agg_out_path_parquet}")
                saved_parquet = True
            except Exception as e:
                print(f"[All] Parquet save failed for {target_name} ({e}); writing CSV instead.")
            if not saved_parquet:
                agg_df.to_csv(agg_out_path_csv, index=False)
                print(f"[All] Wrote aggregates (csv): {agg_out_path_csv}")

            # Plot (top-K by frequency)
            agg_sorted = agg_df.sort_values("freq", ascending=False)
            top_k_df = agg_sorted.head(topk).reset_index(drop=True)

            x_norm = top_k_df["mean_squared_norm"].astype(float).to_numpy()
            y_ic = top_k_df["info_content"].astype(float).to_numpy()
            y_log10 = np.log10(top_k_df["freq"].astype(float).to_numpy())

            fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            try:
                fig.suptitle(f"{target_name} — {model_id}", fontsize=12)
            except Exception:
                pass
            # Norm vs IC
            axes[0].scatter(x_norm, y_ic, s=6, alpha=0.3)
            m_ic, b_ic = np.polyfit(x_norm, y_ic, 1)
            # Pearson correlation (Norm vs IC)
            corr_ic = float(np.corrcoef(x_norm, y_ic)[0, 1]) if x_norm.size and y_ic.size else float("nan")
            x_line = np.linspace(x_norm.min(), x_norm.max(), 200)
            axes[0].plot(x_line, m_ic * x_line + b_ic, color="crimson", linewidth=2, label=f"fit: y={m_ic:.3g}x+{b_ic:.3g}")
            axes[0].set_xlabel("Centered Squared Norms of Last Layer activations")
            axes[0].set_ylabel("Information content (ln frequency)")
            axes[0].set_title(f"Norm vs IC (top {len(top_k_df)} tokens)")
            axes[0].text(
                0.02,
                0.98,
                f"r = {corr_ic:.3f}",
                transform=axes[0].transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=(1, 1, 1, 0.8), edgecolor="none"),
            )
            axes[0].legend(frameon=False)
            # log10(freq) vs Norm
            x_log = y_log10
            y_norm = x_norm
            axes[1].scatter(x_log, y_norm, s=6, alpha=0.3, color="tab:blue")
            m_flip, b_flip = np.polyfit(x_log, y_norm, 1)
            # Pearson correlation (log10(freq) vs Norm)
            corr_log = float(np.corrcoef(x_log, y_norm)[0, 1]) if x_log.size and y_norm.size else float("nan")
            x_line2 = np.linspace(x_log.min(), x_log.max(), 200)
            axes[1].plot(x_line2, m_flip * x_line2 + b_flip, color="red", linewidth=2, label=f"fit: y={m_flip:.3g}x+{b_flip:.3g}")
            axes[1].set_xlabel("log10 frequency")
            axes[1].set_ylabel("Centered Squared Norms of Last Layer activations")
            axes[1].set_title(f"log10(freq) vs Norm (top {len(top_k_df)} tokens)")
            axes[1].text(
                0.02,
                0.98,
                f"r = {corr_log:.3f}",
                transform=axes[1].transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=(1, 1, 1, 0.8), edgecolor="none"),
            )
            axes[1].legend(frameon=False)

            # Include key params in filename for traceability
            suffix_bits = [
                f"mt{max_tokens}",
                f"eb{effective_block}",
                f"bsz{batch_size}",
                f"cf{int(center_fraction * 100)}",
                f"tk{topk}",
            ]
            if revision:
                safe_rev = str(revision).replace("/", "_").replace(":", "_")
                suffix_bits.append(f"rev{safe_rev}")
            fig_base = os.path.join(out_dir, f"{safe_model}_hidden_gain_scatter_{'_'.join(suffix_bits)}")
            png_path = fig_base + ".png"
            svg_path = fig_base + ".svg"
            fig.savefig(png_path, dpi=200)
            # SVG saving disabled
            plt.close(fig)
            print(f"[All] Saved figure for {target_name} to: {png_path} and {svg_path}")
        except Exception as e:
            print(f"[All] Skipping {target_name}: {e}")
        finally:
            # Explicit cleanup between models to avoid CUDA OOM
            try:
                del model
            except Exception:
                pass
            try:
                del tok
            except Exception:
                pass
            try:
                del center_vec
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

def main():
    """CLI entrypoint that dispatches to subcommands based on `command`."""
    parser = argparse.ArgumentParser(description="Vector norm replication scripts")
    parser.add_argument("command", choices=[
        "setup",
        "embeddings",
        "norms",
        "tokenize",
        "tokenize-bigram",
        "plot-first",
        "plot-binned",
        "plot-rank-l2",
        "plot-rank-l2-raw",
        "plot-rank-l2-by-family",
        "plot-rank-l2-raw-by-family",
        "plot-l2logf",
        "plot-l2logf-by-family",
        "plot-zlogf",
        "plot-zpct",
        "plot-combined-one",
        "plot-combined-all",
        "plot-combined-all-centered",
        "plot-next-probs",
        "plot-next-probs-freq",
        "plot-next-probs-freq-all",
        "plot-next-probs-all",
        "plot-rankrank-probs",
        "plot-rankrank-probs-all",
        "plot-rankrank-slopes",
        "plot-rankrank-slopes-all",
        "plot-rankrank-slopes-logprob",
        "plot-rankrank-slopes-logprob-all",
        "plot-hidden-gain",
        "plot-hidden-gain-all",
        "report-topfreq",
    ], help="Action to run")
    parser.add_argument("--model", default=None, help="Target model name (for plot-first)")
    parser.add_argument("--config", default=None, help="Path to model config JSON (defaults to CONFIG_PATH from tokenizer_util)")
    parser.add_argument("--bins", default=20, help="Number of bins (for plot-binned / rank-l2 / zlogf)")
    parser.add_argument("--quantiles", default=100, help="Number of quantiles (for plot-zpct)")
    parser.add_argument("--log-bins", action="store_true", help="Use log-spaced rank bins (for plot-rank-l2)")
    parser.add_argument("--x-log", action="store_true", help="Log-scale x-axis (for plot-rank-l2)")
    parser.add_argument("--include-root", action="store_true", help="Include 1/8-root columns (for plot-combined-one)")
    parser.add_argument("--ymin", type=float, default=None, help="Fixed lower y-axis limit for combined scatter plots")
    parser.add_argument("--ymax", type=float, default=None, help="Fixed upper y-axis limit for combined scatter plots")
    parser.add_argument("--no-cache", action="store_true", help="Disable tokenization cache")
    parser.add_argument("--topk", default=100, help="Top-K most frequent tokens to compare (for report-topfreq)")
    parser.add_argument("--freq-type", default="unigram", choices=["unigram", "bigram"], help="Frequency definition: unigram counts or distinct bigram neighbors")
    # Args for plot-next-probs
    parser.add_argument("--seeds", type=int, default=1000, help="Number of random seeds to average (for plot-next-probs)")
    parser.add_argument("--seq-length", type=int, default=10, help="White-noise sequence length (for plot-next-probs)")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Scale factor for noise magnitude (for plot-next-probs)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (for plot-next-probs)")
    parser.add_argument("--noise-mode", default="white", choices=["white", "data", "ancestral_sampling"], help="Noise source: white (random embeddings), data (sampled sequences), or ancestral_sampling (dataset frequencies as probabilities)")
    parser.add_argument("--freq-dataset-dir", default=None, help="Directory containing datasets to derive probabilities from (for noise_mode=ancestral_sampling)")
    parser.add_argument("--rank-center", type=int, default=1000, help="Center rank to focus on (for plot-next-probs)")
    parser.add_argument("--rank-width", type=int, default=1000, help="Number of tokens in rank window (for plot-next-probs)")
    # Sweep options for plot-rankrank-probs
    parser.add_argument("--rank-sweep-start", type=int, default=1, help="Start frequency rank (inclusive) for sweeping centers")
    parser.add_argument("--rank-sweep-end", type=int, default=50000, help="End frequency rank (inclusive) for sweeping centers")
    parser.add_argument("--rank-sweep-step", type=int, default=10, help="Step size when sweeping centers")
    parser.add_argument("--log-file", default=None, help="Optional path to append slope logs (defaults to FIG_DIR/rankrank_slopes.log)")
    parser.add_argument("--avg-probs-file", default="/home/vec_norm/scripts/results/prob_tests/llama3_2_1b_avg_probs.npy", help="Optional path to .npy file with precomputed average next-token probabilities")
    # Fit options for slope vs center
    parser.add_argument("--fit-on-values", action="store_true", help="For rankrank-slopes: fit using actual (norm,prob) values instead of windowed ranks")
    parser.add_argument("--embedding-kind", default="input", choices=["input", "output"], help="Embedding type for combined plots: input (default) or output")
    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "embeddings":
        cmd_embeddings(args)
    elif args.command == "norms":
        cmd_norms(args)
    elif args.command == "tokenize":
        cmd_tokenize(args)
    elif args.command == "tokenize-bigram":
        cmd_tokenize_bigram(args)
    elif args.command == "plot-first":
        cmd_plot_first(args)
    elif args.command == "plot-binned":
        cmd_plot_binned(args)
    elif args.command == "plot-rank-l2":
        cmd_plot_rank_vs_l2(args)
    elif args.command == "plot-rank-l2-raw":
        cmd_plot_rank_vs_l2_raw(args)
    elif args.command == "plot-rank-l2-by-family":
        cmd_plot_rank_vs_l2_by_family(args)
    elif args.command == "plot-rank-l2-raw-by-family":
        cmd_plot_rank_vs_l2_raw_by_family(args)
    elif args.command == "plot-l2logf":
        cmd_plot_l2_vs_logfreq(args)
    elif args.command == "plot-l2logf-by-family":
        cmd_plot_l2_vs_logfreq_by_family(args)
    elif args.command == "plot-zlogf":
        cmd_plot_zscore_vs_logfreq(args)
    elif args.command == "plot-zpct":
        cmd_plot_zscore_vs_percentile(args)
    elif args.command == "plot-combined-one":
        cmd_plot_combined_single(args)
    elif args.command == "plot-combined-all":
        cmd_plot_combined_all(args)
    elif args.command == "plot-combined-all-centered":
        cmd_plot_combined_all_centered(args)
    elif args.command == "plot-next-probs":
        cmd_plot_next_probs(args)
    elif args.command == "plot-next-probs-freq":
        cmd_plot_next_probs_freq(args)
    elif args.command == "plot-next-probs-freq-all":
        cmd_plot_next_probs_freq_all(args)
    elif args.command == "plot-next-probs-all":
        cmd_plot_next_probs_all(args)
    elif args.command == "plot-rankrank-probs":
        cmd_plot_rankrank_probs(args)
    elif args.command == "plot-rankrank-probs-all":
        cmd_plot_rankrank_probs_all(args)
    elif args.command == "plot-rankrank-slopes":
        cmd_plot_rankrank_slopes(args)
    elif args.command == "plot-rankrank-slopes-all":
        cmd_plot_rankrank_slopes_all(args)
    elif args.command == "plot-rankrank-slopes-logprob":
        cmd_plot_rankrank_slopes_logprob(args)
    elif args.command == "plot-rankrank-slopes-logprob-all":
        cmd_plot_rankrank_slopes_logprob_all(args)
    elif args.command == "plot-hidden-gain":
        cmd_plot_hidden_gain(args)
    elif args.command == "plot-hidden-gain-all":
        cmd_plot_hidden_gain_all(args)
    elif args.command == "report-topfreq":
        cmd_report_topfreq(args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


