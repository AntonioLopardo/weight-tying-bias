"""CLI entrypoint for generating Figure 5 (norm-frequency comparison).

Compares L2 norms of input, output, and tied embeddings against token frequency.
"""

import os
import argparse
from collections import Counter

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
    RESULTS_DIR,
    FIG_DIR,
    CONFIG_PATH,
    DATA_PATH,
    TOKEN_CACHE_DIR,
    EMBEDDING_CACHE_DIR,
)

import torch
from plotting_util import plot_figure5_comparison


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
    """Resolve a model identifier from the config spec."""
    return spec.get("path") or spec.get("model") or name


def _get_revision_from_spec(spec: dict) -> str | None:
    """Resolve an optional Hugging Face revision from the config spec."""
    return spec.get("branch") or spec.get("revision")


def _get_tokenizer_from_spec(spec: dict) -> str | None:
    """Get optional tokenizer override from the config spec."""
    return spec.get("tokenizer")


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


def _load_output_embeddings_for_spec(spec: dict, model_id: str):
    """Load output (lm_head) embeddings for a model via transformers or native OLMo."""
    revision = _get_revision_from_spec(spec)
    model_class = spec.get("class", "huggingface")

    if model_class == "olmo_native":
        from huggingface_hub import hf_hub_download  # type: ignore
        try:
            ckpt_path = hf_hub_download(repo_id=model_id, filename="rank0.pt", revision=revision)
        except Exception as e:
            raise FileNotFoundError(f"Could not download rank0.pt from {model_id}: {e}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "model" in state and isinstance(state["model"], dict):
            inner_state = state["model"]
        else:
            inner_state = state

        emb = None
        out_candidates = [
            "transformer.ff_out.weight",
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


# ----------------------------------------
# Command: plot-figure5
# ----------------------------------------
def cmd_plot_figure5(args):
    """Generate Figure 5: 1x3 comparison of untied input, untied output, and tied embeddings.

    Requires a config with exactly two models: one tied and one untied.
    The config should have keys containing 'tied' and 'untied' to identify them.
    """
    _ensure_dir(FIG_DIR)
    config_path = args.config or CONFIG_PATH
    cfg = load_model_config(config_path)

    # Find tied and untied models
    tied_spec = None
    untied_spec = None
    tied_name = None
    untied_name = None

    for name, spec in cfg.items():
        if spec.get("class") not in ("huggingface", "olmo", "olmo_native", "olmo_local"):
            continue
        name_lower = name.lower()
        if 'tied' in name_lower and 'untied' not in name_lower:
            tied_name = name
            tied_spec = spec
        elif 'untied' in name_lower:
            untied_name = name
            untied_spec = spec

    if tied_spec is None or untied_spec is None:
        raise ValueError("Config must contain both a 'tied' and 'untied' model (by name)")

    tied_model_id = _get_model_id(tied_name, tied_spec)
    untied_model_id = _get_model_id(untied_name, untied_spec)

    print(f"[Figure 5] Tied model: {tied_name} -> {tied_model_id}")
    print(f"[Figure 5] Untied model: {untied_name} -> {untied_model_id}")

    # Load embeddings
    print("[Figure 5] Loading untied input embeddings...")
    untied_input_emb, _ = _load_embeddings_for_spec(untied_spec, untied_model_id)
    print("[Figure 5] Loading untied output embeddings...")
    untied_output_emb, _ = _load_output_embeddings_for_spec(untied_spec, untied_model_id)
    print("[Figure 5] Loading tied embeddings...")
    tied_emb, _ = _load_embeddings_for_spec(tied_spec, tied_model_id)

    # Compute norms
    untied_input_norms = compute_l2_norms(untied_input_emb.detach().cpu().float())
    untied_output_norms = compute_l2_norms(untied_output_emb.detach().cpu().float())
    tied_norms = compute_l2_norms(tied_emb.detach().cpu().float())

    # Build frequencies (use untied model's tokenizer)
    tokenizer_id = _get_tokenizer_from_spec(untied_spec)
    print("[Figure 5] Building token frequencies...")
    freq_counts = _build_freqs_for_model_id(
        untied_model_id,
        data_path=DATA_PATH,
        use_cache=True,
        cache_dir=TOKEN_CACHE_DIR,
        revision=_get_revision_from_spec(untied_spec),
        tokenizer_id=tokenizer_id,
    )

    # Generate output path
    out_path = args.output or os.path.join(FIG_DIR, "figure5_norm_frequency.png")

    # Plot
    plot_figure5_comparison(
        untied_input_norms=untied_input_norms,
        untied_output_norms=untied_output_norms,
        tied_norms=tied_norms,
        freq_counts=freq_counts,
        out_path=out_path,
        untied_name=untied_name.replace("-9k", "").replace("-10k", ""),
        tied_name=tied_name.replace("-9k", "").replace("-10k", ""),
        steps=args.steps or "10k steps",
        ymin=args.ymin if args.ymin is not None else 0.8,
        ymax=args.ymax if args.ymax is not None else 2.0,
        xmin=args.xmin if args.xmin is not None else 0.0,
        xmax=args.xmax if args.xmax is not None else 7.0,
    )
    print(f"[Figure 5] Saved to: {out_path}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Norm-frequency analysis (Figure 5)")
    parser.add_argument("command", choices=["plot-figure5"], help="Action to run")
    parser.add_argument("--config", default=None, help="Path to model config JSON")
    parser.add_argument("--ymin", type=float, default=None, help="Fixed lower y-axis limit")
    parser.add_argument("--ymax", type=float, default=None, help="Fixed upper y-axis limit")
    parser.add_argument("--xmin", type=float, default=None, help="Fixed lower x-axis limit")
    parser.add_argument("--xmax", type=float, default=None, help="Fixed upper x-axis limit")
    parser.add_argument("--output", default=None, help="Output path for figure")
    parser.add_argument("--steps", default=None, help="Training steps label (e.g. '10k steps')")
    args = parser.parse_args()

    if args.command == "plot-figure5":
        cmd_plot_figure5(args)


if __name__ == "__main__":
    main()
