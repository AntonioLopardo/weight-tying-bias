import os
import sys
import json
import subprocess
from typing import Tuple, Dict, Optional
import pickle


def ensure(pkg_name: str) -> None:
    try:
        __import__(pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])


from importlib import import_module
for _pkg in ("packaging", "torch", "transformers", "safetensors", "huggingface_hub", "datasets", "sentencepiece", "ai2-olmo"):
    ensure(_pkg)

import torch
from huggingface_hub import hf_hub_download 
from safetensors import safe_open 
from collections import Counter 
from datasets import load_dataset 
from typing import Iterator, List, Tuple, Optional, Dict

# Resolve paths relative to this util's directory
UTIL_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(UTIL_DIR, "..", ".."))

RESULTS_DIR = os.path.join(UTIL_DIR, "embeddings")
FIG_DIR = os.path.join(UTIL_DIR, "results", "Figures", "Dec13th")
CONFIG_PATH = os.path.join(UTIL_DIR, "configs", "tok_config_olmo_1B_0724_GeLU.json") #//home/vec_norm/scripts/configs/tok_config_pythia_1b.json
DATA_PATH = os.path.join(UTIL_DIR, "text_data", "eng_latn_300mb.txt")
#DATA_PATH = os.path.join("/home/vec_norm/scripts/results/generated_freqs/olmo2-1b-0425.txt")
TOKEN_CACHE_DIR = os.path.join(UTIL_DIR, "tokenized_data", "revisions_wtv_lol")
EMBEDDING_CACHE_DIR = os.path.join(UTIL_DIR, "embeddings", "revisions_wtv")


def load_model_config(config_path: Optional[str] = None) -> Dict:
    path = config_path or CONFIG_PATH
    with open(path, "r") as f:
        return json.load(f)


def find_embedding_tensor_key(state_dict_keys):
    # Minimal exact-match list to support common models; keep simple
    candidates = [
        "model.transformer.wte.weight",          # OLMo-1B (observed)
        "transformer.wte.weight",                # GPT-style
        "model.embed_tokens.weight",             # LLaMA-style
        "language_model.model.embed_tokens.weight",  # Gemma-3
        "lm_head.weight",                        # sometimes tied (last-resort)
        "wte.weight",
        "embeddings.word_embeddings.weight",
        "model.decoder.embed_tokens.weight",
        "gpt_neox.embed_in.weight",
    ]
    for key in candidates:
        if key in state_dict_keys:
            return key
    raise KeyError("Could not locate embedding weight key")


def _cache_filename(model_id: str, revision: Optional[str] = None) -> str:
    safe_model = model_id.replace("/", "__")
    if revision and isinstance(revision, str) and len(revision) > 0:
        safe_rev = revision.replace("/", "_").replace(":", "_")
        return f"{safe_model}__{safe_rev}.pt"
    return f"{safe_model}.pt"


def load_input_embeddings_from_repo(model_id: str, revision: Optional[str] = None, use_cache: bool = False, cache_dir: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
    # Try local cache first
    cache_root = cache_dir or EMBEDDING_CACHE_DIR
    if use_cache:
        try:
            os.makedirs(cache_root, exist_ok=True)
            cache_path = os.path.join(cache_root, _cache_filename(model_id, revision))
            if os.path.exists(cache_path):
                emb_cached = torch.load(cache_path, map_location="cpu")
                # Build minimal meta without requiring network
                try:
                    # Local import to avoid global import-time errors
                    from transformers import AutoConfig  # type: ignore
                    cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
                    config_hidden = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
                    config_vocab = getattr(cfg, "vocab_size", None)
                except Exception:
                    config_hidden = None
                    config_vocab = None
                meta_cached = {
                    "model_id": model_id,
                    "vocab_size": emb_cached.shape[0],
                    "hidden_size": emb_cached.shape[1],
                    "revision": revision,
                    "config_hidden_size": config_hidden,
                    "config_vocab_size": config_vocab,
                }
                return emb_cached, meta_cached
        except Exception:
            pass

    # Local import to avoid global import-time errors; tolerate broken/missing configs
    try:
        from transformers import AutoConfig  # type: ignore
        cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    except Exception:
        cfg = None

    filenames = [
        "model.safetensors",
        "pytorch_model.bin",
    ]

    file_path = None
    for filename in filenames:
        try:
            file_path = hf_hub_download(repo_id=model_id, filename=filename, revision=revision)
            break
        except Exception:
            continue

    emb = None

    if file_path is not None:
        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                keys = list(f.keys())
                emb_key = find_embedding_tensor_key(keys)
                print(f"[load_input_embeddings_from_repo] Using embedding key '{emb_key}' from '{os.path.basename(file_path)}'")
                emb = f.get_tensor(emb_key)
        else:
            state = torch.load(file_path, map_location="cpu")
            keys = list(state.keys())
            emb_key = find_embedding_tensor_key(keys)
            print(f"[load_input_embeddings_from_repo] Using embedding key '{emb_key}' from '{os.path.basename(file_path)}'")
            emb = state[emb_key]
    else:
        index_path = None
        for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            try:
                index_path = hf_hub_download(repo_id=model_id, filename=idx_name, revision=revision)
                break
            except Exception:
                continue
        if index_path is None:
            raise FileNotFoundError(f"No weight file found for {model_id}")
        with open(index_path, "r") as f:
            index_json = json.load(f)
        weight_map = index_json.get("weight_map", {})
        emb_key = find_embedding_tensor_key(weight_map.keys())
        shard_filename = weight_map[emb_key]
        shard_path = hf_hub_download(repo_id=model_id, filename=shard_filename, revision=revision)
        print(f"[load_input_embeddings_from_repo] Using embedding key '{emb_key}' from shard '{shard_filename}'")
        if shard_filename.endswith(".safetensors"):
            with safe_open(shard_path, framework="pt") as f:
                emb = f.get_tensor(emb_key)
        else:
            state = torch.load(shard_path, map_location="cpu")
            emb = state[emb_key]

    if emb is None or emb.ndim != 2:
        raise ValueError(f"Unexpected embedding tensor shape for {model_id}: {None if emb is None else tuple(emb.shape)}")

    # Save to cache
    if use_cache:
        try:
            os.makedirs(cache_root, exist_ok=True)
            cache_path = os.path.join(cache_root, _cache_filename(model_id, revision))
            torch.save(emb.detach().cpu(), cache_path)
        except Exception:
            pass

    meta = {
        "model_id": model_id,
        "vocab_size": emb.shape[0],
        "hidden_size": emb.shape[1],
        "revision": revision,
        "config_hidden_size": (getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None)) if cfg is not None else None),
        "config_vocab_size": (getattr(cfg, "vocab_size", None) if cfg is not None else None),
    }
    return emb, meta


def load_input_embeddings_from_olmo(model_id: str, revision: Optional[str] = None, use_cache: bool = False, cache_dir: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
    cache_root = cache_dir or EMBEDDING_CACHE_DIR
    if use_cache:
        try:
            os.makedirs(cache_root, exist_ok=True)
            cache_path = os.path.join(cache_root, _cache_filename(model_id, revision))
            if os.path.exists(cache_path):
                emb_cached = torch.load(cache_path, map_location="cpu")
                meta_cached = {
                    "model_id": model_id,
                    "vocab_size": emb_cached.shape[0],
                    "hidden_size": emb_cached.shape[1],
                    "revision": revision,
                }
                return emb_cached, meta_cached
        except Exception:
            pass

    from hf_olmo import OLMoForCausalLM  # pip install ai2-olmo

    if revision:
        model = OLMoForCausalLM.from_pretrained(model_id, revision=revision)
    else:
        model = OLMoForCausalLM.from_pretrained(model_id)
    emb = model.get_input_embeddings().weight.detach().cpu()

    if emb is None or emb.ndim != 2:
        raise ValueError(f"Unexpected embedding tensor shape for {model_id}: {None if emb is None else tuple(emb.shape)}")

    if use_cache:
        try:
            os.makedirs(cache_root, exist_ok=True)
            cache_path = os.path.join(cache_root, _cache_filename(model_id, revision))
            torch.save(emb, cache_path)
        except Exception:
            pass

    meta = {
        "model_id": model_id,
        "vocab_size": emb.shape[0],
        "hidden_size": emb.shape[1],
        "revision": revision,
    }
    return emb, meta


def load_input_embeddings_from_olmo_native(model_id: str, revision: Optional[str] = None, use_cache: bool = False, cache_dir: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
    """Load input embeddings from native OLMo checkpoint format (config.yaml + rank0.pt)."""
    cache_root = cache_dir or EMBEDDING_CACHE_DIR
    if use_cache:
        try:
            os.makedirs(cache_root, exist_ok=True)
            cache_path = os.path.join(cache_root, _cache_filename(model_id, revision))
            if os.path.exists(cache_path):
                emb_cached = torch.load(cache_path, map_location="cpu")
                meta_cached = {
                    "model_id": model_id,
                    "vocab_size": emb_cached.shape[0],
                    "hidden_size": emb_cached.shape[1],
                    "revision": revision,
                }
                return emb_cached, meta_cached
        except Exception:
            pass

    # Download native OLMo checkpoint
    try:
        # Try downloading just the rank0.pt file
        ckpt_path = hf_hub_download(repo_id=model_id, filename="rank0.pt", revision=revision)
    except Exception as e:
        raise FileNotFoundError(f"Could not download rank0.pt from {model_id}: {e}")

    # Load checkpoint and extract embedding weights
    # Use weights_only=False for native OLMo checkpoints (they contain pathlib objects)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Native OLMo checkpoints store embeddings under various keys
    emb = None
    emb_candidates = [
        "model.transformer.wte.weight",  # OLMo standard
        "transformer.wte.weight",
        "model.embed_tokens.weight",
        "wte.weight",
    ]

    # Handle nested state dict (some checkpoints wrap in 'model' key)
    if "model" in state and isinstance(state["model"], dict):
        inner_state = state["model"]
    else:
        inner_state = state

    for key in emb_candidates:
        if key in inner_state:
            emb = inner_state[key]
            print(f"[load_input_embeddings_from_olmo_native] Using key '{key}' from rank0.pt")
            break

    if emb is None:
        # Try to find any key with 'embed' or 'wte' in it
        for key in inner_state.keys():
            if 'embed' in key.lower() or 'wte' in key.lower():
                if isinstance(inner_state[key], torch.Tensor) and inner_state[key].ndim == 2:
                    emb = inner_state[key]
                    print(f"[load_input_embeddings_from_olmo_native] Using fallback key '{key}' from rank0.pt")
                    break

    if emb is None or emb.ndim != 2:
        available_keys = [k for k in inner_state.keys() if isinstance(inner_state.get(k), torch.Tensor)]
        raise ValueError(f"Could not find embedding tensor in {model_id}. Available tensor keys: {available_keys[:20]}")

    emb = emb.detach().cpu()

    # Save to cache
    if use_cache:
        try:
            os.makedirs(cache_root, exist_ok=True)
            cache_path = os.path.join(cache_root, _cache_filename(model_id, revision))
            torch.save(emb, cache_path)
        except Exception:
            pass

    meta = {
        "model_id": model_id,
        "vocab_size": emb.shape[0],
        "hidden_size": emb.shape[1],
        "revision": revision,
    }
    return emb, meta


def load_input_embeddings_from_olmo_local(checkpoint_path: str, use_cache: bool = False, cache_dir: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
    """Load input embeddings from a local OLMo checkpoint directory (with model.pt)."""
    import os
    
    model_pt_path = os.path.join(checkpoint_path, "model.pt")
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"Could not find model.pt in {checkpoint_path}")
    
    # Create a cache key from the checkpoint path
    cache_key = checkpoint_path.replace("/", "__").replace("\\", "__")
    cache_root = cache_dir or EMBEDDING_CACHE_DIR
    
    if use_cache:
        try:
            os.makedirs(cache_root, exist_ok=True)
            cache_path = os.path.join(cache_root, f"{cache_key}_emb.pt")
            if os.path.exists(cache_path):
                emb_cached = torch.load(cache_path, map_location="cpu")
                meta_cached = {
                    "model_id": checkpoint_path,
                    "vocab_size": emb_cached.shape[0],
                    "hidden_size": emb_cached.shape[1],
                }
                print(f"[load_input_embeddings_from_olmo_local] Loaded from cache: {cache_path}")
                return emb_cached, meta_cached
        except Exception:
            pass
    
    print(f"[load_input_embeddings_from_olmo_local] Loading from {model_pt_path}")
    state = torch.load(model_pt_path, map_location="cpu", weights_only=False)
    
    # Find embedding weights
    emb = None
    emb_candidates = [
        "model.transformer.wte.weight",
        "transformer.wte.weight", 
        "model.embed_tokens.weight",
        "wte.weight",
    ]
    
    # Handle nested state dict
    if "model" in state and isinstance(state["model"], dict):
        inner_state = state["model"]
    else:
        inner_state = state
    
    for key in emb_candidates:
        if key in inner_state:
            emb = inner_state[key]
            print(f"[load_input_embeddings_from_olmo_local] Using key '{key}'")
            break
    
    if emb is None:
        for key in inner_state.keys():
            if 'embed' in key.lower() or 'wte' in key.lower():
                if isinstance(inner_state[key], torch.Tensor) and inner_state[key].ndim == 2:
                    emb = inner_state[key]
                    print(f"[load_input_embeddings_from_olmo_local] Using fallback key '{key}'")
                    break
    
    if emb is None or emb.ndim != 2:
        available_keys = [k for k in inner_state.keys() if isinstance(inner_state.get(k), torch.Tensor)]
        raise ValueError(f"Could not find embedding tensor. Available keys: {available_keys[:20]}")
    
    emb = emb.detach().cpu()
    
    # Save to cache
    if use_cache:
        try:
            os.makedirs(cache_root, exist_ok=True)
            cache_path = os.path.join(cache_root, f"{cache_key}_emb.pt")
            torch.save(emb, cache_path)
            print(f"[load_input_embeddings_from_olmo_local] Saved to cache: {cache_path}")
        except Exception:
            pass
    
    meta = {
        "model_id": checkpoint_path,
        "vocab_size": emb.shape[0],
        "hidden_size": emb.shape[1],
    }
    return emb, meta


def load_output_embeddings_from_olmo_local(checkpoint_path: str, use_cache: bool = False, cache_dir: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
    """Load output embeddings from a local OLMo checkpoint directory (with model.pt)."""
    import os
    
    model_pt_path = os.path.join(checkpoint_path, "model.pt")
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"Could not find model.pt in {checkpoint_path}")
    
    cache_key = checkpoint_path.replace("/", "__").replace("\\", "__")
    cache_root = cache_dir or EMBEDDING_CACHE_DIR
    
    if use_cache:
        try:
            os.makedirs(cache_root, exist_ok=True)
            cache_path = os.path.join(cache_root, f"{cache_key}_out_emb.pt")
            if os.path.exists(cache_path):
                emb_cached = torch.load(cache_path, map_location="cpu")
                meta_cached = {
                    "model_id": checkpoint_path,
                    "vocab_size": emb_cached.shape[0],
                    "hidden_size": emb_cached.shape[1],
                }
                print(f"[load_output_embeddings_from_olmo_local] Loaded from cache: {cache_path}")
                return emb_cached, meta_cached
        except Exception:
            pass
    
    print(f"[load_output_embeddings_from_olmo_local] Loading from {model_pt_path}")
    state = torch.load(model_pt_path, map_location="cpu", weights_only=False)
    
    # Find output embedding weights
    emb = None
    out_candidates = [
        "model.transformer.ff_out.weight",
        "transformer.ff_out.weight",
        "lm_head.weight",
        "ff_out.weight",
    ]
    
    if "model" in state and isinstance(state["model"], dict):
        inner_state = state["model"]
    else:
        inner_state = state
    
    for key in out_candidates:
        if key in inner_state:
            emb = inner_state[key]
            print(f"[load_output_embeddings_from_olmo_local] Using key '{key}'")
            break
    
    if emb is None:
        for key in inner_state.keys():
            if 'ff_out' in key.lower() or 'lm_head' in key.lower():
                if isinstance(inner_state[key], torch.Tensor) and inner_state[key].ndim == 2:
                    emb = inner_state[key]
                    print(f"[load_output_embeddings_from_olmo_local] Using fallback key '{key}'")
                    break
    
    if emb is None or emb.ndim != 2:
        available_keys = [k for k in inner_state.keys() if isinstance(inner_state.get(k), torch.Tensor)]
        raise ValueError(f"Could not find output embedding tensor. Available keys: {available_keys[:20]}")
    
    emb = emb.detach().cpu()
    
    if use_cache:
        try:
            os.makedirs(cache_root, exist_ok=True)
            cache_path = os.path.join(cache_root, f"{cache_key}_out_emb.pt")
            torch.save(emb, cache_path)
            print(f"[load_output_embeddings_from_olmo_local] Saved to cache: {cache_path}")
        except Exception:
            pass
    
    meta = {
        "model_id": checkpoint_path,
        "vocab_size": emb.shape[0],
        "hidden_size": emb.shape[1],
    }
    return emb, meta


def compute_l2_norms(emb: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(emb, ord=2, dim=1)


def _cache_key_for(model_id: str, data_path: str, revision: Optional[str] = None) -> str:
    # Normalize model_id for filename safety, store under unigrams/ subdir
    safe_model = model_id.replace("/", "__")
    data_base = os.path.basename(data_path)
    if revision and isinstance(revision, str) and len(revision) > 0:
        safe_rev = revision.replace("/", "_").replace(":", "_")
        return os.path.join("unigrams", f"{safe_model}__{safe_rev}__{data_base}.pkl")
    return os.path.join("unigrams", f"{safe_model}__{data_base}.pkl")


def _fallback_tokenizer_id(model_id: str) -> Optional[str]:
    """Return a sensible tokenizer repo id fallback for known model_id patterns."""
    lower = model_id.lower()
    if "pythia-410m-seed" in lower:
        return "EleutherAI/pythia-410m"
    if "pythia-1b-seed" in lower:
        return "EleutherAI/pythia-1b"
    return None


def _load_tokenizer_with_fallback(model_id: str, revision: Optional[str], tokenizer_id: Optional[str] = None):
    """Load AutoTokenizer for model_id; if unavailable, fall back to a base tokenizer.
    
    If tokenizer_id is provided, it will be used instead of model_id for loading the tokenizer.
    """
    from transformers import AutoTokenizer  # type: ignore
    last_err = None
    # Use explicit tokenizer_id if provided
    tok_repo = tokenizer_id or model_id
    tok_revision = None if tokenizer_id else revision  # Don't use model revision for separate tokenizer
    # Try requested repo (fast then slow)
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(tok_repo, use_fast=use_fast, trust_remote_code=True, revision=tok_revision)
        except Exception as e:
            last_err = e
    # Try fallback repo (without revision override)
    fb = _fallback_tokenizer_id(tok_repo)
    if fb:
        for use_fast in (True, False):
            try:
                return AutoTokenizer.from_pretrained(fb, use_fast=use_fast, trust_remote_code=True)
            except Exception:
                pass
    if last_err:
        raise last_err
    raise RuntimeError(f"Failed to load tokenizer for {tok_repo}")


def build_parallel_freqs_for_model(model_id: str, data_path: Optional[str] = None, num_proc: Optional[int] = None, batch_size: int = 1024, use_cache: bool = False, cache_dir: Optional[str] = None, revision: Optional[str] = None, tokenizer_id: Optional[str] = None) -> Counter:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tok = _load_tokenizer_with_fallback(model_id, revision, tokenizer_id=tokenizer_id)
    try:
        tok.model_max_length = int(1e12)
    except Exception:
        pass

    data_file = data_path or DATA_PATH

    # Try cache
    if use_cache:
        cache_root = cache_dir or TOKEN_CACHE_DIR
        os.makedirs(cache_root, exist_ok=True)
        key = _cache_key_for(model_id, data_file, revision=revision)
        cache_path = os.path.join(cache_root, key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached: Counter = pickle.load(f)
                print(f"Loaded cached token frequencies: {cache_path}")
                return cached
            except Exception:
                pass

    ds = load_dataset("text", data_files=data_file, split="train")

    def tok_batch(batch):
        enc = tok(batch["text"], add_special_tokens=False)
        return {"ids": enc["input_ids"]}

    nproc = num_proc if (isinstance(num_proc, int) and num_proc > 0) else max(1, (os.cpu_count() or 2) - 1)

    tok_ds = ds.map(
        tok_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=nproc,
        remove_columns=["text"],
        desc=f"Tokenizing {model_id}",
    )

    c = Counter()
    for row in tok_ds:
        c.update(row["ids"])  # list[int]

    # Save cache
    if use_cache:
        try:
            cache_root = cache_dir or TOKEN_CACHE_DIR
            os.makedirs(cache_root, exist_ok=True)
            key = _cache_key_for(model_id, data_file, revision=revision)
            cache_path = os.path.join(cache_root, key)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(c, f)
            print(f"Saved token frequencies cache: {cache_path}")
        except Exception:
            pass

    return c


def _bigram_cache_key_for(model_id: str, data_path: str, revision: Optional[str] = None) -> str:
    safe_model = model_id.replace("/", "__")
    data_base = os.path.basename(data_path)
    # Store bigram caches under a dedicated subdirectory
    if revision and isinstance(revision, str) and len(revision) > 0:
        safe_rev = revision.replace("/", "_").replace(":", "_")
        return os.path.join("distinct_bigrams", f"{safe_model}__{safe_rev}__{data_base}.pkl")
    return os.path.join("distinct_bigrams", f"{safe_model}__{data_base}.pkl")


def build_distinct_bigram_counts_for_model(
    model_id: str,
    data_path: Optional[str] = None,
    num_proc: Optional[int] = None,
    batch_size: int = 1024,
    use_cache: bool = False,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
) -> Counter:
    """Compute distinct bigram neighbor counts per token id.

    For each token id t, counts how many distinct neighbors appear adjacent to t
    in the tokenized corpus. Uses the union of left and right neighbors.

    Returns a Counter mapping token_id -> number of distinct neighbors.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tok = _load_tokenizer_with_fallback(model_id, revision, tokenizer_id=tokenizer_id)
    try:
        tok.model_max_length = int(1e12)
    except Exception:
        pass

    data_file = data_path or DATA_PATH

    # Try cache
    if use_cache:
        cache_root = cache_dir or TOKEN_CACHE_DIR
        os.makedirs(cache_root, exist_ok=True)
        key = _bigram_cache_key_for(model_id, data_file, revision=revision)
        cache_path = os.path.join(cache_root, key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached: Counter = pickle.load(f)
                print(f"Loaded cached distinct bigram counts: {cache_path}")
                return cached
            except Exception:
                pass

    ds = load_dataset("text", data_files=data_file, split="train")

    def tok_batch(batch):
        enc = tok(batch["text"], add_special_tokens=False)
        return {"ids": enc["input_ids"]}

    nproc = num_proc if (isinstance(num_proc, int) and num_proc > 0) else max(1, (os.cpu_count() or 2) - 1)

    tok_ds = ds.map(
        tok_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=nproc,
        remove_columns=["text"],
        desc=f"Tokenizing for bigrams {model_id}",
    )

    # Use dict of sets to accumulate unique neighbors
    neighbor_sets: Dict[int, set] = {}

    for row in tok_ds:
        ids: list = row["ids"]
        if not ids:
            continue
        # Right neighbors
        for a, b in zip(ids[:-1], ids[1:]):
            s = neighbor_sets.get(a)
            if s is None:
                s = set()
                neighbor_sets[a] = s
            s.add(b)
        # Left neighbors
        for a, b in zip(ids[1:], ids[:-1]):
            s = neighbor_sets.get(a)
            if s is None:
                s = set()
                neighbor_sets[a] = s
            s.add(b)

    out = Counter()
    for tid, s in neighbor_sets.items():
        out[tid] = len(s)

    # Save cache
    if use_cache:
        try:
            cache_root = cache_dir or TOKEN_CACHE_DIR
            os.makedirs(cache_root, exist_ok=True)
            key = _bigram_cache_key_for(model_id, data_file, revision=revision)
            cache_path = os.path.join(cache_root, key)
            # Ensure subdirectory exists (e.g., distinct_bigrams/)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(out, f)
            print(f"Saved distinct bigram counts cache: {cache_path}")
        except Exception:
            pass

    return out


# ------------------------------
# Hidden-gain analysis utilities
# ------------------------------
def get_effective_block_size(model, requested_block_size: int) -> int:
    """Return min(requested_block_size, model's max sequence length), with fallbacks."""
    try:
        max_pos = getattr(model.config, "max_position_embeddings", None)
        if max_pos is None:
            max_pos = getattr(model.config, "n_positions", requested_block_size)
        effective = int(min(requested_block_size, int(max_pos)))
    except Exception:
        effective = requested_block_size
    return max(1, int(effective))


def token_blocks_from_file(path: str, tokenizer, block_size: int) -> Iterator[List[int]]:
    """Yield contiguous token id blocks up to block_size from a text file."""
    buffer_ids: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            ids = tokenizer.encode(line, add_special_tokens=False)
            if not ids:
                continue
            buffer_ids.extend(ids)
            while len(buffer_ids) >= block_size:
                yield buffer_ids[:block_size]
                buffer_ids = buffer_ids[block_size:]
        if buffer_ids:
            yield buffer_ids


def make_hidden_gain_batches(
    blocks_iter: Iterator[List[int]],
    batch_size: int,
    block_size: int,
    tokenizer,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Batch blocks into right-padded tensors; yields (input_ids, attention_mask)."""
    ids_batch: List[List[int]] = []
    for block in blocks_iter:
        ids_batch.append(block[:block_size])
        if len(ids_batch) == batch_size:
            max_len = max(len(x) for x in ids_batch)
            input_ids = []
            attn = []
            for seq in ids_batch:
                pad_len = max_len - len(seq)
                if pad_len > 0:
                    seq = seq + [tokenizer.pad_token_id] * pad_len
                input_ids.append(seq)
                attn.append([1] * (len(seq) - pad_len) + [0] * pad_len)
            yield torch.tensor(input_ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)
            ids_batch = []
    if ids_batch:
        max_len = max(len(x) for x in ids_batch)
        input_ids = []
        attn = []
        for seq in ids_batch:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                seq = seq + [tokenizer.pad_token_id] * pad_len
            input_ids.append(seq)
            attn.append([1] * (len(seq) - pad_len) + [0] * pad_len)
        yield torch.tensor(input_ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


def hidden_gain_estimate_center_vector(
    model,
    tokenizer,
    data_path: str,
    effective_block_size: int,
    batch_size: int,
    target_tokens: int,
) -> Optional[torch.Tensor]:
    """Estimate a global mean vector over target_tokens last-layer hidden states."""
    if target_tokens <= 0:
        return None
    device = next(model.parameters()).device
    counted = 0
    center_vec: Optional[torch.Tensor] = None
    with torch.no_grad():
        for input_ids_t, attn_t in make_hidden_gain_batches(
            token_blocks_from_file(data_path, tokenizer, effective_block_size),
            batch_size,
            effective_block_size,
            tokenizer,
        ):
            if counted >= target_tokens:
                break
            input_ids = input_ids_t.to(device)
            attention_mask = attn_t.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1].float()  # [B, T, H]
            valid_mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            batch_sum = (last_hidden * valid_mask).sum(dim=(0, 1)).to(torch.float64)
            to_add = int(attention_mask.sum().item())
            if counted + to_add > target_tokens:
                ratio = (target_tokens - counted) / max(to_add, 1)
                batch_sum = batch_sum * ratio
                to_add = target_tokens - counted
            if center_vec is None:
                center_vec = torch.zeros(last_hidden.size(-1), dtype=torch.float64, device=last_hidden.device)
            center_vec += batch_sum
            counted += to_add
    if counted <= 0 or center_vec is None:
        return None
    return (center_vec / counted).to(dtype=torch.float32)


def hidden_gain_aggregate_per_token(
    model,
    tokenizer,
    data_path: str,
    effective_block_size: int,
    batch_size: int,
    max_tokens: int,
    center_vec: Optional[torch.Tensor] = None,
    topk_for_corr: int = 10000,
):
    """Aggregate per-token frequency and mean squared norm; return (agg_df, meta)."""
    # Local imports to keep top-level deps light
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    device = next(model.parameters()).device
    sum_sq_by_id: Dict[int, float] = {}
    count_by_id: Dict[int, int] = {}
    token_str_by_id: Dict[int, str] = {}
    total_count = 0

    with torch.no_grad():
        for input_ids_t, attn_t in make_hidden_gain_batches(
            token_blocks_from_file(data_path, tokenizer, effective_block_size),
            batch_size,
            effective_block_size,
            tokenizer,
        ):
            if max_tokens > 0 and total_count >= max_tokens:
                break
            input_ids = input_ids_t.to(device)
            attention_mask = attn_t.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1].float()  # [B, T, H]
            if center_vec is not None:
                last_hidden = last_hidden - center_vec.view(1, 1, -1)
            norms_sq = (last_hidden ** 2).sum(dim=-1).cpu()  # [B, T]
            ids_cpu = input_ids.cpu()
            attn_cpu = attention_mask.cpu()
            for b in range(ids_cpu.size(0)):
                seq_ids = ids_cpu[b].tolist()
                seq_mask = attn_cpu[b].tolist()
                seq_norms = norms_sq[b].tolist()
                for tid, mask_val, n2 in zip(seq_ids, seq_mask, seq_norms):
                    if mask_val == 0:
                        continue
                    if max_tokens > 0 and total_count >= max_tokens:
                        break
                    total_count += 1
                    sum_sq_by_id[tid] = sum_sq_by_id.get(tid, 0.0) + float(n2)
                    count_by_id[tid] = count_by_id.get(tid, 0) + 1
                    if tid not in token_str_by_id:
                        token_str_by_id[tid] = tokenizer.convert_ids_to_tokens(tid)
                if max_tokens > 0 and total_count >= max_tokens:
                    break

    if total_count <= 0:
        # Return empty structures
        empty = pd.DataFrame(columns=["token_id", "token", "freq", "mean_squared_norm"])
        return empty, {"total_count": 0, "unique_tokens": 0, "corr_topk": None}

    rows = []
    for tid, cnt in count_by_id.items():
        mean_sq = (sum_sq_by_id.get(tid, 0.0) / max(cnt, 1))
        rows.append((tid, token_str_by_id.get(tid, ""), cnt, mean_sq))
    agg_df = pd.DataFrame(rows, columns=["token_id", "token", "freq", "mean_squared_norm"])
    N_total = float(agg_df["freq"].sum())
    agg_df["info_content"] = np.log(N_total / agg_df["freq"].astype(float))

    agg_sorted = agg_df.sort_values("freq", ascending=False)
    if topk_for_corr is not None and topk_for_corr > 0:
        top_k_df = agg_sorted.head(int(topk_for_corr)).reset_index(drop=True)
        corr = top_k_df[["mean_squared_norm", "info_content"]].corr().iloc[0, 1]
    else:
        corr = None

    meta = {
        "total_count": int(total_count),
        "unique_tokens": int(len(count_by_id)),
        "corr_topk": None if corr is None else float(corr),
    }
    return agg_df, meta


