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
for _pkg in ("packaging", "torch", "transformers", "safetensors", "huggingface_hub", "datasets", "sentencepiece"):
    ensure(_pkg)

import torch
from huggingface_hub import hf_hub_download 
from safetensors import safe_open 
from collections import Counter 
from datasets import load_dataset 
from typing import Tuple, Optional, Dict

# Resolve paths relative to this util's directory
UTIL_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(UTIL_DIR, "..", ".."))

RESULTS_DIR = os.path.join(UTIL_DIR, "embeddings")
FIG_DIR = os.path.join(UTIL_DIR, "results", "Figures", "Dec13th")
CONFIG_PATH = os.path.join(UTIL_DIR, "configs", "tok_config_figure5_local.json")
DATA_PATH = os.path.join(UTIL_DIR, "..", "text_data", "eng_latn_300mb.txt")
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
        # Check for weight tying: if no separate output embedding, use input embedding
        # Weight tying means the model uses transformer.wte.weight for both input and output
        input_emb_keys = ["model.transformer.wte.weight", "transformer.wte.weight", "wte.weight"]
        for key in input_emb_keys:
            if key in inner_state:
                candidate = inner_state[key]
                if isinstance(candidate, torch.Tensor) and candidate.ndim == 2:
                    # Check shape: output embedding should be [vocab_size, hidden_dim]
                    # FFN weights are [hidden_dim, ffn_dim] which is different
                    if candidate.shape[0] > candidate.shape[1]:  # vocab > hidden (typical)
                        emb = candidate
                        print(f"[load_output_embeddings_from_olmo_local] Weight tying detected, using input embedding '{key}'")
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




