"""
Custom lm-evaluation-harness model wrapper for native OLMo checkpoints.

Loads models via OLMo.from_checkpoint() and implements the LM interface
required by lm-eval for perplexity and downstream task evaluation.

Usage:
    from olmo_native_lm import OLMoNativeLM
    lm = OLMoNativeLM(checkpoint_dir="/path/to/checkpoint", batch_size=8)
"""

import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance
from lm_eval.models.utils import Collator, handle_stop_sequences, normalize_gen_kwargs

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(os.environ.get("WTB_ROOT", _SCRIPT_DIR.parent))
_OLMO_CODE = _SCRIPT_DIR.parent / "OLMo"  # always use repo's OLMo for code
if str(_OLMO_CODE) not in sys.path:
    sys.path.insert(0, str(_OLMO_CODE))

from olmo.model import OLMo  # noqa: E402

logger = logging.getLogger(__name__)

_TOKENIZER_FILENAME = "allenai_eleuther-ai-gpt-neox-20b-pii-special.json"
_TOKENIZER_DIR = Path(os.environ.get(
    "OLMO_TOKENIZER_DIR",
    _REPO_ROOT / "tokenizers",
))
DEFAULT_TOKENIZER_PATH = _TOKENIZER_DIR / _TOKENIZER_FILENAME


class OLMoNativeLM(TemplateLM):
    """lm-eval wrapper for native OLMo checkpoints (model.pt + config.yaml)."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        batch_size: int = 8,
        max_gen_toks: int = 256,
        tokenizer_path: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks
        self._dtype = dtype

        logger.info(f"Loading OLMo model from {checkpoint_dir}")
        self._model = OLMo.from_checkpoint(checkpoint_dir, device="cpu")
        self._model = self._model.to(self._dtype).to(self._device)
        self._model.eval()

        self._max_length = self._model.config.max_sequence_length
        # OLMo pads vocab to a multiple of 128 for kernel efficiency;
        # we slice logits to real vocab size to avoid spurious predictions
        self._vocab_size = self._model.config.vocab_size
        self._eos_token_id = self._model.config.eos_token_id

        tok_path = str(tokenizer_path or DEFAULT_TOKENIZER_PATH)
        logger.info(f"Loading tokenizer from {tok_path}")
        base_tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            eos_token="<|endoftext|>",
            pad_token="<|padding|>",
        )
        self._eos_str = self.tok_decode(
            self._eos_token_id, skip_special_tokens=False
        )

        logger.info(
            f"Model loaded: vocab_size={self._vocab_size}, "
            f"max_seq_len={self._max_length}, eos_id={self._eos_token_id}, "
            f"dtype={self._dtype}"
        )

    # ---- TemplateLM required properties / methods ----

    @property
    def eot_token_id(self) -> int:
        return self._eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        if add_special_tokens is None:
            add_special_tokens = False
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_batch_encode(
        self,
        strings: list[str],
        left_truncate_len: int | None = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self.tokenizer(
            strings,
            padding="longest",
            truncation=truncation,
            max_length=left_truncate_len if truncation else None,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"]
        attn_mask = encoding["attention_mask"]

        if left_truncate_len is not None:
            input_ids = input_ids[:, -left_truncate_len:]
            attn_mask = attn_mask[:, -left_truncate_len:]

        return input_ids, attn_mask

    # ---- Core model calls ----

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return logits [batch, seq, vocab]."""
        with torch.no_grad(), torch.autocast(
            device_type=self._device.type, dtype=self._dtype
        ):
            output = self._model(input_ids)
            logits = output.logits[:, :, : self._vocab_size]
            return logits

    # ---- loglikelihood ----

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        res = []

        def _collate(req):
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        n_reordered = len(re_ord)
        batch_size = override_bs if override_bs is not None else self._batch_size
        chunks = re_ord.get_batched(n=batch_size)

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            for _, context_enc, continuation_enc in chunk:
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self._max_length

                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self._max_length + 1):][:-1],
                    dtype=torch.long,
                    device=self._device,
                )
                inplen = inp.shape[0]

                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # Pad to the longest in the batch
            max_len = max(t.shape[0] for t in inps)
            padded = torch.full(
                (len(inps), max_len),
                self.tokenizer.pad_token_id or 0,
                dtype=torch.long,
                device=self._device,
            )
            for i, inp in enumerate(inps):
                padded[i, : inp.shape[0]] = inp

            multi_logits = F.log_softmax(
                self._model_call(padded), dim=-1
            )  # [batch, seq, vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                contlen = len(cont_toks)
                # Select continuation logits only
                logits = logits[inplen - contlen : inplen]  # [contlen, vocab]
                logits = logits.unsqueeze(0)  # [1, contlen, vocab]

                greedy_tokens = logits.argmax(dim=-1)

                cont_toks_t = torch.tensor(
                    cont_toks, dtype=torch.long, device=self._device
                ).unsqueeze(0)  # [1, contlen]

                max_equal = (greedy_tokens == cont_toks_t).all()

                log_probs = torch.gather(
                    logits, 2, cont_toks_t.unsqueeze(-1)
                ).squeeze(-1)  # [1, contlen]

                answer = (float(log_probs.sum()), bool(max_equal))
                res.append(answer)

                if request_str is not None:
                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    # ---- loglikelihood_rolling (for perplexity) ----

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        all_windows = []
        request_window_counts = []

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=disable_tqdm,
                desc="Preparing rolling windows",
            )
        ):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self._max_length,
                        context_len=1,
                    ),
                )
            )
            windows = [(None,) + x for x in rolling_token_windows]
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        all_nlls = []
        for i in range(0, len(all_windows), self._batch_size):
            batch = all_windows[i : i + self._batch_size]
            batch_indices, batch_windows = zip(*batch)

            batch_nlls = self._loglikelihood_tokens(
                requests=list(batch_windows),
                disable_tqdm=True,
                override_bs=len(batch_windows),
            )
            all_nlls.extend(zip(batch_indices, batch_nlls))

        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )
            current_idx += window_count

        return loglikelihoods

    # ---- generate_until (for generative tasks) ----

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        res = []

        def _collate(req):
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        re_ords = Collator(
            [req.args for req in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running generate_until requests",
        )

        for chunk in re_ords.get_batched(n=self._batch_size):
            contexts, all_gen_kwargs = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            assert isinstance(gen_kwargs, dict)

            kwargs = normalize_gen_kwargs(gen_kwargs, self._max_gen_toks)
            until = handle_stop_sequences(
                kwargs.pop("until", None), eos=self._eos_str
            )
            max_gen_toks = kwargs.pop("max_gen_toks")
            max_ctx_len = self._max_length - max_gen_toks
            max_stop_len = max((len(s) for s in until), default=0)

            for context in contexts:
                ctx_tokens = self.tok_encode(context)
                ctx_tokens = ctx_tokens[-max_ctx_len:]
                input_ids = torch.tensor(
                    [ctx_tokens], dtype=torch.long, device=self._device
                )

                generated_tokens = []
                decoded_so_far = ""
                past_key_values = None

                for _ in range(max_gen_toks):
                    with torch.no_grad(), torch.autocast(
                        device_type=self._device.type, dtype=self._dtype
                    ):
                        fwd_ids = (
                            input_ids[:, -1:]
                            if past_key_values is not None
                            else input_ids
                        )
                        output = self._model(
                            fwd_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                            last_logits_only=True,
                        )

                    past_key_values = output.attn_key_values
                    logits = output.logits[:, -1, : self._vocab_size]
                    next_token = logits.argmax(dim=-1)
                    next_token_id = next_token.item()

                    if next_token_id == self._eos_token_id:
                        break

                    generated_tokens.append(next_token_id)
                    # Feed next token for KV-cache step
                    input_ids = next_token.unsqueeze(0).unsqueeze(0)

                    # Incremental stop-sequence check on the tail
                    decoded_so_far += self.tok_decode(
                        [next_token_id], skip_special_tokens=False
                    )
                    tail = decoded_so_far[-max_stop_len * 2 :] if max_stop_len else ""
                    if any(s in tail for s in until):
                        break

                s = self.tok_decode(generated_tokens)
                for stop_seq in until:
                    idx = s.find(stop_seq)
                    if idx != -1:
                        s = s[:idx]

                res.append(s)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), s
                )
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
