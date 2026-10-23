"""Single-consumer generation with native request-local KV state and measurements."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import torch
from transformers import AutoTokenizer

from disk_moe.contracts import GenerationEvent

from .artifact import Artifact
from .model import load_model


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class _TextStream:
    """HF TextStreamer-style word buffering with byte-fallback safety.

Decode the entire generated token prefix without cleanup and emit only complete
words/lines (or CJK characters). Holding incomplete UTF-8 fallback characters and
the final word avoids broken multi-token Unicode and whitespace cleanup rewrites.
"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokens = []
        self.printed = ""

    def add(self, token_id: int) -> str:
        self.tokens.append(token_id)
        return self._emit(final=False)

    def finish(self) -> str:
        return self._emit(final=True)

    def _emit(self, final):
        text = self.tokenizer.decode(self.tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not text.startswith(self.printed):
            raise ValueError("Tokenizer rewrote an already emitted text prefix; this tokenizer is unsupported for append-only streaming")
        if final:
            cut = len(text)
        elif "\ufffd" in text[len(self.printed):]:
            cut = len(self.printed)
        elif text.endswith("\n"):
            cut = len(text)
        elif text and (0x4E00 <= ord(text[-1]) <= 0x9FFF):
            cut = len(text)
        else:
            cut = text.rfind(" ") + 1
        cut = max(cut, len(self.printed))
        delta = text[len(self.printed):cut]
        self.printed = text[:cut]
        return delta


class Glm4Engine:
    def __init__(self, artifact_dir, *, tokenizer=None, tokenizer_name=None, device="cpu", cache_budget_bytes=268435456, attn_implementation="eager"):
        self.artifact = Artifact(artifact_dir)
        if tokenizer is None:
            source = tokenizer_name or Path(artifact_dir) / "tokenizer"
            if tokenizer_name is None and not Path(source).is_dir():
                raise ValueError("This artifact has no tokenizer; pass tokenizer_name or an explicit tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
        self.tokenizer = tokenizer
        self.model = load_model(self.artifact, device=device, cache_budget_bytes=cache_budget_bytes, attn_implementation=attn_implementation)
        self.device = torch.device(device)
        self.max_context = self.model.config.max_position_embeddings

    def format_messages(self, messages: Sequence[Mapping[str, str]]) -> str:
        if not messages or any(not isinstance(message, Mapping) or message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("messages must contain role/content strings")
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("Tokenizer has no native chat template; use the prompt input instead")
        return self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)

    def _encode(self, prompt):
        # Prompt is either already formatted by the checkpoint's chat template or
        # explicitly supplied text. Avoid inserting a second BOS/EOS around it.
        result = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        ids = result["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Prompt must encode to at least one token in a single sequence")
        if ids.min().item() < 0 or ids.max().item() >= self.model.config.vocab_size:
            raise ValueError("Tokenizer IDs exceed this model's vocabulary")
        return ids

    def validate_request(self, prompt: str, max_new_tokens: int) -> None:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        ids = self._encode(prompt)
        if ids.shape[1] + max_new_tokens > self.max_context:
            raise ValueError(f"Prompt plus output limit exceeds the {self.max_context}-token native context window")

    def describe(self) -> dict[str, object]:
        return {
            "backend": "glm4_moe_lite_disk",
            "native_transformers_version": self.artifact.manifest["transformers_version"],
            "dtype": self.artifact.manifest["dtype"],
            "device": str(self.device),
            "native_top_k": self.model.config.num_experts_per_tok,
            "context_window": self.max_context,
            "resident_tensor_bytes": self.artifact.resident_tensor_bytes,
            "resident_bytes_scope": "serialized resident tensors; excludes KV/activations/staging/allocator/OS cache",
            **self.model.expert_cache.stats(),
        }

    def generate(self, prompt: str, max_new_tokens: int) -> Iterator[GenerationEvent]:
        self.validate_request(prompt, max_new_tokens)
        ids = self._encode(prompt).to(self.device)
        prompt_tokens = ids.shape[1]
        cache_before = self.model.expert_cache.stats()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
        stream = _TextStream(self.tokenizer)
        past = None
        prefill_seconds = 0.0
        decode_seconds = 0.0
        decode_tokens = 0
        generated = 0
        reason = "length"
        # The context manager stays inside the iterator; closing it unwinds local
        # KV/tensor references at the next cooperative consumer boundary.
        try:
            with torch.inference_mode():
                for step in range(max_new_tokens):
                    _synchronize(self.device)
                    start = time.perf_counter()
                    output = self.model(ids, past_key_values=past, use_cache=True, logits_to_keep=1)
                    logits = output.logits[:, -1, :]
                    if not torch.isfinite(logits).all():
                        raise ValueError("Model produced non-finite logits")
                    token_id = int(logits.argmax(dim=-1).item())
                    _synchronize(self.device)
                    elapsed = time.perf_counter() - start
                    if step == 0:
                        prefill_seconds = elapsed
                    else:
                        decode_seconds += elapsed
                        decode_tokens += 1
                    past = output.past_key_values
                    del output, logits
                    generated += 1
                    if token_id in eos_ids:
                        reason = "stop"
                        yield GenerationEvent(token_id=token_id)
                        break
                    delta = stream.add(token_id)
                    yield GenerationEvent(delta=delta, token_id=token_id)
                    ids = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
                tail = stream.finish()
                if tail:
                    yield GenerationEvent(delta=tail)
                after = self.model.expert_cache.stats()
                counters = {key: int(after[key]) - int(cache_before[key]) for key in ("cache_hits", "cache_misses", "cache_evictions", "cache_bypasses", "logical_bytes_read")}
                rate = decode_tokens / decode_seconds if decode_tokens and decode_seconds else 0.0
                if not math.isfinite(rate):
                    rate = 0.0
                yield GenerationEvent(finish_reason=reason, metrics={
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated,
                    "prefill_seconds": prefill_seconds,
                    "decode_seconds": decode_seconds,
                    "decode_tokens": decode_tokens,
                    "decode_tokens_per_second": rate,
                    "timing_scope": "synchronized model forward + greedy selection; excludes tokenization, text decode and consumer waits",
                    "cache_retained_bytes": int(after["cache_retained_bytes"]),
                    "cache_budget_bytes": int(after["cache_budget_bytes"]),
                    "resident_tensor_bytes": self.artifact.resident_tensor_bytes,
                    "io_measurement": str(after["io_measurement"]),
                    **counters,
                })
        finally:
            past = None
