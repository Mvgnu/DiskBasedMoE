"""Single-stream model benchmark; generation requires an explicit local export."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

from disk_moe.contracts import InferenceEngine


def benchmark_engine(
    engine: InferenceEngine,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    repeats: int = 1,
    include_text: bool = False,
) -> dict:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    engine.validate_request(prompt, max_new_tokens)
    runs = []
    for index in range(repeats):
        token_times = []
        chunks = []
        terminal = None
        before = engine.describe()
        started = time.perf_counter()
        iterator = iter(engine.generate(prompt, max_new_tokens))
        try:
            for event in iterator:
                if terminal is not None:
                    raise RuntimeError("engine yielded data after its terminal event")
                if event.token_id is not None:
                    token_times.append(time.perf_counter())
                chunks.append(event.delta)
                if event.finish_reason is not None:
                    terminal = event
        finally:
            close = getattr(iterator, "close", None)
            if close:
                close()
        elapsed = time.perf_counter() - started
        if terminal is None:
            raise RuntimeError("engine ended without a terminal event")
        intervals = [b - a for a, b in zip(token_times, token_times[1:])]
        decode_seconds = sum(intervals)
        ordered = sorted(intervals)
        p95 = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)] if ordered else None
        row = {
            "run": index + 1,
            "elapsed_seconds": elapsed,
            "observed_generated_tokens": len(token_times),
            "observed_ttft_seconds": token_times[0] - started if token_times else None,
            "observed_decode_tokens": len(intervals),
            "observed_decode_seconds": decode_seconds,
            "observed_decode_tokens_per_second": len(intervals) / decode_seconds
            if decode_seconds
            else None,
            "observed_decode_interval_p50_seconds": statistics.median(intervals)
            if intervals
            else None,
            "observed_decode_interval_p95_seconds": p95,
            "observed_decode_interval_max_seconds": max(intervals) if intervals else None,
            "finish_reason": terminal.finish_reason,
            "engine_metrics": dict(terminal.metrics),
            "engine_before": before,
            "engine_after": engine.describe(),
        }
        if include_text:
            row["generated_text"] = "".join(chunks)
        runs.append(row)
    versions = {}
    for name in ("disk-based-moe", "torch", "transformers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "kind": "measured_single_stream_generation",
        "batch_size": 1,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_characters": len(prompt),
        "max_new_tokens": max_new_tokens,
        "cache_state": "application cache persists across repeats; OS page cache is uncontrolled",
        "timing_convention": "TTFT from generate start to first token ID; decode uses N-1 subsequent token intervals. Validation and model setup are excluded. EOS token IDs count; rendered characters do not.",
        "peak_memory": None,
        "physical_disk_bytes": None,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": versions,
        },
        "runs": runs,
        "limits": "Logical loader bytes are not physical SSD reads. Process/GPU peak memory is not measured here. A tiny random fixture validates this harness, not pretrained model quality or trillion-parameter speed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", help="optional local tokenizer path; defaults to export")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--cache-mib", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--include-text", action="store_true")
    args = parser.parse_args()
    try:
        # Composition boundary: the planner/IO modules remain dependency-free.
        from disk_moe.runtime import Glm4Engine

        prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
        started = time.perf_counter()
        engine = Glm4Engine(
            args.model_dir,
            tokenizer_name=args.tokenizer,
            device=args.device,
            cache_budget_bytes=args.cache_mib * 2**20,
        )
        setup_seconds = time.perf_counter() - started
        result = benchmark_engine(
            engine,
            prompt,
            max_new_tokens=args.max_new_tokens,
            repeats=args.repeats,
            include_text=args.include_text,
        )
        result["model_setup_seconds"] = setup_seconds
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
