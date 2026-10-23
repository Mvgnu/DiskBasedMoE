"""Read-only, bounded-buffer file benchmark. Buffered reads may hit OS page cache."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import stat
import time
from pathlib import Path


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def benchmark_file(
    path: Path,
    *,
    block_bytes: int = 8 * 2**20,
    max_bytes: int = 2**30,
    passes: int = 1,
    pattern: str = "sequential",
    seed: int = 0,
) -> dict:
    """Read an existing regular file without writes, cache drops or prewarming.

    Random mode samples aligned blocks with replacement across the whole file.
    max_bytes bounds *each pass*. Repeated passes are explicitly page-cache-prone.
    """
    for name, value in (("block_bytes", block_bytes), ("max_bytes", max_bytes), ("passes", passes)):
        _positive(name, value)
    if block_bytes > 64 * 2**20:
        raise ValueError("block_bytes must be <= 64 MiB (bounded allocation)")
    if pattern not in {"sequential", "random"}:
        raise ValueError("pattern must be sequential or random")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    runs = []
    path = Path(path).resolve(strict=True)
    # fstat prevents accidental reads from pipes/devices even after path resolution.
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
            raise ValueError("benchmark requires a nonempty existing regular file")
        size = info.st_size
        buffer = bytearray(min(block_bytes, max_bytes, size))
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as handle:
            for pass_number in range(1, passes + 1):
                consumed = reads = 0
                goal = min(max_bytes, size) if pattern == "sequential" else max_bytes
                started = time.perf_counter()
                while consumed < goal:
                    offset = (
                        consumed
                        if pattern == "sequential"
                        else rng.randrange(math.ceil(size / len(buffer))) * len(buffer)
                    )
                    wanted = min(len(buffer), goal - consumed, size - offset)
                    handle.seek(offset)
                    got = handle.readinto(memoryview(buffer)[:wanted])
                    if got != wanted:
                        raise OSError("file changed/truncated during benchmark (short read)")
                    consumed += got
                    reads += 1
                elapsed = time.perf_counter() - started
                runs.append(
                    {
                        "pass": pass_number,
                        "bytes_read": consumed,
                        "read_calls": reads,
                        "elapsed_seconds": elapsed,
                        "gb_per_second": consumed / elapsed / 10**9,
                        "gib_per_second": consumed / elapsed / 2**30,
                    }
                )
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns) != (info.st_size, info.st_mtime_ns):
            raise OSError("file changed during benchmark; discard result")
    finally:
        os.close(descriptor)
    return {
        "kind": "measured_buffered_file_reads_not_inference",
        "path": str(path),
        "file_bytes": size,
        "block_bytes": block_bytes,
        "max_bytes_per_pass": max_bytes,
        "pattern": pattern,
        "seed": seed,
        "io_mode": "buffered OS reads (Python userspace buffering disabled; O_DIRECT not used)",
        "cache_state": "uncontrolled; first pass may already be cached; later passes may be warmer",
        "physical_disk_bytes": None,
        "buffer_bytes": len(buffer),
        "environment": {"system": platform.platform(), "python": platform.python_version()},
        "runs": runs,
        "limits": "Do not use a RAM-fitting or sparse file result as measured SSD bandwidth. No system caches were dropped. No file was created, modified or deleted.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--block-mib", type=int, default=8)
    parser.add_argument("--max-mib", type=int, default=1024, help="bytes read per pass, in MiB")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--pattern", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        result = benchmark_file(
            args.path,
            block_bytes=args.block_mib * 2**20,
            max_bytes=args.max_mib * 2**20,
            passes=args.passes,
            pattern=args.pattern,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
