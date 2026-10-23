"""Capacity and bandwidth arithmetic, explicitly not an inference benchmark."""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

GB = 10**9
GIB = 2**30


def finite(name: str, value: float, *, minimum: float = 0, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum or (positive and value == minimum):
        op = ">" if positive else ">="
        raise ValueError(f"{name} must be finite and {op} {minimum}")


def integer(name: str, value: int, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class MemoryBudget:
    """One actual memory arena, never an implicit sum of host RAM and VRAM."""

    capacity_gib: float
    resident_weights_gib: float
    kv_gib: float
    staging_gib: float
    reserve_gib: float
    expert_cache_gib: float
    arena: str = "device"

    def report(self) -> dict:
        for name, value in asdict(self).items():
            if name != "arena":
                finite(name, value)
        if self.arena not in {"host", "device", "unified"}:
            raise ValueError("arena must be host, device or unified; do not pool separate memory")
        used = sum(
            (
                self.resident_weights_gib,
                self.kv_gib,
                self.staging_gib,
                self.reserve_gib,
                self.expert_cache_gib,
            )
        )
        if used > self.capacity_gib:
            raise ValueError(
                f"{self.arena} memory does not fit: {used:g} GiB > {self.capacity_gib:g} GiB"
            )
        return {
            **asdict(self),
            "allocated_gib": used,
            "unallocated_gib": self.capacity_gib - used,
            "scope": "one memory arena; other arenas require their own budget",
        }


@dataclass(frozen=True)
class PlanInput:
    total_params: int
    routed_params_per_token: int
    effective_bits: float
    read_gbps: float
    cache_hit_rate: float = 0
    other_seconds_per_token: float = 0
    target_tps: float = 1
    read_amplification: float = 1
    extra_storage_bytes: int = 0
    memory: MemoryBudget | None = None


def _rate(seconds: float) -> float | None:
    return 1 / seconds if seconds else None


def estimate(inputs: PlanInput) -> dict:
    integer("total_params", inputs.total_params, minimum=1)
    integer("routed_params_per_token", inputs.routed_params_per_token, minimum=1)
    integer("extra_storage_bytes", inputs.extra_storage_bytes)
    if inputs.routed_params_per_token > inputs.total_params:
        raise ValueError("routed_params_per_token cannot exceed total_params")
    for name in ("effective_bits", "read_gbps", "target_tps"):
        finite(name, getattr(inputs, name), positive=True)
    finite("other_seconds_per_token", inputs.other_seconds_per_token)
    finite("read_amplification", inputs.read_amplification, minimum=1)
    finite("cache_hit_rate", inputs.cache_hit_rate)
    if inputs.cache_hit_rate > 1:
        raise ValueError("cache_hit_rate must be in [0, 1]")
    if inputs.memory is not None and not isinstance(inputs.memory, MemoryBudget):
        raise ValueError("memory must be a MemoryBudget")
    memory = inputs.memory.report() if inputs.memory else None

    payload = math.ceil(inputs.total_params * inputs.effective_bits / 8)
    selected = math.ceil(inputs.routed_params_per_token * inputs.effective_bits / 8)
    cold_reads = selected * inputs.read_amplification
    misses = cold_reads * (1 - inputs.cache_hit_rate)
    io_seconds = misses / (inputs.read_gbps * GB)
    serial = io_seconds + inputs.other_seconds_per_token
    overlapped = max(io_seconds, inputs.other_seconds_per_token)
    time_left = 1 / inputs.target_tps - inputs.other_seconds_per_token
    required_hit = (
        None if time_left < 0 else max(0.0, 1 - time_left * inputs.read_gbps * GB / cold_reads)
    )
    return {
        "kind": "analytical_scenario_not_a_benchmark",
        "inputs": asdict(inputs),
        "capacity": {
            "payload_bytes": payload,
            "extra_storage_bytes": inputs.extra_storage_bytes,
            "total_gb": (payload + inputs.extra_storage_bytes) / GB,
            "total_gib": (payload + inputs.extra_storage_bytes) / GIB,
        },
        "traffic": {
            "selected_routed_bytes_per_token": selected,
            "cold_read_bytes_per_token": cold_reads,
            "miss_bytes_per_token": misses,
            "miss_gb_per_token": misses / GB,
        },
        "latency": {
            "io_seconds_per_token": io_seconds,
            "serial_seconds_per_token": serial,
            "serial_tokens_per_second": _rate(serial),
            "perfect_overlap_seconds_per_token": overlapped,
            "perfect_overlap_tokens_per_second": _rate(overlapped),
        },
        "target": {
            "tokens_per_second": inputs.target_tps,
            "required_byte_hit_rate_serial": required_hit,
            "serial_target_possible_given_other_time": required_hit is not None,
            "scenario_meets_serial_target": serial <= 1 / inputs.target_tps,
        },
        "memory": memory,
        "limits": [
            "Decode only. Prefill, TTFT and model output quality are not predicted.",
            "Cache hit rate is an explicit byte-weighted assumption, not inferred from capacity.",
            "Assumes attention, shared/dense weights and KV fit in their declared memory arena.",
            "Read bandwidth must be measured at representative request sizes; page-cache speed is not SSD speed.",
            "Other time must include uncovered compute, H2D, routing, allocation and synchronization.",
            "Perfect overlap is an optimistic bound and may be unattainable with sequential layer dependencies.",
            "Uniform effective bits is a scenario; mixed precision and duplicate preparation artifacts add capacity.",
            "A null rate means the inputs model zero latency, not measured infinite throughput.",
        ],
    }


@dataclass(frozen=True)
class ExpertAccess:
    token: int
    layer: int
    expert: int
    bytes: int


def replay_trace(accesses: Iterable[ExpertAccess], budget_bytes: int) -> dict:
    """Cold-start LRU replay of ordered logical accesses, including oversize bypass."""
    integer("budget_bytes", budget_bytes)
    cache: OrderedDict[tuple[int, int], int] = OrderedDict()
    sizes: dict[tuple[int, int], int] = {}
    retained = peak = hits = misses = evictions = selected = read_bytes = count = 0
    tokens = []
    current = None
    for access in accesses:
        for name in ("token", "layer", "expert", "bytes"):
            integer(name, getattr(access, name), minimum=1 if name == "bytes" else 0)
        if current is not None and access.token < current["token"]:
            raise ValueError("trace tokens must be nondecreasing; preserve execution order")
        if current is None or current["token"] != access.token:
            current = {"token": access.token, "selected_bytes": 0, "read_bytes": 0}
            tokens.append(current)
        key = (access.layer, access.expert)
        if key in sizes and sizes[key] != access.bytes:
            raise ValueError(f"expert {key} changed size within one trace")
        sizes[key] = access.bytes
        selected += access.bytes
        current["selected_bytes"] += access.bytes
        count += 1
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            continue
        misses += 1
        read_bytes += access.bytes
        current["read_bytes"] += access.bytes
        if access.bytes <= budget_bytes:
            while retained + access.bytes > budget_bytes:
                _, size = cache.popitem(last=False)
                retained -= size
                evictions += 1
            cache[key] = access.bytes
            retained += access.bytes
            peak = max(peak, retained)
    return {
        "kind": "logical_trace_simulation_not_physical_io",
        "policy": "global LRU, initially empty; oversize experts bypass cache",
        "budget_bytes": budget_bytes,
        "accesses": count,
        "tokens": len(tokens),
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "retained_bytes": retained,
        "peak_retained_bytes": peak,
        "selected_bytes": selected,
        "read_bytes": read_bytes,
        "byte_hit_rate": 1 - read_bytes / selected if selected else 0,
        "per_token": tokens,
        "limits": "No prefetch, batching, OS page cache, transfer overlap or admission prediction is simulated.",
    }


def read_trace(path: Path) -> Iterable[ExpertAccess]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield ExpertAccess(**json.loads(line))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"trace line {line_number}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=Path, help="JSON with a parameters object (see examples)")
    group.add_argument("--trace", type=Path, help="JSONL: token, layer, expert, bytes")
    parser.add_argument("--cache-bytes", type=int, default=0, help="global trace cache budget")
    parser.add_argument("--read-gbps", type=float, help="override decimal GB/s")
    parser.add_argument(
        "--cache-hit-rate", type=float, help="override byte-weighted hit rate [0,1]"
    )
    parser.add_argument("--other-seconds", type=float, help="override non-IO time/token")
    args = parser.parse_args()
    try:
        if args.trace:
            if any(
                value is not None
                for value in (args.read_gbps, args.cache_hit_rate, args.other_seconds)
            ):
                raise ValueError("bandwidth/latency overrides apply only to --scenario")
            result = replay_trace(read_trace(args.trace), args.cache_bytes)
        else:
            scenario = json.loads(args.scenario.read_text())
            parameters = dict(scenario["parameters"])
            for name, value in (
                ("read_gbps", args.read_gbps),
                ("cache_hit_rate", args.cache_hit_rate),
                ("other_seconds_per_token", args.other_seconds),
            ):
                if value is not None:
                    parameters[name] = value
            if parameters.get("memory") is not None:
                parameters["memory"] = MemoryBudget(**parameters["memory"])
            result = estimate(PlanInput(**parameters))
            result["scenario_description"] = scenario.get("description", "")
            result["sources"] = scenario.get("sources", [])
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
