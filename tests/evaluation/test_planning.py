import json
from dataclasses import replace
from pathlib import Path

import pytest

from disk_moe.evaluation.planning import (
    ExpertAccess,
    MemoryBudget,
    PlanInput,
    estimate,
    replay_trace,
)


def test_named_kimi_profile_counts_all_native_geometry():
    # Independent parameter ledger from the cited model config/implementation.
    scenario = Path(__file__).resolve().parents[2] / "examples/kimi-k2-q4-scenario.json"
    parameters = json.loads(scenario.read_text())["parameters"]
    routed = 60 * 384 * 3 * 7168 * 2048
    shared = 60 * 3 * 7168 * 2048
    routers = 60 * 384 * 7168
    attention_per_layer = (
        7168 * 1536 + 1536 + 1536 * 64 * 192 + 7168 * 576 + 512 + 512 * 64 * 256 + 7168 * 64 * 128
    )
    dense = 3 * 7168 * 18432
    norms = 61 * 2 * 7168 + 7168
    embeddings_and_head = 2 * 163840 * 7168
    expected = (
        routed + shared + routers + 61 * attention_per_layer + dense + norms + embeddings_and_head
    )
    assert parameters["total_params"] == expected
    assert parameters["extra_storage_bytes"] == 60 * 384 * 4
    result = estimate(PlanInput(**parameters))
    assert result["capacity"]["payload_bytes"] > routed * 18 // 32


def kimi():
    # Independent derivation: 60 layers, 8 selected, 3 SwiGLU matrices, 7168x2048.
    return PlanInput(10**12, 60 * 8 * 3 * 7168 * 2048, 4.5, 7, other_seconds_per_token=0.25)


def test_kimi_capacity_and_required_hit_rate():
    result = estimate(kimi())
    assert result["capacity"]["total_gb"] == 562.5
    assert result["traffic"]["selected_routed_bytes_per_token"] == 11_890_851_840
    assert result["latency"]["serial_seconds_per_token"] == pytest.approx(11.89085184 / 7 + 0.25)
    threshold = result["target"]["required_byte_hit_rate_serial"]
    assert threshold == pytest.approx(0.5584841127748843)
    assert estimate(replace(kimi(), cache_hit_rate=threshold + 1e-9))["target"][
        "scenario_meets_serial_target"
    ]
    assert not estimate(replace(kimi(), cache_hit_rate=threshold - 1e-9))["target"][
        "scenario_meets_serial_target"
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"cache_hit_rate": 1.1},
        {"cache_hit_rate": float("nan")},
        {"read_gbps": 0},
        {"effective_bits": float("inf")},
        {"total_params": True},
        {"routed_params_per_token": 10**13},
        {"other_seconds_per_token": -1},
        {"read_amplification": 0.9},
    ],
)
def test_nonphysical_inputs_rejected(changes):
    with pytest.raises(ValueError):
        estimate(replace(kimi(), **changes))


def test_compute_bound_target_and_zero_latency_are_not_speed_claims():
    slow = estimate(replace(kimi(), cache_hit_rate=1, other_seconds_per_token=2))
    assert slow["target"]["required_byte_hit_rate_serial"] is None
    assert not slow["target"]["scenario_meets_serial_target"]
    zero = estimate(replace(kimi(), cache_hit_rate=1, other_seconds_per_token=0))
    assert zero["latency"]["serial_tokens_per_second"] is None
    json.dumps(zero, allow_nan=False)


def test_memory_is_single_arena_and_impossible_fit_fails():
    fits = MemoryBudget(32, 10, 2, 1, 3, 16)
    assert estimate(replace(kimi(), memory=fits))["memory"]["unallocated_gib"] == 0
    with pytest.raises(ValueError, match="does not fit"):
        estimate(replace(kimi(), memory=replace(fits, expert_cache_gib=17)))
    with pytest.raises(ValueError, match="arena"):
        replace(fits, arena="host+device").report()


def test_trace_is_byte_weighted_and_layer_scoped():
    accesses = [
        ExpertAccess(0, 0, 0, 4),
        ExpertAccess(0, 1, 0, 8),
        ExpertAccess(1, 0, 0, 4),
        ExpertAccess(1, 1, 0, 8),
    ]
    report = replay_trace(accesses, 12)
    assert report["hits"] == 2
    assert report["read_bytes"] == 12
    assert report["byte_hit_rate"] == 0.5
    assert report["peak_retained_bytes"] == 12
    assert replay_trace(accesses, 8)["hits"] == 0


def test_trace_oversize_bypass_does_not_evict_fitting_entry():
    accesses = [ExpertAccess(0, 0, 0, 4), ExpertAccess(0, 0, 1, 20), ExpertAccess(1, 0, 0, 4)]
    report = replay_trace(accesses, 4)
    assert report["hits"] == 1
    assert report["read_bytes"] == 24
    assert report["peak_retained_bytes"] == 4
    assert replay_trace(accesses, 0)["retained_bytes"] == 0


def test_trace_rejects_changed_sizes_and_unordered_tokens():
    with pytest.raises(ValueError, match="changed size"):
        replay_trace([ExpertAccess(0, 0, 0, 4), ExpertAccess(1, 0, 0, 5)], 4)
    with pytest.raises(ValueError, match="nondecreasing"):
        replay_trace([ExpertAccess(1, 0, 0, 4), ExpertAccess(0, 0, 0, 4)], 4)
