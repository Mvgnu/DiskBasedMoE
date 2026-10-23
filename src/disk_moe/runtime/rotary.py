"""Preserve static native rotary buffers, including dtype-conversion rounding.

These buffers are nonpersistent in Hugging Face checkpoints. A live model that
has undergone ``model.to(bfloat16)`` can therefore differ from reloading its
checkpoint, whose rotary constructor creates FP32 frequencies. In-memory export
must not erase that distinction. Arbitrarily modified or dynamic rotary state
requires a separate contract and is rejected by this baseline.
"""

from __future__ import annotations

import math

import torch
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteRotaryEmbedding

from .artifact import DTYPES

ROTARY_BUFFERS = {"inv_freq", "original_inv_freq"}


def validate_static_rope_config(config) -> None:
    """Reject unsupported RoPE policy from metadata without touching buffers."""
    rope_type = config.rope_parameters["rope_type"]
    if not isinstance(rope_type, str):
        raise ValueError("Native rotary type must be a supported static RoPE name")
    if "dynamic" in rope_type or rope_type == "longrope":
        raise ValueError("Dynamic/longrope rotary state is unsupported; use a supported static native RoPE checkpoint")


def validate_static_rotary(rotary, config) -> None:
    if type(rotary) is not Glm4MoeLiteRotaryEmbedding:
        raise ValueError("Export requires an unmodified native Glm4MoeLiteRotaryEmbedding")
    validate_static_rope_config(config)
    reference = Glm4MoeLiteRotaryEmbedding(config)
    if rotary.rope_type != reference.rope_type:
        raise ValueError("Native rotary type differs from its config; restore the model's native static rotary state")
    if set(dict(rotary.named_buffers())) != ROTARY_BUFFERS:
        raise ValueError("Unexpected native rotary buffers; only static inv_freq/original_inv_freq are supported")
    if rotary.max_seq_len_cached != reference.max_seq_len_cached or rotary.original_max_seq_len != reference.original_max_seq_len:
        raise ValueError("Mutated rotary sequence state is unsupported; reload the checkpoint before export")
    if not isinstance(rotary.attention_scaling, (float, int)) or not math.isfinite(rotary.attention_scaling) or rotary.attention_scaling != reference.attention_scaling:
        raise ValueError("Mutated rotary attention scaling is unsupported; reload the checkpoint before export")
    for name in ROTARY_BUFFERS:
        actual = getattr(rotary, name)
        if actual.device.type == "meta" or actual.dtype not in DTYPES.values():
            raise ValueError(f"Unsupported native rotary {name} device or dtype")
        expected = getattr(reference, name).to(dtype=actual.dtype)
        if actual.shape != expected.shape or not torch.equal(actual.detach().cpu(), expected):
            raise ValueError(f"Mutated native rotary {name} is unsupported; reload the checkpoint before export")


def snapshot_static_rotary(rotary, config) -> dict[str, torch.Tensor]:
    validate_static_rotary(rotary, config)
    return {name: getattr(rotary, name).detach().cpu().contiguous().clone() for name in sorted(ROTARY_BUFFERS)}


def restore_static_rotary(config, buffers: dict[str, torch.Tensor]):
    if set(buffers) != ROTARY_BUFFERS:
        raise ValueError("Artifact must contain both native static rotary buffers")
    rotary = Glm4MoeLiteRotaryEmbedding(config)
    for name, tensor in buffers.items():
        rotary.register_buffer(name, tensor, persistent=False)
    validate_static_rotary(rotary, config)
    return rotary
