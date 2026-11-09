"""Native GLM4 model with only its routed-expert delivery replaced."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Glm4MoeLiteForCausalLM
from transformers.activations import ACT2FN

from .artifact import Artifact
from .cache import ExpertCache
from .export import _check_config, _native_version
from .rotary import restore_static_rotary


class DiskExperts(nn.Module):
    """Apply the same native SwiGLU expert computation to demand-loaded weights.

Routing indices and coefficients come from the untouched native router. Expert
order matches native eager execution; each expert is released before the next
cache lookup. CPU/GPU transient workspaces are not part of retained cache bytes.
"""

    def __init__(self, config, layer: int, cache: ExpertCache):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer = layer
        self.cache = cache

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        # Avoid a prompt-length x expert-count one-hot allocation. Native eager
        # executes touched experts in increasing ID order; preserve that order.
        for expert in torch.unique(top_k_index, sorted=True).tolist():
            if not 0 <= expert < self.num_experts:
                raise ValueError(f"Native router returned invalid expert {expert}")
            # Native eager enumerates top-k positions before token rows. Avoid
            # a separate one-hot allocation without changing that enumeration.
            top_k_positions, token_indices = torch.where(top_k_index.transpose(0, 1) == expert)
            current = hidden_states[token_indices]
            weights = self.cache.get(self.layer, expert)
            gate, up = F.linear(current, weights["gate_up_proj"]).chunk(2, dim=-1)
            values = F.linear(self.act_fn(gate) * up, weights["down_proj"])
            values = values * top_k_weights[token_indices, top_k_positions, None]
            result.index_add_(0, token_indices, values.to(result.dtype))
            del weights
        return result


def load_model(artifact_dir, *, device="cpu", cache_budget_bytes=268435456, attn_implementation="eager"):
    """Return a native CausalLM with a shared demand-loaded expert cache.

Constructing on meta avoids ever allocating all expert weights. This supported
baseline uses eager/SDPA attention; quantized/fused expert kernels are not claimed.
    """
    _native_version()
    artifact = artifact_dir if isinstance(artifact_dir, Artifact) else Artifact(artifact_dir)
    _check_config(artifact.config)
    if attn_implementation not in {"eager", "sdpa"}:
        raise ValueError("Supported attention implementations are eager and sdpa")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if str(device).startswith("mps") and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    artifact.config._attn_implementation = attn_implementation
    artifact.config._experts_implementation = "eager"
    cache = ExpertCache(artifact, cache_budget_bytes, device=device)
    with torch.device("meta"):
        model = Glm4MoeLiteForCausalLM(artifact.config)
    for index, layer in enumerate(model.model.layers):
        if artifact.config.mlp_layer_types[index] == "sparse":
            layer.mlp.experts = DiskExperts(artifact.config, index, cache)
    expected = model.state_dict()
    seen = set()
    for entry in artifact.resident:
        tensors = artifact.read(entry)
        for name, tensor in tensors.items():
            if name not in expected or tensor.shape != expected[name].shape:
                raise ValueError(f"Resident tensor not expected by native model: {name}")
            wanted_dtype = torch.float32 if name.endswith("e_score_correction_bias") else artifact.dtype
            if tensor.dtype != wanted_dtype:
                raise ValueError(f"Resident tensor has incompatible dtype: {name}")
            seen.add(name)
        model.load_state_dict(tensors, strict=False, assign=True)
        del tensors
    if seen != set(expected):
        raise ValueError(f"Resident tensor set is incomplete: {sorted(set(expected) - seen)[:5]}")
    # Nonpersistent rotary buffers can carry BF16/FP16 conversion rounding in
    # an in-memory source. Restore that recorded state rather than normalizing
    # it to freshly constructed FP32 frequencies and silently changing logits.
    model.model.rotary_emb = restore_static_rotary(artifact.config, artifact.read(artifact.rotary))
    if any(value.device.type == "meta" for value in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Native model contains an unmaterialized tensor")
    if artifact.config.tie_word_embeddings:
        if not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
            raise ValueError("Tied embedding/head checkpoint contains different weights")
        model.tie_weights()
    model.to(device=device)
    model.eval()
    model.requires_grad_(False)
    model.generation_config = artifact.generation_config
    model.expert_cache = cache
    model.disk_artifact = artifact
    return model
