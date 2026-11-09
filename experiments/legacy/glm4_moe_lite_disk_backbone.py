#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteRMSNorm,
    Glm4MoeLiteRotaryEmbedding,
    Glm4MoeLiteTopkRouter,
    Glm4MoeLiteMLP,
    create_causal_mask,
)


def _load_pt(path: Path, device: str):
    return torch.load(path, map_location=device)


class DiskBackedGlm4MoeLiteExpertCache:
    def __init__(self, expert_dir: Path, device: str, cache_size: int):
        self.expert_dir = expert_dir
        self.device = device
        self.cache_size = cache_size
        self.cache: OrderedDict[int, dict] = OrderedDict()

    def _load_expert(self, idx: int) -> dict:
        state = _load_pt(self.expert_dir / f"expert_{idx}.pt", device="cpu")
        if "gate_up_proj" in state:
            return {
                "format": "gate_up",
                "gate_up_proj": state["gate_up_proj"].to(self.device),
                "down_proj": state["down_proj"].to(self.device),
            }
        return {
            "format": "separate",
            "gate_proj": state["gate_proj"].to(self.device),
            "up_proj": state["up_proj"].to(self.device),
            "down_proj": state["down_proj"].to(self.device),
        }

    def get(self, idx: int) -> dict:
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        weights = self._load_expert(idx)
        self.cache[idx] = weights
        if len(self.cache) > self.cache_size:
            _, evicted = self.cache.popitem(last=False)
            del evicted
        return weights

    def prefetch(self, indices: list[int]) -> None:
        for idx in indices:
            if idx in self.cache:
                self.cache.move_to_end(idx)
                continue
            weights = self._load_expert(idx)
            self.cache[idx] = weights
            if len(self.cache) > self.cache_size:
                _, evicted = self.cache.popitem(last=False)
                del evicted


class DiskBackedGlm4MoeLiteNaiveMoe(nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig, expert_dir: Path, device: str, cache_size: int):
        super().__init__()
        self.config = config
        self.num_experts = getattr(config, "num_local_experts", None) or config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.cache = DiskBackedGlm4MoeLiteExpertCache(expert_dir, device=device, cache_size=cache_size)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            weights = self.cache.get(expert_idx)
            if weights["format"] == "gate_up":
                gate, up = F.linear(current_state, weights["gate_up_proj"]).chunk(2, dim=-1)
                down = weights["down_proj"]
            else:
                gate = F.linear(current_state, weights["gate_proj"])
                up = F.linear(current_state, weights["up_proj"])
                down = weights["down_proj"]
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, down)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def prefetch_experts(self, expert_ids: list[int]) -> None:
        self.cache.prefetch(expert_ids)


class DiskBackedGlm4MoeLiteMoE(nn.Module):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        expert_dir: Path,
        device: str,
        cache_size: int,
        top_k_override: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.experts = DiskBackedGlm4MoeLiteNaiveMoe(config, expert_dir, device=device, cache_size=cache_size)
        self.gate = Glm4MoeLiteTopkRouter(config)
        self.shared_experts = Glm4MoeLiteMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = top_k_override if top_k_override is not None else config.num_experts_per_tok
        if self.top_k < 1 or self.top_k > self.n_routed_experts:
            raise ValueError(f"top_k must be between 1 and {self.n_routed_experts}, got {self.top_k}")

    def load_gate(self, state_dict: dict) -> None:
        self.gate.load_state_dict(state_dict)

    def load_shared_experts(self, state_dict: dict) -> None:
        self.shared_experts.load_state_dict(state_dict)

    def set_top_k(self, top_k: int) -> None:
        if top_k < 1 or top_k > self.n_routed_experts:
            raise ValueError(f"top_k must be between 1 and {self.n_routed_experts}, got {top_k}")
        self.top_k = top_k

    def prefetch_experts(self, expert_ids: list[int]) -> None:
        unique_ids = sorted(set(int(x) for x in expert_ids if 0 <= int(x) < self.n_routed_experts))
        if not unique_ids:
            return
        self.experts.prefetch_experts(unique_ids)

    def route_tokens_to_experts(self, router_logits: torch.Tensor):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor, return_router_inputs: bool = False):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        if return_router_inputs:
            router_inputs = residuals
            return hidden_states, router_logits, router_inputs
        return hidden_states, router_logits, None


class DiskBackedGlm4MoeLiteDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        layer_idx: int,
        expert_dir: Path,
        device: str,
        cache_size: int,
        mlp_type: str,
        router_top_k: Optional[int] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Glm4MoeLiteAttention(config, layer_idx)
        self.input_layernorm = Glm4MoeLiteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4MoeLiteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_type = mlp_type
        if mlp_type == "sparse":
            self.mlp = DiskBackedGlm4MoeLiteMoE(
                config,
                expert_dir=expert_dir,
                device=device,
                cache_size=cache_size,
                top_k_override=router_top_k,
            )
        else:
            self.mlp = Glm4MoeLiteMLP(config)

    def load_weights(self, layer_state: dict) -> None:
        self.self_attn.load_state_dict(layer_state["self_attn"])
        self.input_layernorm.load_state_dict(layer_state["input_layernorm"])
        self.post_attention_layernorm.load_state_dict(layer_state["post_attention_layernorm"])
        if self.mlp_type == "sparse":
            self.mlp.load_gate(layer_state["gate"])
            self.mlp.load_shared_experts(layer_state["shared_experts"])
        else:
            self.mlp.load_state_dict(layer_state["mlp"])

    def prefetch_experts(self, expert_ids: list[int]) -> None:
        if self.mlp_type == "sparse" and hasattr(self.mlp, "prefetch_experts"):
            self.mlp.prefetch_experts(expert_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        return_router_inputs: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=None,
            cache_position=None,
        )
        if isinstance(attn_out, tuple):
            if len(attn_out) == 3:
                hidden_states, _, _ = attn_out
            else:
                hidden_states, _ = attn_out
        else:
            hidden_states = attn_out
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.mlp_type == "sparse":
            hidden_states, router_logits, router_inputs = self.mlp(hidden_states, return_router_inputs=return_router_inputs)
        else:
            router_logits = None
            router_inputs = None
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, router_logits, router_inputs


@dataclass
class DiskBackboneOutput:
    hidden_states: torch.Tensor
    router_logits: Optional[torch.Tensor] = None
    router_inputs: Optional[torch.Tensor] = None


class DiskBackedGlm4MoeLiteBackbone(nn.Module):
    def __init__(
        self,
        backbone_dir: Path,
        device: str = "cpu",
        cache_size: int = 2,
        attn_impl: str = "eager",
        router_top_k: Optional[int] = None,
    ):
        super().__init__()
        self.backbone_dir = Path(backbone_dir)
        self.device = device
        self.config = Glm4MoeLiteConfig.from_pretrained(self.backbone_dir)
        self.config._attn_implementation = attn_impl
        self.router_top_k = router_top_k
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.rotary_emb = Glm4MoeLiteRotaryEmbedding(self.config)
        self.norm = Glm4MoeLiteRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        emb_state = _load_pt(self.backbone_dir / "embeddings.pt", device="cpu")
        self.embed_tokens.load_state_dict(emb_state)
        norm_state = _load_pt(self.backbone_dir / "final_norm.pt", device="cpu")
        self.norm.load_state_dict(norm_state)

        layers_dir = self.backbone_dir / "layers"
        self.layers = nn.ModuleList()
        for i in range(self.config.num_hidden_layers):
            expert_dir = self.backbone_dir / f"layer_{i}_experts"
            mlp_type = "sparse"
            if hasattr(self.config, "mlp_layer_types"):
                mlp_type = self.config.mlp_layer_types[i]
            else:
                layer_state = _load_pt(layers_dir / f"layer_{i}.pt", device="cpu")
                mlp_type = layer_state.get("mlp_type", "sparse")
            layer = DiskBackedGlm4MoeLiteDecoderLayer(
                config=self.config,
                layer_idx=i,
                expert_dir=expert_dir,
                device=self.device,
                cache_size=cache_size,
                mlp_type=mlp_type,
                router_top_k=router_top_k,
            )
            layer_state = _load_pt(layers_dir / f"layer_{i}.pt", device="cpu")
            layer.load_weights(layer_state)
            self.layers.append(layer)

        self.embed_tokens.to(self.device)
        self.norm.to(self.device)
        self.layers.to(self.device)
        self.eval()

    def set_router_top_k(self, top_k: int) -> None:
        for layer in self.layers:
            if layer.mlp_type == "sparse" and hasattr(layer.mlp, "set_top_k"):
                layer.mlp.set_top_k(top_k)
        self.router_top_k = top_k

    def prefetch_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return
        layer = self.layers[layer_idx]
        if hasattr(layer, "prefetch_experts"):
            layer.prefetch_experts(expert_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        return_router_layer: Optional[int] = None,
        stop_at_layer: bool = False,
    ) -> DiskBackboneOutput:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        hidden_states = self.embed_tokens(input_ids)
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        router_logits = None
        router_inputs = None
        for i, layer in enumerate(self.layers):
            wants_router = return_router_layer is not None and i == return_router_layer
            hidden_states, layer_router_logits, layer_router_inputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                return_router_inputs=wants_router,
            )
            if wants_router:
                router_logits = layer_router_logits
                router_inputs = layer_router_inputs
                if stop_at_layer:
                    return DiskBackboneOutput(
                        hidden_states=hidden_states,
                        router_logits=router_logits,
                        router_inputs=router_inputs,
                    )

        hidden_states = self.norm(hidden_states)
        return DiskBackboneOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            router_inputs=router_inputs,
        )


class DiskBackedGlm4MoeLiteModel(nn.Module):
    def __init__(
        self,
        backbone_dir: Path,
        device: str = "cpu",
        cache_size: int = 2,
        attn_impl: str = "eager",
        router_top_k: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = DiskBackedGlm4MoeLiteBackbone(
            backbone_dir,
            device=device,
            cache_size=cache_size,
            attn_impl=attn_impl,
            router_top_k=router_top_k,
        )
        self.config = self.backbone.config
        lm_head_state = _load_pt(Path(backbone_dir) / "lm_head.pt", device="cpu")
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.load_state_dict(lm_head_state)
        self.lm_head.to(device)
        self.device = device
        self.eval()

    def set_router_top_k(self, top_k: int) -> None:
        self.backbone.set_router_top_k(top_k)

    @torch.no_grad()
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        out = self.backbone(input_ids)
        return self.lm_head(out.hidden_states)
