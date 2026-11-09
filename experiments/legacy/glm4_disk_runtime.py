#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer

from glm4_moe_lite_disk_backbone import DiskBackedGlm4MoeLiteModel
from disk_moe_runtime import WarmupOracle


class Glm4DiskRuntime:
    def __init__(
        self,
        backbone_dir: Path,
        tokenizer_name: str,
        device: str = "cpu",
        cache_size: int = 2,
        router_top_k: Optional[int] = None,
        oracle_weights: Optional[Path] = None,
        oracle_hidden: int = 512,
        oracle_top_k: Optional[int] = None,
        oracle_layer: Optional[int] = None,
        oracle_interval: int = 1,
        max_new_tokens: int = 32,
        attn_impl: str = "eager",
    ) -> None:
        self.backbone_dir = Path(backbone_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = device
        self.cache_size = cache_size
        self.seq_len = max_new_tokens
        self.router_top_k = router_top_k
        self.model = DiskBackedGlm4MoeLiteModel(
            self.backbone_dir,
            device=device,
            cache_size=cache_size,
            attn_impl=attn_impl,
            router_top_k=router_top_k,
        )
        self.config = self.model.config
        self.model.eval()
        self.oracle = None
        self.oracle_top_k = oracle_top_k
        self.oracle_interval = max(1, int(oracle_interval))
        self.oracle_step = 0
        self.oracle_layer = oracle_layer if oracle_layer is not None else self._default_oracle_layer()
        if oracle_weights:
            oracle_weights = Path(oracle_weights)
            d_model = self.config.hidden_size
            n_experts = getattr(self.config, "n_routed_experts", None) or getattr(self.config, "num_experts")
            self.oracle = WarmupOracle(d_model, oracle_hidden, n_experts)
            state = torch.load(oracle_weights, map_location="cpu")
            self.oracle.load_state_dict(state)
            self.oracle.to(self.device).eval()

    def _default_oracle_layer(self) -> int:
        if hasattr(self.config, "mlp_layer_types"):
            sparse_layers = [i for i, t in enumerate(self.config.mlp_layer_types) if t == "sparse"]
            if sparse_layers:
                return sparse_layers[-1]
        return self.config.num_hidden_layers - 1

    def set_router_top_k(self, top_k: int) -> None:
        self.model.set_router_top_k(top_k)
        self.router_top_k = top_k

    def set_oracle_top_k(self, top_k: int) -> None:
        self.oracle_top_k = top_k

    def cache_stats(self) -> dict:
        return {
            "type": "glm4",
            "cache_size": self.cache_size,
            "router_top_k": self.router_top_k,
            "oracle_top_k": self.oracle_top_k,
            "oracle_enabled": self.oracle is not None,
        }

    def _generate_for_prompt(self, prompt: str) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        eos_id = self.tokenizer.eos_token_id
        output_text = prompt
        with torch.inference_mode():
            for _ in range(self.seq_len):
                if self.oracle is not None:
                    out = self.model.backbone(ids, return_router_layer=self.oracle_layer)
                    logits = self.model.lm_head(out.hidden_states)
                    if self.oracle_top_k and (self.oracle_step % self.oracle_interval == 0):
                        router_inputs = out.router_inputs
                        if router_inputs is not None:
                            chunk_emb = router_inputs.mean(dim=1)
                            scores = self.oracle(chunk_emb.float())
                            k = min(self.oracle_top_k, scores.shape[-1])
                            topk = torch.topk(scores, k=k, dim=-1).indices
                            expert_ids = torch.unique(topk).tolist()
                            self.model.backbone.prefetch_experts(self.oracle_layer, expert_ids)
                    self.oracle_step += 1
                else:
                    logits = self.model(ids)
                next_id = torch.argmax(logits[:, -1, :], dim=-1)
                next_token_id = int(next_id.item())
                if eos_id is not None and next_token_id == eos_id:
                    break
                token = self.tokenizer.decode([next_token_id])
                output_text += token
                ids = torch.cat([ids, next_id[:, None]], dim=-1)
        return output_text

    async def run(self, prompts: List[str]) -> List[str]:
        outputs = []
        for prompt in prompts:
            outputs.append(self._generate_for_prompt(prompt))
        return outputs

    async def stream_generate(self, prompts: List[str]):
        if len(prompts) != 1:
            raise ValueError("GLM4 streaming only supports a single prompt per request")
        prompt = prompts[0]
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        eos_id = self.tokenizer.eos_token_id
        with torch.inference_mode():
            for step in range(self.seq_len):
                if self.oracle is not None:
                    out = self.model.backbone(ids, return_router_layer=self.oracle_layer)
                    logits = self.model.lm_head(out.hidden_states)
                    if self.oracle_top_k and (self.oracle_step % self.oracle_interval == 0):
                        router_inputs = out.router_inputs
                        if router_inputs is not None:
                            chunk_emb = router_inputs.mean(dim=1)
                            scores = self.oracle(chunk_emb.float())
                            k = min(self.oracle_top_k, scores.shape[-1])
                            topk = torch.topk(scores, k=k, dim=-1).indices
                            expert_ids = torch.unique(topk).tolist()
                            self.model.backbone.prefetch_experts(self.oracle_layer, expert_ids)
                    self.oracle_step += 1
                else:
                    logits = self.model(ids)
                next_id = torch.argmax(logits[:, -1, :], dim=-1)
                next_token_id = int(next_id.item())
                if eos_id is not None and next_token_id == eos_id:
                    break
                token = self.tokenizer.decode([next_token_id])
                ids = torch.cat([ids, next_id[:, None]], dim=-1)
                yield step, [token]
