#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import wikitext_dataset
from disk_moe_runtime import RoutingLogger, RoutingLoggerMemmap
from glm4_moe_lite_disk_backbone import DiskBackedGlm4MoeLiteBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate native routing logs with GLM4 MoE Lite disk backbone.")
    parser.add_argument("--backbone_dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--cache_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--router_top_k", type=int, default=4)
    parser.add_argument("--target_layer", type=int, default=None)
    parser.add_argument("--memmap", action="store_true")
    parser.add_argument("--log_path", type=str, default="oracle_log_native_glm4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backbone_dir = Path(args.backbone_dir)
    backbone = DiskBackedGlm4MoeLiteBackbone(
        backbone_dir=backbone_dir,
        device=args.device,
        cache_size=args.cache_size,
        router_top_k=args.router_top_k,
    )

    target_layer = args.target_layer
    if target_layer is None:
        if hasattr(backbone.config, "mlp_layer_types"):
            sparse_layers = [i for i, t in enumerate(backbone.config.mlp_layer_types) if t == "sparse"]
            target_layer = sparse_layers[-1] if sparse_layers else backbone.config.num_hidden_layers - 1
        else:
            target_layer = backbone.config.num_hidden_layers - 1

    log_path = Path(args.log_path)
    if args.memmap:
        logger = RoutingLoggerMemmap(
            d_model=backbone.config.hidden_size,
            n_experts=backbone.config.n_routed_experts,
            max_samples=args.max_samples,
            log_path=str(log_path),
        )
    else:
        logger = RoutingLogger(
            d_model=backbone.config.hidden_size,
            n_experts=backbone.config.n_routed_experts,
            memmap=False,
        )

    loader = wikitext_dataset.WikiText103DataLoader(
        split="train",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        tokenizer_name=args.tokenizer,
        max_samples=args.max_samples,
        buffer_size=1000,
    )

    total = 0
    for tokens, _ in loader:
        tokens = tokens.to(args.device)
        out = backbone(tokens, return_router_layer=target_layer, stop_at_layer=True)
        router_logits = out.router_logits
        router_inputs = out.router_inputs
        if router_logits is None or router_inputs is None:
            raise RuntimeError("Router outputs missing. Check target_layer.")

        router_logits = router_logits.view(tokens.size(0), tokens.size(1), -1)
        topk = torch.topk(router_logits, args.router_top_k, dim=-1).indices
        for b in range(tokens.size(0)):
            expert_ids = torch.unique(topk[b].flatten()).tolist()
            chunk_emb = router_inputs[b].mean(dim=0).detach().cpu()
            logger.log(chunk_emb, expert_ids)
            total += 1
        if total % 1000 == 0:
            print(f"[LOG] {total} samples")
        if total >= args.max_samples:
            break

    if args.memmap:
        logger.flush()
        print(f"[DONE] Wrote {total} samples to {log_path}.npz")
    else:
        logger.flush(log_path)
        print(f"[DONE] Wrote {total} samples to {log_path}")


if __name__ == "__main__":
    main()
