#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def format_bytes(num: int) -> str:
    step = 1024.0
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num < step:
            return f"{num:.2f} {unit}"
        num /= step
    return f"{num:.2f} PiB"


def list_always_on_files(backbone_dir: Path) -> list[Path]:
    files = []
    for name in ["config.json", "embeddings.pt", "final_norm.pt", "lm_head.pt"]:
        path = backbone_dir / name
        if path.exists():
            files.append(path)
    layers_dir = backbone_dir / "layers"
    if layers_dir.exists():
        files.extend(sorted(layers_dir.glob("layer_*.pt")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect always-on weights required for GLM4 MoE Lite routing."
    )
    parser.add_argument("--backbone_dir", type=str, required=True)
    args = parser.parse_args()

    backbone_dir = Path(args.backbone_dir)
    if not backbone_dir.exists():
        raise SystemExit(f"Backbone dir not found: {backbone_dir}")

    files = list_always_on_files(backbone_dir)
    if not files:
        raise SystemExit("No backbone files found. Did you export the model?")

    total = 0
    print("Always-on files (routing + attention + shared experts):")
    for path in files:
        size = path.stat().st_size
        total += size
        rel = path.relative_to(backbone_dir)
        print(f"  {rel}  {format_bytes(size)}")

    print(f"Total always-on size (approx): {format_bytes(total)}")
    print("On-demand expert weights live under layer_*_experts/")


if __name__ == "__main__":
    main()
