# Disk-Backed MoE Chat App

A local web app + API for running large MoE models on limited hardware by keeping only a small set of experts in memory.
Supports:
- OLMoE sharded experts (legacy path)
- GLM4 MoE Lite disk exports (recommended)
- Optional oracle prefetch to warm likely experts
- Web UI at `/` + JSON API at `/api/chat` + SSE streaming at `/api/chat/stream`

## Quick Start (GLM4 Disk-Backed)

1) Export GLM4 MoE Lite weights to a disk-backed layout (run where the full model is available, works with limited RAM below model total size):

```bash
python moe_app/export_glm4_moe_lite_backbone.py \
  --model_name_or_path zai-org/GLM-4.7-Flash \
  --output_dir glm4_flash_disk \
  --dtype bfloat16 \
  --streaming
```

2) Generate native routing logs (slow but RAM-light):

```bash
python moe_app/generate_native_routing_log_glm4.py \
  --backbone_dir glm4_flash_disk \
  --tokenizer zai-org/GLM-4.7-Flash \
  --device mps \
  --router_top_k 4 \
  --cache_size 2 \
  --memmap \
  --log_path oracle_log_native_glm4
```

3) Train a tiny oracle from the log:

```bash
python - <<'PY'
import json, torch
from pathlib import Path

from moe_app.disk_moe_runtime import train_oracle

cfg = json.load(open("glm4_flash_disk/config.json"))
d_model = cfg["hidden_size"]
n_experts = cfg.get("n_routed_experts", cfg.get("num_experts"))

model = train_oracle(
    log_path=Path("oracle_log_native_glm4.npz"),
    d_in=d_model,
    d_hidden=512,
    n_experts=n_experts,
    device="mps",
    memmap=True,
    epochs=3,
    batch_size=256,
    lr=3e-4,
)

torch.save(model.state_dict(), "oracle_glm4.pt")
print("saved oracle_glm4.pt")
PY
```

4) Run the web app with oracle prefetch:

```bash
python moe_app/server.py \
  --backend glm4 \
  --model_dir glm4_flash_disk \
  --tokenizer zai-org/GLM-4.7-Flash \
  --device mps \
  --router_top_k 4 \
  --oracle_weights oracle_glm4.pt \
  --oracle_top_k 4 \
  --cache_size 2 \
  --seq_len 32
```

Open `http://127.0.0.1:8000`.

## Routing & Prefetch Notes

- **router_top_k** should match `num_experts_per_tok` in `glm4_flash_disk/config.json`.
- `seq_len` controls how many new tokens are generated per request.
- The GLM4 disk runtime recomputes the prefix each step (no KV cache), so long generations are slow.
- The oracle prefetch runs every token by default; adjust with `--oracle_interval` if needed.

## Always-On Weights (GLM4 Routing)

For correct routing you must keep these weights resident:
- Embeddings + final norm
- Every layer's attention + norms
- Each sparse layer's router gate + shared experts (dense layers keep their MLP)
Experts are loaded on-demand from `layer_*_experts/`.

Inspect the always-on footprint for an exported backbone:

```bash
python moe_app/inspect_glm4_always_on.py --backbone_dir glm4_flash_disk
```

## API Usage

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain why routing top-k matters in MoE.",
    "max_new_tokens": 64,
    "router_top_k": 4,
    "oracle_top_k": 4
  }'
```

Streaming (Server-Sent Events):

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello from the streaming endpoint.",
    "max_new_tokens": 32
  }'
```

## Notes

- The UI hides the oracle control unless oracle weights are loaded.
- If you have a local tokenizer cache and want offline use, set `HF_HUB_OFFLINE=1`.
- For higher `router_top_k`, increase `--cache_size` to keep more experts warm.

## Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install --pre -r moe_app/deps/requirements-moe.txt
```

> Runs a 27B parameter MoE model on a 16GB MacBook by disk-sharding experts and prefetching with a learned oracle. Because why not.

Inspired by a Tweet, forgot who. Anyways.
