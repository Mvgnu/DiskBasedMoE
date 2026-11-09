# DiskBasedMoE

A small, inspectable experiment in **faithful disk-backed MoE inference**. The
supported reference backend keeps native GLM4 MoE Lite attention, routing and KV
caching, while loading routed experts from a global byte-bounded cache backed by
local safetensors files. Capacity/traffic planning and measurement tools help
decide whether a larger experiment is feasible.

**This repository has not demonstrated a trillion-parameter model at 1 token/s.**
The reference adapter is GLM4 MoE Lite, with unquantized FP32/FP16/BF16 tensors.
It is a correctness and measurement baseline, not a fast quantized GPU kernel.
See [the 1T feasibility analysis](docs/1t-feasibility.md) and
[the target-hardware acceptance protocol](specs/1t-acceptance.md).

## Start with the smallest reproducible experiment

Python 3.11+; the recorded CPU reference environment uses Python 3.12. In a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[runtime,serve,dev]'
```

The explicit first install selects CPU PyTorch for the offline test fixture. For
CUDA or Apple MPS, install a suitable wheel using the
[official instructions](https://pytorch.org/get-started/locally/) before the
extras. The pinned Transformers release matters: expert-bank replacement uses
an internal native-model interface tested at that exact version.

No real model download is needed for the reproducibility drill:

```bash
python -m disk_moe.runtime tiny-demo --output artifacts/tiny-glm4
disk-moe-benchmark --model-dir artifacts/tiny-glm4 --prompt 'Hello world' \
  --max-new-tokens 16 --cache-mib 1 --include-text
disk-moe-serve --model-dir artifacts/tiny-glm4 --cache-mib 1 --max-new-tokens 16
```

Open `http://127.0.0.1:8000`. The fixture has random native-model weights and a
small local tokenizer: it tests export, native parity, generation and serving.
Its output is not a useful language model, and its speed does not predict 1T
performance. Choose a new output directory for another export.

## Use a real GLM4 MoE Lite checkpoint

Export a local compatible Hugging Face safetensors checkpoint, or explicitly
name an indexed Hugging Face repository. Remote unsharded sources must be
downloaded locally first. A remote source may download the full checkpoint;
allow disk space for both source and exported weights. Streaming export avoids
loading all routed experts into RAM simultaneously; resident non-expert weights
still need to fit at inference time.

```bash
disk-moe-export /path/to/glm4-moe-lite-checkpoint artifacts/my-glm4 --dtype bfloat16
disk-moe-serve --model-dir artifacts/my-glm4 --tokenizer /path/to/tokenizer \
  --device cpu --cache-mib 512
```

Use `disk-moe-export --help` and `disk-moe-serve --help` for the exact interface.
Unsupported model families and quantized sources fail explicitly. This adapter
does not load Kimi, DeepSeek, GGUF or arbitrary MoE checkpoints, substitute
weights, or reduce the checkpoint's expert top-k.

Tensor names are checked before payload transfer. The declared GLM-4.7-Flash
auxiliary prediction layer that the pinned native decoder does not use is
explicitly omitted and listed in the export manifest; arbitrary extra tensors
are rejected. Its public index was checked, but a full-size export/inference run
has not been performed. Static rotary state is preserved, including low-precision
rounding; dynamic/LongRoPE or incompatible mutated state is unsupported. Older
artifacts without the rotary snapshot require re-export.

`--cache-mib` limits retained routed-expert tensors **globally**, not per layer
and not total process memory. Resident attention/shared weights, KV, temporary
loads, activations, allocator overhead and OS cache are additional allocations.
A zero or undersized budget uses a transient expert without retaining it.

## Chat API

```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello world","max_new_tokens":16}'
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello world","max_new_tokens":16}'
```

Requests accept either a prompt or role/content messages. Messages use the
tokenizer's own chat template; a missing template is rejected. JSON returns
generated text, finish reason and metrics. SSE returns text deltas/token IDs,
finish reason/metrics and `[DONE]`. Both exclude the prompt. Unknown fields are
rejected, including old router/oracle controls. Native routing is immutable.

The server serializes model access, runs inference off the event loop and bounds
queued delivery. Disconnect cancellation is cooperative at token boundaries: a
running model operation must finish before the next request takes ownership.
The UI uses no remote fonts or JavaScript. Default binding is localhost; this is
not an authenticated public service. Health's engine description is a startup
snapshot; completion metrics describe the request that just ran.

## Plan capacity and traffic without installing ML

The base package needs no third-party dependencies. In another environment,
`python -m pip install -e .` installs just the planning/IO tools.

```bash
disk-moe-plan --scenario examples/kimi-k2-q4-scenario.json
disk-moe-plan --scenario examples/kimi-k2-q4-scenario.json \
  --read-gbps 7 --cache-hit-rate 0.75 --other-seconds 0.25
disk-moe-plan --trace examples/route-trace.jsonl --cache-bytes 74317824
```

The geometry example is an **analytical scenario**, not a supported model or
benchmark. It distinguishes decimal GB from GiB and effective bits including
scale overhead. It reports serialized and perfect-overlap bounds plus the hit
rate needed for a target. A memory budget covers one real memory arena and cannot
pool host RAM with discrete GPU memory. Without a supplied memory budget,
placement feasibility remains unevaluated.

The included trace is tiny and synthetic. Real traces use JSONL `token`, `layer`,
`expert`, `bytes` fields in execution order. Replay uses a cold global LRU and
byte-weighted hit rate. It does not predict routes or model OS cache/prefetch.

## Measure rather than assume

Read an existing representative file without modifying it:

```bash
disk-moe-io-bench /path/to/existing/large/model-file \
  --block-mib 24 --max-mib 4096 --passes 2 --pattern random --seed 0
```

This measures **buffered file reads**, possibly served by OS page cache. It does
not use `O_DIRECT`, drop caches, create files, or measure physical SSD bytes. A
small, RAM-fitting or sparse file cannot establish the SSD bandwidth needed by a
1T model. Use a dedicated tool such as
[fio](https://fio.readthedocs.io/en/latest/fio_doc.html) for controlled IO tests.

`disk-moe-benchmark` reports setup time, first-token latency, subsequent token
intervals, tail latency and engine counters separately. It is single-stream;
decode throughput uses intervals after the first token. Output labels unmeasured
peak memory/physical storage bytes and persistent cache state. Preserve raw JSON
from the actual target hardware before making a performance claim.

## Development: Lop + Gauntlet

Read [AGENTS.md](AGENTS.md), [Lop](lop.md), [Gauntlet](gauntlet.md), and the
[system spec](specs/00_system.md). Domains have disjoint write ownership:

| Domain | Code | Acceptance bar |
|---|---|---|
| [Runtime](specs/domains/runtime.md) | `src/disk_moe/runtime/` | Native Transformers logits and cached decode |
| [Service](specs/domains/service.md) | `src/disk_moe/service/` | Native chat formatting, responsive stream/cancel/next-request |
| [Evaluation](specs/domains/evaluation.md) | `src/disk_moe/evaluation/` | Transfer bounds and cache-state reporting |

```bash
python scripts/domain_check.py all
python -m pytest -q
python -m ruff check src scripts tests server.py export_glm4_moe_lite_backbone.py
python -m build --no-isolation
```

Tests compare a tiny native model against exported weights, including misses,
eviction and cached decode; service checks use real localhost HTTP with
deterministic engines for cancellation and responsiveness. Those fixtures prove
their specific invariants, not quality on real prompts. Independent reviews and
remaining limits are in `docs/reviews/`.

## What happened to the original code?

Original scripts/dependencies remain under
[`experiments/legacy/`](experiments/legacy/README.md), excluded from the installed
package. The default OLMoE path used random model components; the old GLM4 path
repeated prefixes and had export/dtype defects. See the
[reconnaissance record](docs/recon.md). Legacy `.pt` exports require re-exporting
the original checkpoint. Root `server.py` and the exporter are thin launchers;
old flags are not silently reinterpreted.
