# DiskBasedMoE: system contract

## Goal and current milestone

Determine whether exact-routing sparse MoE inference can achieve useful single-
stream decode with most weights on local storage. The first milestone repairs
the original GLM4 experiment into a trustworthy small reference implementation
and supplies tools/specs for a measured 1T feasibility decision. **It does not
claim a working 1T backend, quantized kernels, CUDA overlap or measured 1 t/s.**

Native GLM4 MoE Lite is the executable correctness reference. Keep the installed
Transformers implementation's attention, router, shared experts, norms and KV
cache; replace only routed expert delivery. No speed result may use reduced
top-k, skipped experts, random fallbacks or changed checkpoint mathematics.

## Domains

| Domain | Owns | Consumes |
|---|---|---|
| runtime | `src/disk_moe/runtime/`, `tests/runtime/` | shared generation protocol, pinned native Transformers |
| service | `src/disk_moe/service/`, `tests/service/` | shared generation protocol; runtime at composition boundary only |
| evaluation | `src/disk_moe/evaluation/`, `tests/evaluation/` | shared generation protocol; runtime only for optional model benchmark |

The PM owns packaging, root compatibility launchers, architecture checks, docs,
examples and history migration. See `lop.md` for the full gate. Python packaging
uses `src/`; the capacity planner and file benchmark require no ML dependencies.

## Invariants

1. Each request owns fresh KV state and immutable output limits. The service
   serializes runtime access until a worker has really stopped, even on cancel.
2. An export becomes usable only after all weights are written and validated.
   Fail closed on missing, malformed or incompatible artifacts. Use safetensors
   for new exports, isolated contiguous expert tensors and explicit dtype.
3. Cache capacity is one global byte budget. Distinguish retained expert bytes,
   active loading/compute tensors, resident backbone, KV, staging and OS cache.
   No cache policy may evict weights still needed by an in-flight computation.
4. Preserve checkpoint top-k. Learned prefetch is deferred until native route
   traces and miss/waste accounting can test its incremental benefit.
5. Evidence has a type: executed check, measured benchmark, analytical estimate,
   published result, or proposed work. These are never interchangeable.
6. The archived original implementation is not importable by supported code.

## Road to a useful 1T experiment

M0: native parity, bounded global cache, fresh KV per request, honest API and
reproducible evaluation. M1: source-pinned quantized 1T engine integration and
expert-contiguous storage, measured SSD/transfer/compute traces. M2: target-
hardware cold and warm runs, prompt/decode separation, quality parity to the same
quantized checkpoint, >=1 token/s single-stream decode over a declared workload.
Only then consider overlapped prefetch, admission policies or more model types.

Existing SGLang/llama.cpp-family offload work is a benchmark/integration candidate,
not a mandate to reproduce their kernels in Python. Disk capacity alone is never
the acceptance criterion. A consumer GPU result does not establish Mac support.
