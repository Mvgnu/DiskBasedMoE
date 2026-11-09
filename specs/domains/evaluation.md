---
name: evaluation
description: Falsifiable capacity, traffic and throughput measurements
paths:
  owned: [src/disk_moe/evaluation/]
  tests: [tests/evaluation/]
exports: [capacity planner, route trace replay, file-read benchmark, model benchmark]
consumes: [disk_moe.contracts.InferenceEngine, disk_moe.contracts.GenerationEvent]
verification:
  check: "python -m ruff check src/disk_moe/evaluation tests/evaluation"
  test: "python -m pytest tests/evaluation -q"
  build: "python -m build --no-isolation"
bar:
  reference: "fio cache-state reporting and the roofline transfer lower bound"
  property: "An operator can distinguish model capacity, unavoidable miss traffic, measured file IO and actual single-stream decode without a misleading speed claim."
---

# Evaluation

The planner is pure Python. Inputs specify total and active routed parameters,
effective bits including quantization overhead, resident weight/KV/staging/OS
reserve, available memory, measured read bandwidth, byte-weighted cache hit rate,
and non-IO token time. Report decimal GB and GiB explicitly. Explain serialized
and perfect-overlap bounds, impossible memory fits and required cache hit rate
for a target. A cache hit rate is an input/trace result, never inferred from cache
capacity. Validate finite physical values. No hardware recommendations by slogan.

Replay explicit layer/expert access traces against an LRU byte budget; disclose
trace assumptions and exact read totals. The file benchmark reads only existing
user-selected files, uses bounded buffers, labels buffered/page-cache behavior,
records seed/access pattern/passes/environment, and never silently drops system
caches or writes/deletes user data. It is not an inference benchmark.

The optional real-model benchmark separately reports setup, prompt prefill/TTFT,
decode intervals, token count, native routing and cache state; no download or
synthetic substitution on failure. CPU tiny model runs prove harness behavior,
not 1T speed. The 1T acceptance protocol lives in PM-owned documentation.
