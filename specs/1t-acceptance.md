# 1T acceptance contract — future milestone, not yet passed

## User outcome

One user can generate useful text with a declared approximately 1T sparse MoE
while most routed weights live on local storage. Minimum decode target:
**at least 1.0 generated token/s in a single stream**, preserving native routing
and the same quantized tensors used by the comparison reference. TTFT and stalls
remain separately visible.

## Integration boundary

The reference GLM4 adapter is not a 1T engine. A backend needs a separate scoped
spec and PM contract change if necessary. Prefer an established native engine
with expert-contiguous delivery and quantized kernels. Pin source, checkpoint,
tokenizer/config revisions and artifact hashes. Verify license/build requirements
before importing implementation code. Do not vendor unreviewed kernels or
implicitly run downloaded repository code.

Minimum adapter functions: exact provenance; request validation; native message
formatting; explicit finish/error/cancel; separately timed prefill/decode and
cache/IO counters. Unknown external-server cache/physical-IO state stays unknown.

## Reproducible run record

Every run preserves:

* Exact command, source revisions, Python/compiler/driver/library versions.
* CPU/GPU, RAM/VRAM, storage model/interface/filesystem and free space.
* Weight/tokenizer/config hashes; formats per tensor group; native top-k; resident,
  cache, staging and KV placements with byte budgets.
* Prompt identity/token count, output count, sampling/seed, batch, context, EOS,
  early termination and failures. Count tokens rather than characters.
* Load/packing time separately from validation, prefill, TTFT and decode.
* All inter-token intervals, total decode time and p50/p95/max. Decode counts the
  subsequent N−1 tokens, excluding the first token produced by prefill.
* Peak host RSS/staging and allocated/reserved GPU memory, including how measured.
* Logical requested/read/prefetched bytes; physical disk/H2D bytes when available;
  hits/misses/evictions and prefetch waste. Distinguish expert cache from OS cache.

## Workload matrix

At least five fixed versioned prompts per relevant category: short chat, unrelated
domains that shift routing, and a realistic long coding prompt. Target 256 output
tokens; retain early EOS cases and obtain at least 128 post-prefill tokens for the
sustained decode sample. One request at a time. Do not discard slow runs. Report
each request as well as aggregates.

Run with empty application expert cache and after warm-up. Record OS cache control
independently: an empty Python LRU is not a cold SSD run. Privileged cache dropping
requires explicit operator consent and a dedicated system; default tools never
perform it.

## Correctness and recovery

Compare to the same quantized reference at a documented tolerance under forced
misses, eviction and warm hits. Verify no expert is dropped or substituted. Exercise
missing/truncated shards, failed reads, cancellation during a miss, exhausted
budgets and a second independent request. Fail explicitly instead of replacing
weights with zeros or changing routing.

## Decision

Pass only when fidelity and sustained decode target hold on declared workload and
hardware. Publish TTFT and tail latency alongside that narrow pass. A broader
interactive-usability claim needs an operator-approved prompt-latency budget;
do not infer one from the decode target. Unmeasured hardware stays unverified.

A fresh critic executes the artifact against the chosen native engine and
[llama-bench's phase-reporting bar](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md).
The single largest gap drives the next cycle. Without weights or target hardware,
record the remaining hardware gate rather than calling this milestone passed.
