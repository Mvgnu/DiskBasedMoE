# Is a 1T model at one token per second plausible?

**For a sufficiently sparse MoE, yes. For a dense model streamed wholesale from
one consumer SSD, the transfer budget is radically different.** Capacity and
throughput are separate questions. This repository does not yet run a 1T model;
its reference runtime and tools make the next experiment falsifiable.

## Published evidence, not a local result

[SSD-LLaMA](https://arxiv.org/html/2609.18110v1), submitted 16 September 2026,
reports 1.03 tokens/s for Kimi-K2.7-Code Q4_0 with an RTX 5090 and 32 GB host RAM.
Its NVMe testbed has a stated 9 GiB/s ceiling; the quantized checkpoint is 543.6
GiB. This is batch-one decode. Prompt processing is a separate constraint. We
inspected the paper; we did not reproduce that result or establish Mac support.

[SGLang's SSD Expert Pack article](https://www.lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/)
describes expert-contiguous blocks, aligned direct reads into pinned staging and
a byte-budgeted GPU cache. Its GPU-centric path distinguishes asynchronous
delivery from hiding all miss latency behind compute. The cited source commit
[`81c9f837f19ff8dfe1a9fcd1abfc6069dd28d2ec`](https://github.com/sgl-project/sglang/commit/81c9f837f19ff8dfe1a9fcd1abfc6069dd28d2ec)
was verified to exist through GitHub. We did not build or benchmark that engine.

These sources justify a hardware experiment. Their throughput is not a result
for this Python GLM4 reference adapter.

## Work out bytes before buying hardware

The supplied profile uses the **original Kimi K2**, not the paper's K2.7 variant.
The [official configuration](https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json)
and [implementation](https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/modeling_deepseek.py)
give 60 routed layers after one dense layer, 384 experts per routed layer, 8
selected per token, hidden width 7168 and expert intermediate width 2048.
Our calculation for three-matrix SwiGLU experts is:

```
parameters/expert = 3 × 7168 × 2048 = 44,040,192
selected routed parameters/token = 60 × 8 × 44,040,192 = 21,139,292,160
```

[GGML Q4_0](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
uses 18 bytes for each 32 weights including a scale (4.5 effective bits/weight).
Applying that hypothetical uniform format gives 24,772,608 bytes/expert and
**11,890,851,840 routed bytes/token before cache reuse**. Shared/dense weights and
KV are assumed resident. This is not total active-parameter traffic.

### Capacity ledger

Do not round the model to exactly one trillion when evaluating a storage budget.
The following parameter counts are derived from that native geometry:

| Component | Parameters |
|---|---:|
| Routed experts | 1,014,686,023,680 |
| Shared experts | 2,642,411,520 |
| Router weights | 165,150,720 |
| Attention projections and attention-internal norms | 6,168,569,856 |
| Initial dense MLP | 396,361,728 |
| Layer/final norms | 881,664 |
| Embeddings and untied LM head | 2,348,810,240 |
| **Total** | **1,026,408,209,408** |

The derived count was independently checked by constructing the matching native
DeepSeekV3 geometry on the meta device (no parameter storage). There are also
23,040 routing-correction buffer elements, budgeted as 92,160 FP32 bytes.
The hypothetical uniform Q4_0 payload plus those buffers is **577.355 decimal GB
/ 537.703 GiB**, excluding headers, padding and preparation copies. A real mixed-
precision artifact can differ; this is a planning profile, not a file-size claim.

For a generic model of exactly 1T parameters, bare four-bit storage is 500 GB and
uniform Q4_0 is 562.5 GB. That generic calculation must not replace the named
model's full geometry. The initial Gauntlet round caught precisely that mismatch.

### An explicit throughput scenario

Let `h` be the byte-weighted memory-tier hit rate and `B` delivered read bandwidth
in decimal GB/s. For a serialized schedule:

```
time/token = 11.89085184 × (1 − h) / B + other_seconds_per_token
```

At hypothetical 7 GB/s and 0.25 s of uncovered compute/transfer/synchronization:

| Hit rate | Modeled seconds/token | Modeled tokens/s |
|---:|---:|---:|
| 0% | 1.949 | 0.513 |
| 25% | 1.524 | 0.656 |
| 50% | 1.099 | 0.910 |
| 75% | 0.675 | 1.482 |

The target needs about 55.85% byte reuse under those assumptions. These are not
measured rates. The planner emits this arithmetic, reports perfect-overlap bounds
separately, rejects impossible memory budgets and labels unknown placement.

**Capacity does not determine cache hit rate.** A global LRU can thrash across
layers, and unrelated prompts can replace a hot working set. Replay actual native
routes before choosing `h`; the included trace illustrates only the arithmetic.

A dense exactly-1T model with almost no resident weights would require roughly
562.5 GB/token under uniform Q4_0, or 80.4 seconds of reads at 7 GB/s before compute.
Sparse activation, useful reuse and independently addressable experts change
that problem.

## What the execution path needs

Keep native routing and always-needed weights resident. Make a complete routed
expert independently addressable. Quantized payloads should reach kernels that
consume them without allocating a full-precision copy of the whole model. Bound
retained memory and staging separately; protect in-flight allocations. Keep
native KV caching so each decode step processes one token rather than the prefix.

Measure actual miss traffic. Loader calls do not establish physical SSD reads:
OS page cache may serve them, and tensor views can share pages. Direct IO requires
alignment and suitable buffers, not just a flag on a pickle loader. Failed
prefetch consumes bandwidth and must be counted even when later demand hits.

A mean decode rate above one token/s can coexist with multi-second stalls or
long prompt ingestion. Report median, p95 and maximum inter-token intervals,
TTFT, cold/warm state and output fidelity. Long coding prompts affect memory and
prompt-processing time differently from short repeated questions.

## Recommended next milestone

Use this repository's contracts and measurements with a source-pinned native
quantized engine. Evaluate an existing expert-pack implementation before writing
a competing GPU runtime. Integration must be optional, preserve artifact identity,
retain raw logs, and reject unsupported formats. An external server connection
does not prove its weights stayed on disk.

First measure the target machine's IO and each memory arena. Then compare a
smaller supported checkpoint against its reference under forced misses. Attempt
the 1T artifact after preparation storage, resident weights, KV and staging fit.
Follow [`specs/1t-acceptance.md`](../specs/1t-acceptance.md).

CPU-only, discrete CUDA and Apple unified-memory systems have different kernels
and transfer constraints. The old README's 16 GB MacBook is historical experiment
context, not a proven configuration for the 1T goal.
