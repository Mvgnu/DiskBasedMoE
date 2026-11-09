---
name: runtime
description: Faithful native GLM4 inference with disk-resident routed experts
paths:
  owned: [src/disk_moe/runtime/]
  tests: [tests/runtime/]
exports: [Glm4Engine, export_model, export_checkpoint, load_model]
consumes: [disk_moe.contracts.InferenceEngine, disk_moe.contracts.GenerationEvent, transformers.Glm4MoeLiteForCausalLM]
verification:
  check: "python -m ruff check src/disk_moe/runtime tests/runtime"
  test: "python -m pytest tests/runtime -q"
  build: "python -m build --no-isolation"
bar:
  reference: "Hugging Face Transformers 5.20.2 native GLM4 MoE Lite"
  property: "The same local weights produce matching prefill/cached logits and greedy tokens with a globally bounded expert cache and no changed routing."
---

# Runtime

Implement export/load, a global byte-bounded cache, native expert-bank replacement
and a blocking `Glm4Engine` matching the shared protocol. Internal modules may be
split for artifact, export, cache, model and generation; they form one ownership
domain. Preserve the native model instead of copying the decoder/router.

Export from an in-memory tiny native model for deterministic parity checks and
from local/HF safetensors incrementally for real artifacts. Preserve dtype, clone
views, reject unsupported checkpoint shapes and path escapes, and write completion
metadata last/atomically. Unsupported layouts must fail with actionable errors.
An export should not load an entire model merely to split its experts.

Global cache reports hits, misses, retained bytes, evictions and logical file
bytes read. Explicitly account for one transient expert outside retained cache
if zero/undersized budgets use a bypass. Reject negative budgets. Logs must not
pretend logical file reads prove physical SSD reads.

Generation: one prompt prefill, then one token per decode forward with native KV;
fresh KV every request; native EOS and tokenizer chat template; stream decodes
must preserve text across token boundaries. Return terminal metrics (token count,
prefill/decode timing with documented convention, cache counters). No oracle or
request-level router top-k controls in the supported baseline.

Test native parity in FP32 and BF16 as supported on CPU, cached multi-step decode,
export dtype/storage isolation, cache bound/zero budget, corrupt or partial
artifacts, terminal EOS/limits and new-request KV isolation. No external weights
required. Record GPU/MPS and large checkpoint measurements as not run unless run.

## Public entry points

`disk_moe.runtime` exports `Glm4Engine`, `export_model`, `export_checkpoint`, and
`load_model`. `Glm4Engine(artifact_dir, *, tokenizer=None, tokenizer_name=None,
device="cpu", cache_budget_bytes=268435456, attn_implementation="eager")` follows
the shared blocking protocol. A saved tokenizer or explicit tokenizer is needed.
Plain prompts are encoded without injected special tokens; message lists use the
checkpoint's native chat template, with generation prompt enabled. Missing chat
templates are rejected rather than invented.

`export_model(model, output_dir, *, dtype=None, tokenizer=None, source=None)` is
for an already loaded native model. `export_checkpoint(source, output_dir, *,
dtype=None, revision=None)` streams local or revision-pinned, indexed Hub
safetensors; packed and separate expert source layouts are accepted. Local
unsharded files are supported through their safetensors headers. A remote source
without an index must be downloaded locally first: this baseline does not add a
custom range/authentication/downloading contract. It never executes remote model
code. Quantized sources and unknown tensors fail explicitly. Exports do not
overwrite existing directories. Disk space must hold source plus export.

The complete source tensor-name layout is validated before dtype inference,
tensor materialization or source weight-shard download. Missing resident/expert
names, mixed bank layouts and arbitrary extra names fail this metadata-only
preflight. Shape checks then follow incrementally. One narrow exception matches
the pinned native decoder's unused auxiliary MTP state: exactly 47 live decoder
layers, `num_nextn_predict_layers=1`, and the complete enumerated separate-expert
auxiliary layout at index 47. The native unused-key policy must also cover every
excluded name. This never excludes a live decoder layer, incomplete auxiliary
layout or unrecognized name under the auxiliary prefix.

The external schema reference is the [official GLM-4.7-Flash index at revision
7dd20894a642a0aa287e9827cb1a1f7f91386b67](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/7dd20894a642a0aa287e9827cb1a1f7f91386b67/model.safetensors.index.json):
9,703 source names, 9,491 decoder names, and 212 unused auxiliary names. Its
metadata was inspected without fetching the weight payloads; this is not a full
GLM-4.7-Flash execution result. The manifest's `checkpoint_preflight` records
counts, exact excluded names, policy and reason; export CLIs report the excluded
count. Auxiliary tensors are never materialized or exported, and auxiliary-only
shards are never fetched. A shard also containing live decoder weights can still
contain unused auxiliary bytes when that shared source shard is downloaded.

The v1 artifact includes hashed model/generation configs, resident safetensors
shards, separate expert safetensors, a native static rotary-buffer snapshot, and
a final manifest. Startup validates file
existence, size, header shapes/dtypes and expected expert coverage. Each actual
weight load verifies its content hash. A content corruption in an untouched
expert is therefore detected on first demand, not by eagerly reading all expert
payloads at startup. Output completion is atomic on the local filesystem; no
power-loss durability guarantee is asserted.

In-memory export preserves the native rotary buffers' actual dtype and values.
This matters after `model.to(bfloat16)` or `model.to(float16)`: the rounded
nonpersistent frequency buffers can produce different logits from fresh FP32
ones. Checkpoint export instead snapshots the native constructor's FP32 rotary
state, matching native checkpoint loading. Explicit export dtype conversion
applies to learned weights, not to the captured in-memory rotary buffers.
Supported static state must match the configured native frequencies at its
declared buffer precision, with native scaling and sequence limits. Dynamic or
longrope state, unexpected buffers and incompatible mutations are rejected with
an actionable error before a completed export is created. Artifacts lacking a
rotary snapshot must be re-exported; no silent precision normalization is used.
Native numerical comparisons use the same attention implementation and eager
expert execution. Different fused/grouped expert kernels are separate numerical
comparisons, not assumed bitwise-identical execution.

The byte budget bounds **retained expert tensors**, globally across sparse
layers. One demand-loaded expert can bypass a zero/undersized cache; caller
references pin active computations. Serialized bytes, CPU decoding/staging,
device transfer buffers, backbone, KV, activations and allocator/OS caches are
separate allocations. Statistics report logical successful expert-file reads,
not measured physical storage traffic or process peak memory.

Terminal timing uses synchronized model forward plus greedy selection.
Prefill produces the first token; `decode_tokens` counts subsequent token steps,
including native EOS if generated. Decode rate is zero with zero decode steps.
Tokenization, text decoding and consumer wait time are excluded. Request cache
counters are deltas; resident/cache byte counts describe the final snapshot.

Offline artifact journey: `python -m disk_moe.runtime tiny-demo --output PATH`.
This explicitly creates random tiny weights for correctness, with no language
quality or large-model throughput claim. GPU/MPS and full checkpoints require
separate measured verification.
