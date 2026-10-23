# Initial reconnaissance

Baseline: `Mvgnu/DiskBasedMoE` commit `d345650` (initial repository commit).
Inspection date: 2026-10-23. No open issues or pull requests were returned by the
GitHub connector. This explains architectural decisions, not performance on the
author's original hardware.

| Evidence in baseline | Effect | Decision |
|---|---|---|
| `HybridMoERuntime` constructs random attention and can use random router weights | OLMoE path does not faithfully execute the pretrained model | Archive it outside supported imports |
| GLM4 passes the growing prefix every step; attention receives no KV | Repeated prompt attention, routing and expert loads | Keep native HF model and request-local KV |
| Handwritten native decoder/router copies | API and numerical drift | Replace only expert banks; pin native implementation |
| FP32 construction then ordinary copying of BF16 weights | Dtype expansion and possible mismatches | Preserve dtype on resident materialization |
| Nonstream export saves expert views without isolating backing storage | Each shard can carry an entire expert bank | Independent contiguous safetensors payloads |
| Streaming export records dtype without converting all tensors | Metadata disagrees with actual weights | Validate actual dtype and shape |
| Per-layer count caches without byte metrics | Memory use opaque and scales with layer count | Global byte budget and explicit transient allocations |
| Blocking oracle warmup after execution | Extra IO without established overlap | Defer until native trace evidence exists |
| UI/CLI top-k=1; restoration changes only Python attribute | Altered model semantics and leaked request state | Checkpoint owns routing cardinality |
| Blocking inference in async functions; cleanup outside lock | Health stalls and ownership races | Serialized off-loop worker, bounded delivery |
| Handwritten prompt format and prompt echo | Template/response inconsistency | Native templates and generated text only |
| Three dependency lists contain finance/trading packages | Unrelated installation surface | One pyproject with scoped extras |
| README points at nonexistent `moe_app/`; no tests | No reproducible verification route | Package, offline tiny drill, CI and scoped gates |

The legacy tree preserves evidence of the original attempt. New code never
imports it. Re-export trusted checkpoints rather than automatically unpickling
old exports with unknown storage/dtype defects.

A new model family needs its own native seam and parity checks; matching config
field names does not establish compatibility. Quantized kernels and SSD/GPU
overlap are separate milestones with hardware acceptance requirements. Lop owns
write boundaries; Gauntlet independently executes artifacts against external
bars. Test count is not the definition of quality.
