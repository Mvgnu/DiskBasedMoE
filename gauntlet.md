---
description: Build, inspect the executable artifact against an external bar, repeat
---

# Gauntlet: independent quality review

The PM chooses a user capability and a named outside reference/property. A scoped
builder implements it. A **fresh critic** gets only the goal, spec, rules and
artifact location, not builder conversation or its report. Critics may inspect
source as well as execute the artifact, but must actually run the relevant
journey. They never fix code. If execution is unavailable, the round is blocked,
not passed. A failed attempt is investigated before declaring a blind spot.

Return the single largest gap with a reproducer and the reference property it
violates. The PM sends that gap to the builder; iterate until the critic argues
for a pass or the PM explicitly stops with the remaining gap recorded. There is
no fixed round count. Each pass must say exactly what was inspected and what the
result can establish. Never convert a tiny CPU result into a GPU/1T claim.

## Artifacts and bars

* Runtime: installed/exported tiny native GLM4 model; compare full logits and
  cached decode to pinned Hugging Face Transformers, and inspect memory/IO stats.
* Service: a running localhost HTTP server; drive normal and streaming chat,
  errors, cancellation and a second request. A deterministic test engine can
  establish concurrency behavior; only a real model can establish model output.
* Evaluation: execute the planner, trace replay and file-read benchmark; compare
  arithmetic to explicit transfer lower bounds and fio's cache-state discipline.

At a milestone, a fresh smoothing critic follows the README from installation
through tiny export/generation and planning. Every undocumented fix needed is a
gap. Record reviews in `docs/reviews/` with date, artifact revision/tree, commands,
outcome and limits. PM-owned final evidence records reflect actual executions.

## Definition of done

Domain checks + whole-repo architecture check + package build + spec acceptance,
and independent artifact review, or an explicit scoped PM stop. Hardware-specific
performance remains unverified until the exact model artifact, quantization,
hardware, cold/warm state, prompt length, generated length, TTFT, decode latency,
cache traffic and peak memory are recorded. A paper's result is external evidence.
