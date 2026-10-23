---
description: Scoped multi-agent development for disk-backed MoE inference
---

# Lop: scoped build cycles

Lop checks whether a domain implements its spec. Read `gauntlet.md` too: numerical
correctness, useful behavior and credible performance require an independent bar.

## Ownership and write gate

Each domain has `specs/domains/<name>.md` with owned paths, tests, exports,
consumes, verification commands and a named external reference plus one compared
property. Builders may edit only their owned/test paths and their domain spec.
Everything else is read-only until the PM changes the ownership map.

PM-owned: `AGENTS.md`, `lop.md`, `gauntlet.md`, `specs/00_system.md`, root files,
`pyproject.toml`, dependency locks, `.github/`, `scripts/`, `tests/architecture/`,
`src/disk_moe/__init__.py`, and `src/disk_moe/contracts.py`. Build output, weights
and virtual environments are never source. No domain may copy a competing shape
of the shared protocol. Runtime model internals stay inside the runtime domain.

## Cycle

1. Read the domain spec, its external bar and consumed contracts.
2. Start a branch/worktree from the PM milestone. State your write scope.
3. Implement within it. A cross-domain problem goes to the PM before edits.
4. Run the spec commands, then `python scripts/check_architecture.py`.
5. Update public surface and verification records when they change.
6. State disagreements plainly. Do not silently implement a different model,
   relax an assertion, or replace failed loading with random data.
7. Report: changed paths; commands and results; disagreements; remaining gaps.

Green gates start review; they do not establish model quality or practical speed.
Native output parity, byte accounting and a real user journey outrank test count.

## Anti-patterns

| Shortcut | Required alternative |
|---|---|
| Reimplement pretrained attention/router to fit offloading | Replace the expert delivery boundary and compare native logits |
| Tune top-k down for a speed claim | Preserve checkpoint semantics; label approximation as a separate experiment |
| Treat cache count as a memory budget | Account bytes globally and name transient/resident allocations separately |
| Call page-cache throughput SSD throughput | Record IO mode, cache state and what was actually measured |
| Call an estimate a benchmark | Label assumptions, equations, omissions and measured fields |
| Add an unbounded future/work queue | Bound it and test cancellation/error cleanup |
| Mark unsupported hardware passed | Record not run and the evidence still needed |
