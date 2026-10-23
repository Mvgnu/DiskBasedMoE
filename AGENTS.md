# DiskBasedMoE development

Read `lop.md`, `gauntlet.md`, `specs/00_system.md`, and your domain SPEC before
editing. This project is a research prototype, not a demonstrated trillion-
parameter inference engine. Claims require reproducible measurements.

The PM owns contracts, root packaging, CI, architecture tests, migration and
cross-domain changes. A builder works in its assigned branch/worktree and writes
only its domain paths. Read access is repository-wide. Never change native
routing, skip experts, or substitute random weights to make a speed check pass.

Never commit model weights, datasets, credentials, local caches or build output.
The archived `experiments/legacy` tree is historical evidence, not a supported
backend. Do not import it from the package. External engines are comparison bars,
not claims that their measured performance applies to this implementation.
