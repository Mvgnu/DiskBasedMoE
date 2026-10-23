---
name: service
description: Responsive local chat with isolated requests and honest model controls
paths:
  owned: [src/disk_moe/service/]
  tests: [tests/service/]
exports: [create_app, service CLI, local static UI]
consumes: [disk_moe.contracts.InferenceEngine, disk_moe.contracts.GenerationEvent]
verification:
  check: "python -m ruff check src/disk_moe/service tests/service"
  test: "python -m pytest tests/service -q"
  build: "python -m build --no-isolation"
bar:
  reference: "Hugging Face native chat templates and FastAPI concurrency guidance"
  property: "Chat/stream/cancel/next-request keeps native prompt semantics and model ownership while health stays responsive."
---

# Service

Package FastAPI app, CLI and static UI under the domain. Dependency injection of
the shared engine permits deterministic API/concurrency tests; import ML only at
the CLI composition boundary. GLM4 is the only supported runtime. Default bind
localhost. No remote assets are needed by the UI.

Accept prompt or supported-role messages (exactly one), bounded input and output
limits. Use engine.format_messages for model chat semantics. Reject unsupported
fields including router/oracle overrides, and validate before SSE headers. Both
JSON and SSE return only generated text with terminal finish reason and metrics.

Blocking inference runs off the event loop through a serialized worker with a
bounded stream queue. Cancellation is cooperative: do not release runtime
ownership until the generator/worker stops. Handle a disconnected/slow reader,
exceptions and validation without deadlocks or leaked tasks. Health remains live.

Expose native model and memory/cache limits without calling an estimate measured
throughput. Tests must exercise concurrency, disconnect/error cleanup, invalid
stream requests, defaults, template usage and stream/nonstream equivalence. A
critic must drive a real localhost server, with a named deterministic test engine
when no checkpoint exists; model quality is outside that fixture's evidence.

## HTTP and CLI contract

`create_app(engine, default_max_new_tokens=128, max_pending_requests=8,
stream_queue_size=16)` accepts an injected `InferenceEngine`. `POST /api/chat`
returns `{text, finish_reason, metrics}`. `/api/chat/stream` emits JSON SSE data
events with `delta` and optional `token_id`, a terminal event with native finish
reason and metrics, then `[DONE]`. A failed generation emits `event: error`
instead of a successful completion. JSON requests return HTTP 500 on generation
failure. Model validation and strict-schema failures return HTTP 422 before any
SSE headers. Full admission returns HTTP 503 with `Retry-After: 1`.

Request schemas allow exactly one raw `prompt` or a sequence of native-template
`messages`, plus optional `max_new_tokens`. Input content is limited to 32,768
characters, history to 128 messages, and output to 1..2,048 tokens. Roles are
`system`, `user`, and `assistant`. Unknown fields are errors. Health uses an
engine description captured at app construction so it never touches a model
during generation; request metrics come from terminal generation events.

The CLI accepts `--model-dir`, optional `--tokenizer`, `--device`, `--cache-mib`
(one global expert budget), `--max-new-tokens`, `--host`, and `--port`. Its default
host is `127.0.0.1`. Model imports occur only after argument parsing. The local UI
uses native-template messages and SSE, exposes Stop, and needs no remote assets.

Stop is cooperative: an in-progress native model step must finish before model
ownership can transfer. No forceful thread termination is attempted. A
deterministic fixture establishes HTTP/concurrency behavior only; real model
output, model memory and practical throughput require runtime/evaluation checks.
