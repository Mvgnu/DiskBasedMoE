"""PM-owned interfaces between native inference, serving, and evaluation.

Changes require a spec update before domain implementation. Importing this
module never imports torch, transformers, datasets, or the archived experiment.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass(frozen=True)
class GenerationEvent:
    """An append-only text delta or a terminal event; never includes the prompt.

Every generated token may carry a token_id even when its decoded delta is empty.
The final event has a finish_reason and includes measured, named metrics. End of
sequence tokens are counted as generated tokens but need not render as text.
"""

    delta: str = ""
    token_id: int | None = None
    finish_reason: Literal["stop", "length"] | None = None
    metrics: Mapping[str, int | float | str] = field(default_factory=dict)


class InferenceEngine(Protocol):
    """Blocking, single-consumer engine; the caller owns serialization.

Each generate call owns fresh KV state. Closing its iterator ends generation at
the next cooperative boundary. Runtime routing cardinality is checkpoint-owned.
Validation is read-only and must reject impossible requests before HTTP headers.
"""

    def format_messages(self, messages: Sequence[Mapping[str, str]]) -> str: ...

    def validate_request(self, prompt: str, max_new_tokens: int) -> None: ...

    def generate(self, prompt: str, max_new_tokens: int) -> Iterator[GenerationEvent]: ...

    def describe(self) -> dict[str, object]: ...
