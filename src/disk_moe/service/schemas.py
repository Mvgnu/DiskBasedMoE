"""Strict request boundaries shared by JSON and streaming endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_INPUT_CHARACTERS = 32_768
MAX_NEW_TOKENS = 2_048


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_INPUT_CHARACTERS)

    @field_validator("content")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content must not be blank")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    prompt: str | None = Field(default=None, min_length=1, max_length=MAX_INPUT_CHARACTERS)
    messages: tuple[ChatMessage, ...] | None = Field(default=None, min_length=1, max_length=128)
    max_new_tokens: int | None = Field(default=None, ge=1, le=MAX_NEW_TOKENS)

    @field_validator("messages", mode="before")
    @classmethod
    def immutable_messages(cls, value):
        # JSON arrays become immutable at the request boundary.
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_input(self):
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("Provide exactly one of prompt or messages")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("Prompt must not be blank")
        if self.messages is not None:
            if sum(len(message.content) for message in self.messages) > MAX_INPUT_CHARACTERS:
                raise ValueError(f"Total message content exceeds {MAX_INPUT_CHARACTERS} characters")
        return self
