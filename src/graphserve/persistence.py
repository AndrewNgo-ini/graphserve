"""Pluggable OpenAI-level conversation metadata store.

Message/thread history lives in the LangGraph checkpointer, NOT here.
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from graphserve._ids import parse_conv_id


class ConversationNotFoundError(Exception):
    def __init__(self, conv_id: uuid.UUID) -> None:
        super().__init__(f"Conversation {conv_id} not found")
        self.conv_id = conv_id


@dataclass(frozen=True)
class Conversation:
    id: uuid.UUID
    model: str
    user: str | None
    created_at: int


@runtime_checkable
class ConversationStore(Protocol):
    async def create(self, *, model: str, user: str | None, created_at: int) -> Conversation: ...
    async def get(self, conv_id: uuid.UUID) -> Conversation: ...
    async def delete(self, conv_id: uuid.UUID) -> None: ...
    async def resolve_previous(self, previous_response_id: str) -> Conversation | None: ...


class InMemoryConversationStore:
    """Default store. Not durable; for dev/test and as the reference impl."""

    def __init__(self) -> None:
        self._items: dict[uuid.UUID, Conversation] = {}

    async def create(self, *, model: str, user: str | None, created_at: int) -> Conversation:
        conv = Conversation(id=uuid.uuid4(), model=model, user=user, created_at=created_at)
        self._items[conv.id] = conv
        return conv

    async def get(self, conv_id: uuid.UUID) -> Conversation:
        try:
            return self._items[conv_id]
        except KeyError:
            raise ConversationNotFoundError(conv_id) from None

    async def delete(self, conv_id: uuid.UUID) -> None:
        self._items.pop(conv_id, None)

    async def resolve_previous(self, previous_response_id: str) -> Conversation | None:
        try:
            conv_id = parse_conv_id(previous_response_id)
        except ValueError:
            return None
        return self._items.get(conv_id)
