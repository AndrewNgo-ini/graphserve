"""Conversation/response id helpers.

Two id shapes cross the wire:

* ``conv_<hex>`` — a conversation id (the LangGraph ``thread_id``). Minted by the
  consumer's Conversations API and passed back via the ``conversation`` field.
* ``resp_<model>.<hex>`` — a response id. GraphServe is storeless, so the model
  is encoded in the id: ``GET/DELETE /responses/{id}`` parse it back to resolve
  which registered graph owns the thread. The ``<hex>`` is the same thread uuid.
"""
from __future__ import annotations
import uuid

CONV_PREFIX = "conv_"
RESP_PREFIX = "resp_"


def format_conv_id(conv_uuid: uuid.UUID) -> str:
    return f"{CONV_PREFIX}{conv_uuid.hex}"


def parse_conv_id(value: str) -> uuid.UUID:
    raw = value.removeprefix(CONV_PREFIX)
    return uuid.UUID(raw)


def format_resp_id(model: str, conv_uuid: uuid.UUID) -> str:
    return f"{RESP_PREFIX}{model}.{conv_uuid.hex}"


def parse_resp_id(value: str) -> tuple[str, uuid.UUID]:
    """Split ``resp_<model>.<hex>`` into ``(model, thread_uuid)``.

    The 32-char hex never contains a dot, so the *last* dot separates model from
    thread — model names may themselves contain dots/hyphens (``gpt-4.1-mini``).
    Raises ``ValueError`` on a malformed id.
    """
    raw = value.removeprefix(RESP_PREFIX)
    model, hex_ = raw.rsplit(".", 1)  # ponytail: model w/ trailing ".<32hex>" tail can't occur (registry-controlled)
    return model, uuid.UUID(hex_)


def thread_uuid_from_anchor(value: str) -> uuid.UUID:
    """Extract the thread uuid from a ``resp_`` id, ``conv_`` id, or bare uuid.

    The ``conversation`` / ``previous_response_id`` anchor may be either a
    conversation id or a prior response id; both point at the same thread.
    """
    if value.startswith(RESP_PREFIX):
        return parse_resp_id(value)[1]
    return parse_conv_id(value)
