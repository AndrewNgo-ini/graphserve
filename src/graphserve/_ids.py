"""Public conversation/response id helpers (``conv_<hex>``)."""
from __future__ import annotations
import uuid

CONV_PREFIX = "conv_"

def format_conv_id(conv_uuid: uuid.UUID) -> str:
    return f"{CONV_PREFIX}{conv_uuid.hex}"

def parse_conv_id(value: str) -> uuid.UUID:
    raw = value.removeprefix(CONV_PREFIX)
    return uuid.UUID(raw)
