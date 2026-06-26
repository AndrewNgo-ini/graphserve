"""OpenAI-shaped error envelope. The mounting app maps exceptions to HTTP."""
from __future__ import annotations

def openai_error_body(message: str, *, type: str, code: str | None = None) -> dict:
    return {"error": {"message": message, "type": type, "code": code}}
