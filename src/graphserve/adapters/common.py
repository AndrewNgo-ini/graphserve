"""Helpers shared between the Chat Completions and Responses translators."""

from __future__ import annotations

from typing import Any


def extract_text(content: Any) -> str:
    """Extract plain text from a string or list of content blocks.

    Handles ``str`` directly and lists of dicts with ``type`` in
    ``{"text", "output_text", "input_text"}``.  All other block types are
    silently ignored.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type in ("text", "output_text", "input_text"):
                    parts.append(block.get("text", ""))
    return "".join(parts)


def result_to_text(result: Any) -> str:
    """Extract the response text from a LangGraph result dict.

    Uses the LAST message's text content. This is correct for a normal turn
    (trailing assistant message) and for ``return_direct`` tools (trailing
    tool message whose content IS the response).
    """
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        return ""
    return extract_text(messages[-1].content)


def request_to_context(request: Any) -> dict | None:
    """Build LangGraph runtime context from an OpenAI request, generically.

    Maps the standard OpenAI request fields onto a context envelope every
    graph's middleware can read:
      - ``user`` -> ``context["user_id"]``
      - ``instructions`` (Responses API) -> ``context["metadata"]["custom_instructions"]``
      - ``metadata`` -> ``context["metadata"]`` (passed through verbatim)

    Returns ``None`` when the request carries none of these, so the graph runs
    with no runtime context.
    """
    metadata: dict[str, Any] = dict(getattr(request, "metadata", None) or {})
    instructions = getattr(request, "instructions", None)
    if instructions:
        metadata["custom_instructions"] = instructions

    context: dict[str, Any] = {}
    user = getattr(request, "user", None)
    if user:
        context["user_id"] = user
    if metadata:
        context["metadata"] = metadata

    return context or None
