"""Adapters between LangChain/LangGraph messages and the OpenAI APIs.

- ``common``    — text extraction and request→context helpers (both APIs)
- ``responses`` — Responses API item conversion and SSE streaming
- ``chat``      — Chat Completions SSE streaming
"""

from graphserve.adapters.chat import chat_completion_chunks
from graphserve.adapters.common import extract_text, request_to_context, result_to_text
from graphserve.adapters.responses import (
    emit_response_sse,
    emit_response_sse_from_astream,
    encode_sse,
    lc_messages_to_openai_items,
    messages_to_response_dict,
)

__all__ = [
    "chat_completion_chunks",
    "extract_text",
    "request_to_context",
    "result_to_text",
    "emit_response_sse",
    "emit_response_sse_from_astream",
    "encode_sse",
    "lc_messages_to_openai_items",
    "messages_to_response_dict",
]
