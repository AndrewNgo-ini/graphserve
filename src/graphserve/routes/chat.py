"""OpenAI Chat Completions API route handler."""
from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_messages
from pydantic import BaseModel

from graphserve.errors import openai_error_body
from graphserve.registry import GraphRegistry, UnknownModelError
from graphserve.adapters import chat_completion_chunks, request_to_context, result_to_text


def _openai_tool_calls(message: AIMessage | None) -> list[dict]:
    """Map a LangChain AIMessage's tool_calls to OpenAI chat tool_calls dicts.

    OpenAI requires ``function.arguments`` as a JSON string and a tool-call
    ``id``; LangChain stores args as a dict and the id may be absent.
    """
    if message is None:
        return []
    calls = []
    for tc in getattr(message, "tool_calls", None) or []:
        calls.append({
            "id": tc.get("id") or f"call_{uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": tc.get("name", ""),
                "arguments": json.dumps(tc.get("args") or {}),
            },
        })
    return calls


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = False
    user: str | None = None
    metadata: dict[str, Any] | None = None


def build_chat_router(
    registry: GraphRegistry,
) -> APIRouter:
    """Build the /chat/completions sub-router (private — called by create_openai_router)."""
    router = APIRouter()

    @router.post("/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        # 1. Resolve model config
        try:
            cfg = registry.resolve(request.model)
        except UnknownModelError as exc:
            raise HTTPException(
                status_code=404,
                detail=openai_error_body(
                    str(exc),
                    type="invalid_request_error",
                    code="model_not_found",
                ),
            ) from exc

        # 2. Resolve graph
        graph = cfg.graph

        # 3. Build input from messages
        graph_input = {"messages": convert_to_messages(request.messages)}

        # 4. Build context
        context = request_to_context(request)

        # 5. Build LangGraph config — use conversation_id from metadata as thread_id if provided,
        #    otherwise fall back to a fresh uuid (stateless per-request).
        conv_id = (request.metadata or {}).get("conversation_id")
        thread_id = str(conv_id) if conv_id else uuid4().hex
        run_config: dict = {"configurable": {"thread_id": thread_id}}

        completion_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())

        # 6a. Streaming path
        if request.stream:
            async def _message_stream():
                async for item in graph.astream(
                    graph_input,
                    config=run_config,
                    context=context,
                    stream_mode="messages",
                ):
                    # stream_mode="messages" (string) yields (chunk, metadata) tuples.
                    # NB: a LIST stream_mode would instead yield (mode, (chunk, metadata)),
                    # whose [0] is the mode string — which silently drops all content.
                    if isinstance(item, tuple) and len(item) == 2:
                        yield item

            return StreamingResponse(
                chat_completion_chunks(
                    _message_stream(),
                    completion_id=completion_id,
                    model=request.model,
                    created=created,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # 6b. Non-streaming path
        result = await graph.ainvoke(graph_input, config=run_config, context=context)

        messages = result.get("messages", []) if isinstance(result, dict) else []
        text = result_to_text(result)

        # Surface a pending tool call only when the LAST message is an AIMessage
        # requesting one (a return_direct tool already executed -> trailing
        # ToolMessage -> no pending call).
        last = messages[-1] if messages else None
        last_ai = last if isinstance(last, AIMessage) else None
        tool_calls = _openai_tool_calls(last_ai)
        message: dict[str, Any] = {
            "role": "assistant",
            # OpenAI sets content null on a tool-call turn with no text.
            "content": text or (None if tool_calls else ""),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    return router
