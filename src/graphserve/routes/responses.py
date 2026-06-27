"""OpenAI Responses API route handler."""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from graphserve._ids import format_conv_id, parse_conv_id
from graphserve.errors import openai_error_body
from graphserve.persistence import ConversationNotFoundError, ConversationStore
from graphserve.registry import GraphRegistry, UnknownModelError
from graphserve.translate import (
    encode_sse,
    emit_response_sse,
    extract_text,
    lc_messages_to_openai_items,
    messages_to_response_dict,
    request_to_context,
)


class ResponseCreateRequest(BaseModel):
    model: str
    input: str | list
    stream: bool = False
    user: str | None = None
    previous_response_id: str | None = None
    conversation: str | None = None
    instructions: str | None = None
    metadata: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


def _input_to_messages(input_val: Any) -> list:
    """Convert a plain string or list of turn objects to LangChain messages."""
    if isinstance(input_val, str):
        return [HumanMessage(content=input_val)]
    if isinstance(input_val, list):
        msgs = []
        for item in input_val:
            role = item.get("role", "user") if isinstance(item, dict) else "user"
            content = item.get("content", "") if isinstance(item, dict) else str(item)
            text = extract_text(content)
            if role == "assistant":
                msgs.append(AIMessage(content=text))
            elif role in ("system", "developer"):
                msgs.append(SystemMessage(content=text))
            else:  # user or unknown
                msgs.append(HumanMessage(content=text))
        return msgs
    # fallback
    return [HumanMessage(content=str(input_val))]


def build_responses_router(
    registry: GraphRegistry,
    store: ConversationStore,
) -> APIRouter:
    """Build the /responses sub-router (private — called by create_openai_router)."""
    router = APIRouter()

    @router.post("/responses")
    async def create_response(request: ResponseCreateRequest):
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

        # 2. Resolve or create conversation
        # `conversation` takes precedence over `previous_response_id` when both are set.
        anchor = request.conversation or request.previous_response_id
        if anchor:
            conv = await store.resolve_previous(anchor)
            if conv is None:
                conv = await store.create(
                    model=request.model,
                    user=request.user,
                    created_at=int(time.time()),
                )
        else:
            conv = await store.create(
                model=request.model,
                user=request.user,
                created_at=int(time.time()),
            )

        # 3. Resolve graph
        graph = cfg.graph

        # 4. Build input
        graph_input = {"messages": _input_to_messages(request.input)}

        # 5. Build context
        context = request_to_context(request)

        # 6. Build LangGraph config
        run_config: dict = {"configurable": {"thread_id": str(conv.id)}}
        if request.chat_template_kwargs:
            enable = request.chat_template_kwargs.get("enable_thinking", False)
            run_config["configurable"]["extra_body"] = {
                "chat_template_kwargs": request.chat_template_kwargs,
                "reasoning": {"enabled": enable},
            }

        resp_id = format_conv_id(conv.id)

        # 7a. Streaming path
        if request.stream:
            events = graph.astream_events(
                graph_input,
                config=run_config,
                version="v2",
                context=context,
            )
            sse_stream = emit_response_sse(
                events,
                resp_id=resp_id,
                model=request.model,
                created_at=conv.created_at,
            )
            return StreamingResponse(
                encode_sse(sse_stream),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # 7b. Non-streaming path
        result = await graph.ainvoke(graph_input, config=run_config, context=context)
        return messages_to_response_dict(
            result.get("messages", []) if isinstance(result, dict) else [],
            conversation_id=conv.id,
            model=request.model,
            created_at=conv.created_at,
        )

    @router.get("/responses/{response_id}")
    async def get_response(response_id: str):
        try:
            conv_uuid = parse_conv_id(response_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail=openai_error_body(
                    f"Response {response_id!r} not found",
                    type="invalid_request_error",
                    code="response_not_found",
                ),
            ) from exc

        try:
            conv = await store.get(conv_uuid)
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=openai_error_body(
                    f"Response {response_id!r} not found",
                    type="invalid_request_error",
                    code="response_not_found",
                ),
            ) from exc

        # Resolve the graph and replay thread state from the checkpointer.
        cfg = registry.resolve(conv.model)
        graph = cfg.graph
        try:
            state = await graph.aget_state({"configurable": {"thread_id": str(conv_uuid)}})
            messages = state.values.get("messages", []) if state and state.values else []
        except ValueError:
            # Graph was compiled without a checkpointer — no persisted state.
            messages = []

        return messages_to_response_dict(
            messages,
            conversation_id=conv.id,
            model=conv.model,
            created_at=conv.created_at,
        )

    @router.get("/responses/{response_id}/input_items")
    async def list_response_input_items(response_id: str, limit: int = 100):
        try:
            conv_uuid = parse_conv_id(response_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail=openai_error_body(
                    f"Response {response_id!r} not found",
                    type="invalid_request_error",
                    code="response_not_found",
                ),
            ) from exc

        try:
            conv = await store.get(conv_uuid)
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=openai_error_body(
                    f"Response {response_id!r} not found",
                    type="invalid_request_error",
                    code="response_not_found",
                ),
            ) from exc

        # Replay thread state from the checkpointer (same source as get_response).
        cfg = registry.resolve(conv.model)
        graph = cfg.graph
        try:
            state = await graph.aget_state({"configurable": {"thread_id": str(conv_uuid)}})
            messages = state.values.get("messages", []) if state and state.values else []
        except ValueError:
            messages = []

        items = lc_messages_to_openai_items(messages)[:limit]
        return {
            "object": "list",
            "data": items,
            "first_id": items[0]["id"] if items else None,
            "last_id": items[-1]["id"] if items else None,
            "has_more": len(items) == limit,
        }

    @router.delete("/responses/{response_id}")
    async def delete_response(response_id: str):
        try:
            conv_uuid = parse_conv_id(response_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail=openai_error_body(
                    f"Response {response_id!r} not found",
                    type="invalid_request_error",
                    code="response_not_found",
                ),
            ) from exc

        await store.delete(conv_uuid)
        return {"id": response_id, "object": "response", "deleted": True}

    return router
