"""OpenAI Responses API route handler.

Storeless: conversation state lives entirely in the registered graph's LangGraph
checkpointer, keyed by ``thread_id``. The response id encodes the model
(``resp_<model>.<hex>``) so stateless GET/DELETE can resolve the owning graph.
"""
from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from graphserve._ids import (
    format_resp_id,
    parse_resp_id,
    thread_uuid_from_anchor,
)
from graphserve.errors import openai_error_body
from graphserve.registry import GraphRegistry, UnknownModelError
from graphserve.translate import (
    encode_sse,
    emit_response_sse_from_astream,
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


def _not_found(response_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=openai_error_body(
            f"Response {response_id!r} not found",
            type="invalid_request_error",
            code="response_not_found",
        ),
    )


def build_responses_router(registry: GraphRegistry) -> APIRouter:
    """Build the /responses sub-router (private — called by create_openai_router)."""
    router = APIRouter()

    @router.post("/responses")
    async def create_response(request: ResponseCreateRequest):
        # 1. Resolve model -> graph (model is always present on create).
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

        # 2. Resolve the thread. `conversation` takes precedence over
        #    `previous_response_id`; both anchor to an existing thread. Absent
        #    either, mint a fresh thread (ephemeral conversation).
        anchor = request.conversation or request.previous_response_id
        if anchor:
            try:
                thread_uuid = thread_uuid_from_anchor(anchor)
            except ValueError:
                thread_uuid = uuid4()
        else:
            thread_uuid = uuid4()

        graph = cfg.graph
        graph_input = {"messages": _input_to_messages(request.input)}
        context = request_to_context(request)
        created_at = int(time.time())

        # 3. LangGraph config.
        run_config: dict = {"configurable": {"thread_id": str(thread_uuid)}}
        if request.chat_template_kwargs:
            enable = request.chat_template_kwargs.get("enable_thinking", False)
            run_config["configurable"]["extra_body"] = {
                "chat_template_kwargs": request.chat_template_kwargs,
                "reasoning": {"enabled": enable},
            }

        resp_id = format_resp_id(request.model, thread_uuid)

        # 4a. Streaming path
        if request.stream:
            sse_stream = emit_response_sse_from_astream(
                graph,
                graph_input,
                config=run_config,
                context=context,
                streamable_node_names=cfg.streamable_node_names,
                resp_id=resp_id,
                model=request.model,
                created_at=created_at,
            )
            return StreamingResponse(
                encode_sse(sse_stream),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # 4b. Non-streaming path
        # Track initial message count to extract only new messages from this turn
        try:
            initial_state = await graph.aget_state({"configurable": {"thread_id": str(thread_uuid)}})
            initial_message_count = len(initial_state.values.get("messages", []) or []) if initial_state else 0
        except (ValueError, AttributeError):
            initial_message_count = 0

        result = await graph.ainvoke(graph_input, config=run_config, context=context)
        all_messages = result.get("messages", []) if isinstance(result, dict) else []
        # Extract only the new messages generated in this turn (not full conversation history)
        new_messages = all_messages[initial_message_count:] if initial_message_count < len(all_messages) else all_messages
        return messages_to_response_dict(
            new_messages,
            conversation_id=thread_uuid,
            model=request.model,
            created_at=created_at,
        )

    def _resolve(response_id: str):
        """Parse the response id and resolve (model, thread_uuid, graph)."""
        try:
            model, thread_uuid = parse_resp_id(response_id)
            cfg = registry.resolve(model)
        except (ValueError, UnknownModelError) as exc:
            raise _not_found(response_id) from exc
        return model, thread_uuid, cfg.graph

    @router.get("/responses/{response_id}")
    async def get_response(response_id: str):
        model, thread_uuid, graph = _resolve(response_id)
        try:
            state = await graph.aget_state({"configurable": {"thread_id": str(thread_uuid)}})
            messages = state.values.get("messages", []) if state and state.values else []
        except ValueError:
            # Graph compiled without a checkpointer (and none injected).
            messages = []
        return messages_to_response_dict(
            messages,
            conversation_id=thread_uuid,
            model=model,
            created_at=int(time.time()),
        )

    @router.get("/responses/{response_id}/input_items")
    async def list_response_input_items(response_id: str, limit: int = 100):
        _model, thread_uuid, graph = _resolve(response_id)
        try:
            state = await graph.aget_state({"configurable": {"thread_id": str(thread_uuid)}})
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
        _model, thread_uuid, graph = _resolve(response_id)
        checkpointer = getattr(graph, "checkpointer", None)
        if checkpointer is not None and hasattr(checkpointer, "adelete_thread"):
            await checkpointer.adelete_thread(str(thread_uuid))
        return {"id": response_id, "object": "response", "deleted": True}

    return router
