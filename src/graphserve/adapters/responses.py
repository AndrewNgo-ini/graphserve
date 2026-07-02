"""Adapters between LangChain messages and the OpenAI Responses API.

Covers both directions:
  - checkpoint messages -> Responses item dicts / non-streaming payload
  - LangGraph streams -> Responses-API SSE events
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from fastapi.encoders import jsonable_encoder
from fastapi.sse import ServerSentEvent, format_sse_event
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import convert_to_openai_messages

from graphserve._ids import format_conv_id, format_resp_id
from graphserve.adapters.common import extract_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private id helpers (deterministic, stable across restarts)
# ---------------------------------------------------------------------------

def _short_id(raw: str, prefix: str) -> str:
    digest = hashlib.blake2s(raw.encode(), digest_size=12).hexdigest()
    return f"{prefix}{digest}"


def _message_id(message: BaseMessage) -> str:
    return _short_id(message.id or str(id(message)), "msg_")


def _reasoning_id(message: BaseMessage) -> str:
    return _short_id((message.id or str(id(message))) + ":reasoning", "rs_")


def _function_call_id(message: BaseMessage, call_id: str) -> str:
    return _short_id((message.id or str(id(message))) + ":fc:" + call_id, "fc_")


def _function_call_output_id(message: BaseMessage) -> str:
    return _short_id((message.id or str(id(message))) + ":fco", "fco_")


# ---------------------------------------------------------------------------
# Private conversion helpers
# ---------------------------------------------------------------------------

def _message_to_openai_dict(message: BaseMessage) -> dict[str, Any]:
    converted = convert_to_openai_messages(
        [message],
        text_format="block",
        include_id=True,
    )
    return cast(list[dict[str, Any]], converted)[0]


def _ai_message_to_items(msg: AIMessage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    oai_msg = _message_to_openai_dict(msg)

    # Reasoning block — three possible formats:
    # 1. Claude: additional_kwargs["reasoning"] dict with encrypted_content/summary
    # 2. LiteLLM/vLLM: additional_kwargs["reasoning_content"] flat string
    # 3. Content blocks with type="thinking" (LiteLLM key="thinking"; Anthropic key="text")
    reasoning_kwarg = msg.additional_kwargs.get("reasoning")
    reasoning_content = msg.additional_kwargs.get("reasoning_content")
    if reasoning_kwarg and isinstance(reasoning_kwarg, dict):
        items.append({
            "type": "reasoning",
            "id": _reasoning_id(msg),
            "summary": reasoning_kwarg.get("summary", []),
            "encrypted_content": reasoning_kwarg.get("encrypted_content"),
        })
    elif reasoning_content:
        items.append({
            "type": "reasoning",
            "id": _reasoning_id(msg),
            "content": [{"type": "reasoning_text", "text": reasoning_content}],
            "summary": [],
            "encrypted_content": None,
        })
    else:
        # Fall back to content blocks with type="thinking"
        content = oai_msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_text = block.get("thinking") or block.get("text") or ""
                    if thinking_text:
                        items.append({
                            "type": "reasoning",
                            "id": _reasoning_id(msg),
                            "content": [{"type": "reasoning_text", "text": thinking_text}],
                            "summary": [],
                            "encrypted_content": None,
                        })

    tool_calls = cast(list[dict[str, Any]], oai_msg.get("tool_calls") or [])
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        call_id = str(tool_call.get("id") or "")
        items.append({
            "type": "function_call",
            "id": _function_call_id(msg, call_id),
            "call_id": call_id,
            "name": function.get("name", ""),
            "arguments": function.get("arguments", "{}"),
            "status": "completed",
        })

    text = extract_text(oai_msg.get("content", ""))
    if text or not tool_calls:
        content_parts = [{"type": "output_text", "text": text, "annotations": []}] if text else []
        items.append({
            "type": "message",
            "id": _message_id(msg),
            "role": "assistant",
            "status": "completed",
            "content": content_parts,
        })

    return items


def _lc_messages_to_items(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain checkpoint messages to OpenAI Responses item dicts."""
    items: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, SystemMessage)):
            oai_msg = _message_to_openai_dict(msg)
            role = oai_msg.get("role")
            if role not in {"user", "system", "developer"}:
                continue
            items.append({
                "type": "message",
                "id": _message_id(msg),
                "role": role,
                "content": [{"type": "input_text", "text": extract_text(oai_msg.get("content", ""))}],
            })
        elif isinstance(msg, AIMessage):
            items.extend(_ai_message_to_items(msg))
        elif isinstance(msg, ToolMessage):
            oai_msg = _message_to_openai_dict(msg)
            items.append({
                "type": "function_call_output",
                "id": _function_call_output_id(msg),
                "call_id": oai_msg.get("tool_call_id") or msg.tool_call_id or "",
                "output": extract_text(oai_msg.get("content", "")),
                "status": "completed",
            })
    return items


# ---------------------------------------------------------------------------
# Public conversion API
# ---------------------------------------------------------------------------

def lc_messages_to_openai_items(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to OpenAI Responses API item dicts.

    Returns a flat list of dicts with ``type`` in
    ``{"message", "function_call", "function_call_output", "reasoning"}``.
    """
    return _lc_messages_to_items(messages)


def messages_to_response_dict(
    messages: list[BaseMessage],
    *,
    conversation_id: UUID,
    model: str,
    created_at: int,
) -> dict:
    """Build a non-streaming Responses API payload dict from checkpoint messages.

    The caller supplies ``created_at`` (epoch seconds); no ``time.time()`` is
    called inside this function.

    Returns a dict with keys: ``id``, ``object``, ``created_at``, ``model``,
    ``output``, ``parallel_tool_calls``, ``tool_choice``, ``tools``,
    ``metadata``, ``status``.
    """
    all_items = _lc_messages_to_items(messages)
    output_items = [
        item
        for item in all_items
        if not (
            item.get("type") == "message"
            and item.get("role") in ("user", "system", "developer")
        )
    ]
    conv_id = format_conv_id(conversation_id)
    return {
        "id": format_resp_id(model, conversation_id),
        "object": "response",
        "created_at": created_at,
        "model": model,
        "output": output_items,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "metadata": {"conversation_id": conv_id},
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

_EVENT_MODELS: dict[str, type[Any]] = {
    "response.created": ResponseCreatedEvent,
    "response.in_progress": ResponseInProgressEvent,
    "response.output_item.added": ResponseOutputItemAddedEvent,
    "response.content_part.added": ResponseContentPartAddedEvent,
    "response.output_text.delta": ResponseTextDeltaEvent,
    "response.output_text.done": ResponseTextDoneEvent,
    "response.content_part.done": ResponseContentPartDoneEvent,
    "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": ResponseFunctionCallArgumentsDoneEvent,
    "response.output_item.done": ResponseOutputItemDoneEvent,
    "response.completed": ResponseCompletedEvent,
    "response.failed": ResponseFailedEvent,
}


@dataclass
class _OutputItemState:
    item_id: str
    output_index: int
    item_type: str
    text: str = ""
    call_id: str = ""
    name: str = ""
    arguments: str = ""


class _ResponseEventBuilder:
    def __init__(self, *, resp_id: str, model: str, created_at: int) -> None:
        self.resp_id = resp_id
        self.model = model
        self.created_at = int(created_at)
        self.sequence_number = 0
        self.next_output_index = 0

    def response(
        self,
        status: str,
        *,
        output: list[dict[str, Any]] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.resp_id,
            "object": "response",
            "created_at": self.created_at,
            "model": self.model,
            "output": output or [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": status,
        }
        if error is not None:
            payload["error"] = error
        return payload

    def allocate_output_index(self) -> int:
        output_index = self.next_output_index
        self.next_output_index += 1
        return output_index

    def event(self, event_type: str, **payload: Any) -> ServerSentEvent:
        sequence_number = self.sequence_number
        self.sequence_number += 1
        data: dict[str, Any] = {"type": event_type, "sequence_number": sequence_number, **payload}

        model_cls = _EVENT_MODELS.get(event_type)
        if model_cls is not None:
            data = model_cls(**data).model_dump(exclude_none=True)
            # The SDK's Response sub-model coerces created_at to float; restore int.
            resp = data.get("response")
            if isinstance(resp, dict) and "created_at" in resp:
                resp["created_at"] = int(resp["created_at"])

        return ServerSentEvent(
            data=data,
            event=event_type,
            id=str(sequence_number),
        )


def _new_item_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _chunk_text(chunk: Any) -> str:
    """Extract text from a streaming chunk, checking .text first then .content."""
    text = getattr(chunk, "text", "")
    if text:
        return text
    return extract_text(getattr(chunk, "content", ""))


def _tool_output_text(output_val: Any) -> str:
    """Extract text from a tool output value.

    Unwraps ``.content`` if present (e.g. LangChain ToolMessage), delegates to
    ``extract_text`` for str/list content, and falls back to ``str()`` for any
    other type so callers always get a string.
    """
    if output_val is None:
        return ""
    content = getattr(output_val, "content", output_val)
    if isinstance(content, (str, list)):
        return extract_text(content)
    return str(content)


def _tool_call_chunk_key(tool_call_chunk: dict[str, Any]) -> str:
    index = tool_call_chunk.get("index")
    if index is not None:
        return f"index:{index}"
    call_id = tool_call_chunk.get("id")
    if call_id:
        return f"id:{call_id}"
    return f"anon:{uuid.uuid4().hex[:12]}"


def _event_to_bytes(event: ServerSentEvent) -> bytes:
    if event.raw_data is not None:
        data_str: str | None = event.raw_data
    elif event.data is not None:
        if hasattr(event.data, "model_dump_json"):
            data_str = event.data.model_dump_json()
        else:
            data_str = json.dumps(jsonable_encoder(event.data))
    else:
        data_str = None

    return format_sse_event(
        data_str=data_str,
        event=event.event,
        id=event.id,
        retry=event.retry,
        comment=event.comment,
    )


async def encode_sse(
    events: AsyncIterable[ServerSentEvent],
) -> AsyncIterator[bytes]:
    """Encode ``ServerSentEvent`` objects for explicit response instances."""
    async for event in events:
        yield _event_to_bytes(event)


async def emit_response_sse_from_astream(
    graph: Any,
    graph_input: dict[str, Any],
    *,
    config: dict[str, Any],
    context: dict[str, Any] | None,
    streamable_node_names: list[str] | None,
    resp_id: str,
    model: str,
    created_at: int,
) -> AsyncIterator[ServerSentEvent]:
    """Stream using astream instead of astream_events.

    Uses astream with stream_mode=["messages"] to get AIMessageChunk updates,
    then converts them to Responses API SSE format.
    """
    builder = _ResponseEventBuilder(resp_id=resp_id, model=model, created_at=created_at)

    yield builder.event("response.created", response=builder.response("in_progress"))
    yield builder.event("response.in_progress", response=builder.response("in_progress"))

    try:
        current_message = _OutputItemState(
            item_id=_new_item_id("msg"),
            output_index=-1,
            item_type="message",
        )
        completed_output: list[dict] = []
        function_calls: dict[str, _OutputItemState] = {}

        async for event in graph.astream(
            graph_input,
            config=config,
            context=context,
            stream_mode=["messages"],
            subgraphs=True,
            version="v2",
        ):
            if event.get("type") != "messages":
                continue

            message, metadata = event["data"]
            if not isinstance(message, AIMessageChunk):
                continue

            # Filter by streamable node names
            if streamable_node_names:
                node_name = metadata.get("langgraph_node")
                if node_name not in streamable_node_names:
                    continue

            # Extract reasoning if present (LangChain doesn't yet extract vLLM reasoning field)
            extra = getattr(message, "additional_kwargs", None)
            reasoning = ""
            if isinstance(extra, dict):
                reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""

            if not reasoning:
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") in ("thinking", "reasoning"):
                            # LiteLLM uses "thinking" key; Anthropic uses "text"
                            reasoning = block.get("thinking") or block.get("text") or ""
                            break

            if reasoning and current_message:
                yield builder.event(
                    "response.reasoning_text.delta",
                    item_id=current_message.item_id,
                    delta=reasoning,
                )

            # Extract text
            text = _chunk_text(message)
            if text and current_message is not None:
                # Lazy open on first text
                if current_message.output_index == -1:
                    current_message.output_index = builder.allocate_output_index()
                    yield builder.event(
                        "response.output_item.added",
                        output_index=current_message.output_index,
                        item={
                            "id": current_message.item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "status": "in_progress",
                        },
                    )
                    yield builder.event(
                        "response.content_part.added",
                        output_index=current_message.output_index,
                        item_id=current_message.item_id,
                        content_index=0,
                        part={"type": "output_text", "text": "", "annotations": []},
                    )

                current_message.text += text
                yield builder.event(
                    "response.output_text.delta",
                    item_id=current_message.item_id,
                    output_index=current_message.output_index,
                    content_index=0,
                    delta=text,
                    logprobs=[],
                )

            for tcc in getattr(message, "tool_call_chunks", None) or []:
                key = _tool_call_chunk_key(tcc)
                fc = function_calls.get(key)
                if fc is None:
                    fc = _OutputItemState(
                        item_id=_new_item_id("fc"),
                        output_index=builder.allocate_output_index(),
                        item_type="function_call",
                        call_id=str(tcc.get("id") or _new_item_id("call")),
                        name=tcc.get("name") or "",
                    )
                    function_calls[key] = fc
                    yield builder.event(
                        "response.output_item.added",
                        output_index=fc.output_index,
                        item={
                            "id": fc.item_id,
                            "type": "function_call",
                            "call_id": fc.call_id,
                            "name": fc.name,
                            "arguments": "",
                            "status": "in_progress",
                        },
                    )
                else:
                    fc.call_id = str(tcc.get("id") or fc.call_id)
                    fc.name = tcc.get("name") or fc.name
                args_delta = tcc.get("args") or ""
                if args_delta:
                    fc.arguments += args_delta
                    yield builder.event(
                        "response.function_call_arguments.delta",
                        item_id=fc.item_id,
                        output_index=fc.output_index,
                        delta=args_delta,
                    )

        for fc in function_calls.values():
            yield builder.event(
                "response.function_call_arguments.done",
                item_id=fc.item_id,
                output_index=fc.output_index,
                arguments=fc.arguments,
                name=fc.name,
            )
            fc_item = {
                "id": fc.item_id,
                "type": "function_call",
                "call_id": fc.call_id,
                "name": fc.name,
                "arguments": fc.arguments,
                "status": "completed",
            }
            completed_output.append(fc_item)
            yield builder.event(
                "response.output_item.done",
                output_index=fc.output_index,
                item=fc_item,
            )

        # Finalize message
        if current_message and current_message.output_index != -1:
            yield builder.event(
                "response.output_text.done",
                item_id=current_message.item_id,
                output_index=current_message.output_index,
                content_index=0,
                text=current_message.text,
                logprobs=[],
            )
            yield builder.event(
                "response.content_part.done",
                output_index=current_message.output_index,
                item_id=current_message.item_id,
                content_index=0,
                part={"type": "output_text", "text": current_message.text, "annotations": []},
            )
            item = {
                "id": current_message.item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": current_message.text, "annotations": []}] if current_message.text else [],
            }
            completed_output.append(item)
            yield builder.event(
                "response.output_item.done",
                output_index=current_message.output_index,
                item=item,
            )

        yield builder.event(
            "response.completed",
            response=builder.response("completed", output=completed_output),
        )
    except Exception as exc:
        logger.exception("Error in streaming response")
        yield builder.event(
            "response.failed",
            response=builder.response(
                "failed",
                error={"code": "server_error", "message": str(exc)},
            ),
        )


async def emit_response_sse(
    events: AsyncIterable[dict[str, Any]],
    *,
    resp_id: str,
    model: str,
    created_at: int,
    output_sink: list[str] | None = None,
    streamable_node_names: list[str] | None = None,
) -> AsyncIterator[ServerSentEvent]:
    """Translate LangGraph ``astream_events`` into OpenAI Responses SSE events.

    If ``output_sink`` is provided, each completed assistant message's text is
    appended to it so the caller can record the full output after the stream.

    ``created_at`` is an epoch-seconds integer supplied by the caller; no
    ``time.time()`` is called inside this function.
    """

    builder = _ResponseEventBuilder(resp_id=resp_id, model=model, created_at=created_at)
    completed_output: list[dict[str, Any]] = []
    current_message: _OutputItemState | None = None
    function_calls: dict[str, _OutputItemState] = {}
    tool_outputs: dict[str, _OutputItemState] = {}
    error_occurred = False

    yield builder.event(
        "response.created",
        response=builder.response("in_progress"),
    )
    yield builder.event(
        "response.in_progress",
        response=builder.response("in_progress"),
    )

    try:
        async for event in events:
            kind = event.get("event")
            data = event.get("data", {})

            if kind == "on_chat_model_start":
                # Lazy: do NOT emit output_item.added / content_part.added yet.
                # output_index=-1 signals "not yet opened".
                current_message = _OutputItemState(
                    item_id=_new_item_id("msg"),
                    output_index=-1,
                    item_type="message",
                )

            elif kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk is None:
                    continue

                # Filter by streamable_node_names if configured
                if streamable_node_names:
                    metadata = event.get("metadata", {})
                    node_name = metadata.get("langgraph_node")
                    if node_name not in streamable_node_names:
                        continue

                # Forward native reasoning (vLLM enable_thinking) as a dedicated
                # event so clients can show "thinking". Check multiple locations:
                # 1. additional_kwargs["reasoning_content"] or ["reasoning"] (vLLM)
                # 2. content field with type="thinking" (OpenRouter streaming)
                extra = getattr(chunk, "additional_kwargs", None)
                reasoning = ""
                if isinstance(extra, dict):
                    reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""

                # Also check content blocks for type="thinking"
                if not reasoning:
                    content = getattr(chunk, "content", None)
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "thinking":
                                reasoning = block.get("text", "")
                                break

                if reasoning:
                    yield builder.event(
                        "response.reasoning_text.delta",
                        item_id=current_message.item_id if current_message else None,
                        delta=reasoning,
                    )

                text = _chunk_text(chunk)
                if text and current_message is not None:
                    # Lazy open: emit header events on the FIRST text delta.
                    if current_message.output_index == -1:
                        current_message.output_index = builder.allocate_output_index()
                        yield builder.event(
                            "response.output_item.added",
                            output_index=current_message.output_index,
                            item={
                                "id": current_message.item_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "status": "in_progress",
                            },
                        )
                        yield builder.event(
                            "response.content_part.added",
                            output_index=current_message.output_index,
                            item_id=current_message.item_id,
                            content_index=0,
                            part={"type": "output_text", "text": "", "annotations": []},
                        )
                    current_message.text += text
                    yield builder.event(
                        "response.output_text.delta",
                        item_id=current_message.item_id,
                        output_index=current_message.output_index,
                        content_index=0,
                        delta=text,
                        logprobs=[],
                    )

                for tool_call_chunk in getattr(chunk, "tool_call_chunks", []) or []:
                    key = _tool_call_chunk_key(tool_call_chunk)
                    state = function_calls.get(key)
                    if state is None:
                        state = _OutputItemState(
                            item_id=_new_item_id("fc"),
                            output_index=builder.allocate_output_index(),
                            item_type="function_call",
                            call_id=str(tool_call_chunk.get("id") or _new_item_id("call")),
                            name=tool_call_chunk.get("name") or "",
                        )
                        function_calls[key] = state
                        yield builder.event(
                            "response.output_item.added",
                            output_index=state.output_index,
                            item={
                                "id": state.item_id,
                                "type": "function_call",
                                "call_id": state.call_id,
                                "name": state.name,
                                "arguments": "",
                                "status": "in_progress",
                            },
                        )
                    else:
                        state.call_id = str(tool_call_chunk.get("id") or state.call_id)
                        state.name = tool_call_chunk.get("name") or state.name

                    args_delta = tool_call_chunk.get("args") or ""
                    if args_delta:
                        state.arguments += args_delta
                        yield builder.event(
                            "response.function_call_arguments.delta",
                            item_id=state.item_id,
                            output_index=state.output_index,
                            delta=args_delta,
                        )

            elif kind == "on_chat_model_end":
                if current_message is not None:
                    if current_message.output_index != -1:
                        # Message item was opened (text was received).
                        if output_sink is not None and current_message.text:
                            output_sink.append(current_message.text)
                        yield builder.event(
                            "response.output_text.done",
                            item_id=current_message.item_id,
                            output_index=current_message.output_index,
                            content_index=0,
                            text=current_message.text,
                            logprobs=[],
                        )
                        yield builder.event(
                            "response.content_part.done",
                            output_index=current_message.output_index,
                            item_id=current_message.item_id,
                            content_index=0,
                            part={"type": "output_text", "text": current_message.text, "annotations": []},
                        )
                        item = {
                            "id": current_message.item_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": current_message.text,
                                    "annotations": [],
                                }
                            ]
                            if current_message.text
                            else [],
                        }
                        completed_output.append(item)
                        yield builder.event(
                            "response.output_item.done",
                            output_index=current_message.output_index,
                            item=item,
                        )
                    # else: output_index == -1 → tool-only turn, emit nothing for message
                    current_message = None

                for state in function_calls.values():
                    yield builder.event(
                        "response.function_call_arguments.done",
                        item_id=state.item_id,
                        output_index=state.output_index,
                        arguments=state.arguments,
                        name=state.name,
                    )
                    item = {
                        "id": state.item_id,
                        "type": "function_call",
                        "call_id": state.call_id,
                        "name": state.name,
                        "arguments": state.arguments,
                        "status": "completed",
                    }
                    completed_output.append(item)
                    yield builder.event(
                        "response.output_item.done",
                        output_index=state.output_index,
                        item=item,
                    )
                function_calls.clear()

            elif kind == "on_tool_start":
                run_id = str(event.get("run_id") or uuid.uuid4().hex)
                state = _OutputItemState(
                    item_id=_new_item_id("fco"),
                    output_index=builder.allocate_output_index(),
                    item_type="function_call_output",
                    call_id=run_id,
                    name=event.get("name") or data.get("name") or "",
                )
                tool_outputs[run_id] = state
                yield builder.event(
                    "response.output_item.added",
                    output_index=state.output_index,
                    item={
                        "id": state.item_id,
                        "type": "function_call_output",
                        "call_id": state.call_id,
                        "output": "",
                        "status": "in_progress",
                    },
                )

            elif kind == "on_tool_end":
                run_id = str(event.get("run_id") or "")
                state = tool_outputs.pop(run_id, None)
                if state is None:
                    state = _OutputItemState(
                        item_id=_new_item_id("fco"),
                        output_index=builder.allocate_output_index(),
                        item_type="function_call_output",
                        call_id=run_id or _new_item_id("call"),
                    )

                output_val = data.get("output")
                state.call_id = str(getattr(output_val, "tool_call_id", None) or state.call_id)
                item = {
                    "id": state.item_id,
                    "type": "function_call_output",
                    "call_id": state.call_id,
                    "output": _tool_output_text(output_val),
                    "status": "completed",
                }
                completed_output.append(item)
                yield builder.event(
                    "response.output_item.done",
                    output_index=state.output_index,
                    item=item,
                )

    except Exception as exc:
        logger.error("Responses streaming error: %s", exc, exc_info=True)
        error_occurred = True
        yield builder.event(
            "response.failed",
            response=builder.response(
                "failed",
                output=completed_output,
                error={"code": "server_error", "message": str(exc)},
            ),
        )

    if not error_occurred:
        yield builder.event(
            "response.completed",
            response=builder.response("completed", output=completed_output),
        )
