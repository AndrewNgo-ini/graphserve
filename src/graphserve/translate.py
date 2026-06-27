"""Adapters between LangChain messages and OpenAI Responses API item dicts."""

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
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
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

from graphserve._ids import format_conv_id


# ---------------------------------------------------------------------------
# Public text extractor
# ---------------------------------------------------------------------------

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

    # Reasoning block (Claude extended thinking / native reasoning)
    # Claude format: additional_kwargs["reasoning"] dict with encrypted_content/summary
    reasoning = msg.additional_kwargs.get("reasoning")
    if reasoning:
        encrypted_content = (
            reasoning.get("encrypted_content")
            if isinstance(reasoning, dict)
            else None
        )
        summary = reasoning.get("summary", []) if isinstance(reasoning, dict) else []
        items.append({
            "type": "reasoning",
            "id": _reasoning_id(msg),
            "summary": summary,
            "encrypted_content": encrypted_content,
        })
    else:
        # OpenRouter/vLLM format: content blocks with type="thinking"
        content = oai_msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_text = block.get("text", "")
                    if thinking_text:
                        items.append({
                            "type": "reasoning",
                            "id": _reasoning_id(msg),
                            "summary": [thinking_text],
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
# Public API
# ---------------------------------------------------------------------------

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
        "id": conv_id,
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
# Responses-API SSE streaming
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

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
            logger.debug(f"astream_events: event={kind}")

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


# ---------------------------------------------------------------------------
# Chat Completions chunk stream
# ---------------------------------------------------------------------------

async def chat_completion_chunks(
    message_stream: AsyncIterable[tuple],
    *,
    completion_id: str,
    model: str,
    created: int,
) -> AsyncIterator[bytes]:
    """Translate a LangGraph ``stream_mode="messages"`` stream to Chat Completions SSE.

    ``message_stream`` is an async iterable of ``(msg_chunk, metadata)`` tuples
    already filtered to the ``"messages"`` stream mode by the caller.  No
    ``astream`` call is made inside this function.

    Yields ``data: {...}\\n\\n`` bytes, ending with the ``finish_reason="stop"``
    chunk and then ``data: [DONE]\\n\\n``.

    ``created`` is an epoch-seconds integer supplied by the caller; no
    ``time.time()`` is called inside this function.
    """
    # Leading role chunk
    role_chunk = ChatCompletionChunk(
        id=completion_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant"), finish_reason=None)],
    )
    yield f"data: {role_chunk.model_dump_json()}\n\n".encode()

    def _chunk(delta: ChoiceDelta, finish_reason: str | None = None) -> bytes:
        chunk = ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
        )
        return f"data: {chunk.model_dump_json()}\n\n".encode()

    # Per-chunk content + tool-call deltas
    saw_tool_call = False
    tool_index: dict[str, int] = {}
    next_index = 0
    async for msg_chunk, _metadata in message_stream:
        if not isinstance(msg_chunk, AIMessageChunk):
            continue

        text_content = extract_text(msg_chunk.content) if msg_chunk.content else ""
        if text_content:
            yield _chunk(ChoiceDelta(content=text_content))

        for tcc in getattr(msg_chunk, "tool_call_chunks", None) or []:
            saw_tool_call = True
            # OpenAI deltas correlate fragments by a stable integer index. Use the
            # chunk's own index when present; otherwise assign one per call id.
            idx = tcc.get("index")
            if idx is None:
                key = tcc.get("id") or f"_pos{next_index}"
                idx = tool_index.setdefault(key, next_index)
                if idx == next_index:
                    next_index += 1
            else:
                next_index = max(next_index, idx + 1)
            call_id = tcc.get("id")
            yield _chunk(ChoiceDelta(tool_calls=[ChoiceDeltaToolCall(
                index=idx,
                id=call_id or None,
                # id+type appear on the first fragment for an index, args thereafter.
                type="function" if call_id else None,
                function=ChoiceDeltaToolCallFunction(
                    name=tcc.get("name") or None,
                    arguments=tcc.get("args") or "",
                ),
            )]))

    # Final chunk: tool_calls turns finish with "tool_calls", text turns with "stop".
    yield _chunk(ChoiceDelta(), finish_reason="tool_calls" if saw_tool_call else "stop")

    # SSE terminator
    yield b"data: [DONE]\n\n"
