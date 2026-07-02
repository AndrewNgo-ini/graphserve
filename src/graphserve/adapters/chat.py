"""Adapter from LangGraph message streams to the OpenAI Chat Completions API."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from langchain_core.messages import AIMessageChunk

from graphserve.adapters.common import extract_text


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

    # Per-chunk content + tool-call deltas.
    # pending_tool_call: True while the current model turn is emitting tool calls
    # with no subsequent text. Reset to False when text arrives after a tool call
    # so that internal tool use (model calls tool → gets result → replies with text)
    # ends with finish_reason="stop" rather than "tool_calls".
    pending_tool_call = False
    tool_index: dict[str, int] = {}
    next_index = 0
    async for msg_chunk, _metadata in message_stream:
        if not isinstance(msg_chunk, AIMessageChunk):
            continue

        text_content = extract_text(msg_chunk.content) if msg_chunk.content else ""
        if text_content:
            pending_tool_call = False  # text after a tool call → agent handled it
            yield _chunk(ChoiceDelta(content=text_content))

        for tcc in getattr(msg_chunk, "tool_call_chunks", None) or []:
            pending_tool_call = True
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
                type="function" if call_id else None,
                function=ChoiceDeltaToolCallFunction(
                    name=tcc.get("name") or None,
                    arguments=tcc.get("args") or "",
                ),
            )]))

    # finish_reason="tool_calls" only when the stream ended mid-tool-call
    # (i.e. the client is expected to execute the tool). If the agent handled
    # the tool internally and produced final text, finish with "stop".
    yield _chunk(ChoiceDelta(), finish_reason="tool_calls" if pending_tool_call else "stop")

    # SSE terminator
    yield b"data: [DONE]\n\n"
