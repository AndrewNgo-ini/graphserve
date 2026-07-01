"""Conformance tests: drive GraphServe with the REAL `openai` Python SDK.

The official SDK is the spec. These tests mount the GraphServe FastAPI app
in-process via httpx ASGITransport (no network) and exercise it through
`openai.AsyncOpenAI`. Every assertion relies on the SDK successfully parsing
our wire output into its own typed models — if the SDK parses it, we conform.

Covers: Responses (non-streaming + streaming text + streaming tool calls) and
Chat Completions (non-streaming + streaming).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
import openai
from fastapi import FastAPI
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from graphserve import GraphRegistry, create_openai_router
from tests.fakes import (
    plain_text_llm_graph,
    streaming_text_graph,
    streaming_tool_call_graph,
    tool_call_graph,
    tool_then_text_graph,
)


@asynccontextmanager
async def sdk_client(name: str, graph):
    """Yield an openai.AsyncOpenAI bound to a GraphServe app for *graph*, in-process."""
    registry = GraphRegistry()
    registry.register(name, graph)
    app = FastAPI()
    app.include_router(create_openai_router(registry), prefix="/v1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gs"
    ) as hx:
        yield openai.AsyncOpenAI(api_key="test", base_url="http://gs/v1", http_client=hx)


# ── Responses API ─────────────────────────────────────────────────────────────


async def test_responses_nonstreaming_parses_into_typed_response():
    async with sdk_client("plain", plain_text_llm_graph()) as client:
        resp = await client.responses.create(model="plain", input="hi")
    # SDK parsed our payload into a typed Response with a completed assistant message.
    assert resp.object == "response"
    assert resp.status == "completed"
    assert resp.output[0].type == "message"
    assert resp.output[0].role == "assistant"
    assert resp.output[0].content[0].text == "hello world"


async def test_responses_streaming_text_event_sequence_parses():
    async with sdk_client("plain", plain_text_llm_graph()) as client:
        stream = await client.responses.create(model="plain", input="hi", stream=True)
        events = [event async for event in stream]

    # First/last envelope events are the typed start/finish.
    assert isinstance(events[0], ResponseCreatedEvent)
    assert isinstance(events[-1], ResponseCompletedEvent)

    # Token deltas accumulate to the final text, and the done event agrees.
    deltas = [e.delta for e in events if isinstance(e, ResponseTextDeltaEvent)]
    done = [e for e in events if isinstance(e, ResponseTextDoneEvent)]
    assert "".join(deltas) == "hello world"
    assert done and done[0].text == "hello world"

    # The terminal response carries the full assembled output.
    assert events[-1].response.output[0].content[0].text == "hello world"


async def test_responses_streaming_tool_call_parses():
    async with sdk_client("tools", tool_then_text_graph()) as client:
        stream = await client.responses.create(model="tools", input="hi", stream=True)
        events = [event async for event in stream]

    # A function_call output item is announced and parsed by the SDK.
    added = [e for e in events if isinstance(e, ResponseOutputItemAddedEvent)]
    assert any(getattr(e.item, "type", None) == "function_call" for e in added)

    # Arguments stream completes and is valid JSON the SDK accepted.
    arg_done = [e for e in events if isinstance(e, ResponseFunctionCallArgumentsDoneEvent)]
    assert arg_done, "expected response.function_call_arguments.done"
    assert json.loads(arg_done[0].arguments) == {"q": "x"}
    assert isinstance(events[-1], ResponseCompletedEvent)


# ── Chat Completions API ────────────────────────────────────────────────────


async def test_chat_completions_nonstreaming_parses():
    async with sdk_client("plain", plain_text_llm_graph()) as client:
        cc = await client.chat.completions.create(
            model="plain", messages=[{"role": "user", "content": "hi"}]
        )
    assert cc.object == "chat.completion"
    assert cc.choices[0].message.role == "assistant"
    assert cc.choices[0].message.content == "hello world"
    assert cc.choices[0].finish_reason == "stop"


async def test_chat_completions_streaming_parses():
    async with sdk_client("plain", streaming_text_graph()) as client:
        stream = await client.chat.completions.create(
            model="plain", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        chunks = [chunk async for chunk in stream]

    # Leading role chunk, accumulated content, terminal finish_reason=stop.
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text == "hello world"
    assert chunks[-1].choices[0].finish_reason == "stop"


async def test_chat_completions_nonstreaming_tool_call_parses():
    async with sdk_client("toolcall", tool_call_graph()) as client:
        cc = await client.chat.completions.create(
            model="toolcall", messages=[{"role": "user", "content": "weather?"}]
        )
    msg = cc.choices[0].message
    assert cc.choices[0].finish_reason == "tool_calls"
    assert msg.tool_calls is not None and len(msg.tool_calls) == 1
    call = msg.tool_calls[0]
    assert call.type == "function"
    assert call.id == "call_abc"
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Hanoi"}


async def test_chat_completions_streaming_tool_call_parses():
    async with sdk_client("stream-tool", streaming_tool_call_graph()) as client:
        stream = await client.chat.completions.create(
            model="stream-tool", messages=[{"role": "user", "content": "weather?"}], stream=True
        )
        chunks = [chunk async for chunk in stream]

    # Reassemble the streamed tool call exactly as an OpenAI client would.
    name, args, call_id = "", "", None
    for c in chunks:
        for tc in c.choices[0].delta.tool_calls or []:
            assert tc.index == 0
            if tc.id:
                call_id = tc.id
            if tc.function and tc.function.name:
                name = tc.function.name
            if tc.function and tc.function.arguments:
                args += tc.function.arguments
    assert call_id == "call_1"
    assert name == "lookup"
    assert json.loads(args) == {"q": "x"}
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
