"""Reasoning (enable_thinking) test cases for streaming and non-streaming responses.

Scenarios:
  S1: stream=True,  enable_thinking=True  → response.reasoning_text.delta present
  S2: stream=True,  enable_thinking=False → NO reasoning events
  S3: stream=True,  no chat_template_kwargs → no reasoning events (default)
  S4: stream=False, enable_thinking=True  → output has reasoning + message items
  S5: stream=False, enable_thinking=False → output has ONLY message item
  S6: stream=False, no chat_template_kwargs → no reasoning item (default)
"""
from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from graphserve.adapters import (
    emit_response_sse,
    emit_response_sse_from_astream,
    lc_messages_to_openai_items,
    messages_to_response_dict,
)


async def _drain(events):
    return [e async for e in events]


# ---------------------------------------------------------------------------
# Fake stream helpers — astream_events format (used by emit_response_sse)
# ---------------------------------------------------------------------------

async def _events_with_reasoning():
    """LangGraph astream_events: one reasoning chunk then one text chunk."""
    yield {"event": "on_chat_model_start", "data": {}}

    class ReasoningChunk:
        text = ""
        content = ""
        additional_kwargs = {"reasoning_content": "I need to think about this"}
        tool_call_chunks = []

    yield {"event": "on_chat_model_stream", "data": {"chunk": ReasoningChunk()}}

    class TextChunk:
        text = "Hello"
        content = "Hello"
        additional_kwargs = {}
        tool_call_chunks = []

    yield {"event": "on_chat_model_stream", "data": {"chunk": TextChunk()}}
    yield {"event": "on_chat_model_end", "data": {}}


async def _events_no_reasoning():
    """LangGraph astream_events: text only, no reasoning."""
    yield {"event": "on_chat_model_start", "data": {}}

    class TextChunk:
        text = "Hello"
        content = "Hello"
        additional_kwargs = {}
        tool_call_chunks = []

    yield {"event": "on_chat_model_stream", "data": {"chunk": TextChunk()}}
    yield {"event": "on_chat_model_end", "data": {}}


# ---------------------------------------------------------------------------
# Fake graph — astream format (used by emit_response_sse_from_astream)
# ---------------------------------------------------------------------------

class _FakeGraph:
    """Fake LangGraph graph whose astream yields pre-canned message events."""

    def __init__(self, message_events: list[tuple[Any, dict]]) -> None:
        self._events = message_events

    async def astream(self, *args, **kwargs):
        for msg, metadata in self._events:
            yield {"type": "messages", "data": (msg, metadata)}


def _graph_with_reasoning() -> _FakeGraph:
    reasoning_chunk = AIMessageChunk(
        content="",
        additional_kwargs={"reasoning_content": "step by step"},
    )
    text_chunk = AIMessageChunk(content="Answer")
    metadata = {"langgraph_node": "model"}
    return _FakeGraph([(reasoning_chunk, metadata), (text_chunk, metadata)])


def _graph_no_reasoning() -> _FakeGraph:
    text_chunk = AIMessageChunk(content="Answer")
    metadata = {"langgraph_node": "model"}
    return _FakeGraph([(text_chunk, metadata)])


# ---------------------------------------------------------------------------
# S1 — stream=True, enable_thinking=True → reasoning delta events present
# ---------------------------------------------------------------------------

async def test_s1_stream_with_reasoning_emits_reasoning_delta():
    evts = await _drain(
        emit_response_sse(_events_with_reasoning(), resp_id="r1", model="m", created_at=1)
    )
    types = [e.event for e in evts]
    assert "response.reasoning_text.delta" in types, "S1: reasoning delta must appear"


async def test_s1_stream_astream_with_reasoning_emits_reasoning_delta():
    evts = await _drain(
        emit_response_sse_from_astream(
            _graph_with_reasoning(),
            {"messages": []},
            config={},
            context=None,
            streamable_node_names=["model"],
            resp_id="r1",
            model="m",
            created_at=1,
        )
    )
    types = [e.event for e in evts]
    assert "response.reasoning_text.delta" in types, "S1 (astream): reasoning delta must appear"


async def test_s1_stream_with_reasoning_reasoning_text_content():
    """The reasoning delta event carries the thinking text."""
    evts = await _drain(
        emit_response_sse(_events_with_reasoning(), resp_id="r1", model="m", created_at=1)
    )
    reasoning_evts = [e for e in evts if e.event == "response.reasoning_text.delta"]
    assert reasoning_evts
    data = reasoning_evts[0].data
    delta = data.get("delta") if isinstance(data, dict) else getattr(data, "delta", None)
    assert delta == "I need to think about this"


# ---------------------------------------------------------------------------
# S2 — stream=True, enable_thinking=False → NO reasoning events
# ---------------------------------------------------------------------------

async def test_s2_stream_no_reasoning_no_delta():
    evts = await _drain(
        emit_response_sse(_events_no_reasoning(), resp_id="r2", model="m", created_at=1)
    )
    types = [e.event for e in evts]
    assert "response.reasoning_text.delta" not in types, "S2: no reasoning delta"


async def test_s2_stream_astream_no_reasoning_no_delta():
    evts = await _drain(
        emit_response_sse_from_astream(
            _graph_no_reasoning(),
            {"messages": []},
            config={},
            context=None,
            streamable_node_names=["model"],
            resp_id="r2",
            model="m",
            created_at=1,
        )
    )
    types = [e.event for e in evts]
    assert "response.reasoning_text.delta" not in types, "S2 (astream): no reasoning delta"


# ---------------------------------------------------------------------------
# S3 — stream=True, no chat_template_kwargs → default, no reasoning
# ---------------------------------------------------------------------------

async def test_s3_stream_default_no_reasoning_delta():
    """Default stream with no chat_template_kwargs: no reasoning events."""
    evts = await _drain(
        emit_response_sse(_events_no_reasoning(), resp_id="r3", model="m", created_at=1)
    )
    types = [e.event for e in evts]
    assert "response.reasoning_text.delta" not in types, "S3: default stream has no reasoning"


# ---------------------------------------------------------------------------
# S4 — stream=False, enable_thinking=True → output has reasoning + message
# ---------------------------------------------------------------------------

def test_s4_nonstream_with_reasoning_has_both_items():
    msg = AIMessage(
        content="Answer",
        additional_kwargs={"reasoning_content": "my thoughts"},
    )
    items = lc_messages_to_openai_items([HumanMessage(content="question"), msg])
    item_types = [i["type"] for i in items]
    assert "reasoning" in item_types, "S4: reasoning item missing"
    assert "message" in item_types, "S4: message item missing"


def test_s4_nonstream_reasoning_before_message():
    """reasoning item must precede the message item in output."""
    msg = AIMessage(
        content="Answer",
        additional_kwargs={"reasoning_content": "my thoughts"},
    )
    items = lc_messages_to_openai_items([msg])
    item_types = [i["type"] for i in items]
    assert item_types.index("reasoning") < item_types.index("message"), (
        "S4: reasoning must come before message"
    )


def test_s4_nonstream_reasoning_item_content():
    """reasoning item carries reasoning_text with the correct text."""
    msg = AIMessage(
        content="Answer",
        additional_kwargs={"reasoning_content": "deep thinking"},
    )
    items = lc_messages_to_openai_items([msg])
    reasoning = next(i for i in items if i["type"] == "reasoning")
    content = reasoning.get("content", [])
    assert any(c.get("type") == "reasoning_text" for c in content)
    assert any(c.get("text") == "deep thinking" for c in content)


def test_s4_messages_to_response_dict_with_reasoning():
    conv_id = uuid.uuid4()
    msg = AIMessage(
        content="Final answer",
        additional_kwargs={"reasoning_content": "step by step"},
    )
    result = messages_to_response_dict(
        [HumanMessage(content="q"), msg],
        conversation_id=conv_id,
        model="m",
        created_at=1,
    )
    output_types = [o["type"] for o in result["output"]]
    assert "reasoning" in output_types, "S4 (response dict): reasoning item missing"
    assert "message" in output_types, "S4 (response dict): message item missing"


# ---------------------------------------------------------------------------
# S5 — stream=False, enable_thinking=False → ONLY message item
# ---------------------------------------------------------------------------

def test_s5_nonstream_no_reasoning_only_message():
    msg = AIMessage(content="Simple answer")
    items = lc_messages_to_openai_items([HumanMessage(content="q"), msg])
    item_types = [i["type"] for i in items]
    assert "reasoning" not in item_types, "S5: no reasoning item expected"
    assert "message" in item_types, "S5: message item missing"


def test_s5_messages_to_response_dict_without_reasoning():
    conv_id = uuid.uuid4()
    msg = AIMessage(content="Simple answer")
    result = messages_to_response_dict(
        [HumanMessage(content="q"), msg],
        conversation_id=conv_id,
        model="m",
        created_at=1,
    )
    output_types = [o["type"] for o in result["output"]]
    assert "reasoning" not in output_types, "S5 (response dict): no reasoning item"
    assert "message" in output_types, "S5 (response dict): message item missing"


# ---------------------------------------------------------------------------
# S6 — stream=False, no chat_template_kwargs → default, no reasoning
# ---------------------------------------------------------------------------

def test_s6_nonstream_default_no_reasoning():
    """Default non-streaming with no chat_template_kwargs: no reasoning item."""
    msg = AIMessage(content="Default response")
    items = lc_messages_to_openai_items([msg])
    item_types = [i["type"] for i in items]
    assert "reasoning" not in item_types, "S6: default non-stream has no reasoning"
    assert "message" in item_types, "S6: message item missing"
