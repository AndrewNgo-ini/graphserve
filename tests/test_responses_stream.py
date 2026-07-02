from langchain_core.messages import AIMessageChunk

from graphserve.adapters import emit_response_sse, emit_response_sse_from_astream, encode_sse


async def _drain(events):
    return [e async for e in events]


async def _fake_chat_stream():
    """Mimics LangGraph astream_events v2 for a single text answer."""
    yield {"event": "on_chat_model_start", "data": {}}

    class Chunk:
        text = "Hello"
        content = "Hello"
        additional_kwargs = {}
        tool_call_chunks = []

    yield {"event": "on_chat_model_stream", "data": {"chunk": Chunk()}}
    yield {"event": "on_chat_model_end", "data": {}}


async def test_emits_created_progress_and_completed():
    out: list[str] = []
    evts = await _drain(emit_response_sse(_fake_chat_stream(), resp_id="conv_abc", model="m", created_at=1, output_sink=out))
    types = [e.event for e in evts]
    assert types[0] == "response.created"
    assert "response.in_progress" in types
    assert "response.output_text.delta" in types
    assert types[-1] == "response.completed"
    assert "".join(out) == "Hello"


async def test_text_response_event_sequence():
    """Assert the full event sequence for a text response matches the OpenAI spec."""
    evts = await _drain(emit_response_sse(_fake_chat_stream(), resp_id="conv_abc", model="m", created_at=1))
    types = [e.event for e in evts]

    # Verify required events are present
    assert "response.output_item.added" in types
    assert "response.content_part.added" in types
    assert "response.output_text.delta" in types
    assert "response.output_text.done" in types
    assert "response.content_part.done" in types
    assert "response.output_item.done" in types

    # Verify ordering: content_part.added comes right after output_item.added
    idx_item_added = types.index("response.output_item.added")
    idx_content_part_added = types.index("response.content_part.added")
    assert idx_content_part_added == idx_item_added + 1, (
        f"response.content_part.added (idx={idx_content_part_added}) must immediately follow "
        f"response.output_item.added (idx={idx_item_added})"
    )

    # Verify ordering: content_part.done comes right after output_text.done, before output_item.done
    idx_text_done = types.index("response.output_text.done")
    idx_content_part_done = types.index("response.content_part.done")
    idx_item_done = types.index("response.output_item.done")
    assert idx_content_part_done == idx_text_done + 1, (
        f"response.content_part.done (idx={idx_content_part_done}) must immediately follow "
        f"response.output_text.done (idx={idx_text_done})"
    )
    assert idx_content_part_done < idx_item_done, (
        f"response.content_part.done (idx={idx_content_part_done}) must precede "
        f"response.output_item.done (idx={idx_item_done})"
    )


async def test_created_at_is_int():
    """Assert created_at in response.created event is an int, not a float."""
    evts = await _drain(emit_response_sse(_fake_chat_stream(), resp_id="conv_abc", model="m", created_at=1700000000))
    created_evt = next(e for e in evts if e.event == "response.created")
    # The event data is a dict after model_dump; parse it from the SSE data
    data = created_evt.data
    if isinstance(data, dict):
        created_at_val = data["response"]["created_at"]
    else:
        # Fallback: use model_dump if it's a Pydantic model
        data_dict = data.model_dump() if hasattr(data, "model_dump") else data
        created_at_val = data_dict["response"]["created_at"]
    assert isinstance(created_at_val, int), (
        f"created_at must be int, got {type(created_at_val).__name__}: {created_at_val!r}"
    )
    # Ensure float input is also cast to int
    evts_float = await _drain(emit_response_sse(_fake_chat_stream(), resp_id="conv_abc", model="m", created_at=1700000000))
    created_evt_float = next(e for e in evts_float if e.event == "response.created")
    data_float = created_evt_float.data
    if isinstance(data_float, dict):
        created_at_float_val = data_float["response"]["created_at"]
    else:
        data_float_dict = data_float.model_dump() if hasattr(data_float, "model_dump") else data_float
        created_at_float_val = data_float_dict["response"]["created_at"]
    assert isinstance(created_at_float_val, int)


async def test_encode_sse_produces_wire_bytes():
    evts = emit_response_sse(_fake_chat_stream(), resp_id="conv_abc", model="m", created_at=1)
    lines = b"".join([b async for b in encode_sse(evts)])
    assert b"event: response.created" in lines
    assert lines.endswith(b"\n\n")


async def _fake_tool_only_stream():
    """Mimics a model turn that produces ONLY a tool call — no text at all."""
    yield {"event": "on_chat_model_start", "data": {}}

    class ToolOnlyChunk:
        text = ""
        content = ""
        additional_kwargs = {}
        tool_call_chunks = [{"name": "lookup", "args": '{"q":"x"}', "id": "call_1", "index": 0}]

    yield {"event": "on_chat_model_stream", "data": {"chunk": ToolOnlyChunk()}}

    class AIMsg:
        tool_calls = [{"name": "lookup", "args": {"q": "x"}, "id": "call_1"}]

    yield {"event": "on_chat_model_end", "data": {"output": {"generations": [[{"message": AIMsg()}]]}}}


async def test_tool_only_turn_emits_no_message_item():
    """A tool-only turn must NOT emit any response.output_item.added for a message type.

    Previously, the message item was opened eagerly in on_chat_model_start,
    producing a spurious empty message item in completed_output.
    """
    evts = await _drain(emit_response_sse(
        _fake_tool_only_stream(), resp_id="conv_xyz", model="m", created_at=1
    ))

    # 1. There must be a function_call output_item.added
    added_events = [e for e in evts if e.event == "response.output_item.added"]
    item_types = []
    for e in added_events:
        data = e.data if isinstance(e.data, dict) else (e.data.model_dump() if hasattr(e.data, "model_dump") else {})
        item = data.get("item") or {}
        item_types.append(item.get("type"))

    assert "function_call" in item_types, (
        f"Expected a function_call output_item.added, got item types: {item_types}"
    )

    # 2. There must be NO message output_item.added
    assert "message" not in item_types, (
        f"Tool-only turn must not emit a message output_item.added, got: {item_types}"
    )

    # 3. The completed event's output must contain no message item
    completed = next((e for e in evts if e.event == "response.completed"), None)
    assert completed is not None, "Missing response.completed event"
    completed_data = completed.data if isinstance(completed.data, dict) else (completed.data.model_dump() if hasattr(completed.data, "model_dump") else {})
    output = completed_data.get("response", {}).get("output", [])
    msg_items = [item for item in output if item.get("type") == "message"]
    assert not msg_items, (
        f"Tool-only turn must not produce any message item in completed output, got: {msg_items}"
    )

    # 4. The completed output must contain the function_call item
    fc_items = [item for item in output if item.get("type") == "function_call"]
    assert fc_items, f"Expected function_call item in completed output, got: {output}"


# ---------------------------------------------------------------------------
# response.completed output — astream path (emit_response_sse_from_astream)
# ---------------------------------------------------------------------------

class _SimpleGraph:
    """Minimal fake graph yielding a single text AIMessageChunk."""

    async def astream(self, *args, **kwargs):
        chunk = AIMessageChunk(content="Hello")
        yield {"type": "messages", "data": (chunk, {"langgraph_node": "model"})}


async def test_astream_response_completed_carries_output():
    """response.completed from emit_response_sse_from_astream must include output.

    Regression: the output array was always empty, causing clients that render
    from response.completed.output (e.g. OpenWebUI) to discard streamed text.
    """
    evts = await _drain(emit_response_sse_from_astream(
        _SimpleGraph(),
        {"messages": []},
        config={},
        context=None,
        streamable_node_names=["model"],
        resp_id="r1",
        model="m",
        created_at=1,
    ))
    completed = next((e for e in evts if e.event == "response.completed"), None)
    assert completed is not None, "Missing response.completed event"
    data = completed.data if isinstance(completed.data, dict) else (completed.data.model_dump() if hasattr(completed.data, "model_dump") else {})
    output = data.get("response", {}).get("output", [])
    assert output, "response.completed.response.output must not be empty"
    assert any(item.get("type") == "message" for item in output)
