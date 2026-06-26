"""End-to-end roundtrip tests: tool-call streaming + GET-after-create."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from graphserve import GraphRegistry, GraphConfig, create_openai_router
from tests.fakes import echo_graph, echo_graph_with_checkpointer, plain_text_llm_graph, tool_then_text_graph


def _client(graph):
    reg = GraphRegistry()
    reg.register("m", GraphConfig(graph=graph))
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    return TestClient(app)


def test_tool_call_streams_function_call_events():
    c = _client(tool_then_text_graph())
    with c.stream("POST", "/v1/responses", json={"model": "m", "input": "go", "stream": True}) as r:
        body = "".join(r.iter_text())
    assert "event: response.output_item.added" in body
    assert "function_call" in body
    assert "event: response.completed" in body


def test_get_after_create_returns_non_empty_output():
    """GET /responses/{id} replays checkpoint state and returns the assistant message."""
    c = _client(echo_graph_with_checkpointer())
    created = c.post("/v1/responses", json={"model": "m", "input": "hi", "stream": False}).json()
    got = c.get(f"/v1/responses/{created['id']}")
    assert got.status_code == 200
    body = got.json()
    assert body["id"] == created["id"]
    # output must be non-empty and contain an assistant message with the echo text
    assert body["output"], "Expected non-empty output from checkpoint replay"
    assistant_items = [item for item in body["output"] if item.get("type") == "message" and item.get("role") == "assistant"]
    assert assistant_items, "Expected at least one assistant message item in output"
    # Verify the echo text appears in the assistant content
    content_text = "".join(
        block.get("text", "")
        for item in assistant_items
        for block in item.get("content", [])
    )
    assert "echo: hi" in content_text, f"Expected 'echo: hi' in content, got: {content_text!r}"


def test_get_after_create_no_checkpointer_returns_empty_output():
    """GET /responses/{id} on a graph without a checkpointer returns 200 with empty output."""
    c = _client(echo_graph())
    created = c.post("/v1/responses", json={"model": "m", "input": "hi", "stream": False}).json()
    got = c.get(f"/v1/responses/{created['id']}")
    assert got.status_code == 200
    body = got.json()
    assert body["id"] == created["id"]
    assert body["output"] == [], f"Expected empty output for checkpointer-less graph, got: {body['output']}"


def test_no_double_emission_single_assistant_message():
    """Regression guard: a graph with a real LLM node must emit exactly ONE assistant message.

    Prior to the fix, the on_chain_stream branch in emit_response_sse would emit
    an AIMessage a second time alongside the on_chat_model_* events, doubling
    assistant output items in the stream and corrupting conversation history.
    """
    c = _client(plain_text_llm_graph())
    with c.stream("POST", "/v1/responses", json={"model": "m", "input": "hi", "stream": True}) as r:
        body = "".join(r.iter_text())

    # Count how many times response.output_item.added events carry a message item
    # Each occurrence of the pattern means one assistant message was added to the stream
    import re
    added_blocks = re.findall(r'event: response\.output_item\.added\ndata: (\{[^\n]+\})', body)
    message_added_count = sum(
        1 for block in added_blocks
        if '"type":"message"' in block or '"type": "message"' in block
    )
    assert message_added_count == 1, (
        f"Expected exactly 1 assistant message in stream, got {message_added_count}. "
        "Double-emission regression detected."
    )


def test_tool_only_first_turn_has_no_empty_message_item():
    """Regression: tool-only turns must not emit an empty message item in completed output.

    tool_then_text_graph's first turn (HumanMessage in) emits only a tool call;
    the second turn (ToolMessage in) emits plain text "done".
    The first turn must produce NO message item in the stream.
    Because the graph node calls model.invoke() synchronously (not astream),
    we see on_chat_model_stream events via astream_events callbacks.
    Assert: the completed event's response.output contains the function_call
    item and NO message item with empty content from the first turn.
    """
    import re
    import json as _json
    c = _client(tool_then_text_graph())
    with c.stream("POST", "/v1/responses", json={"model": "m", "input": "go", "stream": True}) as r:
        body = "".join(r.iter_text())

    # Find the response.completed event's data
    completed_match = re.search(
        r'event: response\.completed\ndata: (\{[^\n]+\})', body
    )
    assert completed_match, "No response.completed event found in stream"
    completed_data = _json.loads(completed_match.group(1))
    output = completed_data.get("response", {}).get("output", [])

    # Must have at least one function_call item
    fc_items = [item for item in output if item.get("type") == "function_call"]
    assert fc_items, f"Expected function_call in completed output, got: {output}"

    # Must NOT have any message item with empty content from the tool-only first turn.
    # (A message item with non-empty content from the text turn IS acceptable if present.)
    empty_msg_items = [
        item for item in output
        if item.get("type") == "message" and not item.get("content")
    ]
    assert not empty_msg_items, (
        f"Tool-only first turn emitted a spurious empty message item: {empty_msg_items}"
    )
