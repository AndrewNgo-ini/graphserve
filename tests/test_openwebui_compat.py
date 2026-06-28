"""Open WebUI drop-in compatibility tests.

Open WebUI only ever calls GET /models and POST /chat/completions on an
OpenAI-compatible backend, is stateless toward it (replays the full messages
array each turn), and detects streaming via the text/event-stream content-type.
These tests replay that exact traffic against create_openai_router.
"""
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from graphserve import GraphRegistry, GraphConfig, create_openai_router
from tests.fakes import echo_graph, streaming_text_graph, recording_graph, internal_tool_graph


def _client(model: str, graph) -> TestClient:
    reg = GraphRegistry()
    reg.register(model, GraphConfig(graph=graph))
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    return TestClient(app)


def test_models_list_shape_owui_parses():
    """GET /v1/models returns OpenAI list shape; OWUI reads data[].id.

    OWUI's verify-connection and model picker both read response["data"] and
    each item's ["id"]. Registered model names must appear as ids.
    """
    reg = GraphRegistry()
    reg.register("medical", GraphConfig(graph=echo_graph()))
    reg.register("triage", GraphConfig(graph=echo_graph()))
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    client = TestClient(app)

    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert set(ids) == {"medical", "triage"}
    assert all(m["object"] == "model" for m in body["data"])


def test_full_history_replayed_to_graph():
    """OWUI sends the full messages array every turn; the graph must see all of it.

    OWUI keeps conversation state in its own DB and replays system+prior turns
    on each call. The backend is stateless: every message in the request must
    reach the graph, with roles mapped to the correct LangChain message types.
    """
    graph, received = recording_graph()
    client = _client("medical", graph)

    payload = {
        "model": "medical",
        "messages": [
            {"role": "system", "content": "You are a helpful clinic assistant."},
            {"role": "user", "content": "I have a headache."},
            {"role": "assistant", "content": "How long have you had it?"},
            {"role": "user", "content": "Two days."},
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text

    seen = received["messages"]
    assert len(seen) == 4
    assert isinstance(seen[0], SystemMessage)
    assert isinstance(seen[1], HumanMessage)
    assert isinstance(seen[2], AIMessage)
    assert isinstance(seen[3], HumanMessage)
    assert seen[0].content == "You are a helpful clinic assistant."
    assert seen[3].content == "Two days."


def test_each_request_is_self_contained():
    """A second request does NOT accumulate the first request's messages.

    With no checkpointer (default), each call is an independent thread; the graph
    must see only the messages from the current request — matching OWUI's
    stateless replay model where it always sends the complete history itself.
    """
    graph, received = recording_graph()
    client = _client("medical", graph)

    client.post("/v1/chat/completions", json={
        "model": "medical",
        "messages": [{"role": "user", "content": "first"}],
    })
    assert len(received["messages"]) == 1

    client.post("/v1/chat/completions", json={
        "model": "medical",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ],
    })
    # Sees exactly the 3 messages from THIS request, not 1 + 3 accumulated.
    assert len(received["messages"]) == 3
    assert received["messages"][-1].content == "second"


def test_tool_role_history_accepted():
    """OWUI replays prior tool-call turns: assistant(tool_calls) + tool(result).

    A conversation that previously used tools is replayed verbatim. The backend
    must accept the tool role (with tool_call_id) without erroring and forward a
    ToolMessage to the graph.
    """
    graph, received = recording_graph()
    client = _client("medical", graph)

    payload = {
        "model": "medical",
        "messages": [
            {"role": "user", "content": "weather in Hanoi?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Hanoi\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "31C, sunny"},
            {"role": "user", "content": "and tomorrow?"},
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    assert any(isinstance(m, ToolMessage) for m in received["messages"])


def test_multimodal_content_array_accepted():
    """OWUI sends content as an array of parts (text + image_url) for vision turns.

    The backend must accept list-form content without a 422/500. (Whether the
    graph uses the image is the graph's concern; the bind layer must not reject.)
    """
    graph, received = recording_graph()
    client = _client("medical", graph)

    payload = {
        "model": "medical",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        }],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    assert isinstance(received["messages"][0], HumanMessage)


def test_non_streaming_envelope_owui_fields():
    """Non-streaming response carries the exact chat.completion fields OWUI reads."""
    client = _client("medical", echo_graph())
    r = client.post("/v1/chat/completions", json={
        "model": "medical",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "medical"
    assert isinstance(body["id"], str) and body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert "usage" in body


def test_task_call_json_content_passthrough():
    """A graph returning a JSON-object string (title gen) reaches OWUI verbatim.

    OWUI's title/tags/follow-up task calls parse the content as a JSON object.
    The bind layer must not wrap, escape, or mutate the assistant content.
    """
    title_json = '{"title": "\U0001F3AF Headache Triage Chat"}'

    class TitleGraph:
        async def ainvoke(self, graph_input, config=None, context=None):
            return {"messages": [AIMessage(content=title_json)]}

    client = _client("medical", TitleGraph())
    r = client.post("/v1/chat/completions", json={
        "model": "medical",
        "messages": [{"role": "user", "content": "Create a concise title"}],
        "stream": False,
    })
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content == title_json
    # OWUI then json.loads(content) — verify it survives a round trip.
    assert json.loads(content)["title"] == "\U0001F3AF Headache Triage Chat"


def _collect_sse(client: TestClient, payload: dict):
    """POST a streaming chat completion; return (content_type, chunks, raw)."""
    with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        content_type = r.headers["content-type"]
        raw = "".join(r.iter_text())
    chunks = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            if data == "[DONE]":
                continue
            chunks.append(json.loads(data))
    return content_type, chunks, raw


def test_streaming_content_type_is_event_stream():
    """OWUI only streams to the client when content-type is text/event-stream."""
    client = _client("medical", streaming_text_graph())
    content_type, _, _ = _collect_sse(client, {
        "model": "medical",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert content_type.startswith("text/event-stream")


def test_streaming_chunk_shape_and_terminator():
    """SSE chunks match OpenAI chat.completion.chunk; stream ends with [DONE]."""
    client = _client("medical", streaming_text_graph())
    _, chunks, raw = _collect_sse(client, {
        "model": "medical",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })

    assert raw.rstrip().endswith("data: [DONE]")
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)

    # First chunk announces the assistant role.
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"

    # Content deltas reconstruct the streamed text.
    text = "".join(
        c["choices"][0]["delta"].get("content") or "" for c in chunks
    )
    assert text == "hello world"

    # Final chunk carries finish_reason="stop" for a plain text turn.
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_internal_tool_call_finish_reason_is_stop():
    """When the agent handles a tool internally and replies with text, finish_reason must be "stop".

    A graph that makes a tool call and then produces a final text response must NOT
    end the stream with finish_reason="tool_calls" — that would tell OpenWebUI to
    execute the tool itself, causing it to discard the streamed text response.
    """
    client = _client("medical", internal_tool_graph())
    _, chunks, _ = _collect_sse(client, {
        "model": "medical",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })

    # The final streamed text ("done") must be present.
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert "done" in text, f"Expected final text in stream, got: {text!r}"

    # finish_reason must be "stop", not "tool_calls".
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop", (
        "Internal tool call must not leak finish_reason='tool_calls' to the client"
    )


def test_extra_openai_params_tolerated():
    """OWUI-forwarded OpenAI params not in graphserve's schema are ignored, not rejected."""
    client = _client("medical", echo_graph())
    r = client.post("/v1/chat/completions", json={
        "model": "medical",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "stop": ["\n\n"],
        "max_tokens": 256,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": 42,
        "response_format": {"type": "json_object"},
        "tools": [{"type": "function", "function": {"name": "noop", "parameters": {}}}],
        "tool_choice": "auto",
        "stream_options": {"include_usage": True},
    })
    assert r.status_code == 200, r.text


def test_authorization_header_does_not_break():
    """OWUI sends Authorization: Bearer <key>; graphserve has no auth and must 200."""
    client = _client("medical", echo_graph())
    r = client.post(
        "/v1/chat/completions",
        json={"model": "medical", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer sk-owui-test-key"},
    )
    assert r.status_code == 200


def test_stream_with_include_usage_completes():
    """stream_options.include_usage must not break streaming.

    KNOWN GAP: graphserve does not emit a trailing usage-only chunk that the
    OpenAI spec defines for include_usage=True. OWUI tolerates its absence (usage
    display is best-effort). This test pins current behavior: the stream still
    completes with [DONE] and no chunk reports usage. If graphserve later emits a
    usage chunk, update this assertion deliberately.
    """
    client = _client("medical", streaming_text_graph())
    _, chunks, raw = _collect_sse(client, {
        "model": "medical",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert raw.rstrip().endswith("data: [DONE]")
    assert all(c.get("usage") is None for c in chunks)


def test_unknown_model_returns_openai_404():
    """Requesting an unregistered model returns a 404 with an OpenAI error body."""
    client = _client("medical", echo_graph())
    r = client.post("/v1/chat/completions", json={
        "model": "does-not-exist",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 404
    # FastAPI wraps the handler's detail under "detail".
    error = r.json()["detail"]["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "model_not_found"
    assert isinstance(error["message"], str) and error["message"]
