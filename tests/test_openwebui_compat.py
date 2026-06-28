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
from tests.fakes import echo_graph, streaming_text_graph, recording_graph


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
