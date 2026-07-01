"""Tests for POST /chat/completions — non-streaming and streaming."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from graphserve import GraphRegistry, create_openai_router
from tests.fakes import echo_graph, echo_graph_with_checkpointer


def _client():
    reg = GraphRegistry()
    reg.register("medical", echo_graph())
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    return TestClient(app)


def test_chat_non_streaming():
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "medical", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"].startswith("echo:")


def test_chat_streaming():
    with _client().stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "medical", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        body = "".join(r.iter_text())
    assert "data: " in body and "[DONE]" in body


def test_chat_non_streaming_returns_last_message_text():
    """The last state message's text becomes the completion content.

    A return_direct-style trailing ToolMessage is surfaced as the response,
    with no per-graph output hook.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class FakeGraph:
        async def ainvoke(self, graph_input, config=None, context=None):
            return {"messages": [
                HumanMessage(content="summarize"),
                AIMessage(content="", tool_calls=[{
                    "id": "c1", "name": "summarize_medical_history",
                    "args": {}, "type": "tool_call",
                }]),
                ToolMessage(content='{"khoa": "Khoa Noi"}', tool_call_id="c1",
                            name="summarize_medical_history"),
            ]}

    reg = GraphRegistry()
    reg.register("custom", FakeGraph())
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={"model": "custom", "messages": [{"role": "user", "content": "summarize"}]},
    )
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content == '{"khoa": "Khoa Noi"}'


# ---------------------------------------------------------------------------
# metadata field tests
# ---------------------------------------------------------------------------

def test_chat_metadata_reaches_graph_context():
    """Request metadata is forwarded to the graph as generic runtime context."""
    from langchain_core.messages import AIMessage

    captured: dict = {}

    class FakeGraph:
        async def ainvoke(self, graph_input, config=None, context=None):
            captured["context"] = context
            return {"messages": [AIMessage(content="ok")]}

    reg = GraphRegistry()
    reg.register("meta-model", FakeGraph())
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "meta-model",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"foo": "bar"},
        },
    )
    assert r.status_code == 200
    assert captured["context"]["metadata"]["foo"] == "bar"


def test_chat_metadata_none_accepted():
    """A request without metadata field is still accepted (200) — metadata defaults to None."""
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "medical", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# metadata.conversation_id threading tests
# ---------------------------------------------------------------------------

def test_chat_conversation_id_same_thread_no_error():
    """Two non-streaming calls with the same metadata.conversation_id both succeed (200).

    Uses echo_graph_with_checkpointer so the thread_id is valid against the
    MemorySaver checkpointer and state is preserved across calls.
    """
    reg = GraphRegistry()
    reg.register("threaded", echo_graph_with_checkpointer())
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    client = TestClient(app)

    payload_first = {
        "model": "threaded",
        "messages": [{"role": "user", "content": "turn one"}],
        "metadata": {"conversation_id": "thread-x"},
    }
    payload_second = {
        "model": "threaded",
        "messages": [{"role": "user", "content": "turn two"}],
        "metadata": {"conversation_id": "thread-x"},
    }

    r1 = client.post("/v1/chat/completions", json=payload_first)
    assert r1.status_code == 200, r1.text

    r2 = client.post("/v1/chat/completions", json=payload_second)
    assert r2.status_code == 200, r2.text

    # Both calls should return echo responses.
    assert r1.json()["choices"][0]["message"]["content"].startswith("echo:")
    assert r2.json()["choices"][0]["message"]["content"].startswith("echo:")


def test_chat_different_conversation_ids_are_independent():
    """Two calls with different conversation_ids use separate threads."""
    reg = GraphRegistry()
    reg.register("threaded2", echo_graph_with_checkpointer())
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    client = TestClient(app)

    r1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "threaded2",
            "messages": [{"role": "user", "content": "alpha"}],
            "metadata": {"conversation_id": "thread-a"},
        },
    )
    r2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "threaded2",
            "messages": [{"role": "user", "content": "beta"}],
            "metadata": {"conversation_id": "thread-b"},
        },
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Each thread echoes its own message.
    assert "alpha" in r1.json()["choices"][0]["message"]["content"]
    assert "beta" in r2.json()["choices"][0]["message"]["content"]
