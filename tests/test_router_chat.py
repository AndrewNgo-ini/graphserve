"""Tests for POST /chat/completions — non-streaming and streaming."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from graphserve import GraphRegistry, GraphConfig, create_openai_router
from tests.fakes import echo_graph, echo_graph_with_checkpointer


def _client():
    reg = GraphRegistry()
    reg.register("medical", GraphConfig(graph=echo_graph()))
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


def test_chat_non_streaming_output_to_text():
    """output_to_text callable is applied to the full result dict in non-streaming mode."""
    reg = GraphRegistry()
    reg.register(
        "custom",
        GraphConfig(
            graph=echo_graph(),
            output_to_text=lambda out: "CUSTOM:" + out["messages"][-1].content,
        ),
    )
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={"model": "custom", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content == "CUSTOM:echo: hi"


# ---------------------------------------------------------------------------
# metadata field tests
# ---------------------------------------------------------------------------

def test_chat_metadata_accepted_and_reaches_context_factory():
    """A request with metadata={"foo": "bar"} is accepted (200) and the full metadata
    dict is available to the context_factory via request.metadata."""
    captured: dict = {}

    reg = GraphRegistry()
    reg.register(
        "meta-model",
        GraphConfig(
            graph=echo_graph(),
            context_factory=lambda req: captured.update(req.metadata or {}),
        ),
    )
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
    assert captured == {"foo": "bar"}


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
    reg.register("threaded", GraphConfig(graph=echo_graph_with_checkpointer()))
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
    reg.register("threaded2", GraphConfig(graph=echo_graph_with_checkpointer()))
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
