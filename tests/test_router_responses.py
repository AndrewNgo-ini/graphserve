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


def test_models_lists_registered():
    r = _client().get("/v1/models")
    assert r.status_code == 200
    assert "medical" in [m["id"] for m in r.json()["data"]]


def test_non_streaming_response():
    r = _client().post("/v1/responses", json={"model": "medical", "input": "hi", "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "response" and body["model"] == "medical"
    assert any(i["type"] == "message" for i in body["output"])


def test_streaming_response_sse():
    with _client().stream("POST", "/v1/responses", json={"model": "medical", "input": "hi", "stream": True}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert "event: response.created" in body
    assert "event: response.completed" in body


def test_conversation_anchor():
    """POSTing with conversation=<prior id> continues the same thread without error."""
    client = _client()
    # Create an initial response to get an id.
    r1 = client.post("/v1/responses", json={"model": "medical", "input": "hello", "stream": False})
    assert r1.status_code == 200
    prior_id = r1.json()["id"]

    # Continue using the `conversation` field instead of `previous_response_id`.
    r2 = client.post(
        "/v1/responses",
        json={"model": "medical", "input": "follow-up", "stream": False, "conversation": prior_id},
    )
    assert r2.status_code == 200
    assert r2.json()["object"] == "response"


def test_structured_block_input():
    """Structured input list with content blocks must be text-extracted, not repr'd."""
    payload = {
        "model": "medical",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "stream": False,
    }
    r = _client().post("/v1/responses", json=payload)
    assert r.status_code == 200
    body = r.json()
    # The echo graph echoes the last human message text.
    # If structured input was repr'd the output would contain "[{" not "echo: hi".
    output_texts = [
        part["text"]
        for item in body["output"]
        if item["type"] == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    ]
    assert any("echo: hi" in t for t in output_texts), (
        f"Expected 'echo: hi' in output_texts but got: {output_texts}"
    )


# ---------------------------------------------------------------------------
# GET /responses/{id}/input_items
# ---------------------------------------------------------------------------

def _ckpt_client():
    reg = GraphRegistry()
    reg.register("medical", GraphConfig(graph=echo_graph_with_checkpointer()))
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    return TestClient(app)


def test_input_items_lists_conversation():
    """GET /responses/{id}/input_items returns the conversation's items as a list object."""
    client = _ckpt_client()
    r1 = client.post("/v1/responses", json={"model": "medical", "input": "hi", "stream": False})
    resp_id = r1.json()["id"]

    r = client.get(f"/v1/responses/{resp_id}/input_items")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert any(i.get("role") == "user" for i in body["data"])
    assert body["first_id"] == body["data"][0]["id"]
    assert body["last_id"] == body["data"][-1]["id"]
    assert body["has_more"] is False


def test_input_items_limit_sets_has_more():
    client = _ckpt_client()
    resp_id = client.post(
        "/v1/responses", json={"model": "medical", "input": "hi", "stream": False}
    ).json()["id"]

    r = client.get(f"/v1/responses/{resp_id}/input_items", params={"limit": 1})
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 1
    assert body["has_more"] is True


def test_input_items_unknown_id_404():
    r = _ckpt_client().get("/v1/responses/resp_not-a-real-id/input_items")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# metadata field tests
# ---------------------------------------------------------------------------

def test_responses_metadata_reaches_graph_context():
    """A responses request forwards metadata to the graph as generic runtime context."""
    from langchain_core.messages import AIMessage

    captured: dict = {}

    class FakeGraph:
        async def ainvoke(self, graph_input, config=None, context=None):
            captured["context"] = context
            return {"messages": [AIMessage(content="ok")]}

    reg = GraphRegistry()
    reg.register("meta-resp-model", GraphConfig(graph=FakeGraph()))
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    client = TestClient(app)

    r = client.post(
        "/v1/responses",
        json={
            "model": "meta-resp-model",
            "input": "hello",
            "stream": False,
            "metadata": {"patient_id": "P-12345", "session": "abc"},
        },
    )
    assert r.status_code == 200
    assert captured["context"]["metadata"]["patient_id"] == "P-12345"
    assert captured["context"]["metadata"]["session"] == "abc"
