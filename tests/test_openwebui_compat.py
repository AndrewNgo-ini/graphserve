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
