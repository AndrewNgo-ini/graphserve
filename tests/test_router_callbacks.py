from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from graphserve import GraphConfig, GraphRegistry, create_openai_router


class FakeGraph:
    def __init__(self, sink):
        self.sink = sink

    async def ainvoke(self, graph_input, config=None, context=None):
        self.sink["callbacks"] = (config or {}).get("callbacks")
        return {"messages": [AIMessage(content="ok")]}


def test_router_callbacks_provider_invoked_per_request():
    sink = {}
    sentinel = object()
    reg = GraphRegistry()
    reg.register("m", GraphConfig(graph=FakeGraph(sink)))

    app = FastAPI()
    app.include_router(create_openai_router(reg, callbacks=lambda: [sentinel]))
    client = TestClient(app)

    client.post("/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
    })
    assert sink["callbacks"] == [sentinel]


def test_router_no_callbacks_provider_means_no_callbacks_key():
    sink = {}
    reg = GraphRegistry()
    reg.register("m", GraphConfig(graph=FakeGraph(sink)))

    app = FastAPI()
    app.include_router(create_openai_router(reg))
    client = TestClient(app)

    client.post("/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
    })
    assert sink["callbacks"] is None
