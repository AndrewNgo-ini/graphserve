import pytest
from graphserve.registry import GraphRegistry, GraphConfig, UnknownModelError

def test_register_and_resolve():
    reg = GraphRegistry()
    cfg = GraphConfig(graph=object())
    reg.register("medical", cfg)
    assert reg.resolve("medical") is cfg
    assert reg.list_models() == ["medical"]

def test_resolve_unknown_raises():
    reg = GraphRegistry()
    with pytest.raises(UnknownModelError):
        reg.resolve("nope")

def test_duplicate_registration_raises():
    reg = GraphRegistry()
    reg.register("m", GraphConfig(graph=object()))
    with pytest.raises(ValueError):
        reg.register("m", GraphConfig(graph=object()))

async def test_resolve_graph_direct_and_callable():
    sentinel = object()
    assert await GraphConfig(graph=sentinel).resolve_graph() is sentinel
    assert await GraphConfig(graph=lambda: sentinel).resolve_graph() is sentinel
    async def factory():
        return sentinel
    assert await GraphConfig(graph=factory).resolve_graph() is sentinel
