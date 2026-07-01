import pytest
from graphserve.registry import GraphRegistry, UnknownModelError


def test_register_and_resolve():
    reg = GraphRegistry()
    sentinel = object()
    reg.register("medical", sentinel)
    resolved = reg.resolve("medical")
    assert resolved.graph is sentinel
    assert resolved.streamable_node_names is None
    assert reg.list_models() == ["medical"]

def test_register_streamable_node_names():
    reg = GraphRegistry()
    reg.register("m", object(), streamable_node_names=["a", "b"])
    assert reg.resolve("m").streamable_node_names == ["a", "b"]

def test_resolve_unknown_raises():
    reg = GraphRegistry()
    with pytest.raises(UnknownModelError):
        reg.resolve("nope")

def test_duplicate_registration_raises():
    reg = GraphRegistry()
    reg.register("m", object())
    with pytest.raises(ValueError):
        reg.register("m", object())
