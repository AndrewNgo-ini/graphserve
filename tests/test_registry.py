import dataclasses

import pytest
from graphserve.registry import GraphRegistry, GraphConfig, UnknownModelError


def test_graphconfig_fields():
    fields = {f.name for f in dataclasses.fields(GraphConfig)}
    assert fields == {"graph", "streamable_node_names"}

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

def test_graph_is_stored_as_is():
    sentinel = object()
    assert GraphConfig(graph=sentinel).graph is sentinel
