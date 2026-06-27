"""Declarative graph registration. No auto-discovery, no inheritance."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


class UnknownModelError(KeyError):
    pass


@dataclass
class GraphConfig:
    """An already-compiled graph to serve.

    ``graph`` must be an already-compiled graph — GraphServe never constructs
    it. Build/compile the graph in your application and pass it in. Per-request
    concerns (runtime context, callbacks, output extraction) are handled
    generically by GraphServe, not per graph.
    """

    graph: Any
    streamable_node_names: list[str] | None = None


class GraphRegistry:
    def __init__(self) -> None:
        self._configs: dict[str, GraphConfig] = {}

    def register(self, model_name: str, config: GraphConfig) -> None:
        if model_name in self._configs:
            raise ValueError(f"Model {model_name!r} already registered")
        self._configs[model_name] = config

    def resolve(self, model_name: str) -> GraphConfig:
        try:
            return self._configs[model_name]
        except KeyError:
            raise UnknownModelError(
                f"Unknown model {model_name!r}. Available: {self.list_models()}"
            ) from None

    def list_models(self) -> list[str]:
        return list(self._configs)
