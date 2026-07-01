"""Declarative graph registration. No auto-discovery, no inheritance."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


class UnknownModelError(KeyError):
    pass


@dataclass
class _Registered:
    """Internal record. ``graph`` must be an already-compiled graph —
    GraphServe never constructs it."""

    graph: Any
    streamable_node_names: list[str] | None = None


class GraphRegistry:
    def __init__(self) -> None:
        self._configs: dict[str, _Registered] = {}

    def register(
        self,
        model_name: str,
        graph: Any,
        *,
        streamable_node_names: list[str] | None = None,
    ) -> None:
        if model_name in self._configs:
            raise ValueError(f"Model {model_name!r} already registered")
        self._configs[model_name] = _Registered(graph, streamable_node_names)

    def resolve(self, model_name: str) -> _Registered:
        try:
            return self._configs[model_name]
        except KeyError:
            raise UnknownModelError(
                f"Unknown model {model_name!r}. Available: {self.list_models()}"
            ) from None

    def list_models(self) -> list[str]:
        return list(self._configs)
