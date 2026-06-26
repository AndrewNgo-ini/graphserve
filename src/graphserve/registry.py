"""Declarative graph registration. No auto-discovery, no inheritance."""
from __future__ import annotations
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class UnknownModelError(KeyError):
    pass


@dataclass
class GraphConfig:
    graph: Any | Callable[[], Any]
    request_to_input: Callable[..., dict] | None = None
    context_factory: Callable[..., Any] | None = None
    output_to_text: Callable[..., str] | None = None
    callbacks_factory: Callable[..., list] | None = None

    async def resolve_graph(self) -> Any:
        if callable(self.graph) and not hasattr(self.graph, "astream"):
            result = self.graph()
            if inspect.isawaitable(result):
                return await result
            return result
        return self.graph


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
