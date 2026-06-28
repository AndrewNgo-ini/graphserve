"""Public builder: mount OpenAI-compatible routes on a consumer's FastAPI app."""
from __future__ import annotations

import logging

from fastapi import APIRouter
from langgraph.checkpoint.memory import InMemorySaver

from graphserve.registry import GraphRegistry
from graphserve.routes.chat import build_chat_router
from graphserve.routes.responses import build_responses_router

logger = logging.getLogger(__name__)


def create_openai_router(registry: GraphRegistry) -> APIRouter:
    """Build an APIRouter with OpenAI-compatible /models, /responses, /chat routes.

    GraphServe is a pure OpenAI↔LangGraph converter: it holds no conversation
    store and persists nothing itself. All stateful Responses logic runs against
    the registered graph's LangGraph checkpointer, keyed by ``thread_id``.

    Cross-cutting concerns are the consumer's responsibility, applied with
    standard FastAPI tools:

    - **Auth**: pass ``dependencies=[Depends(...)]`` to ``app.include_router``.
    - **Callbacks / tracing**: attach to the graph when you construct it.

    Each registered graph MUST be compiled with a checkpointer so stateful
    GET / ``previous_response_id`` continuity works. If a graph was compiled
    without one, an ``InMemorySaver`` is injected here (state is lost on restart)
    and a warning is logged.
    """
    for model in registry.list_models():
        graph = registry.resolve(model).graph
        # Real compiled graphs expose ``checkpointer`` (None if compiled without
        # one). Plain test doubles lack the attribute — leave those untouched.
        if getattr(graph, "checkpointer", "missing") is None:
            graph.checkpointer = InMemorySaver()
            logger.warning(
                "Graph %r had no checkpointer; injected InMemorySaver "
                "(conversation state is lost on restart).",
                model,
            )

    router = APIRouter()

    @router.get("/models")
    async def list_models() -> dict:
        return {
            "object": "list",
            "data": [{"id": m, "object": "model"} for m in registry.list_models()],
        }

    router.include_router(build_responses_router(registry))
    router.include_router(build_chat_router(registry))

    return router
