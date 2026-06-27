"""Public builder: mount OpenAI-compatible routes on a consumer's FastAPI app."""
from __future__ import annotations

from fastapi import APIRouter

from graphserve.persistence import ConversationStore, InMemoryConversationStore
from graphserve.registry import GraphRegistry
from graphserve.routes.chat import build_chat_router
from graphserve.routes.responses import build_responses_router


def create_openai_router(
    registry: GraphRegistry,
    *,
    store: ConversationStore | None = None,
) -> APIRouter:
    """Build an APIRouter with OpenAI-compatible /models and /responses routes.

    GraphServe is a pure bind layer: it serves the registered graphs over the
    OpenAI APIs and nothing else. Cross-cutting concerns are the consumer's
    responsibility, applied with standard FastAPI tools:

    - **Auth**: pass ``dependencies=[Depends(...)]`` to ``app.include_router``.
    - **Callbacks / tracing**: attach to the graph when you construct it.

    Parameters
    ----------
    registry:
        The ``GraphRegistry`` mapping model names to ``GraphConfig`` objects.
    store:
        Optional conversation store. Defaults to a fresh ``InMemoryConversationStore``.

    Notes
    -----
    Stateful GET / ``previous_response_id`` continuity requires the registered
    graph to be compiled with a LangGraph checkpointer — this is the consumer's
    responsibility (e.g. ``graph.compile(checkpointer=MemorySaver())``).
    """
    store = store or InMemoryConversationStore()
    router = APIRouter()

    @router.get("/models")
    async def list_models() -> dict:
        return {
            "object": "list",
            "data": [{"id": m, "object": "model"} for m in registry.list_models()],
        }

    router.include_router(build_responses_router(registry, store))
    router.include_router(build_chat_router(registry))

    return router
