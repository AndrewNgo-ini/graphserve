"""Public builder: mount OpenAI-compatible routes on a consumer's FastAPI app."""
from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends

from graphserve.persistence import ConversationStore, InMemoryConversationStore
from graphserve.registry import GraphRegistry
from graphserve.routes.chat import build_chat_router
from graphserve.routes.responses import build_responses_router


def create_openai_router(
    registry: GraphRegistry,
    *,
    store: ConversationStore | None = None,
    auth: Callable | None = None,
    callbacks: Callable[[], list] | None = None,
) -> APIRouter:
    """Build an APIRouter with OpenAI-compatible /models and /responses routes.

    Parameters
    ----------
    registry:
        The ``GraphRegistry`` mapping model names to ``GraphConfig`` objects.
    store:
        Optional conversation store. Defaults to a fresh ``InMemoryConversationStore``.
    auth:
        Optional callable used as a FastAPI dependency on the router.
    callbacks:
        Optional zero-arg provider invoked per request to produce a fresh
        LangChain callbacks list (e.g. a per-request tracing handler).

    Notes
    -----
    Stateful GET / ``previous_response_id`` continuity requires the registered
    graph to be compiled with a LangGraph checkpointer — this is the consumer's
    responsibility (e.g. ``graph.compile(checkpointer=MemorySaver())``).
    """
    store = store or InMemoryConversationStore()
    deps = [Depends(auth)] if auth else []
    router = APIRouter(dependencies=deps)

    @router.get("/models")
    async def list_models() -> dict:
        return {
            "object": "list",
            "data": [{"id": m, "object": "model"} for m in registry.list_models()],
        }

    # Include the Responses API sub-router.
    router.include_router(build_responses_router(registry, store, auth, callbacks))

    # Include the Chat Completions sub-router (Task 11).
    router.include_router(build_chat_router(registry, auth, callbacks))

    return router
