# GraphServe

GraphServe is an open-source Python library that serves any LangGraph graph over the OpenAI Chat Completions and Responses APIs, enabling seamless integration of agent workflows into OpenAI-compatible applications.

## Installation

```bash
pip install graphserve
```

## Quickstart

```python
from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END
from graphserve import GraphRegistry, GraphConfig, create_openai_router

# 1. Build your LangGraph graph
graph = StateGraph(...)
# ... add nodes and edges ...
compiled = graph.compile()

# 2. Register it under a model name
registry = GraphRegistry()
registry.register("my-agent", GraphConfig(graph=compiled))

# 3. Mount the OpenAI-compatible router on your FastAPI app
app = FastAPI()
app.include_router(create_openai_router(registry), prefix="/v1")
```

Your app now exposes:

| Route | Description |
|---|---|
| `GET /v1/models` | List registered graphs |
| `POST /v1/responses` | Create a response (streaming or non-streaming) |
| `GET /v1/responses/{id}` | Retrieve a previous response |
| `DELETE /v1/responses/{id}` | Delete a response |
| `POST /v1/chat/completions` | Chat Completions API |

## Public API

| Export | Description |
|---|---|
| `GraphRegistry` | Registry mapping model names to graph configs |
| `GraphConfig` | Configuration for a single graph (graph, factories, hooks) |
| `create_openai_router` | Builds a FastAPI `APIRouter` with all OpenAI-compatible routes |
| `ConversationStore` | Protocol for plugging in a custom conversation metadata backend |

## `create_openai_router` options

```python
create_openai_router(
    registry,          # GraphRegistry — required
    store=None,        # ConversationStore — defaults to InMemoryConversationStore
    auth=None,         # FastAPI dependency for authentication
)
```

> **Stateful GET / `previous_response_id` continuity** requires the registered
> graph to be compiled with a LangGraph checkpointer — this is the consumer's
> responsibility (e.g. `graph.compile(checkpointer=MemorySaver())`).
> GraphServe reads thread state via `graph.aget_state(...)` at GET time.

## Streaming

Pass `"stream": true` in the request body to receive Server-Sent Events following the OpenAI Responses API event schema (`response.created`, `response.output_item.added`, `response.output_text.delta`, `response.completed`, etc.).

## License

MIT
