# GraphServe

GraphServe is an open-source Python library that serves any LangGraph graph over the OpenAI Chat Completions and Responses APIs, enabling seamless integration of agent workflows into OpenAI-compatible applications.

## The problem

We want LangGraph agents driven from off-the-shelf open-source chat UIs like Open WebUI. Three schema mismatches sit in between:

1. **LangGraph speaks its own streaming protocol, not OpenAI's.** `graph.stream()` emits typed chunks chosen by `stream_mode` (`values`/`updates`/`messages`/`custom`/`debug`) — `(message_chunk, metadata)` pairs tagged by node name, not OpenAI SSE deltas. ([docs](https://docs.langchain.com/oss/python/langgraph/streaming))
2. **The open-source chat-UI ecosystem standardized on the OpenAI API.** Not just Open WebUI — [LibreChat](https://www.librechat.ai/docs/quick_start/custom_endpoints), [LobeChat](https://lobehub.com/docs/self-hosting/environment-variables/model-provider), [AnythingLLM](https://docs.anythingllm.com/setup/llm-configuration/cloud/openai-generic), and [HuggingFace Chat UI](https://huggingface.co/docs/chat-ui/en/configuration/models/providers/openai) all integrate arbitrary backends the same way: give them a base URL that serves `GET /v1/models` and `POST /v1/chat/completions`, or the UI can't connect. ([Open WebUI docs](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/))
3. **Backends emit "thinking"/reasoning in incompatible response shapes.** The same reasoning output arrives differently depending on the model behind LangChain: Claude puts a `reasoning` dict (`summary`/`encrypted_content`) in `additional_kwargs`, vLLM/LiteLLM put a flat `reasoning_content` string, and others emit content blocks with `type: "thinking"`. None of these match the OpenAI reasoning schema that clients expect. ([OpenRouter](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens), [vLLM](https://docs.vllm.ai/en/latest/features/reasoning_outputs/))

GraphServe sits between the compiled graph and the client: it presents each graph as an OpenAI model, translates LangGraph's token stream into OpenAI SSE, and normalizes the different backend reasoning shapes into OpenAI's reasoning schema.

Detailed, provider-specific evidence and citations for each problem live in [`docs/problems/`](docs/problems/).

## Installation

Into a uv project (needs an existing `pyproject.toml`):

```bash
uv add graphserve
```

Or into a plain environment / with pip:

```bash
uv pip install graphserve   # or: pip install graphserve
```

## Quickstart

```python
from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END
from graphserve import GraphRegistry, create_openai_router

# 1. Build your LangGraph graph
graph = StateGraph(...)
# ... add nodes and edges ...
compiled = graph.compile()

# 2. Register it under a model name
registry = GraphRegistry()
registry.register("my-agent", compiled)

# Optional: only stream tokens from specific graph nodes
registry.register("my-agent-2", compiled, streamable_node_names=["responder"])

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
| `GraphRegistry` | Registry mapping model names to compiled graphs |
| `create_openai_router` | Builds a FastAPI `APIRouter` with all OpenAI-compatible routes |

### `GraphRegistry.register`

```python
registry.register(
    model_name,                    # str — the OpenAI "model" name clients request
    graph,                         # an already-compiled LangGraph graph
    *,
    streamable_node_names=None,    # list[str] | None — if set, only stream tokens
                                   # emitted by these node names; None streams all
)
```

GraphServe never compiles graphs for you — build and `.compile()` in your app, then
pass the compiled graph in. Registering the same `model_name` twice raises `ValueError`.

## Scope: a pure OpenAI↔LangGraph converter

`create_openai_router(registry)` takes only the registry — there are no knobs. GraphServe
converts between the OpenAI wire format and LangGraph; everything cross-cutting stays your
job, applied with standard tools:

- **Auth** — apply it where you mount the router:
  ```python
  app.include_router(create_openai_router(registry), prefix="/v1",
                     dependencies=[Depends(verify_api_key)])
  ```
- **Callbacks / tracing** — attach to your graph when you build it.

### Runtime context passed to your graph

Per-request context is derived generically from the OpenAI request and exposed to the graph
as LangGraph runtime `context`: `user` → `context["user_id"]`, `instructions` →
`context["metadata"]["custom_instructions"]`, and `metadata` is passed through as
`context["metadata"]`. Graphs read what they need and ignore the rest.

## Stateful responses and `previous_response_id`

All state is managed through the registered graph's LangGraph checkpointer, keyed by `thread_id`.
If a graph is compiled without a checkpointer, GraphServe automatically injects an `InMemorySaver`
(with a warning). Response IDs encode the model (`resp_<model>.<hex>`) so GET/DELETE routes can
resolve the owning graph without any external metadata store.

To use a persistent checkpointer:

```python
from langgraph.checkpoint.memory import MemorySaver
compiled = graph.compile(checkpointer=MemorySaver())
```

## Streaming

Pass `"stream": true` in the request body to receive Server-Sent Events following the OpenAI Responses API event schema (`response.created`, `response.output_item.added`, `response.output_text.delta`, `response.completed`, etc.).

## License

MIT
