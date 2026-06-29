# Open WebUI Drop-In Compatibility Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove `graphserve` is a drop-in replacement for the OpenAI-compatible backend that Open WebUI (and similar clients) connect to, by replaying Open WebUI's exact HTTP traffic against the router and asserting on exactly the fields/shapes Open WebUI parses.

**Architecture:** Open WebUI's browser never talks to your backend — its own Python backend (`backend/open_webui/routers/openai.py`) proxies to you. It calls only **`GET {base}/models`** and **`POST {base}/chat/completions`**, is **fully stateless toward the backend** (it replays the entire `messages` array every turn, storing history in its own DB), and decides streaming purely by the `text/event-stream` content-type. So compatibility = nail the Chat Completions + models surface under that traffic. The Responses API is out of scope (Open WebUI never calls it). Tests drive the server with FastAPI `TestClient` + small fake LangGraph graphs — fast, deterministic, no network.

**Tech Stack:** pytest + pytest-asyncio (`asyncio_mode = "auto"`), FastAPI `TestClient`, LangGraph fake graphs (`tests/fakes.py`), `graphserve` public API (`GraphRegistry`, `GraphConfig`, `create_openai_router`).

## Global Constraints

- All routes are mounted under the `/v1` prefix in tests: `app.include_router(create_openai_router(reg), prefix="/v1")`. Open WebUI appends `/models` and `/chat/completions` to whatever base URL the admin registers, so the admin registers `http://host/v1` and the real calls are `/v1/models` and `/v1/chat/completions`.
- All new tests live in a single new file: `tests/test_openwebui_compat.py`. One shared fake (`recording_graph`) is added to `tests/fakes.py`.
- Test command from repo root: `pytest tests/test_openwebui_compat.py -v`. Full suite: `pytest`.
- These are **characterization/compatibility tests over already-implemented code** — most will PASS on first run, which *is* the success signal (it confirms compatibility). The plan flags the expected outcome per test. If a test FAILS, that is a genuine Open-WebUI incompatibility: STOP and use `superpowers:systematic-debugging` to decide whether to fix `graphserve` or correct the test's expectation — do not weaken an assertion to make it green.
- Imports used across the file (put at top of `tests/test_openwebui_compat.py`):
  ```python
  import json
  from fastapi import FastAPI
  from fastapi.testclient import TestClient
  from langchain_core.messages import (
      AIMessage, HumanMessage, SystemMessage, ToolMessage,
  )
  from graphserve import GraphRegistry, GraphConfig, create_openai_router
  from tests.fakes import echo_graph, streaming_text_graph, recording_graph
  ```

---

### Task 1: Shared recording fake + `GET /models` shape

**Files:**
- Modify: `tests/fakes.py` (append a new factory after `streaming_text_graph`, ~line 234)
- Create: `tests/test_openwebui_compat.py`
- Test: `tests/test_openwebui_compat.py`

**Interfaces:**
- Consumes: `GraphRegistry`, `GraphConfig`, `create_openai_router` (public); `State`, `AIMessage`, `StateGraph`, `START`, `END` already imported in `tests/fakes.py`.
- Produces:
  - `recording_graph() -> tuple[CompiledGraph, dict]` — returns `(graph, received)` where, after any invoke, `received["messages"]` holds the full list of LangChain `BaseMessage` objects the graph node saw. Used by Tasks 2–3 to assert full-history replay and role mapping.
  - Pattern `_client(model, graph)` helper inside the test file — builds a `TestClient` with one registered model. Used by every later task.

- [ ] **Step 1: Add the recording fake to `tests/fakes.py`**

Append after `streaming_text_graph` (end of file, after line 234):

```python
def recording_graph():
    """Graph that records the messages it received, then echoes the last one.

    Returns ``(graph, received)``. After any invoke, ``received["messages"]`` is
    the full list of LangChain messages the node saw — used to assert that the
    server forwards the ENTIRE OpenAI ``messages`` array each turn (Open WebUI is
    stateless toward the backend) with roles mapped to the right message types.
    """
    received: dict = {}

    def respond(state: State) -> State:
        received["messages"] = list(state["messages"])
        last = state["messages"][-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        return {"messages": [AIMessage(content=f"echo: {text}")]}

    g = StateGraph(State)
    g.add_node("respond", respond)
    g.add_edge(START, "respond")
    g.add_edge("respond", END)
    return g.compile(), received
```

- [ ] **Step 2: Create the test file with the client helper and the `/models` test**

```python
"""Open WebUI drop-in compatibility tests.

Open WebUI only ever calls GET /models and POST /chat/completions on an
OpenAI-compatible backend, is stateless toward it (replays the full messages
array each turn), and detects streaming via the text/event-stream content-type.
These tests replay that exact traffic against create_openai_router.
"""
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from graphserve import GraphRegistry, GraphConfig, create_openai_router
from tests.fakes import echo_graph, streaming_text_graph, recording_graph


def _client(model: str, graph) -> TestClient:
    reg = GraphRegistry()
    reg.register(model, GraphConfig(graph=graph))
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    return TestClient(app)


def test_models_list_shape_owui_parses():
    """GET /v1/models returns OpenAI list shape; OWUI reads data[].id.

    OWUI's verify-connection and model picker both read response["data"] and
    each item's ["id"]. Registered model names must appear as ids.
    """
    reg = GraphRegistry()
    reg.register("medical", GraphConfig(graph=echo_graph()))
    reg.register("triage", GraphConfig(graph=echo_graph()))
    app = FastAPI()
    app.include_router(create_openai_router(reg), prefix="/v1")
    client = TestClient(app)

    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert set(ids) == {"medical", "triage"}
    assert all(m["object"] == "model" for m in body["data"])
```

- [ ] **Step 3: Run the test**

Run: `pytest tests/test_openwebui_compat.py -v`
Expected: PASS (router returns `{"object": "list", "data": [{"id": m, "object": "model"}, ...]}` — see `src/graphserve/router.py:42-47`).

- [ ] **Step 4: Commit**

```bash
git add tests/fakes.py tests/test_openwebui_compat.py
git commit -m "test(owui): models list shape + recording fake"
```

---

### Task 2: Stateless full-history replay + role mapping

**Files:**
- Test: `tests/test_openwebui_compat.py` (append)

**Interfaces:**
- Consumes: `recording_graph` (Task 1), `_client` (Task 1).
- Produces: nothing new.

- [ ] **Step 1: Write the test**

Append to `tests/test_openwebui_compat.py`:

```python
def test_full_history_replayed_to_graph():
    """OWUI sends the full messages array every turn; the graph must see all of it.

    OWUI keeps conversation state in its own DB and replays system+prior turns
    on each call. The backend is stateless: every message in the request must
    reach the graph, with roles mapped to the correct LangChain message types.
    """
    graph, received = recording_graph()
    client = _client("medical", graph)

    payload = {
        "model": "medical",
        "messages": [
            {"role": "system", "content": "You are a helpful clinic assistant."},
            {"role": "user", "content": "I have a headache."},
            {"role": "assistant", "content": "How long have you had it?"},
            {"role": "user", "content": "Two days."},
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text

    seen = received["messages"]
    assert len(seen) == 4
    assert isinstance(seen[0], SystemMessage)
    assert isinstance(seen[1], HumanMessage)
    assert isinstance(seen[2], AIMessage)
    assert isinstance(seen[3], HumanMessage)
    assert seen[0].content == "You are a helpful clinic assistant."
    assert seen[3].content == "Two days."


def test_each_request_is_self_contained():
    """A second request does NOT accumulate the first request's messages.

    With no checkpointer (default), each call is an independent thread; the graph
    must see only the messages from the current request — matching OWUI's
    stateless replay model where it always sends the complete history itself.
    """
    graph, received = recording_graph()
    client = _client("medical", graph)

    client.post("/v1/chat/completions", json={
        "model": "medical",
        "messages": [{"role": "user", "content": "first"}],
    })
    assert len(received["messages"]) == 1

    client.post("/v1/chat/completions", json={
        "model": "medical",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ],
    })
    # Sees exactly the 3 messages from THIS request, not 1 + 3 accumulated.
    assert len(received["messages"]) == 3
    assert received["messages"][-1].content == "second"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_openwebui_compat.py -k "full_history or self_contained" -v`
Expected: PASS. `chat.py:74` builds `{"messages": convert_to_messages(request.messages)}`; with no checkpointer each `ainvoke` starts fresh (`chat.py:118`), so the node sees exactly the request's messages.

- [ ] **Step 3: Commit**

```bash
git add tests/test_openwebui_compat.py
git commit -m "test(owui): stateless full-history replay + role mapping"
```

---

### Task 3: Tool-role and multimodal content acceptance

**Files:**
- Test: `tests/test_openwebui_compat.py` (append)

**Interfaces:**
- Consumes: `recording_graph`, `_client`.
- Produces: nothing new.

- [ ] **Step 1: Write the test**

Append to `tests/test_openwebui_compat.py`:

```python
def test_tool_role_history_accepted():
    """OWUI replays prior tool-call turns: assistant(tool_calls) + tool(result).

    A conversation that previously used tools is replayed verbatim. The backend
    must accept the tool role (with tool_call_id) without erroring and forward a
    ToolMessage to the graph.
    """
    graph, received = recording_graph()
    client = _client("medical", graph)

    payload = {
        "model": "medical",
        "messages": [
            {"role": "user", "content": "weather in Hanoi?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Hanoi\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "31C, sunny"},
            {"role": "user", "content": "and tomorrow?"},
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    assert any(isinstance(m, ToolMessage) for m in received["messages"])


def test_multimodal_content_array_accepted():
    """OWUI sends content as an array of parts (text + image_url) for vision turns.

    The backend must accept list-form content without a 422/500. (Whether the
    graph uses the image is the graph's concern; the bind layer must not reject.)
    """
    graph, received = recording_graph()
    client = _client("medical", graph)

    payload = {
        "model": "medical",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ],
        }],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    assert isinstance(received["messages"][0], HumanMessage)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_openwebui_compat.py -k "tool_role or multimodal" -v`
Expected: PASS. `convert_to_messages` (`chat.py:74`) maps the `tool` role to `ToolMessage` (uses `tool_call_id`) and list-form `content` to a `HumanMessage` with content blocks.
Note: if `test_tool_role_history_accepted` errors on conversion, that is a real gap in role handling — debug per `systematic-debugging`, do not delete the tool turn.

- [ ] **Step 3: Commit**

```bash
git add tests/test_openwebui_compat.py
git commit -m "test(owui): tool-role + multimodal content acceptance"
```

---

### Task 4: Non-streaming `chat.completion` envelope + task-call JSON passthrough

**Files:**
- Test: `tests/test_openwebui_compat.py` (append)

**Interfaces:**
- Consumes: `echo_graph`, `_client`.
- Produces: nothing new.

**Why this matters:** After each chat turn OWUI fires extra **non-streaming** `/chat/completions` calls for title, tags, and follow-up generation, expecting the model's content to be a strict JSON object (`{"title": ...}`, `{"tags": [...]}`, `{"follow_ups": [...]}`). The bind layer must pass the graph's text content through verbatim so that JSON survives.

- [ ] **Step 1: Write the test**

Append to `tests/test_openwebui_compat.py`:

```python
def test_non_streaming_envelope_owui_fields():
    """Non-streaming response carries the exact chat.completion fields OWUI reads."""
    client = _client("medical", echo_graph())
    r = client.post("/v1/chat/completions", json={
        "model": "medical",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "medical"
    assert isinstance(body["id"], str) and body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert "usage" in body


def test_task_call_json_content_passthrough():
    """A graph returning a JSON-object string (title gen) reaches OWUI verbatim.

    OWUI's title/tags/follow-up task calls parse the content as a JSON object.
    The bind layer must not wrap, escape, or mutate the assistant content.
    """
    title_json = '{"title": "\U0001F3AF Headache Triage Chat"}'

    class TitleGraph:
        async def ainvoke(self, graph_input, config=None, context=None):
            return {"messages": [AIMessage(content=title_json)]}

    client = _client("medical", TitleGraph())
    r = client.post("/v1/chat/completions", json={
        "model": "medical",
        "messages": [{"role": "user", "content": "Create a concise title"}],
        "stream": False,
    })
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content == title_json
    # OWUI then json.loads(content) — verify it survives a round trip.
    assert json.loads(content)["title"] == "\U0001F3AF Headache Triage Chat"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_openwebui_compat.py -k "envelope or passthrough" -v`
Expected: PASS. Non-streaming envelope is built at `chat.py:137-154`; content comes from `result_to_text` → `extract_text` which returns a string verbatim (`translate.py:215-225`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_openwebui_compat.py
git commit -m "test(owui): non-streaming envelope + task-call JSON passthrough"
```

---

### Task 5: Streaming SSE conformance

**Files:**
- Test: `tests/test_openwebui_compat.py` (append)

**Interfaces:**
- Consumes: `streaming_text_graph` (emits "hello world" token-level), `_client`.
- Produces: `_collect_sse(client, payload) -> tuple[str, list[dict], str]` helper local to the test file — returns `(content_type, parsed_chunks, raw_text)`.

**Why this matters:** OWUI pipes your body straight to the browser only if the response content-type is `text/event-stream`, splits on `data: `, and stops at the literal `[DONE]`. Each event must be a valid `chat.completion.chunk` with `choices[0].delta`.

- [ ] **Step 1: Write the SSE collector helper + streaming tests**

Append to `tests/test_openwebui_compat.py`:

```python
def _collect_sse(client: TestClient, payload: dict):
    """POST a streaming chat completion; return (content_type, chunks, raw)."""
    with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        content_type = r.headers["content-type"]
        raw = "".join(r.iter_text())
    chunks = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            if data == "[DONE]":
                continue
            chunks.append(json.loads(data))
    return content_type, chunks, raw


def test_streaming_content_type_is_event_stream():
    """OWUI only streams to the client when content-type is text/event-stream."""
    client = _client("medical", streaming_text_graph())
    content_type, _, _ = _collect_sse(client, {
        "model": "medical",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert content_type.startswith("text/event-stream")


def test_streaming_chunk_shape_and_terminator():
    """SSE chunks match OpenAI chat.completion.chunk; stream ends with [DONE]."""
    client = _client("medical", streaming_text_graph())
    _, chunks, raw = _collect_sse(client, {
        "model": "medical",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })

    assert raw.rstrip().endswith("data: [DONE]")
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)

    # First chunk announces the assistant role.
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"

    # Content deltas reconstruct the streamed text.
    text = "".join(
        c["choices"][0]["delta"].get("content") or "" for c in chunks
    )
    assert text == "hello world"

    # Final chunk carries finish_reason="stop" for a plain text turn.
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_openwebui_compat.py -k "streaming" -v`
Expected: PASS. Streaming path sets `media_type="text/event-stream"` (`chat.py:110`); chunk shape and `data: [DONE]\n\n` terminator from `chat_completion_chunks` (`translate.py:907-986`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_openwebui_compat.py
git commit -m "test(owui): streaming SSE conformance"
```

---

### Task 6: OWUI extra-field tolerance, auth header, and the usage-in-stream gap

**Files:**
- Test: `tests/test_openwebui_compat.py` (append)

**Interfaces:**
- Consumes: `echo_graph`, `streaming_text_graph`, `_client`, `_collect_sse` (Task 5).
- Produces: nothing new.

**Why this matters:** OWUI forwards the user's/model's full OpenAI param set (`temperature`, `top_p`, `tools`, `response_format`, `stream_options`, …) and an `Authorization: Bearer` header. The bind layer must tolerate all of them (it ignores unknowns) and never reject on auth (auth is the mounting app's concern). The streaming usage chunk is a known OpenAI-spec gap that is **not** OWUI-blocking — this task characterizes the current behavior so a future change is a conscious decision.

- [ ] **Step 1: Write the test**

Append to `tests/test_openwebui_compat.py`:

```python
def test_extra_openai_params_tolerated():
    """OWUI-forwarded OpenAI params not in graphserve's schema are ignored, not rejected."""
    client = _client("medical", echo_graph())
    r = client.post("/v1/chat/completions", json={
        "model": "medical",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "stop": ["\n\n"],
        "max_tokens": 256,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": 42,
        "response_format": {"type": "json_object"},
        "tools": [{"type": "function", "function": {"name": "noop", "parameters": {}}}],
        "tool_choice": "auto",
        "stream_options": {"include_usage": True},
    })
    assert r.status_code == 200, r.text


def test_authorization_header_does_not_break():
    """OWUI sends Authorization: Bearer <key>; graphserve has no auth and must 200."""
    client = _client("medical", echo_graph())
    r = client.post(
        "/v1/chat/completions",
        json={"model": "medical", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer sk-owui-test-key"},
    )
    assert r.status_code == 200


def test_stream_with_include_usage_completes():
    """stream_options.include_usage must not break streaming.

    KNOWN GAP: graphserve does not emit a trailing usage-only chunk that the
    OpenAI spec defines for include_usage=True. OWUI tolerates its absence (usage
    display is best-effort). This test pins current behavior: the stream still
    completes with [DONE] and no chunk reports usage. If graphserve later emits a
    usage chunk, update this assertion deliberately.
    """
    client = _client("medical", streaming_text_graph())
    _, chunks, raw = _collect_sse(client, {
        "model": "medical",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert raw.rstrip().endswith("data: [DONE]")
    assert all(c.get("usage") is None for c in chunks)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_openwebui_compat.py -k "extra_openai or authorization or include_usage" -v`
Expected: PASS. `ChatCompletionRequest` (`chat.py:41-46`) is a plain pydantic model — pydantic v2 ignores extra fields by default, so unknown params don't 422. No auth is enforced in `graphserve` (`router.py:23` documents auth as the consumer's `Depends`). The streaming chunk generator never emits a usage field, so `c.get("usage") is None` holds.

- [ ] **Step 3: Commit**

```bash
git add tests/test_openwebui_compat.py
git commit -m "test(owui): param tolerance, auth header, usage-stream gap"
```

---

### Task 7: Error path — unknown model returns OpenAI-shaped 404

**Files:**
- Test: `tests/test_openwebui_compat.py` (append)

**Interfaces:**
- Consumes: `echo_graph`, `_client`.
- Produces: nothing new.

**Why this matters:** If OWUI requests a model the backend no longer serves (its picker can lag a model-list refresh), the backend must return a clean, OpenAI-shaped error rather than a 500, so OWUI surfaces a readable message instead of marking the whole connection broken.

- [ ] **Step 1: Write the test**

Append to `tests/test_openwebui_compat.py`:

```python
def test_unknown_model_returns_openai_404():
    """Requesting an unregistered model returns a 404 with an OpenAI error body."""
    client = _client("medical", echo_graph())
    r = client.post("/v1/chat/completions", json={
        "model": "does-not-exist",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 404
    # FastAPI wraps the handler's detail under "detail".
    error = r.json()["detail"]["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "model_not_found"
    assert isinstance(error["message"], str) and error["message"]
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_openwebui_compat.py -k "unknown_model" -v`
Expected: PASS. `chat.py:60-68` raises `HTTPException(status_code=404, detail=openai_error_body(..., type="invalid_request_error", code="model_not_found"))`; `openai_error_body` shape is `{"error": {"message", "type", "code"}}` (`errors.py:4-5`), nested under `detail` by FastAPI.

- [ ] **Step 3: Run the full compat suite + full suite**

Run: `pytest tests/test_openwebui_compat.py -v && pytest`
Expected: all compat tests PASS; full suite remains green (no regressions introduced — only additions).

- [ ] **Step 4: Commit**

```bash
git add tests/test_openwebui_compat.py
git commit -m "test(owui): unknown-model OpenAI 404"
```

---

## Self-Review

**Spec coverage** (against the Open WebUI drop-in checklist from research):
1. `GET /models` OpenAI list shape (listing + verify + test-connection) → Task 1. ✓
2. `POST /chat/completions` full `messages` each call, stateless → Task 2. ✓
3. Streaming: `text/event-stream`, `data:` chunks, `[DONE]` → Task 5. ✓
4. Non-streaming JSON `chat.completion` for title/tags/follow-up (strict JSON content passthrough) → Task 4. ✓
5. Roles system/user/assistant/tool + multimodal content → Tasks 2, 3. ✓
6. Tolerate forwarded OpenAI params + Bearer auth → Task 6. ✓
7. Graceful errors (unknown model) → Task 7. ✓
8. Stateless / self-contained per request → Task 2 (`test_each_request_is_self_contained`). ✓
- Out of scope by decision: Responses API statefulness (OWUI never calls it), embeddings (`/embeddings` — only used if OWUI's embedding connection points here; not part of chat swap), Azure/Anthropic path variants. The streaming `usage` chunk is intentionally characterized as a known gap (Task 6), not implemented.

**Placeholder scan:** No TBD/TODO/"add error handling"/"similar to Task N" — every step contains complete, runnable code and an exact command with expected outcome.

**Type/name consistency:** `recording_graph()` returns `(graph, received)` and is used that way in Tasks 2–3. `_client(model, graph)` signature is consistent across all tasks. `_collect_sse` defined in Task 5 and reused in Task 6. Model name `"medical"` used consistently. `received["messages"]` key consistent between fake and assertions.
