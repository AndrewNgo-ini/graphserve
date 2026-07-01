# Problem 1 · LangGraph — streaming protocol ≠ OpenAI SSE

**Problem.** LangGraph has its own streaming protocol. `graph.stream()` / `astream()`
emit typed chunks selected by `stream_mode` — `values`, `updates`, `messages`, `custom`,
`debug`. Pass a list and each event is a `(mode, chunk)` tuple. Token streaming
(`stream_mode="messages"`) yields `(message_chunk, metadata)` pairs tagged with the
emitting node name.

Nothing outside LangChain reads this: there is no `choices[].delta`, no `data: [DONE]`,
no `/v1/chat/completions` envelope.

**What GraphServe does today.** Translates the `messages`-mode stream into OpenAI
Responses SSE (`response.created`, `response.output_item.added`,
`response.output_text.delta`, `response.completed`, plus `response.reasoning_text.delta`).
`streamable_node_names` optionally restricts which nodes' tokens are surfaced. See
`src/graphserve/translate.py`.

## References

- [LangGraph — Streaming](https://docs.langchain.com/oss/python/langgraph/streaming) (stream modes, `(mode, chunk)` tuples, `messages`-mode `(token, metadata)` pairs)
