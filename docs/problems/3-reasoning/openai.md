# Problem 3 · OpenAI — reasoning schema

**Client request schema.** An effort enum:

```json
{ "reasoning_effort": "low" }
```

or, on the Responses API:

```json
{ "reasoning": { "effort": "low" } }
```

Supported values are model-dependent (`none`/`minimal`/`low`/`medium`/`high`/`xhigh`).

**Backend response schema.** A reasoning *summary* item, not the raw thinking tokens.

**What GraphServe does today.** Emits reasoning as OpenAI `type: "reasoning"` items /
`response.reasoning_text.delta` events (`src/graphserve/translate.py`) — this is the
target schema the other providers are normalized *into*.

## References

- [OpenAI — Reasoning models](https://developers.openai.com/api/docs/guides/reasoning) (`reasoning.effort` values, adaptive reasoning, summaries)
