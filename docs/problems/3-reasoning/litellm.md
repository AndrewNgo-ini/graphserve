# Problem 3 · LiteLLM — reasoning schema

**Client request schema.** LiteLLM exposes a unified `reasoning_effort` param (and
passes provider-specific params like Gemini's `thinking_budget` through).

**Backend response schema.** LiteLLM standardizes reasoning into two fields on the
assistant message: `reasoning_content` (a string) and `thinking_blocks` (a list). For
Anthropic models each block looks like:

```json
{ "type": "thinking", "thinking": "...", "signature": "..." }
```

i.e. the reasoning text is under the `thinking` key (not `text`).

**What GraphServe does today.** Handles the content-block form — a block with
`type: "thinking"`, reading the `thinking` key — and normalizes it into OpenAI reasoning
output. See the "three possible formats" branch in `src/graphserve/translate.py`.

## References

- [LiteLLM — 'Thinking' / 'Reasoning Content'](https://docs.litellm.ai/docs/reasoning_content) (`reasoning_content`, `thinking_blocks`, block `type: "thinking"`)
