# Problem 3 · DeepSeek — reasoning schema

**Backend response schema.** `deepseek-reasoner` returns Chain-of-Thought in a
`reasoning_content` field, at the same level as `content`:

```json
{
  "reasoning_content": "…intermediate thinking…",
  "content": "…final answer…"
}
```

Note: passing `reasoning_content` back in a follow-up request returns HTTP 400 — it must
be stripped from prior messages before the next call.

**What GraphServe does today.** Normalizes the `reasoning_content` string into OpenAI
reasoning output (same path as vLLM), in `src/graphserve/translate.py`.

## References

- [DeepSeek — Reasoning Model (deepseek-reasoner)](https://api-docs.deepseek.com/guides/reasoning_model) (`reasoning_content` field, 400 on echo-back)
