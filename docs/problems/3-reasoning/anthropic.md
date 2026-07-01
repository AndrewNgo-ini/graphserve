# Problem 3 · Anthropic / Claude — reasoning schema

**Client request schema.** A token budget:

```json
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 1024
  }
}
```

Minimum budget is 1,024 tokens and must be less than `max_tokens`. `budget_tokens` is
deprecated on Claude Opus 4.6 / Sonnet 4.6 and being replaced by an adaptive **effort**
knob.

**Backend response schema.** After LangChain parses it, reasoning sits in
`message.additional_kwargs["reasoning"]` as a dict:

```json
{ "summary": [ ... ], "encrypted_content": "..." }
```

(Claude 4 returns *summarized* thinking; you are billed for the full thinking tokens.)

**What GraphServe does today.** Normalizes the `reasoning` dict into OpenAI reasoning
output (`type: "reasoning"` items / `response.reasoning_text.delta`), in
`src/graphserve/translate.py`. It does not translate `reasoning_effort` → `budget_tokens`
on the request side.

## References

- [Anthropic — Extended thinking](https://docs.claude.com/en/docs/build-with-claude/extended-thinking) (`thinking.budget_tokens`, adaptive effort, summarized output)
