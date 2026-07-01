# Problem 3 · OpenRouter — reasoning schema

**Client request schema.** OpenRouter exposes a unified `reasoning` object — either an
effort level or a token budget:

```json
{ "reasoning": { "effort": "low" } }
```

```json
{ "reasoning": { "max_tokens": 2000 } }
```

(It maps these onto whatever the underlying model needs.)

**Backend response schema.** In streaming, OpenRouter emits reasoning as a content block
with `type: "thinking"` where the text lives in the **`text`** key (distinct from
LiteLLM, which uses the `thinking` key):

```json
{ "type": "thinking", "text": "…" }
```

**What GraphServe does today.** Extracts this during streaming — `src/graphserve/translate.py`
checks content blocks for `type == "thinking"` and reads `block.get("text")`, emitting it
as a `response.reasoning_text.delta` event:

```python
# 2. content field with type="thinking" (OpenRouter streaming)
if not reasoning:
    content = getattr(chunk, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                reasoning = block.get("text", "")
                break
```

## References

- [OpenRouter — Reasoning Tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) (unified `reasoning.effort` / `reasoning.max_tokens`, reasoning output)
