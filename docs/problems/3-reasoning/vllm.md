# Problem 3 · vLLM — reasoning schema

**Client request schema.** Thinking is gated by the model's chat template, toggled via
`chat_template_kwargs`. To **disable** thinking on a Qwen3-style model:

```json
{
  "model": "my-agent",
  "input": "…",
  "chat_template_kwargs": { "enable_thinking": false }
}
```

The server must also be launched with `--reasoning-parser <name>` (e.g. `deepseek_r1`,
`qwen3`, `gpt_oss`) or no reasoning is emitted at all.

**Backend response schema.** Reasoning comes back as a flat string in
`message.additional_kwargs["reasoning_content"]` — non-standard, not part of the OpenAI
schema.

**What GraphServe does today.** Passes `chat_template_kwargs` through to the graph run
config (`src/graphserve/routes/responses.py`) and normalizes the `reasoning_content`
string into OpenAI reasoning output (`src/graphserve/translate.py`):

```python
if request.chat_template_kwargs:
    enable = request.chat_template_kwargs.get("enable_thinking", False)
    run_config["configurable"]["extra_body"] = {
        "chat_template_kwargs": request.chat_template_kwargs,
        "reasoning": {"enabled": enable},
    }
```

## References

- [vLLM — Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/) (`--reasoning-parser`, `reasoning_content`, `reasoning_effort` toggle)
