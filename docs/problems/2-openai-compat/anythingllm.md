# Problem 2 · AnythingLLM — only speaks the OpenAI API

**Problem.** AnythingLLM ships a *Generic OpenAI* provider — "an easy way to interact
with any LLM provider that is OpenAI-compatible in both API functionality and inference
response." You configure a **Base URL**, API key, chat model name, context window, and
max tokens.

**What GraphServe does today.** Serves the OpenAI-compatible surface the Generic OpenAI
provider points at: `GET /v1/models` and `POST /v1/chat/completions`.

## References

- [AnythingLLM — OpenAI (Generic) LLM](https://docs.anythingllm.com/setup/llm-configuration/cloud/openai-generic) (Generic OpenAI provider, custom Base URL)
