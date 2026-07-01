# Problem 2 · HuggingFace Chat UI — only speaks the OpenAI API

**Problem.** HuggingFace Chat UI (the code behind HuggingChat) integrates backends via
OpenAI-compatible APIs, configured with `OPENAI_BASE_URL` and the `/models` endpoint.
Any service that speaks the OpenAI protocol (llama.cpp server, Ollama, OpenRouter, etc.)
works by default.

**What GraphServe does today.** Presents graphs as OpenAI models under a base URL that
serves `GET /v1/models` and `POST /v1/chat/completions`.

## References

- [HuggingFace Chat UI — OpenAI provider](https://huggingface.co/docs/chat-ui/en/configuration/models/providers/openai) (`OPENAI_BASE_URL`, `/models`)
