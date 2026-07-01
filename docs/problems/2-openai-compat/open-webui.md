# Problem 2 · Open WebUI — only speaks the OpenAI API

**Problem.** Open WebUI is built around the OpenAI Chat Completions protocol rather than
per-provider modules. Adding a backend = Admin Settings → Connections → OpenAI → a
**base URL** + **API key**. On connect it calls the provider's `/models` endpoint to
verify and list models; a backend without that surface can't be added (though you can
manually allowlist model IDs if only `/models` is missing).

Required surface: `GET /v1/models` and `POST /v1/chat/completions`.

**What GraphServe does today.** Presents each registered graph as an OpenAI model and
serves `GET /v1/models`, `POST /v1/chat/completions`, and `POST /v1/responses`.

## References

- [Open WebUI — OpenAI-Compatible](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/) (built on the OpenAI Chat Completions protocol)
- [Open WebUI — Starting with OpenAI](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai/) (connection verified via `/models`)
