# Problem 2 · LobeChat — only speaks the OpenAI API

**Problem.** LobeChat supports any OpenAI-compatible host, added via env vars or the
*Add Custom Provider* UI. The form's request format (SDK type) is literally `openai`,
plus a proxy/base URL and API key. Whether the base URL needs a trailing `/v1` depends
on the provider.

**What GraphServe does today.** Presents graphs as OpenAI models under a base URL that
serves `GET /v1/models` and `POST /v1/chat/completions`.

## References

- [LobeChat — Model Service Providers](https://lobehub.com/docs/self-hosting/environment-variables/model-provider) (OpenAI-compatible hosts, *Add Custom Provider*, SDK type `openai`)
