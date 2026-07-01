# Problems

The concrete problems GraphServe exists to solve — **one file per provider per
problem**, each with a specific schema/example, what GraphServe does today (with source
pointers), and a **References** section linking the supporting docs. Written so anyone
using this project (configuring a backend, filing an issue, or sending a PR) can find
exactly the provider they're on.

## 1 · LangGraph's streaming protocol ≠ OpenAI SSE

- [LangGraph](1-streaming/langgraph.md)

## 2 · OSS chat UIs only speak the OpenAI API

- [Open WebUI](2-openai-compat/open-webui.md)
- [LibreChat](2-openai-compat/librechat.md)
- [LobeChat](2-openai-compat/lobechat.md)
- [AnythingLLM](2-openai-compat/anythingllm.md)
- [HuggingFace Chat UI](2-openai-compat/hf-chat-ui.md)
- [Jan](2-openai-compat/jan.md)

## 3 · Backends emit reasoning in incompatible shapes

- [vLLM](3-reasoning/vllm.md)
- [Anthropic / Claude](3-reasoning/anthropic.md)
- [OpenAI](3-reasoning/openai.md)
- [LiteLLM](3-reasoning/litellm.md)
- [OpenRouter](3-reasoning/openrouter.md)
- [DeepSeek](3-reasoning/deepseek.md)
