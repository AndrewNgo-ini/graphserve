# Problem 2 · Jan — only speaks the OpenAI API

**Problem.** Jan connects to remote models as OpenAI-compatible providers (base URL +
API key), and itself ships a built-in OpenAI-compatible server at `localhost:1337` — on
the premise that this makes it "a drop-in replacement for any tool that integrates with
the OpenAI API." Either way the contract is the OpenAI surface.

**What GraphServe does today.** Serves `GET /v1/models` and `POST /v1/chat/completions`,
so a graph can be added as a remote OpenAI-compatible provider in Jan.

## References

- [Jan (GitHub)](https://github.com/menloresearch/jan) (built-in OpenAI-compatible server, drop-in replacement)
