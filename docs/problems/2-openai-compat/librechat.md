# Problem 2 · LibreChat — only speaks the OpenAI API

**Problem.** LibreChat integrates non-native backends through "custom endpoints" for
"OpenAI API-compatible services." You declare each in `librechat.yaml` with a `baseURL`
and `apiKey`:

```yaml
endpoints:
  custom:
    - name: "my-agent"
      apiKey: "${MY_KEY}"
      baseURL: "https://my-host/v1"
      models:
        default: ["my-agent"]
        fetch: true          # pulls the list from /v1/models
```

**What GraphServe does today.** Serves exactly the surface a custom endpoint expects:
`GET /v1/models` and `POST /v1/chat/completions`.

## References

- [LibreChat — Custom Endpoints](https://www.librechat.ai/docs/quick_start/custom_endpoints) (`baseURL` + `apiKey`, `fetch` from `/models`)
