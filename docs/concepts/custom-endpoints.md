# Custom Endpoints & Headers

## Beyond Direct Provider Access

By default, majordomo-llm sends requests directly to each provider's API (e.g., `api.openai.com`, `api.anthropic.com`). But many production setups need to route requests through an intermediary:

- **API gateways** like majordomo-gateway for centralized logging, rate limiting, and cost tracking
- **Corporate proxies** that enforce security policies or audit requirements
- **Self-hosted models** served behind an OpenAI-compatible API
- **Regional endpoints** for data residency compliance

Custom endpoints let you point any provider at a different URL and attach custom HTTP headers to every request.

## Two Levels of Headers

majordomo-llm supports headers at two levels:

### Instance-level: `default_headers`

Set once at construction time, sent with every request from that instance. Use these for stable identifiers that don't change between calls.

```python
llm = get_llm_instance(
    "anthropic", "claude-sonnet-5",
    base_url="https://gateway.example.com",
    default_headers={
        "X-Majordomo-Key": "mdm_key_here",
        "X-Majordomo-Feature": "search",
        "X-Majordomo-User-Id": "user_abc",
    },
)
```

### Per-request: `extra_headers`

Set on individual API calls. Merged with instance headers on each request, with per-request values winning on conflict.

```python
response = await llm.get_response(
    "Summarize this document",
    extra_headers={"X-Request-Id": "req_123"},
)
```

In this example, the request is sent with all four headers: the three from `default_headers` plus `X-Request-Id` from `extra_headers`.

## How It Works Per Provider

Each provider SDK handles headers differently. majordomo-llm maps `base_url`, `default_headers`, and `extra_headers` to the right SDK mechanism automatically:

| Provider | Instance headers | Per-request headers |
|----------|-----------------|-------------------|
| OpenAI | `default_headers=` on `AsyncOpenAI` constructor | `extra_headers=` on API calls |
| Anthropic | `default_headers=` on `AsyncAnthropic` constructor | `extra_headers=` on API calls |
| DeepSeek | `default_headers=` on `AsyncOpenAI` constructor | `extra_headers=` on API calls |
| Gemini | `HttpOptions(headers=)` on `genai.Client` | `HttpOptions(headers=)` in `GenerateContentConfig` |
| Cohere | Merged into `request_options` per call | `request_options` with `additional_headers` |

You don't need to know these details. The same `default_headers` and `extra_headers` dictionaries work identically across all providers.

## Composability

Custom endpoints compose naturally with every other feature:

- **Cascade**: Pass `base_url` and `default_headers` to `LLMCascade` to route all providers through the same gateway
- **Logging**: `LoggingLLM` passes `extra_headers` through to the wrapped LLM
- **Streaming**: `extra_headers` works on `get_response_stream()` calls
- **Structured outputs**: `extra_headers` works on `get_structured_json_response()` calls

## Next Steps

See the [Proxy Routing recipe](../recipes/proxy-routing.md) for practical examples including gateway integration, per-request metadata, and cascade routing.
