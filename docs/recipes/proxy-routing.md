# Proxy Routing & Custom Headers

Route LLM requests through a gateway or proxy and attach custom HTTP headers.

## Route Through a Gateway

Point any provider at a custom base URL:

```python
from majordomo_llm import get_llm_instance

llm = get_llm_instance(
    "anthropic", "claude-sonnet-5",
    base_url="https://gateway.example.com",
    default_headers={"X-Majordomo-Key": "mdm_key_here"},
)

response = await llm.get_response("Hello!")
```

The request goes to `gateway.example.com` instead of `api.anthropic.com`, with the `X-Majordomo-Key` header attached.

## Optimal Routing (Let the Gateway Pick the Backend)

The `majordomo` provider is a step beyond pointing a concrete provider at a gateway. Instead of naming a backend, you name a **canonical open-weight model** and let Majordomo Steward select the optimal backend (Fireworks, Together, …) for that model at request time.

```python
import os
from majordomo_llm import get_llm_instance

llm = get_llm_instance(
    "majordomo", "glm-5.2",
    base_url=os.environ["MAJORDOMO_GATEWAY_URL"],
)

response = await llm.get_response("Hello!")

response.routed_provider   # e.g. "fireworks" — the backend the gateway chose
response.routed_model      # e.g. "accounts/fireworks/models/glm-5p2"
response.total_cost        # priced from the routed backend's published rates
```

Canonical models: `deepseek-v4-pro`, `kimi-k2.6`, `kimi-k3`, `glm-5.1`, `glm-5.2`, `inkling` (see `get_supported_models("majordomo")`).

How it differs from the sections below:

- **`base_url` is required** — this provider only operates behind the gateway.
- **`MAJORDOMO_API_KEY` is required** and auto-injected as the `X-Majordomo-Key` header; the gateway injects the backend provider's own key. The canonical model is also sent as the `x-majordomo-model` header so the gateway can route on it.
- **Cost is resolved after the call**, not from a fixed config entry. The gateway returns `X-Majordomo-Routed-Provider` / `X-Majordomo-Routed-Model` response headers, and the usage is priced against that pair's rates in `llm_config.yaml` (e.g. GLM-5.2's cached read is 0.14 on Fireworks vs 0.26 on Together). An unconfigured routed pair degrades to `0.0` cost with a warning — usage counts still stand.

This is **server-side** provider selection, distinct from [`LLMCascade`](cascade.md) (client-side failover on error). The two compose: a cascade entry can itself be `("majordomo", "glm-5.2")`.

## Per-Request Headers

Add headers to individual calls with `extra_headers`. These are merged with `default_headers`, with per-request values winning on conflict:

```python
llm = get_llm_instance(
    "openai", "gpt-4.1",
    base_url="https://gateway.example.com",
    default_headers={
        "X-Majordomo-Key": "mdm_key_here",
        "X-Majordomo-Feature": "search",
    },
)

# This request sends all three headers
response = await llm.get_response(
    "Find recent news about AI",
    extra_headers={"X-Majordomo-Request-Id": "req_abc123"},
)
```

## Override a Default Header

Per-request headers take precedence over instance headers with the same key:

```python
llm = get_llm_instance(
    "anthropic", "claude-sonnet-5",
    base_url="https://gateway.example.com",
    default_headers={
        "X-Majordomo-Key": "mdm_key_here",
        "X-Majordomo-Feature": "search",
    },
)

# Override X-Majordomo-Feature for this one request
response = await llm.get_response(
    "Translate this to Spanish",
    extra_headers={"X-Majordomo-Feature": "translation"},
)
```

## Cascade Through a Gateway

Route all cascade providers through the same gateway:

```python
from majordomo_llm import LLMCascade

cascade = LLMCascade(
    [
        ("anthropic", "claude-sonnet-5"),
        ("openai", "gpt-4.1"),
        ("gemini", "gemini-2.5-flash"),
    ],
    base_url="https://gateway.example.com",
    default_headers={"X-Majordomo-Key": "mdm_key_here"},
)

# All three providers route through the gateway
response = await cascade.get_response(
    "Hello!",
    extra_headers={"X-Majordomo-Request-Id": "req_abc123"},
)
```

## With Logging

`LoggingLLM` passes `extra_headers` through to the wrapped LLM:

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

llm = get_llm_instance(
    "anthropic", "claude-sonnet-5",
    base_url="https://gateway.example.com",
    default_headers={"X-Majordomo-Key": "mdm_key_here"},
)

db = await SqliteAdapter.create("llm_logs.db")
storage = await FileStorageAdapter.create("./request_logs")
logged_llm = LoggingLLM(llm, db, storage)

# extra_headers flows through the logging wrapper to the provider
response = await logged_llm.get_response(
    "Hello!",
    extra_headers={"X-Majordomo-Request-Id": "req_abc123"},
)
```

## With Streaming

`extra_headers` works with streaming responses:

```python
stream = await llm.get_response_stream(
    "Explain quantum computing",
    extra_headers={"X-Majordomo-Request-Id": "req_stream_456"},
)
async for chunk in stream:
    print(chunk, end="", flush=True)
```

## With Structured Outputs

`extra_headers` works with structured output responses:

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]

response = await llm.get_structured_json_response(
    response_model=Summary,
    user_prompt="Summarize the benefits of async programming",
    extra_headers={"X-Majordomo-Request-Id": "req_struct_789"},
)
```

## Notes

- `base_url` and `default_headers` are optional on both `get_llm_instance()` and `LLMCascade`. When omitted, requests go directly to the provider.
- `extra_headers` is optional on every API method (`get_response`, `get_response_stream`, `get_json_response`, `get_structured_json_response`, `get_json_schema_response`). When omitted, only `default_headers` are sent.
- For DeepSeek, a custom `base_url` overrides the default `https://api.deepseek.com` endpoint.
- All providers are supported. The header merging logic is handled internally per SDK.
