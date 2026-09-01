---
name: majordomo-llm
description: |
  How to use the majordomo-llm Python library to make LLM calls in code.
  Load this skill when the user says "use majordomo-llm", "call an LLM with majordomo",
  or when building Python code that needs LLM text, JSON, or structured responses.
  Also load when the user asks about cost tracking, provider failover/cascade,
  or logging LLM requests to a database.
  Do NOT load for Go projects or non-LLM tasks. For routing calls through a Majordomo
  gateway (cost tracking, metadata, agent-run waterfalls), use the majordomo-gateway
  skill — though the "Custom base URL" section below shows the basic hookup.
allowed-tools: Read, Write, Bash
---

## Installation

```bash
# Basic (text + JSON responses)
uv add majordomo-llm

# With request logging to a database
uv add "majordomo-llm[logging]"
```

## Getting an LLM Instance

Use the factory — never instantiate providers directly:

```python
from majordomo_llm import get_llm_instance

llm = get_llm_instance("anthropic", "claude-sonnet-4-6")
llm = get_llm_instance("openai", "gpt-5-mini")
llm = get_llm_instance("gemini", "gemini-2.5-flash")
llm = get_llm_instance("deepseek", "deepseek-v4-pro")
llm = get_llm_instance("cohere", "command-a-03-2025")

# Custom base URL (e.g. route through a Majordomo gateway)
llm = get_llm_instance(
    "anthropic", "claude-sonnet-4-6",
    base_url="http://localhost:7680",
    default_headers={"X-Majordomo-Key": "mdm_sk_your_key_here"},
)
```

To route through a gateway for centralized cost tracking, metadata attribution, and
agent-run observability, see the **majordomo-gateway** skill.

API keys come from environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `CO_API_KEY`. Pass `api_key=` to override.

## Supported Providers and Models

```python
from majordomo_llm import get_supported_providers, get_supported_models

get_supported_providers()          # openai, anthropic, gemini, deepseek, cohere, bedrock,
                                   # fireworks, together, baseten, nebius, deepinfra,
                                   # moonshot, novita, ...
get_supported_models("anthropic")  # ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-sonnet-4-6", ...]
get_supported_models("openai")     # ["gpt-5", "gpt-5-mini", "gpt-4.1", "o3", "o4-mini", ...]
```

## Text Response

```python
response = await llm.get_response(
    user_prompt="Summarize this article: ...",
    system_prompt="You are a concise summarizer.",  # optional
    temperature=0.3,      # optional — unset by default, not sent unless passed
    top_p=1.0,            # optional — unset by default, not sent unless passed
    max_tokens=32000,     # optional — overrides the model's configured cap
)

response.content        # str — the LLM's reply
response.input_tokens   # int
response.output_tokens  # int
response.cached_tokens  # int (Anthropic prompt caching)
response.input_cost     # float USD
response.output_cost    # float USD
response.total_cost     # float USD
response.response_time  # float seconds
response.deprecation_warning  # str | None — set if model was auto-replaced
response.stop_reason    # str | None — "end_turn", "tool_use", "max_tokens", ...
```

## Image Understanding

Anthropic, OpenAI, and Gemini accept in-memory images on the existing response APIs:

```python
from pathlib import Path
from majordomo_llm import ImageInput, get_llm_instance

llm = get_llm_instance("anthropic", "claude-sonnet-5")
response = await llm.get_response(
    "Describe this image.",
    images=(ImageInput(Path("photo.jpg").read_bytes(), "image/jpeg"),),
)
```

`images=` also works with streaming, JSON, raw JSON Schema, and Pydantic structured
responses. Pass bytes with one of `image/jpeg`, `image/png`, `image/gif`, or
`image/webp`; majordomo-llm does not fetch URLs.

## Image Generation and Editing

Use the image factory rather than `get_llm_instance()`:

```python
from pathlib import Path
from majordomo_llm import ImageInput, get_image_instance

model = get_image_instance("openai", "gpt-image-2")
response = await model.generate("A watercolor lighthouse", output_format="png")
Path("lighthouse.png").write_bytes(response.images[0].data)

edited = await model.edit(
    "Make it moonlit",
    images=(ImageInput(Path("source.png").read_bytes(), "image/png"),),
)
```

Use `get_supported_image_providers()` and `get_supported_image_models(provider)`
for discovery. `ImageResponse` reports decoded image bytes, modality-specific
usage, costs, latency, provider, and model.

For ordered failover, use `ImageCascade` with the same provider/model tuple pattern
as `LLMCascade`:

```python
from majordomo_llm import ImageCascade

model = ImageCascade([
    ("openai", "gpt-image-2"),
    ("gemini", "gemini-3.1-flash-image"),
])
response = await model.generate("A watercolor lighthouse")
```

Wrap any image model or cascade with `LoggingImageModel` to use the existing
database and storage adapters. Image logs retain hashes and metadata, never raw
reference or generated bytes.

```python
from majordomo_llm.logging import LoggingImageModel

logged = LoggingImageModel(model, database, storage)
response = await logged.generate("A watercolor lighthouse")
await logged.close()
```

Use `ImageHookPipeline` for typed generation policy and validation. Built-ins are
`ImagePromptRegexHook`, `ImageRequestLimitsHook`, and `ImageIntegrityHook`:

```python
from majordomo_llm import ImageHookPipeline, ImageIntegrityHook, ImageRequestLimitsHook

pipeline = ImageHookPipeline([
    ImageRequestLimitsHook("limits", max_count=1),
    ImageIntegrityHook("integrity"),
])
model = get_image_instance(
    "openai", "gpt-image-2", hook_pipeline=pipeline
)
```

Invalid inputs block before the provider call. `ImageIntegrityHook` rejects corrupt
outputs and advances an `ImageCascade`; on a direct model the same verdict raises
`ImageHookRetryRequested`. Pass `caller_metadata=` to `generate()` or `edit()` for
tenant- or workflow-aware custom hooks.

### Output cap (`max_tokens`)

Anthropic and Bedrock require an output cap on every request; the other providers
send none and inherit the model default. The cap resolves per-request → model config
(`llm_config.yaml`) → library default (**16000** non-streaming, **64000** streaming).

A response cut off at the cap raises `ResponseTruncatedError` rather than returning
truncated content. Raise the ceiling instead of catching it — and remember that with
thinking on, thinking and answer share this budget.

```python
from majordomo_llm import ResponseTruncatedError

try:
    response = await llm.get_response(prompt)
except ResponseTruncatedError as e:
    print(f"hit {e.max_tokens} after {e.output_tokens} tokens")
    print(e.partial_content)   # whatever arrived before the cut
```

## Streaming Response

```python
stream = await llm.get_response_stream(
    user_prompt="Write a long story about...",
    system_prompt="You are a creative writer.",
)

async for chunk in stream:          # chunk: str
    print(chunk, end="", flush=True)

# Usage is available after the stream is fully consumed:
stream.usage.total_cost             # float USD

# Alternatively, collect the full response at once:
response = await stream.collect()   # returns LLMResponse
```

## JSON Response

Automatically strips markdown code fences and parses the response as JSON:

```python
response = await llm.get_json_response(
    user_prompt='Extract: {"name": "John", "age": 30}',
    system_prompt="Return valid JSON only.",
)

response.content   # dict — the parsed JSON
```

## Structured Response (Pydantic)

Returns a validated Pydantic model instance. Uses provider-native structured output
(Anthropic: tool calling; others: response schema or prompt injection):

```python
from typing import Literal
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    summary: str
    sentiment: Literal["positive", "negative", "neutral"]

response = await llm.get_structured_json_response(
    response_model=Article,
    user_prompt="Analyze this article: ...",
    system_prompt="You are a content analyst.",
)

response.content           # Article — validated Pydantic instance
response.content.title     # str
response.content.sentiment # str
response.total_cost        # float USD (usage on structured responses too)
```

## Cascade (Automatic Failover)

Tries providers in order; catches `ProviderError` and falls back to the next:

```python
from majordomo_llm import LLMCascade

cascade = LLMCascade([
    ("anthropic", "claude-sonnet-4-6"),   # primary
    ("openai", "gpt-5-mini"),             # fallback
    ("gemini", "gemini-2.5-flash"),       # last resort
])

response = await cascade.get_response("Hello!")  # same interface as LLM
```

## Optimal Routing (Majordomo Gateway)

The `majordomo` provider does not name a backend. You name a canonical open-weight
model and Majordomo Steward selects the optimal backend (Fireworks, Together, …) at
request time. This is **server-side** selection — distinct from `LLMCascade`, which is
client-side failover on error. They compose.

Requires routing through the gateway (`base_url`) and `MAJORDOMO_API_KEY` (required;
auto-injected as the `X-Majordomo-Key` header):

```python
import os
from majordomo_llm import get_llm_instance

llm = get_llm_instance(
    "majordomo", "glm-5.2",
    base_url=os.environ["MAJORDOMO_GATEWAY_URL"],
)

response = await llm.get_response("Hello!")   # text, JSON, structured, streaming all work
response.routed_provider   # "fireworks" — the backend the gateway actually chose
response.routed_model      # "accounts/fireworks/models/glm-5p2" — its native model id
response.total_cost        # priced from the routed backend's rates in llm_config.yaml
```

Canonical models (`get_supported_models("majordomo")`): `deepseek-v4-pro`, `kimi-k2.6`,
`kimi-k3`, `glm-5.1`, `glm-5.2`, `inkling`.

Because the backend is only known after the call, cost is resolved from the gateway's
`X-Majordomo-Routed-Provider` / `X-Majordomo-Routed-Model` response headers, not a fixed
rate. If the routed pair isn't configured in `llm_config.yaml`, `routed_provider` /
`routed_model` still populate but cost degrades to `0.0` with a warning.

## Named Aliases

Aliases are pre-configured in `llm_config.yaml` and can also be registered at runtime:

```python
from majordomo_llm import get_llm_by_alias, register_alias

llm = get_llm_by_alias("fast")        # single-model alias → LLM instance
llm = get_llm_by_alias("resilient")   # cascade alias → LLMCascade instance

# Register a runtime alias
register_alias("my-fast", ("anthropic", "claude-haiku-4-5-20251001"))
register_alias("my-cascade", [
    ("anthropic", "claude-sonnet-4-6"),
    ("openai", "gpt-5-mini"),
])
```

## Request Logging (optional, requires `[logging]` extra)

`LoggingLLM` wraps any LLM. Logging is fire-and-forget — never blocks the response.

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM
from majordomo_llm.logging.adapters import PostgresAdapter, S3Adapter

llm = get_llm_instance("anthropic", "claude-sonnet-4-6")

db = await PostgresAdapter.create(
    host="localhost", port=5432,
    database="majordomo", user="user", password="pass",
)
storage = await S3Adapter.create(bucket="my-llm-logs")  # optional

logged_llm = LoggingLLM(llm, db, storage)

response = await logged_llm.get_response("Hello!")   # same interface as LLM

await logged_llm.close()   # flush pending log tasks + close connections
```

Other adapters: `MySQLAdapter`, `SqliteAdapter`, `FileStorageAdapter`.

## Error Handling

```python
from majordomo_llm.exceptions import (
    MajordomoError,       # base — catch-all for library errors
    ConfigurationError,   # missing/invalid API key or unknown provider/model
    ProviderError,        # upstream API failure; has .provider and .original_error
    ResponseParsingError, # JSON parse failure; has .raw_content
    ResponseTruncatedError,  # hit the max_tokens cap; has .max_tokens,
                             # .output_tokens, .partial_content
)

try:
    response = await llm.get_response("Hello")
except ProviderError as e:
    print(f"Provider {e.provider} failed: {e.original_error}")
except ConfigurationError as e:
    print(f"Config error: {e}")
```

All public `get_response` and `get_json_response` methods automatically retry up to 3 times
with exponential backoff on transient failures before raising `ProviderError`.

`ResponseTruncatedError` is deliberately outside that policy: it is not retried (the
same budget would hit the same ceiling) and does not trigger `LLMCascade` failover
(the next provider would truncate identically). Fix it by raising `max_tokens`.

## Complete Example

```python
import asyncio
from pydantic import BaseModel
from majordomo_llm import get_llm_instance
from majordomo_llm.exceptions import ProviderError

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    reasoning: str

async def analyze(text: str) -> SentimentResult:
    llm = get_llm_instance("anthropic", "claude-sonnet-4-6")
    response = await llm.get_structured_json_response(
        response_model=SentimentResult,
        user_prompt=f"Analyze the sentiment of: {text}",
    )
    print(f"Cost: ${response.total_cost:.6f}")
    return response.content

result = asyncio.run(analyze("I love this product!"))
print(result.sentiment)  # "positive"
```
