# majordomo-llm

[![PyPI version](https://badge.fury.io/py/majordomo-llm.svg)](https://badge.fury.io/py/majordomo-llm)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-website-blue.svg)](https://superset-studio.github.io/majordomo-llm/)

A unified Python interface for multiple LLM providers with automatic cost tracking, retry logic, and structured output support.

## Features

- **Unified API** - Same interface for OpenAI, Anthropic (Claude), Google Gemini, DeepSeek, Cohere, Amazon Bedrock, and the open-weight inference platforms (Fireworks, Together, Baseten, Nebius, DeepInfra, Moonshot, Novita)
- **Streaming** - Real-time token-by-token output via `get_response_stream()` with async iteration
- **Cost Tracking** - Automatic calculation of input/output token costs per request
- **Structured Outputs** - Native support for Pydantic models and raw JSON Schema dicts
- **Image Understanding** - Send validated JPEG, PNG, GIF, or WebP inputs to Anthropic, OpenAI, and Gemini while keeping the text, JSON, structured, and streaming response APIs
- **Image Generation & Editing** - Generate or edit images through OpenAI and Gemini with decoded byte responses and modality-specific cost tracking
- **Automatic Retries** - Built-in exponential backoff retry logic using tenacity
- **Output Caps** - Per-model `max_tokens` in config plus a per-request override, and a raised `ResponseTruncatedError` when a response is cut off, instead of silently truncated content
- **Automatic Fallback** - Cascade across providers with `LLMCascade` for resilience
- **Optimal Routing** - The `majordomo` provider lets the Majordomo gateway pick the best backend for a canonical open-weight model at request time, with cost resolved from the routed backend
- **Request Logging** - Optional async logging to PostgreSQL/MySQL/SQLite with S3 or local file storage for request/response bodies
- **API Key Tracking** - Log hashed API keys and optional aliases for usage attribution
- **Async First** - Fully async/await compatible for high-performance applications
- **Type Safe** - Complete type annotations and `py.typed` marker for IDE support

## Installation

```bash
pip install majordomo-llm
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add majordomo-llm
```

### Optional: Request Logging

To enable request logging to PostgreSQL, MySQL, or S3:

```bash
pip install majordomo-llm[logging]
```

## Quick Start

### Basic Text Response

```python
import asyncio
from majordomo_llm import get_llm_instance

async def main():
    # Create an LLM instance
    llm = get_llm_instance("anthropic", "claude-sonnet-5")

    # Get a response
    response = await llm.get_response(
        user_prompt="What is the capital of France?",
        system_prompt="You are a helpful geography assistant.",
    )

    print(response.content)
    print(f"Tokens: {response.input_tokens} in, {response.output_tokens} out")
    print(f"Cost: ${response.total_cost:.6f}")

asyncio.run(main())
```

### Image Understanding

```python
from pathlib import Path

from majordomo_llm import ImageInput, get_llm_instance

llm = get_llm_instance("anthropic", "claude-sonnet-5")
response = await llm.get_response(
    "Describe the important details in this image.",
    images=(
        ImageInput(
            data=Path("photo.jpg").read_bytes(),
            media_type="image/jpeg",
        ),
    ),
)
print(response.content)
```

The same `images=` argument works with streaming, JSON, and Pydantic structured
responses. Image bytes are sent directly to the provider; majordomo-llm never
fetches URLs. Request logging records only MIME type, size, and SHA-256—not image
content.

### Image Generation and Editing

```python
from pathlib import Path

from majordomo_llm import ImageInput, get_image_instance

generator = get_image_instance("openai", "gpt-image-2")
generated = await generator.generate(
    "A watercolor lighthouse at dusk",
    aspect_ratio="3:2",
    output_format="png",
)
Path("lighthouse.png").write_bytes(generated.images[0].data)

edited = await generator.edit(
    "Turn the daytime scene into a moonlit scene",
    images=(ImageInput(Path("photo.png").read_bytes(), "image/png"),),
)
Path("edited.jpg").write_bytes(edited.images[0].data)
print(f"Cost: ${edited.total_cost:.6f}")
```

Use `get_supported_image_providers()` and `get_supported_image_models(provider)`
to discover generation models. Image-generation configuration is separate from
text-model discovery because its response and pricing contracts differ.

Use `ImageCascade` for ordered generation or editing failover. Each child model
performs its own retries before the cascade advances. Provider failures, malformed
image responses, and unsupported provider options advance to the next model;
invalid caller input does not.

```python
from majordomo_llm import ImageCascade

generator = ImageCascade([
    ("openai", "gpt-image-2"),
    ("gemini", "gemini-3.1-flash-image"),
])
response = await generator.generate("A watercolor lighthouse")
```

Wrap an image model with `LoggingImageModel` to record metrics asynchronously
through the existing database and storage adapters. Image inputs and generated
outputs are logged as MIME type, byte length, and SHA-256 only; raw bytes are
never copied into logs.

```python
from majordomo_llm.logging import LoggingImageModel

logged = LoggingImageModel(generator, database, storage)
response = await logged.generate("A watercolor lighthouse")
await logged.close()
```

Use a separate typed `ImageHookPipeline` for prompt policy, request limits, and
decoded-image integrity checks. Hooks receive an immutable `ImageHookRequest`
before generation and the typed `ImageResponse` afterward. They can pass, warn,
block, replace the request or response, or explicitly ask an `ImageCascade` to
try its next provider.

```python
from majordomo_llm import (
    ImageHookPipeline,
    ImageIntegrityHook,
    ImagePromptRegexHook,
    ImageRequestLimitsHook,
)

pipeline = ImageHookPipeline([
    ImagePromptRegexHook("secrets", pattern=r"secret", action="block"),
    ImageRequestLimitsHook("limits", max_count=1, max_total_input_bytes=10_000_000),
    ImageIntegrityHook("integrity", max_pixels=20_000_000),
])
generator = get_image_instance(
    "openai", "gpt-image-2", hook_pipeline=pipeline
)
response = await generator.generate(
    "A watercolor lighthouse",
    caller_metadata={"tenant_id": "acme"},
)
```

`ImageIntegrityHook` blocks invalid reference images before spend. If a generated
image is corrupt, mislabeled, or over the pixel limit, it requests the next model
from an `ImageCascade`; on a direct model it raises `ImageHookRetryRequested`.

### JSON Response

```python
response = await llm.get_json_response(
    user_prompt="List the top 3 largest countries by area as JSON",
    system_prompt="Respond with valid JSON only.",
)

# response.content is a parsed Python dict
for country in response.content["countries"]:
    print(country["name"])
```

### Streaming

```python
stream = await llm.get_response_stream(
    user_prompt="Explain quantum computing",
    system_prompt="Be concise.",
)

async for chunk in stream:
    print(chunk, end="", flush=True)

print(f"\nCost: ${stream.usage.total_cost:.6f}")

# Or collect the full response:
stream = await llm.get_response_stream("Summarize this document...")
response = await stream.collect()  # Returns an LLMResponse
print(response.content)
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel

class CountryInfo(BaseModel):
    name: str
    capital: str
    population: int
    area_km2: float

response = await llm.get_structured_json_response(
    response_model=CountryInfo,
    user_prompt="Give me information about Japan",
)

# response.content is a validated CountryInfo instance
country = response.content
print(f"{country.name}: {country.capital}, pop. {country.population:,}")
```

### Structured Output with Raw JSON Schema

```python
import json

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "capital": {"type": "string"},
        "population": {"type": "integer"},
    },
    "required": ["name", "capital", "population"],
}

response = await llm.get_json_schema_response(
    user_prompt="Give me information about Japan",
    response_schema=schema,
    schema_name="CountryInfo",
)

# response.content is canonical JSON: sorted keys, no extra whitespace
country = json.loads(response.content)
print(country["capital"])
```

## Configuration

### Environment Variables

Set API keys for the providers you want to use:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GEMINI_API_KEY="..."

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."

# Cohere
export CO_API_KEY="..."

# Open-weight inference platforms
export FIREWORKS_API_KEY="..."
export TOGETHER_API_KEY="..."
export BASETEN_API_KEY="..."
export NEBIUS_API_KEY="..."
export DEEPINFRA_API_KEY="..."
export MOONSHOT_API_KEY="..."
export NOVITA_API_KEY="..."
```

For local development, copy `.env.example` to `.env` and fill in your keys. Never commit `.env`.

### Available Models

#### OpenAI
- `gpt-5.5`
- `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.4-pro`
- `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- `o3`, `o4-mini`

#### Anthropic
- `claude-opus-5`, `claude-fable-5`, `claude-sonnet-5`
- `claude-opus-4-8` and its effort profiles `claude-opus-4-8-fast`, `claude-opus-4-8-medium`, `claude-opus-4-8-deep`
- `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`
- `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`

The Claude 4 family (`claude-opus-4-1-20250805`, `claude-opus-4-20250514`,
`claude-sonnet-4-20250514`) was retired on 2026-08-25 — the Messages API returns
`404 not_found_error` for all three. They are mapped in `deprecated_models` and
resolve automatically to `claude-opus-5` / `claude-sonnet-5` with a warning.

#### Gemini
- `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`
- `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`

#### DeepSeek
- `deepseek-v4-flash`, `deepseek-v4-pro`
- `deepseek-chat`, `deepseek-reasoner`

#### Cohere
- `command-a-03-2025`, `command-r-plus-08-2024`
- `command-r-08-2024`, `command-r7b-12-2024`

#### Open-Weight Inference Platforms

The same open-weight models are served by several platforms at different prices; pick a
provider directly, or stack them behind an `LLMCascade` for failover.

| Provider | Endpoint | Models |
| --- | --- | --- |
| `baseten` | `https://inference.baseten.co/v1` | `deepseek-ai/DeepSeek-V4-Pro`, `moonshotai/Kimi-K2.6`, `moonshotai/Kimi-K3`, `zai-org/GLM-5.2`, `thinkingmachines/inkling` |
| `nebius` | `https://api.tokenfactory.nebius.com/v1` | `deepseek-ai/DeepSeek-V4-Pro`, `moonshotai/Kimi-K2.6`, `moonshotai/Kimi-K3`, `zai-org/GLM-5.1`, `zai-org/GLM-5.2` |
| `deepinfra` | `https://api.deepinfra.com/v1/openai` | `deepseek-ai/DeepSeek-V4-Pro`, `moonshotai/Kimi-K2.6`, `moonshotai/Kimi-K3`, `zai-org/GLM-5.1`, `zai-org/GLM-5.2`, `thinkingmachines/Inkling` |
| `moonshot` | `https://api.moonshot.ai/v1` | `kimi-k2.6`, `kimi-k3` |
| `novita` | `https://api.novita.ai/openai/v1` | `deepseek/deepseek-v4-pro`, `moonshotai/kimi-k2.6`, `moonshotai/kimi-k3`, `zai-org/glm-5.1`, `zai-org/glm-5.2` |

Notes:

- Model IDs are case-sensitive and passed through verbatim, and the spelling varies by
  platform. Baseten writes Inkling lowercase (`thinkingmachines/inkling`) where Together
  and DeepInfra use `thinkingmachines/Inkling`; Novita uses `deepseek/deepseek-v4-pro` where the
  HF-style platforms use `deepseek-ai/DeepSeek-V4-Pro`.
- Capability flags are per host, not per model, and were measured against the live
  APIs rather than taken from vendor metadata. Some deployments pin their sampling
  parameters (Moonshot's Kimi SKUs require `temperature=1`/`top_p=0.95`; Baseten's
  Kimi-K3 requires `top_p=0.95`) and some reject or ignore strict `json_schema`.
  Those models carry `supports_temperature_top_p: false` or
  `supports_structured_outputs: false` in `llm_config.yaml`.
- Nebius publishes no cached-token rate, so cached reads bill at `input_cost` there
  (as with Fireworks and Together). Its rates come from `GET /v1/models?verbose=true`,
  which is also the quickest way to re-check them.
- Moonshot is the first-party Kimi vendor. The configured rates are the USD
  international platform; mainland-China accounts bill separately in RMB via
  `https://api.moonshot.cn/v1` (reachable by passing it as `base_url`).

#### Majordomo (Optimal Routing)
Canonical open-weight models routed to the optimal backend by the Majordomo gateway (see [Optimal Routing](#optimal-routing-majordomo-gateway)):
- `deepseek-v4-pro`, `kimi-k2.6`, `kimi-k3`
- `glm-5.1`, `glm-5.2`, `inkling`

### Deprecated Model Handling

If you pass a deprecated model to `get_llm_instance()`, it is automatically replaced with the provider-recommended replacement and a warning is logged. The response object includes a `deprecation_warning` field so you can detect this in your application:

```python
llm = get_llm_instance("openai", "gpt-4o")  # deprecated → auto-replaced with gpt-4.1

response = await llm.get_response("Hello!")
if response.deprecation_warning:
    print(response.deprecation_warning)
    # "Model 'gpt-4o' for provider 'openai' is deprecated.
    #  Automatically replaced with 'gpt-4.1'."
```

See the `deprecated_models` section in `llm_config.yaml` for the full mapping.

## API Reference

### Factory Functions

#### `get_llm_instance(provider: str, model: str) -> LLM`

Create an LLM instance for the specified provider and model.

```python
from majordomo_llm import get_llm_instance

llm = get_llm_instance("openai", "gpt-4.1")
```

### LLM Methods

All LLM instances support these async methods:

#### `get_response(user_prompt, system_prompt=None, temperature=None, top_p=None, max_tokens=None) -> LLMResponse`

Get a plain text response.

#### `get_json_response(user_prompt, system_prompt=None, temperature=None, top_p=None, max_tokens=None) -> LLMJSONResponse`

Get a JSON response (automatically parsed).

#### `get_response_stream(user_prompt, system_prompt=None, temperature=None, top_p=None, max_tokens=None) -> LLMStreamResponse`

Get a streaming text response. Yields chunks via async iteration; usage metrics are available after the stream completes.

#### `get_structured_json_response(response_model, user_prompt, system_prompt=None, temperature=None, top_p=None, max_tokens=None) -> LLMStructuredResponse`

Get a response validated against a Pydantic model.

#### `get_json_schema_response(user_prompt, response_schema, system_prompt=None, schema_name="Response", schema_description=None, temperature=None, top_p=None, max_tokens=None) -> LLMResponse`

Get a response validated against a raw JSON Schema dict. `response.content` is canonical JSON.

### Sampling Parameters

`temperature` and `top_p` are optional and unset by default. They are sent only when
you pass them, so each provider applies its own documented default:

```python
await llm.get_response("Hello")                    # neither parameter is sent
await llm.get_response("Hello", temperature=0.2)   # only temperature is sent
```

Models whose deployment rejects these parameters never receive them — that covers
every current OpenAI and Anthropic flagship, plus deployments that pin their values
(Moonshot's Kimi SKUs require `temperature=1` / `top_p=0.95`). Those models set
`supports_temperature_top_p: false` in `llm_config.yaml`. Passing a value to one of
them is silently ignored and the call proceeds without it, rather than failing — a
cascade can legitimately mix models that do and do not accept sampling parameters.

### Output Cap (`max_tokens`)

Anthropic's Messages API and Bedrock's Converse API require an output cap on every
request; the OpenAI-compatible providers and Gemini do not send one and inherit each
model's own default. For the three that need it, the cap resolves in this order:

1. A per-request `max_tokens=` argument
2. The model's `max_tokens` in `llm_config.yaml`
3. The library default — **16000** non-streaming, **64000** streaming

```python
llm = get_llm_instance("anthropic", "claude-sonnet-5")   # config sets 128000
await llm.get_response(prompt)                            # uses 128000
await llm.get_response(prompt, max_tokens=4096)           # uses 4096
```

The non-streaming default is lower because a large cap on a non-streaming call can
exceed the provider SDK's HTTP timeout. Use `get_response_stream()` when you need the
model's full ceiling.

When thinking is enabled, thinking and answer share this budget — a deep-reasoning
profile such as `claude-opus-4-8-deep` needs headroom for both.

#### Truncation

A response cut off at the cap raises `ResponseTruncatedError` rather than returning
partial (or empty) content:

```python
from majordomo_llm import ResponseTruncatedError

try:
    response = await llm.get_response(prompt)
except ResponseTruncatedError as e:
    print(f"hit {e.max_tokens} after {e.output_tokens} tokens")
    print(e.partial_content)   # whatever arrived before the cut
```

The error is not retried — re-sampling would spend the same budget against the same
ceiling — and does not trigger `LLMCascade` failover, since the next provider would
truncate identically. Every response also carries `stop_reason` if you would rather
inspect it than catch.

### Response Objects

All response objects include usage metrics:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` / `dict` / `BaseModel` | The response content |
| `input_tokens` | `int` | Number of input tokens |
| `output_tokens` | `int` | Number of output tokens |
| `cached_tokens` | `int` | Number of cached tokens (if applicable) |
| `input_cost` | `float` | Cost for input tokens (USD) |
| `output_cost` | `float` | Cost for output tokens (USD) |
| `total_cost` | `float` | Total cost (USD) |
| `response_time` | `float` | Response time in seconds |
| `deprecation_warning` | `str \| None` | Warning if a deprecated model was auto-replaced |
| `routed_provider` | `str \| None` | For `majordomo` optimal routing, the backend the gateway selected (`None` otherwise) |
| `routed_model` | `str \| None` | For `majordomo` optimal routing, the routed backend's native model id (`None` otherwise) |
| `stop_reason` | `str \| None` | Why the provider stopped generating (`end_turn`, `tool_use`, `max_tokens`, …); `None` where unreported |

## Advanced Usage

### Automatic Fallback with LLMCascade

Use `LLMCascade` for automatic failover between providers:

```python
from majordomo_llm import LLMCascade

# Providers are tried in order - first is primary, rest are fallbacks
cascade = LLMCascade([
    ("anthropic", "claude-sonnet-5"),  # Primary
    ("openai", "gpt-4.1"),                        # First fallback
    ("gemini", "gemini-2.5-flash"),              # Last resort
])

# If Anthropic fails, automatically tries OpenAI, then Gemini
response = await cascade.get_response("Hello!")
```

All response methods (`get_response`, `get_json_response`, `get_structured_json_response`, `get_response_stream`) support automatic fallback.

### Optimal Routing (Majordomo Gateway)

The `majordomo` provider names a canonical open-weight model and lets the Majordomo gateway select the optimal backend (Fireworks, Together, …) at request time. Unlike `LLMCascade` (client-side failover on error), this is **server-side** provider selection — and the two compose.

It routes through the gateway, so `base_url` is required and `MAJORDOMO_API_KEY` must be set (auto-injected as the `X-Majordomo-Key` header):

```python
import os
from majordomo_llm import get_llm_instance

llm = get_llm_instance(
    "majordomo", "glm-5.2",
    base_url=os.environ["MAJORDOMO_GATEWAY_URL"],
)

response = await llm.get_response("Hello!")

print(response.routed_provider)  # e.g. "fireworks" — the backend the gateway chose
print(response.routed_model)     # e.g. "accounts/fireworks/models/glm-5p2"
print(response.total_cost)       # priced from the routed backend's rates
```

Because the backend is known only after the call, cost is resolved from the gateway's `X-Majordomo-Routed-Provider` / `X-Majordomo-Routed-Model` response headers against that pair's rates in `llm_config.yaml`, rather than a fixed rate. An unconfigured routed pair degrades to `0.0` cost with a warning (token counts still stand). Text, JSON, structured, and streaming calls are all supported.

### Direct Provider Access

You can also instantiate providers directly for more control:

```python
from majordomo_llm import Anthropic

llm = Anthropic(
    model="claude-sonnet-5",
    input_cost=3.0,    # per million tokens
    output_cost=15.0,  # per million tokens
)
```

### Web Search (Anthropic)

Enable web search for supported Claude models:

```python
from majordomo_llm.providers.anthropic import Anthropic

llm = Anthropic(
    model="claude-sonnet-4-5-20250929",
    input_cost=3.0,
    output_cost=15.0,
    use_web_search=True,
)
```

### Request Logging

Log all LLM requests asynchronously to a database with optional storage for request/response bodies. Logging is fire-and-forget and does not block your main request flow.

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM, PostgresAdapter, S3Adapter

async def main():
    # Create your LLM instance
    llm = get_llm_instance("anthropic", "claude-sonnet-5")

    # Set up database adapter (PostgreSQL, MySQL, or SQLite)
    db = await PostgresAdapter.create(
        host="localhost",
        port=5432,
        database="llm_logs",
        user="postgres",
        password="password",
    )

    # Optional: Set up S3 for storing request/response bodies
    storage = await S3Adapter.create(
        bucket="my-llm-logs",
        prefix="requests",  # optional, defaults to "llm-logs"
    )

    # Wrap your LLM with logging
    logged_llm = LoggingLLM(llm, db, storage)

    # Use as normal - all requests are logged automatically
    response = await logged_llm.get_response("Hello!")

    # Don't forget to close connections when done
    await logged_llm.close()
```

#### Local Development Setup

For local development and testing, use SQLite and local file storage:

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

async def main():
    llm = get_llm_instance("anthropic", "claude-sonnet-5")

    # SQLite for metrics (auto-creates database and table)
    db = await SqliteAdapter.create("llm_logs.db")

    # Local file storage for request/response bodies
    storage = await FileStorageAdapter.create("./request_logs")

    logged_llm = LoggingLLM(llm, db, storage)
    response = await logged_llm.get_response("Hello!")

    await logged_llm.close()
```

#### API Key Tracking

Track which API key was used for each request with optional human-readable aliases:

```python
from majordomo_llm.providers.anthropic import Anthropic

# Create LLM with API key alias for attribution
llm = Anthropic(
    model="claude-sonnet-5",
    input_cost=3.0,
    output_cost=15.0,
    api_key_alias="production-team-1",  # Optional human-readable name
)

# The LoggingLLM wrapper automatically logs:
# - api_key_hash: First 16 chars of SHA256 hash (safe for logging)
# - api_key_alias: Your custom name (e.g., "production-team-1")
```

This is useful for:
- Tracking costs per team or application
- Debugging which key was used for specific requests
- Auditing API key usage patterns

#### Database Schema

Create the logging table using the included schema:

```sql
CREATE TABLE IF NOT EXISTS llm_requests (
    request_id VARCHAR(36) PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    response_time FLOAT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    input_cost DECIMAL(10, 8),
    output_cost DECIMAL(10, 8),
    total_cost DECIMAL(10, 8),
    s3_request_key VARCHAR(255),
    s3_response_key VARCHAR(255),
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    api_key_hash VARCHAR(16),
    api_key_alias VARCHAR(100)
);
```

#### Available Adapters

**Database Adapters:**
- **PostgresAdapter** - PostgreSQL via asyncpg
- **MySQLAdapter** - MySQL via aiomysql
- **SqliteAdapter** - SQLite via aiosqlite (great for local development)

**Storage Adapters:**
- **S3Adapter** - AWS S3 via aioboto3
- **FileStorageAdapter** - Local filesystem (great for local development)

## Development

### Setup

```bash
git clone https://github.com/superset-studio/majordomo-llm.git
cd majordomo-llm
uv sync --all-extras
```

### Running Tests

```bash
uv run pytest
```

### Type Checking

```bash
uv run mypy src/majordomo_llm
```

### Linting

```bash
uv run ruff check src/majordomo_llm
```

### Documentation

Build and preview the docs locally:

```bash
uv add --dev mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions
uv run mkdocs serve
```

### Pre-commit Hooks & Checks

Enable local checks (using uvx):

```bash
uvx pre-commit install
uvx pre-commit run --all-files
```

Hooks include private-key detection and basic hygiene checks. See `.pre-commit-config.yaml`.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
