# majordomo-llm

A unified async Python interface for multiple LLM providers with built-in cost tracking, automatic retries, and structured outputs.

## Why majordomo-llm?

Building with LLMs often means dealing with:

- **Different APIs for each provider** — OpenAI, Anthropic, and Gemini all have different client libraries and response formats
- **Hidden costs** — Token usage and spending are hard to track across providers
- **Fragile integrations** — When one provider goes down, your application goes down
- **Inconsistent structured outputs** — Each provider handles JSON schemas differently

majordomo-llm solves these problems with a single, consistent interface that works across all major providers.

## Quick Example

```python
import asyncio
from pydantic import BaseModel
from majordomo_llm import get_llm_instance

class Summary(BaseModel):
    title: str
    key_points: list[str]
    word_count: int

async def main():
    # Works with any provider: openai, anthropic, gemini, deepseek, cohere
    llm = get_llm_instance("anthropic", "claude-sonnet-5")

    response = await llm.get_structured_json_response(
        response_model=Summary,
        user_prompt="Summarize the benefits of async programming in Python",
    )

    print(response.content.title)
    print(response.content.key_points)
    print(f"Cost: ${response.total_cost:.6f}")

asyncio.run(main())
```

## Key Features

### Unified Provider Interface

Write once, run on any provider. Switch between OpenAI, Anthropic, Gemini, DeepSeek, and Cohere with a single line change.

```python
llm = get_llm_instance("openai", "gpt-4.1")
llm = get_llm_instance("anthropic", "claude-sonnet-5")
llm = get_llm_instance("gemini", "gemini-2.5-flash")
```

### Streaming Responses

Stream text token-by-token as it's generated. Usage and cost metrics are available after the stream completes.

```python
stream = await llm.get_response_stream("Explain quantum computing")
async for chunk in stream:
    print(chunk, end="", flush=True)
print(f"\nCost: ${stream.usage.total_cost:.6f}")
```

### Structured Outputs with Pydantic

Get validated, typed Python objects instead of raw JSON. Provider-specific implementation details are handled internally.

```python
response = await llm.get_structured_json_response(
    response_model=MyPydanticModel,
    user_prompt="Extract data from this text...",
)
result: MyPydanticModel = response.content  # Fully typed
```

### Built-in Cost Tracking

Every response includes token counts and calculated costs. No external tracking needed.

```python
print(f"Tokens: {response.input_tokens} in / {response.output_tokens} out")
print(f"Cost: ${response.total_cost:.6f}")
```

### Cascade Failover

Automatically fall back to alternative providers when one fails.

```python
from majordomo_llm import LLMCascade

cascade = LLMCascade([
    ("anthropic", "claude-sonnet-5"),
    ("openai", "gpt-4.1"),
    ("gemini", "gemini-2.5-flash"),
])
response = await cascade.get_response("Hello!")  # Tries each until one succeeds
```

### Custom Endpoints & Proxy Routing

Route requests through any gateway or proxy with custom base URLs and HTTP headers. Pass `api_key` directly or let providers read from environment variables. Headers can be set at instance level (`default_headers`) or per request (`extra_headers`).

```python
llm = get_llm_instance(
    "anthropic", "claude-sonnet-5",
    api_key="sk-ant-...",
    base_url="https://gateway.example.com",
    default_headers={"X-Majordomo-Key": "mdm_key_here"},
)
response = await llm.get_response("Hello!", extra_headers={"X-Request-Id": "req_123"})
```

### Optional Request Logging

Persist all requests for analytics, debugging, and compliance with pluggable database and storage adapters.

```python
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

db = await SqliteAdapter.create("logs.db")
storage = await FileStorageAdapter.create("./request_logs")
logged_llm = LoggingLLM(llm, db, storage)
```

## Supported Providers

| Provider | Recent Models |
|----------|---------------|
| OpenAI | gpt-5.5, gpt-5.4, gpt-5.4-mini, gpt-5, gpt-4.1, o3, o4-mini |
| Anthropic | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5, claude-sonnet-4 |
| Google Gemini | gemini-3.1-pro-preview, gemini-3-flash-preview, gemini-2.5-flash |
| DeepSeek | deepseek-v4-flash, deepseek-v4-pro, deepseek-chat, deepseek-reasoner |
| Cohere | command-a, command-r-plus, command-r |

All providers support structured outputs and streaming. Additional models are available—see `llm_config.yaml` for the complete list with pricing.

## Next Steps

- [Getting Started](getting-started.md) — Installation and quickstart
- [Core Concepts](concepts/index.md) — Understand the key capabilities
- [Recipes](recipes/basic-usage.md) — Practical examples and patterns
- [API Reference](api/base.md) — Detailed API documentation
