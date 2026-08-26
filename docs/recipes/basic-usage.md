# Basic Usage

Get responses from any LLM provider with a unified async interface.

## Simple Request

```python
import asyncio
from majordomo_llm import get_llm_instance

async def main():
    llm = get_llm_instance("openai", "gpt-4.1")
    resp = await llm.get_response(
        user_prompt="What is the capital of France?",
        system_prompt="Answer concisely.",
        temperature=0.3,
    )
    print(resp.content)  # Paris
    print(f"Cost: ${resp.total_cost:.6f}")
    print(f"Tokens: {resp.input_tokens} in / {resp.output_tokens} out")

asyncio.run(main())
```

## Switching Providers

Use the same interface across all supported providers:

```python
# OpenAI
llm = get_llm_instance("openai", "gpt-4.1")

# Anthropic
llm = get_llm_instance("anthropic", "claude-sonnet-5")

# Google Gemini
llm = get_llm_instance("gemini", "gemini-2.5-flash")

# DeepSeek
llm = get_llm_instance("deepseek", "deepseek-chat")

# Cohere
llm = get_llm_instance("cohere", "command-r-plus")
```

## Response Object

Every response includes usage metrics:

```python
resp = await llm.get_response("Hello!")

resp.content        # The response text
resp.input_tokens   # Tokens in the prompt
resp.output_tokens  # Tokens in the response
resp.total_cost     # Cost in USD
resp.response_time  # Time in seconds
```

## Streaming

Stream responses in real time:

```python
stream = await llm.get_response_stream(
    user_prompt="What is the capital of France?",
)
async for chunk in stream:
    print(chunk, end="", flush=True)

print(f"\nCost: ${stream.usage.total_cost:.6f}")
```

Or collect the full response:

```python
stream = await llm.get_response_stream("Summarize this text...")
response = await stream.collect()  # Returns an LLMResponse
print(response.content)
```

See the [Streaming recipe](streaming.md) for more examples.

## JSON Responses

Get raw JSON without Pydantic validation:

```python
resp = await llm.get_json_response(
    user_prompt="List 3 countries as JSON with name and capital fields",
)
print(resp.content)  # dict: {"countries": [...]}
```

## Custom API Key

Pass an API key directly instead of using environment variables:

```python
llm = get_llm_instance("openai", "gpt-4.1", api_key="sk-...")
```

If `api_key` is not provided, the provider falls back to its respective environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

## Custom Base URL & Headers

Route requests through a proxy or gateway:

```python
llm = get_llm_instance(
    "anthropic", "claude-sonnet-5",
    api_key="sk-ant-...",
    base_url="https://gateway.example.com",
    default_headers={"X-Majordomo-Key": "mdm_key_here"},
)

resp = await llm.get_response(
    "Hello!",
    extra_headers={"X-Request-Id": "req_123"},
)
```

`default_headers` are sent on every request. `extra_headers` are per-call and override defaults on conflict. See the [Proxy Routing recipe](proxy-routing.md) for more examples.

## With Logging

Track all requests for analytics:

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

llm = get_llm_instance("anthropic", "claude-sonnet-5")

db = await SqliteAdapter.create("llm_logs.db")
storage = await FileStorageAdapter.create("./request_logs")
logged_llm = LoggingLLM(llm, db, storage)

resp = await logged_llm.get_response("Hello!")

await logged_llm.flush()  # Ensure logs are written
await db.close()
await storage.close()
```

Notes

- Set API keys via environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `CO_API_KEY`) or pass `api_key` directly to `get_llm_instance()`.
- Model costs are loaded from `llm_config.yaml`; add new models there.
- All methods are async; use `asyncio.run()` or an async context.
- See the [Cascade recipe](cascade.md) for automatic provider failover.
