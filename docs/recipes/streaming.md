# Streaming

Stream responses in real time from any provider.

## Real-Time Output

```python
import asyncio
import time
from majordomo_llm import get_llm_instance

async def main():
    llm = get_llm_instance("anthropic", "claude-sonnet-5")

    stream = await llm.get_response_stream(
        user_prompt="Explain why the sky is blue.",
        system_prompt="Be concise. Respond in 2-3 sentences.",
    )

    first_chunk_time = None
    start = time.time()

    async for chunk in stream:
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
        print(chunk, end="", flush=True)

    print(f"\nTime to first chunk: {first_chunk_time:.2f}s")
    print(f"Total: {stream.usage.response_time:.2f}s")
    print(f"Tokens: {stream.usage.input_tokens} in / {stream.usage.output_tokens} out")
    print(f"Cost: ${stream.usage.total_cost:.6f}")

asyncio.run(main())
```

## Collect Full Response

Use `.collect()` when you want streaming's lower latency but need the full response as an `LLMResponse`:

```python
async def main():
    llm = get_llm_instance("openai", "gpt-4.1")

    stream = await llm.get_response_stream(
        user_prompt="What are the three primary colors?",
        system_prompt="Answer in one sentence.",
    )
    response = await stream.collect()

    print(response.content)
    print(f"Cost: ${response.total_cost:.6f}")

asyncio.run(main())
```

## With Cascade Failover

Streaming works with `LLMCascade`. If the primary provider fails to start the stream, it falls back to the next:

```python
from majordomo_llm import LLMCascade

cascade = LLMCascade([
    ("anthropic", "claude-sonnet-5"),
    ("openai", "gpt-4.1"),
    ("gemini", "gemini-2.5-flash"),
])

stream = await cascade.get_response_stream("Hello!")
async for chunk in stream:
    print(chunk, end="")
```

## With Logging

`LoggingLLM` logs streaming requests automatically after the stream completes:

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

llm = get_llm_instance("openai", "gpt-4.1")

db = await SqliteAdapter.create("llm_logs.db")
storage = await FileStorageAdapter.create("./request_logs")
logged_llm = LoggingLLM(llm, db, storage)

stream = await logged_llm.get_response_stream("Hello!")
async for chunk in stream:
    print(chunk, end="")
# Usage is logged automatically after the stream finishes
```

## Notes

- Streaming methods do **not** retry automatically. Handle retries at the application level if needed.
- All providers are supported: OpenAI, Anthropic, Gemini, DeepSeek, and Cohere.
- See the [Streaming concept guide](../concepts/streaming.md) for design details.
