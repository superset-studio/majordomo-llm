# Streaming

majordomo-llm supports streaming responses from all providers via `get_response_stream()`. Responses arrive token-by-token through an async iterator, with full usage and cost metrics available after the stream completes.

## How It Works

`get_response_stream()` returns an `LLMStreamResponse` — an async-iterable wrapper around the provider's native streaming API. Each provider implements streaming differently (SSE events, chunked responses, etc.), but the interface is identical:

```python
stream = await llm.get_response_stream("Explain photosynthesis")
async for chunk in stream:
    print(chunk, end="")  # Each chunk is a string fragment
```

Internally, `LLMStreamResponse` wraps a provider-specific async generator and a shared `_StreamState`. As chunks flow through, the provider populates token counts. When iteration ends, costs are computed automatically.

## Usage Metrics

Usage data (tokens, costs, timing) is available via `stream.usage` after the stream is fully consumed:

```python
stream = await llm.get_response_stream("Hello")
async for chunk in stream:
    print(chunk, end="")

# Available after iteration completes
print(stream.usage.input_tokens)
print(stream.usage.output_tokens)
print(stream.usage.total_cost)
print(stream.usage.response_time)
```

Accessing `stream.usage` before consuming the stream returns `None`.

## Collecting into LLMResponse

If you want the complete response as a single object (like `get_response()` returns), use `.collect()`:

```python
stream = await llm.get_response_stream("Summarize this article")
response = await stream.collect()

# response is a standard LLMResponse
print(response.content)       # Full text
print(response.total_cost)    # Cost
print(response.input_tokens)  # Token counts
```

This is useful when you want streaming's lower time-to-first-token but still need the full response object.

## Retries

Streaming methods do **not** have automatic retries (unlike `get_response()`). Retries don't compose well with async generators — once a stream has started yielding chunks, it can't be transparently restarted. Handle retries at the application level if needed.

## Cascade Failover

Streaming works with `LLMCascade`. Failover happens on stream **creation** errors only — if the primary provider fails to start the stream, the next provider is tried. Mid-stream errors propagate to the caller.

```python
cascade = LLMCascade([
    ("anthropic", "claude-sonnet-5"),
    ("openai", "gpt-4.1"),
])
stream = await cascade.get_response_stream("Hello!")
```

## Logging

`LoggingLLM` supports streaming transparently. It attaches callbacks to the stream that fire after iteration completes, logging usage metrics and content asynchronously without blocking your application.
