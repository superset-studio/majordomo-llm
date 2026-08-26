# Getting Started

## Installation

```bash
uv add majordomo-llm
```

Optional logging extras:

```bash
uv add majordomo-llm[logging]
```

## Configure API Keys

Copy `.env.example` to `.env` and set keys:

```bash
cp .env.example .env
```

## Quickstart

```python
import asyncio
from majordomo_llm import get_llm_instance

async def main():
    llm = get_llm_instance("anthropic", "claude-sonnet-5")
    resp = await llm.get_response("What is the capital of France?")
    print(resp.content)
    print(resp.total_cost)

asyncio.run(main())
```

## Streaming

```python
import asyncio
from majordomo_llm import get_llm_instance

async def main():
    llm = get_llm_instance("anthropic", "claude-sonnet-5")
    stream = await llm.get_response_stream("What is the capital of France?")
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print(f"\nCost: ${stream.usage.total_cost:.6f}")

asyncio.run(main())
```

## Local Docs

```bash
uv add --dev mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions
uv run mkdocs serve
```
