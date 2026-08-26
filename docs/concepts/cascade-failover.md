# Cascade Failover

## The Single-Provider Risk

Relying on a single LLM provider creates availability risks:

- **Outages**: Every provider experiences downtime
- **Rate limits**: High-traffic applications hit quota limits
- **Regional issues**: Some providers have region-specific problems
- **Cost optimization**: Different providers may be cheaper for different use cases

Manual failover logic is error-prone and clutters application code.

## Automatic Multi-Provider Failover

`LLMCascade` wraps multiple provider configurations and automatically fails over when errors occur:

- Tries providers in priority order
- Catches `ProviderError` exceptions and moves to the next provider
- Returns the first successful response
- Supports all LLM methods (`get_response`, `get_json_response`, `get_structured_json_response`)

## Basic Usage

```python
from majordomo_llm import LLMCascade

cascade = LLMCascade([
    ("anthropic", "claude-sonnet-5"),  # Primary: preferred provider
    ("openai", "gpt-4.1"),                        # Secondary: reliable fallback
    ("gemini", "gemini-2.5-flash"),              # Tertiary: cost-effective backup
])

# Automatically tries providers in order until one succeeds
response = await cascade.get_response(
    user_prompt="Summarize this document",
    system_prompt="Be concise.",
)

print(response.content)
```

## Failover Behavior

1. Request sent to Anthropic
2. If Anthropic returns `ProviderError` (rate limit, outage, etc.), try OpenAI
3. If OpenAI fails, try Gemini
4. If all fail, raise the last `ProviderError`

**Important**: Only `ProviderError` triggers failover. Application-level errors (bad prompts, validation failures) are not retried.

## Strategy Tips

- **Diversify vendors**: Don't use multiple models from the same provider
- **Consider capabilities**: Ensure fallback models support your use case (structured outputs, context length, etc.)
- **Monitor which provider serves requests**: Log the provider used for capacity planning

```python
# Each provider has built-in retries (3 attempts with exponential backoff)
# Cascade adds cross-provider resilience on top of per-provider retries
```

## Next Steps

See the [Cascade Failover recipe](../recipes/cascade.md) for more configuration examples.
