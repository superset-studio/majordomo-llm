# majordomo-llm Examples

This directory contains example applications demonstrating majordomo-llm features.

| Demo | What it covers | Providers |
| --- | --- | --- |
| `demo.py` | Multi-provider sweep with SQLite logging and a cost summary | All in `shared.PROVIDERS` |
| `streaming_demo.py` | `get_response_stream()`, TTFT and decode throughput | All in `shared.PROVIDERS` |
| `structured_response_demo.py` | `get_structured_json_response()` with Pydantic schemas | All in `shared.PROVIDERS` |
| `open_weight_demo.py` | One model, every host that serves it — rate, cost, latency | Open-weight hosts |
| `cascade_demo.py` | `LLMCascade` failover and the alias registry | Open-weight hosts |
| `flagship_demo.py` | Frontier closed models vs open-weight alternatives | Anthropic, OpenAI, Gemini, DeepSeek, Bedrock, Fireworks, Together |
| `prompt_caching_demo.py` | Explicit vs automatic prompt caching, cache-aware costs | Anthropic, Bedrock Mantle, OpenAI, Gemini, DeepSeek, open-weight hosts |
| `web_search_demo.py` | Server-side web search and `tool_use_cost` | Anthropic, OpenAI, Gemini |
| `image_demo.py` | Image understanding plus image generation | Anthropic/OpenAI/Gemini understanding; OpenAI/Gemini generation |
| `routing_demo.py` | `majordomo` gateway provider — server-side optimal routing | Requires Majordomo Steward |

Every demo accepts `--provider <name>` to run a single provider and `--gateway` to
route through Majordomo Steward. Entries whose API key is unset are skipped, so a
single key still produces useful output.

`routing_demo.py` is the exception: the flag that matters there is `--model`,
taking a canonical name, because picking the provider is the gateway's job.

`--provider majordomo` works **only** in `routing_demo.py`. Every other demo
rejects it with a pointer to that script — it names a canonical model and lets
Steward choose the backend, so it cannot run anywhere that pins a concrete
provider.

## Two Different "Majordomo" Things

They share `MAJORDOMO_API_KEY` and are otherwise unrelated. Keeping them straight
matters, because only one of them changes which model actually runs.

| | `--gateway` flag | `majordomo` provider |
| --- | --- | --- |
| Purpose | Usage tracking and cost attribution | Server-side model routing |
| You name | A concrete provider and model | A canonical model only |
| What runs | Exactly what you asked for | Whatever backend Steward selects |
| How you find out | You already knew | `routed_provider` / `routed_model` on the response |
| Where | Any demo, via `gateway_kwargs()` | `routing_demo.py` only |

**Nothing runs the `majordomo` provider by default.** It is absent from
`shared.PROVIDERS`, skipped by `get_all_llm_instances()` (see `_GATEWAY_PROVIDERS`
in `factory.py`), referenced by no alias, and `get_available_providers()` filters it
out even if an entry were added by mistake. Reaching it without a gateway URL raises
`ConfigurationError` rather than failing quietly. Running `routing_demo.py` is the
opt-in — merely having `MAJORDOMO_API_KEY` in your `.env` for usage tracking never
routes anything.

## Shared Provider List

`shared.py` holds the `PROVIDERS` list that drives `demo.py`, `streaming_demo.py`,
and `structured_response_demo.py`. Add a `(provider, model, (env_var, ...))` entry
there and all three demos pick it up.

Note the Nebius entry deliberately runs `zai-org/GLM-5.2` rather than its cheaper
`moonshotai/Kimi-K2.6`: that deployment is configured `supports_structured_outputs:
false` (it accepts `json_schema` without enforcing it), so the structured demo would
correctly raise `StructuredOutputUnsupported` and look like a failure.

## Open-Weight Cross-Host Comparison

The `open_weight_demo.py` script is model-major rather than provider-major: it fixes
the weights and fans out across every host that serves them, which is the question
this library exists to answer — for identical output, what does each host charge and
how fast is it?

It prints three separate rankings, and the distinction matters:

- **Rate comparison** — a fixed reference workload (10,000 input + 1,000 output
  tokens) priced from `llm_config.yaml`. This is the like-for-like number.
- **Observed cost** — what the live call actually cost. Useful, but confounded:
  a chattier host emits more output tokens and looks expensive even at an
  identical rate.
- **Latency** — wall-clock for the call.

### Run

```bash
uv run python examples/open_weight_demo.py
uv run python examples/open_weight_demo.py --provider deepinfra
```

## Cascade & Alias Demo

The `cascade_demo.py` script covers `LLMCascade` and the alias registry:

- Listing every alias, single-model and cascade alike
- Running a cascade alias (`glm-5.2`) end to end
- Forced failover — the primary is given an invalid key, proving the fallback serves
- `register_alias()` for a chain assembled at runtime

Note that `LLMResponse` does not record which cascade member answered (unlike
`routed_provider` / `routed_model` for gateway-routed calls). The cascade logs each
failover to the `majordomo_llm.cascade` logger, so the demo captures those warnings
and derives the answering member from them.

### Run

```bash
uv run python examples/cascade_demo.py
```

## Optimal Routing Demo (Majordomo Gateway)

The `routing_demo.py` script exercises the `majordomo` provider, which names a
canonical open-weight model and lets Majordomo Steward pick the backend per request:

- `routed_provider` / `routed_model` on the response — the backend actually used
- Cost resolved from the routed pair's rates rather than a fixed config entry
- Streaming through the router

Unlike every other demo this one **cannot** run without a gateway: the provider
raises `ConfigurationError` without a `base_url`, by design. It has its own CLI
rather than the shared one — `--model` takes a canonical name, and there is no
`--provider` flag because choosing the provider is precisely what you are delegating.

### Run

```bash
uv run python examples/routing_demo.py
uv run python examples/routing_demo.py --model glm-5.2 --model kimi-k3
uv run python examples/routing_demo.py --gateway-url http://steward.internal:7680
```

## Flagship Comparison

The `flagship_demo.py` script runs frontier closed models (Claude Opus, GPT-5.6,
Gemini, DeepSeek reasoner, Claude via Bedrock Mantle) alongside open-weight
alternatives on Fireworks and Together, including the three DeepSeek-V4-Pro
reasoning-effort profiles. Logs to its own SQLite database so it does not clobber
`demo.py`'s.

### Run

```bash
uv run python examples/flagship_demo.py
```

## Web Search Demo

The `web_search_demo.py` script demonstrates `use_web_search=True` and per-provider
tool wiring, plus `tool_use_cost` accounting for Anthropic and Gemini (OpenAI bills
web search through output tokens).

Only Anthropic, OpenAI, Gemini, and Bedrock implement server-side web search. For
every other provider — including all the open-weight hosts — `use_web_search=True`
is accepted and silently ignored.

### Run

```bash
uv run python examples/web_search_demo.py
```

## Structured Response Demo

The `structured_response_demo.py` script showcases the `get_structured_json_response()` method with various Pydantic models:

- **Sentiment Analysis** - Simple model with Enum field
- **Text Analysis** - Nested models with entity extraction
- **Code Review** - Constrained integer fields and booleans
- **Product Recommendations** - Complex nested lists with validation

### Run

```bash
uv run python examples/structured_response_demo.py
```

### Example Output

```
Demo 1: Sentiment Analysis (with Enum)
Result (SentimentAnalysis):
  Sentiment: positive
  Confidence: 95.00%
  Reasoning: The text expresses enthusiasm with words like "thrilled" and "exceeded expectations"
```

## Streaming Demo

The `streaming_demo.py` script showcases the `get_response_stream()` method:

- **Real-time streaming** - Chunks printed as they arrive with time-to-first-chunk metrics
- **Collect into LLMResponse** - Using `.collect()` to consume the stream and get the full response

### Run

```bash
uv run python examples/streaming_demo.py
```

### Example Output

```
Demo 1: Streaming with real-time output

  [anthropic/claude-3-5-haiku-latest]
  The sky appears blue because of Rayleigh scattering...
  Time to first chunk: 0.34s | Total: 1.12s
  Tokens: 28 in / 45 out | Cost: $0.000159

Demo 2: Collect stream into LLMResponse

  [anthropic/claude-3-5-haiku-latest]
  Content: The three primary colors are red, blue, and yellow.
  Tokens: 22 in / 15 out | Cost: $0.000041
```

## Prompt Caching Demo

The `prompt_caching_demo.py` script showcases both prompt caching flavors side
by side, using a large reused system prompt as the cacheable prefix:

- **Explicit caching** (Anthropic `claude-sonnet-5` / `claude-opus-4-8-fast`,
  Bedrock Mantle) — this library controls the `cache_control` breakpoint, so it
  demonstrates cache **creation** (`cache_creation_tokens` > 0 on the cold
  call), cache **read** (`cached_tokens` > 0 on the warm call), and the
  `use_prompt_caching=False` toggle on `get_llm_instance` that suppresses the
  breakpoint entirely.
- **Automatic caching** (OpenAI `gpt-5.6-luna`, Gemini `gemini-3.6-flash`,
  DeepSeek `deepseek-v4-flash`, plus the open-weight hosts) — the provider caches
  repeated prefixes server-side; there is no creation step or toggle, but
  `cached_tokens` populate on the warm call and bill at the discounted
  `cached_input_cost` rate.

  The open-weight entries all run the **same** model (GLM-5.2) on purpose, because
  the cache-read rate for identical weights varies by host: $0.14/M on Baseten and
  DeepInfra, $0.26/M on Novita, and Nebius publishes no discounted cache tier at
  all — its cached reads bill at the full input rate, which the demo calls out
  explicitly rather than silently printing no savings.

Cache-aware cost accounting is shown for both: `input_cost` folds in cache
read/write tokens (additive for the explicit providers, subset re-pricing for
the automatic ones) using the rates in `llm_config.yaml`.

### Run

```bash
uv run python examples/prompt_caching_demo.py
uv run python examples/prompt_caching_demo.py --provider anthropic
```

### Example Output

```
  [anthropic/claude-sonnet-5]  (explicit cache-control), system prompt ~24200 chars

    Flow A — reuse the same system prompt across two calls:
    Call 1 (cold — expect cache WRITE > 0):
      Tokens: 24 in / 18 out | cache write 3050 / cache read 0
      Cost: $0.011... (input $0.011... + output $0.000...)
    Call 2 (warm — expect cache READ > 0):
      Tokens: 26 in / 41 out | cache write 0 / cache read 3050
      Cost: $0.001... (input $0.000... + output $0.000...)
      Cache hit: 3050 tokens read; prompt-side savings vs. uncached ~= $0.008235

    Flow B — caching OFF (use_prompt_caching=False):
    Call (expect cache write/read == 0):
      Tokens: 3074 in / 18 out | cache write 0 / cache read 0
      Cost: $0.009... (input $0.009... + output $0.000...)

  [openai/gpt-5.6-luna]  (automatic caching), system prompt ~24200 chars

    Flow A — reuse the same system prompt across two calls:
    Call 1 (cold — expect cache read 0):
      Tokens: 3072 in / 20 out | cache write 0 / cache read 0
    Call 2 (warm — expect cache READ > 0):
      Tokens: 3074 in / 44 out | cache write 0 / cache read 2944
      Cache hit: 2944 tokens read; prompt-side savings vs. uncached ~= $0.002650

    (No use_prompt_caching toggle — this provider caches automatically ...)
```

## Demo: Multi-Provider Comparison with Logging

The `demo.py` script showcases:

- Running the same prompts across every provider in `shared.PROVIDERS` (OpenAI, Anthropic,
  Gemini, DeepSeek, Cohere, Bedrock, and the open-weight hosts)
- Automatic request logging to SQLite with API key hash tracking
- Local file storage for request/response bodies
- Cost and performance comparison across providers

### Setup

1. Install dependencies with the logging extras:

```bash
uv sync --all-extras
```

2. Set API keys for the providers you want to test. Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=sk-...
CO_API_KEY=...

# Open-weight inference hosts
FIREWORKS_API_KEY=...
TOGETHER_API_KEY=...
BASETEN_API_KEY=...
NEBIUS_API_KEY=...
DEEPINFRA_API_KEY=...
MOONSHOT_API_KEY=...
NOVITA_API_KEY=...
```

Or export them in your shell. You don't need all keys - the demo will skip providers without keys.

### Run

```bash
uv run python examples/demo.py
```

### Output

The demo will:

1. Run 3 prompts (code, content, customer support) against each available provider
2. Display responses, token counts, costs, and timing for each
3. Print a summary table from the logged metrics
4. Save all request/response bodies as JSON files in `examples/request_logs/`

### Files Created

After running, you'll have:

- `llm_logs.db` - SQLite database with request metrics
- `request_logs/` - Directory with JSON files for each request/response

You can query the SQLite database directly:

```bash
# Basic query
sqlite3 examples/llm_logs.db "SELECT provider, model, total_cost, response_time FROM llm_requests"

# Query with API key tracking (api_key_hash is first 16 chars of SHA256)
sqlite3 examples/llm_logs.db "SELECT provider, model, api_key_hash, api_key_alias FROM llm_requests"
```

## Prompts

The `prompts.json` file contains sample prompts across three domains:

- **code-generation**: Rust ownership explanation
- **content-generation**: Marketing tagline creation
- **customer-support**: Ticket classification

Feel free to modify or add prompts to test different scenarios.
