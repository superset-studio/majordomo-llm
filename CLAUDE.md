# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies (includes dev + logging extras)
uv sync --all-extras

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_anthropic.py

# Run a single test
uv run pytest tests/test_anthropic.py::test_anthropic_response

# Type checking
uv run mypy src/majordomo_llm

# Linting
uv run ruff check src/majordomo_llm

# Fix linting issues
uv run ruff check --fix src/majordomo_llm
```

## Architecture

Unified async interface for LLM providers (OpenAI, Anthropic, Gemini, DeepSeek, Cohere, Amazon Bedrock, Amazon Bedrock Mantle, Fireworks, Together, Baseten, Nebius, DeepInfra, Moonshot, Novita) with cost tracking and structured output support.

### Core Components

- **`base.py`**: Abstract `LLM` base class defining the interface (`get_response`, `get_json_response`, `get_structured_json_response`). Response dataclasses (`LLMResponse`, `LLMJSONResponse`, `LLMStructuredResponse`) inherit from `Usage` for cost/token tracking.

- **`factory.py`**: `get_llm_instance(provider, model)` factory function. Loads model configs from `llm_config.yaml` (costs per million tokens, feature flags like `supports_temperature_top_p`).

- **`llm_config.yaml`**: Model configuration (costs, feature flags). Add new models here; the factory will pick them up automatically. `max_tokens` is the per-model output cap, and is only read by the three providers whose API requires one (`anthropic`, `bedrock`, `bedrock_mantle`); everywhere else the model's own default applies and the key is not forwarded. Verify a new value with `scripts/check_max_tokens.py`, which reads each vendor's real ceiling out of its rejection of an over-large cap.

- **`cascade.py`**: `LLMCascade` wraps multiple providers for automatic failover. Tries providers in order; catches `ProviderError` and falls back to next.

- **`providers/`**: Provider implementations extending `LLM`. Each provider:
  - Uses its native async client (e.g., `anthropic.AsyncAnthropic`)
  - Implements `get_response()` with automatic retries via tenacity
  - Overrides `_get_structured_response()` for provider-specific structured output (Anthropic uses tool calling, others use response schemas)

- **`providers/_openai_compatible.py`**: `OpenAICompatibleLLM` base class holding the shared chat-completions machinery (text, streaming, JSON schema) for providers that speak the OpenAI wire protocol. Subclasses (`Baseten`, `Nebius`, `DeepInfra`, `Moonshot`, `Novita`) set four class attributes — `PROVIDER_NAME`, `DISPLAY_NAME`, `DEFAULT_BASE_URL`, `API_KEY_ENV` — and override nothing. A model may set `supports_structured_outputs: false` in `llm_config.yaml` when its deployment accepts `json_schema` without enforcing it; the base then raises `StructuredOutputUnsupported` instead of returning prose intermittently. `Fireworks` and `Together` predate it and still carry their own copies.

- **`logging/`**: Optional request logging subsystem (requires `[logging]` extra):
  - `LoggingLLM`: Wrapper that logs requests fire-and-forget (non-blocking)
  - `interfaces.py`: `DatabaseAdapter` and `StorageAdapter` ABCs
  - `adapters/`: Database (`PostgresAdapter`, `MySQLAdapter`, `SqliteAdapter`) and storage (`S3Adapter`, `FileStorageAdapter`) implementations

- **`examples/`**: Demo app showing multi-provider usage with logging. Run with `uv run python examples/demo.py`

- **`exceptions.py`**: Exception hierarchy: `MajordomoError` (base) → `ConfigurationError`, `ProviderError`, `ResponseParsingError`, `ResponseTruncatedError`

### Key Patterns

- All LLM methods are async and return response objects with embedded usage metrics
- Retry logic: `@retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))`
- Costs calculated per million tokens using `TOKENS_PER_MILLION = 1_000_000`
- Pydantic models for structured output schemas via `model_json_schema()`
- Provider errors are wrapped in `ProviderError` with `original_error` attribute
- Output caps resolve in one place, `LLM._resolve_max_tokens()`: per-request `max_tokens` → the model's config value → `DEFAULT_MAX_TOKENS` (16000) / `DEFAULT_STREAM_MAX_TOKENS` (64000). Providers call it rather than choosing a literal
- Truncation raises `ResponseTruncatedError` via `LLM._check_truncation()`. It subclasses `MajordomoError` rather than `ProviderError` specifically so it is neither retried by `retry_provider_call` nor failed over by `LLMCascade` — both would repeat the same call against the same ceiling. On structured paths the check runs before content extraction, so the cause is reported instead of a downstream parse error
- API key hashing: `_hash_api_key()` uses SHA256 truncated to 16 hex chars for safe logging
- All providers accept `api_key_alias` for human-readable key identification in logs

### Environment Variables

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GEMINI_API_KEY` - Google Gemini API key
- `DEEPSEEK_API_KEY` - DeepSeek API key
- `CO_API_KEY` - Cohere API key
- `AWS_BEARER_TOKEN_BEDROCK` - Amazon Bedrock API key (long-term bearer token). Used by both `Bedrock` (Converse API) and `BedrockMantle` (AWS-native Anthropic Messages API)
- `AWS_REGION` (or `AWS_DEFAULT_REGION`) - AWS region for Bedrock requests (e.g., `us-east-1`)
- `FIREWORKS_API_KEY` - Fireworks AI API key
- `TOGETHER_API_KEY` - Together AI API key
- `BASETEN_API_KEY` - Baseten Model APIs key
- `NEBIUS_API_KEY` - Nebius Token Factory API key
- `DEEPINFRA_API_KEY` - DeepInfra API key
- `MOONSHOT_API_KEY` - Moonshot AI (Kimi) API key
- `NOVITA_API_KEY` - Novita AI API key
