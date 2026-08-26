# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.22.0] - 2026-08-25

### Added

- **`max_tokens` is now configurable.** It was hardcoded at every call site that required it — 1024 for plain text and streaming, 4096/8192 for the structured paths in `providers/anthropic.py`, and 1024/4096 in `providers/bedrock.py` — with no way for a caller to raise it. It is now a per-model key in `llm_config.yaml` and a per-request keyword argument on `get_response`, `get_response_stream`, `get_json_response`, and `get_structured_json_response`. Precedence is per-request → model config → library default, resolved in one place (`LLM._resolve_max_tokens`) rather than chosen per call site
  - Every model in the `anthropic`, `bedrock_mantle`, and `bedrock` blocks now declares its real ceiling, read from the vendor rather than a catalog (see Verification below): 128000 for Opus 5 / Fable 5 / Opus 4.8 (and its effort profiles) / Opus 4.7 / Opus 4.6 / Sonnet 5 / Sonnet 4.6, 64000 for the 4.5 line, 32000 for Opus 4 / 4.1 and Sonnet 4; on Bedrock, 262144 for `moonshotai.kimi-k2.5`, `moonshot.kimi-k2-thinking`, `nvidia.nemotron-nano-3-30b` and `nvidia.nemotron-super-3-120b`, 163840 for `deepseek.v3.2`, 131072 for `nvidia.nemotron-nano-12b-v2`, 32768 for `us.deepseek.r1-v1:0`, and 8192 for the two Llama 4 profiles
  - Only providers whose API *requires* an output cap read the key. The twelve OpenAI-compatible providers and Gemini omit `max_tokens` entirely and inherit each model's own default, so the key is not forwarded to them rather than being silently accepted
- **`stop_reason` on `LLMResponse` and `LLMStreamResponse`** — the provider's verbatim stop reason (`end_turn`, `tool_use`, `max_tokens`, …), or `None` for providers that report none. Also recorded in the request body written by `LoggingLLM`, so a truncated call is visible in the log row
- **`ResponseTruncatedError`**, carrying `max_tokens`, `output_tokens`, and `partial_content`. Exported from `majordomo_llm`

### Fixed

- **Truncated responses no longer return silently.** `_get_response_impl` builds content by joining the response's text blocks; when the output cap was hit before any text block was emitted, the caller received `content == ""` with no exception and no warning, indistinguishable from a model that had nothing to say. Anthropic and Bedrock now raise `ResponseTruncatedError` when the provider reports `stop_reason`/`stopReason` of `max_tokens`, on the plain-text, streaming, and structured paths alike. This mirrors the existing handling of `stop_reason == "refusal"`, which already raised
  - The error is deliberately **not** retried — re-sampling spends the same budget against the same ceiling — and deliberately does **not** trigger `LLMCascade` failover, since the next provider in the chain would truncate identically. It subclasses `MajordomoError` rather than `ProviderError` to get both behaviors without special-casing
  - On the structured paths the check runs *before* content extraction, so a cut-off tool call reports the truncation rather than surfacing as a missing-tool or JSON parse error
- **`_run_hooks_returning_response` no longer drops fields.** When a hook rewrote the response content, the rebuilt `LLMResponse` silently lost `routed_provider` and `routed_model`; both are now carried across, along with the new `stop_reason`

### Changed

- **The plain-text and streaming defaults are no longer 1024.** A model with no configured `max_tokens` now gets 16000 on non-streaming calls and 64000 on streaming ones, following Anthropic's own guidance: 16000 keeps a non-streaming response inside the SDK's HTTP timeout, while streaming has no such constraint. 1024 is roughly 700 words and, with thinking enabled, was shared between thinking and answer — the `claude-opus-4-8-medium` and `claude-opus-4-8-deep` profiles set `thinking: adaptive`, so any nontrivial answer truncated
- The `thinking` docstring in `providers/anthropic.py` pointed at "a dedicated config entry" for raising the cap. That entry did not exist; it now does, and the docstring names it

### Verification

- `scripts/check_max_tokens.py` probes each configured Anthropic and Bedrock model with a deliberately over-large `max_tokens` and reads the true ceiling back out of the vendor's rejection. The request is refused before inference, so the sweep costs nothing. Credentials come from the environment or a local `.env`, matching `scripts/smoke_test_providers.py`
- **The sweep corrected seven of the nine Bedrock ceilings.** They had been seeded from LiteLLM's public catalog, which understated every one of them — `moonshot.kimi-k2-thinking` 8192 → 262144, `nvidia.nemotron-nano-12b-v2` 8192 → 131072, `nvidia.nemotron-nano-3-30b` 8192 → 262144, `nvidia.nemotron-super-3-120b` 32768 → 262144, both Llama 4 profiles 4096 → 8192, and `us.deepseek.r1-v1:0` 4096 → 32768. All 13 reachable Anthropic models and all 4 Bedrock Mantle models confirmed as configured

### Removed

- **The Claude 4 family is retired.** `claude-opus-4-1-20250805`, `claude-opus-4-20250514`, and `claude-sonnet-4-20250514` return HTTP 404 (`not_found_error`) from the Messages API — surfaced by the `max_tokens` sweep, which could not probe them. All three are removed from the `anthropic` models block and mapped in `deprecated_models` to the current flagship of the same tier: the two Opus entries to `claude-opus-5` and Sonnet 4 to `claude-sonnet-5`. Existing callers keep working and get the standard deprecation warning; the Opus mapping also cuts cost from $15/$75 to $5/$25
  - Removal is what activates the mapping — `get_llm_instance()` only consults `deprecated_models` when a model is absent from the registry
  - The shipped `resilient-sonnet` alias targeted `claude-sonnet-4-20250514`. Alias validation resolves against the models block only, not `deprecated_models`, so leaving it would have raised `ConfigurationError` at import and broken the package outright. It now targets `claude-sonnet-5`, and a test guards against any alias naming a retired model
  - `claude-sonnet-4-20250514` was the canonical example throughout the docstrings, README, and docs; those 87 references now name live models. Note this was invisible to the test suite, which mocks every provider SDK client and so never validates that a registered model exists upstream

## [0.21.0] - 2026-08-20

### Added

- **Five OpenAI-compatible inference providers** — `baseten`, `nebius`, `deepinfra`, `moonshot`, and `novita` — broadening the backend set for the canonical open-weight models beyond Fireworks and Together. All five speak the OpenAI chat-completions wire protocol, so no new dependency is required; each supports text, streaming, and JSON-schema structured output, and injects `x-majordomo-provider` when given a `base_url` so a gateway can disambiguate them from vanilla OpenAI traffic
  - **Baseten** (`https://inference.baseten.co/v1`, `BASETEN_API_KEY`) — `deepseek-ai/DeepSeek-V4-Pro` ($1.74/$0.145/$3.48), `moonshotai/Kimi-K2.6` ($0.95/$0.16/$4.00), `moonshotai/Kimi-K3` ($3.00/$0.30/$15.00), `zai-org/GLM-5.2` ($1.40/$0.14/$4.40), `thinkingmachines/inkling` ($1.00/$0.17/$4.05). GLM-5.1 is deliberately absent: its model-library page still advertises Model API pricing, but a live call returns HTTP 410 (deprecated). Note the lowercase Inkling slug, which differs from Together's `thinkingmachines/Inkling`. Dedicated per-deployment endpoints are reached via `base_url`
  - **Nebius Token Factory** (`https://api.tokenfactory.nebius.com/v1`, `NEBIUS_API_KEY`) — `deepseek-ai/DeepSeek-V4-Pro` ($1.75/$3.50), `moonshotai/Kimi-K2.6` ($0.95/$4.00), `moonshotai/Kimi-K3` ($3.00/$15.00), `zai-org/GLM-5.1` ($1.40/$4.40), `zai-org/GLM-5.2` ($1.40/$4.40). Rates and IDs taken from `GET /v1/models?verbose=true`; that response publishes no cached-token rate, so `cached_input_cost` is unset and cached reads bill at `input_cost`. Keys provisioned against the legacy AI Studio host reach it via `base_url`
  - **DeepInfra** (`https://api.deepinfra.com/v1/openai`, `DEEPINFRA_API_KEY`) — `deepseek-ai/DeepSeek-V4-Pro` ($1.30/$0.10/$2.60), `moonshotai/Kimi-K2.6` ($0.75/$0.15/$3.50), `moonshotai/Kimi-K3` ($2.85/$0.285/$14.25), `zai-org/GLM-5.1` ($1.05/$0.205/$3.50), `zai-org/GLM-5.2` ($0.75/$0.14/$2.40), `thinkingmachines/Inkling` ($0.95/$0.16/$4.05). DeepInfra undercuts both Fireworks and Together on every model registered here. Note its OpenAI-compatible routes are nested under `/v1/openai`, not `/v1`. All rates are DeepInfra's **Standard** tier — its Priority (1.5x) and Flex (0.8x) tiers are not modelled, so requests routed to either will be mispriced by that multiplier
  - **Moonshot** (`https://api.moonshot.ai/v1`, `MOONSHOT_API_KEY`) — first-party Kimi: `kimi-k2.6` ($0.95/$0.16/$4.00) and `kimi-k3` ($3.00/$0.30/$15.00). Rates are the USD international platform; mainland-China accounts bill separately in RMB via `api.moonshot.cn`
  - **Novita** (`https://api.novita.ai/openai/v1`, `NOVITA_API_KEY`) — `deepseek/deepseek-v4-pro` ($1.60/$0.135/$3.20), `moonshotai/kimi-k2.6` ($0.80/$0.16/$3.40), `moonshotai/kimi-k3` ($3.00/$0.30/$15.00), `zai-org/glm-5.1` ($1.38/$0.26/$4.40), `zai-org/glm-5.2` ($1.40/$0.26/$4.40). Rates and IDs come from Novita's unauthenticated `/openai/v1/models`, which returns pricing inline. Note the org prefix differs from the HF-style platforms (`deepseek/deepseek-v4-pro`, not `deepseek-ai/DeepSeek-V4-Pro`); Novita also serves the same routes under `/openai` and `/v3/openai`
- **`providers/_openai_compatible.py` with `OpenAICompatibleLLM`** — a shared base holding the chat-completions request/response, streaming, and JSON-schema machinery for providers speaking the OpenAI wire protocol. Subclasses set four class attributes (`PROVIDER_NAME`, `DISPLAY_NAME`, `DEFAULT_BASE_URL`, `API_KEY_ENV`) and override nothing, so each of the five new providers is ~35 lines. A `_provider_request_kwargs()` hook forwards `reasoning_effort` / `thinking` when a model config sets them and is available for providers needing different collapse rules
- All four classes are exported from `majordomo_llm` and `majordomo_llm.providers`, and registered in the factory's `_PROVIDER_CLASSES`

- **`supports_structured_outputs` for OpenAI-compatible providers.** The existing config key now applies to `baseten`, `nebius`, `deepinfra`, `moonshot`, and `novita` (defaulting to `true`, so nothing else changes). When `false`, `OpenAICompatibleLLM` raises `StructuredOutputUnsupported` before issuing the request rather than letting a non-conforming response fail downstream in parsing. `Fireworks` and `Together` predate the shared base and do not accept the kwarg, so they are excluded
- Set `supports_structured_outputs: false` on Nebius `moonshotai/Kimi-K2.6`. Measured against the live API, that deployment accepts a strict `json_schema` response format but is not grammar-constrained: it conformed on only 2 of 5 samples and answered in prose otherwise. Text and streaming on that model are unaffected. Nebius's own `/v1/models` metadata is not a reliable source for this — it also omits `structured_outputs` for GLM-5.2 and Kimi-K3, both of which conform 5/5, and reports empty `supported_sampling_parameters` for both Kimi SKUs even though all five models accept `temperature`/`top_p`

### Fixed

- **Streaming through the `majordomo` gateway now reports the routed backend.** The routing headers arrive mid-stream, so the identity was consumed for pricing and then discarded: a streamed call reported `routed_provider`/`routed_model` as `None` while the same non-streaming call reported them. `_StreamState` now carries both, `LLMStreamResponse` exposes them as properties for callers that iterate, and `collect()` propagates them onto the returned `LLMResponse`

### Changed

- **`temperature` and `top_p` no longer default to `0.3` / `1.0`.** They now default to `None` and are sent only when the caller passes them, so each provider applies its own documented default instead of one this library invented. The parameters are also now independent — passing only `temperature` no longer forces a `top_p` value alongside it
- **Sampling parameters are dropped, with a warning, for models that reject them.** `LLM._sampling_params()` centralizes the rule: a model whose config sets `supports_temperature_top_p: false` never receives them, and an explicitly-passed value logs a warning naming the provider, model, and ignored values rather than failing the call — an `LLMCascade` or alias chain can legitimately mix members that do and do not accept them. This replaces per-provider `if self.supports_temperature_top_p:` branches that duplicated the entire request call, removing roughly 200 lines of duplication across seven providers
- **Gemini now honors `supports_temperature_top_p`.** It previously sent `temperature` and `top_p` on every request regardless of the flag


- `get_llm_instance()` now forwards `reasoning_effort` / `thinking` from model config for the five new providers alongside `deepseek`, `fireworks`, and `together`

- **Per-model capability flags, measured against the live APIs.** Vendor metadata proved unreliable, so every registered model was called directly. Models that pin their sampling parameters carry `supports_temperature_top_p: false` — Moonshot `kimi-k2.6`/`kimi-k3` (`invalid temperature: only 1 is allowed for this model`, `invalid top_p: only 0.95 is allowed`) and Baseten `moonshotai/Kimi-K3` (`Cannot override enforced sampling params`); omitting both parameters succeeds. Models that cannot honor a strict `json_schema` carry `supports_structured_outputs: false` — DeepInfra `thinkingmachines/Inkling` (HTTP 405), Novita `deepseek/deepseek-v4-pro` and `zai-org/glm-5.1` (HTTP 400, `json_object` only), Nebius `moonshotai/Kimi-K2.6` (accepts it but conformed on only 10 of 18 samples), and Baseten `deepseek-ai/DeepSeek-V4-Pro` (accepts it and returned prose on 5 of 5 samples)

### Notes

- **No alias changes.** The cross-vendor cascades (`deepseek-v4-pro`, `kimi-k2.6`, `kimi-k3`, `glm-5.1`, `glm-5.2`, `inkling`) remain Fireworks→Together; wiring the new providers into them is a separate change
- **No reasoning-profile entries** (`deepseek-v4-pro-reasoning` / `-hard`) are registered for the new providers. Whether they honour `reasoning_effort` / `thinking` on their DeepSeek deployments is unverified, and a silently-ignored reasoning flag is worse than an absent one
- `Fireworks` and `Together` were left on their own implementations rather than migrated onto `OpenAICompatibleLLM`

## [0.20.1] - 2026-08-11

### Fixed

- The `majordomo` provider now also sends the canonical model name as the `x-majordomo-model` request header (alongside `x-majordomo-provider`), so the gateway can route on the header rather than parsing the request body's `model` field

## [0.20.0] - 2026-08-11

### Added

- **`majordomo` gateway provider with server-side optimal routing.** A new pseudo-provider (`providers/majordomo.py`, registered in the factory and exported from the package) that names a canonical open-weight model — `deepseek-v4-pro`, `kimi-k2.6`, `kimi-k3`, `glm-5.1`, `glm-5.2`, `inkling` — and lets Majordomo Steward pick the optimal backend at request time, rather than pinning a provider. It signals routing to the gateway with `x-majordomo-provider: majordomo`, **requires** `base_url` (the gateway URL) and `MAJORDOMO_API_KEY` (auto-injected as the `X-Majordomo-Key` header), and speaks the OpenAI-compatible wire protocol. Because the backend is only known after the call, cost is **not** taken from a fixed config entry: the provider reads the gateway's `X-Majordomo-Routed-Provider` / `X-Majordomo-Routed-Model` response headers and prices the usage against that pair's rates in `llm_config.yaml` (e.g. GLM-5.2's cached read is 0.14 on Fireworks vs 0.26 on Together). Missing or unconfigured routed pairs degrade to zero cost with a warning rather than crashing. Supports text, streaming, and structured/JSON-schema output
- **`routed_provider` / `routed_model` on `LLMResponse`** — the concrete backend the gateway selected, surfaced for observability (`None` on direct provider calls)
- **`get_model_pricing(provider, model) -> ModelPricing | None`** in the factory (and exported from the package) — resolves a concrete pair's per-million rates and cache-accounting mode from `llm_config.yaml` without instantiating a client. Used to price gateway-routed calls after the fact
- **`compute_costs(...)` in `base.py`** — the stateless core of `LLM._calculate_costs`, extracted so a request can be priced against rates other than the calling instance's own (the routed backend's). `_calculate_costs` now delegates to it with no behaviour change
- **`_StreamState.price_override`** — an optional pricing hook applied at stream finalization, so a streamed Majordomo call prices its final usage against the routed backend instead of the provider's (empty) rates. Defaults to `None` (existing providers unaffected)

### Changed

- **`get_all_llm_instances()` skips gateway-routed providers** (`majordomo`), which cannot be instantiated without a live gateway `base_url`

## [0.19.0] - 2026-07-28

### Added

- **Kimi K3 and GLM-5.2 across Fireworks and Together** in `llm_config.yaml`, exposed as Fireworks→Together failover cascade aliases (`kimi-k3`, `glm-5.2`) mirroring the existing `kimi-k2.6` / `glm-5.1` pattern. Fireworks: `accounts/fireworks/models/kimi-k3` ($3.00/$0.30/$15.00) and `accounts/fireworks/models/glm-5p2` ($1.40/$0.14/$4.40). Together: `moonshotai/Kimi-K3` ($3.00/$0.30/$15.00) and `zai-org/GLM-5.2` ($1.40/$0.26/$4.40). Unlike the older SKUs on both providers, these expose a discounted cache tier and set `cached_input_cost`; the Fireworks and Together block header comments were updated to document that newer SKUs carry a cache tier while older ones bill cached reads at `input_cost`
- **Inkling (Thinking Machines) across Fireworks and Together** in `llm_config.yaml`, exposed as an `inkling` Fireworks→Together failover cascade alias. Fireworks `accounts/fireworks/models/inkling` and Together `thinkingmachines/Inkling`, both $1.00/$0.17/$4.05 (input/cached/output) with `cached_input_cost` set

## [0.18.0] - 2026-07-28

### Added

- **Claude Opus 5** (`claude-opus-5`, $5/$25 per MTok) registered in `llm_config.yaml` under both the `anthropic` provider and `bedrock_mantle` (`anthropic.claude-opus-5`). Mirrors the `claude-opus-4-8` shape: cache pricing per Anthropic's published convention (`cached_input_cost` 0.10× input = 0.50, `cache_write_cost` 1.25× = 6.25), `supports_temperature_top_p: false` (the 5-generation rejects sampling params). The direct Anthropic entry is additionally flagged `supports_web_search: true` and `supports_structured_outputs: true`. The factory picks it up automatically; no code changes required

## [0.17.0] - 2026-07-23

### Added

- **Configurable prompt caching on the Anthropic-family providers.** `use_prompt_caching` is now a per-model flag in `llm_config.yaml` (default `true`), a constructor argument on `Anthropic`/`BedrockMantle`, and an override on `get_llm_instance(..., use_prompt_caching=…)` (mirroring `use_web_search`; `None` keeps the config default). It gates the ephemeral `cache_control` breakpoint stamped on the system prompt — set it `False` to suppress caching for short, non-reused system prompts where the cache-write premium is wasted. Providers without an explicit cache breakpoint (OpenAI, Gemini, DeepSeek, Fireworks, Together, Cohere, Bedrock Converse) ignore it. Verified live end-to-end against `claude-sonnet-5`, `claude-opus-4-8-fast`, and `bedrock_mantle` Haiku 4.5 (cold write → warm read → toggle-off)
- **Cache-aware cost accounting.** `_calculate_costs` now folds cache read/write tokens into `input_cost` according to a per-provider accounting mode (`_cache_accounting`): **subset** for OpenAI/Gemini/DeepSeek/Fireworks/Together (cached reads are part of `input_tokens` and re-priced down to `cached_input_cost`) and **additive** for Anthropic/Bedrock (cache read/write are reported separately and added on top at `cached_input_cost` / `cache_write_cost`). When a rate is unset the cost is computed exactly as before, so the change is backward compatible. Verified: additive `input_cost = (input·rate + read·read_rate + write·write_rate)/1M` matches the live API to the cent
- **`cached_input_cost` / `cache_write_cost` per-model pricing in `llm_config.yaml`,** populated for every cache-capable provider using each vendor's published, stable convention (Anthropic/Bedrock Mantle: read 0.10× input, 5-min write 1.25×; OpenAI GPT-5 line 0.10×, GPT-4.1/o-series 0.25×; Gemini 0.25×; DeepSeek 0.10×). Fireworks/Together are intentionally left unset (no separate cache tier — cached reads bill at `input_cost`); Bedrock Converse is left unset (no cache points are inserted). A header comment documents the multiplier basis
- **`cache_creation_tokens` on `Usage`** (and `LLMResponse`/`LLMJSONResponse`/`LLMStructuredResponse`, streaming `_StreamState`, and the logging layer). Captured from Anthropic (`cache_creation_input_tokens`) and Bedrock (`cacheWriteInputTokens`). Gemini now also captures cache **reads** (`cached_content_token_count`), which were previously hard-coded to `0`. A `cache_creation_tokens` column was added to the SQLite `CREATE TABLE` and all three log-adapter INSERTs (SQLite/Postgres/MySQL)
- **`examples/prompt_caching_demo.py`** demonstrating both caching flavors side by side: explicit create → read → `use_prompt_caching=False` toggle (Anthropic, Bedrock Mantle) and automatic cache-read + subset discount (OpenAI, Gemini, DeepSeek), with per-call token/cost breakdowns and per-read savings
- **`run_demo(main, description)` in `examples/shared.py`** — factors the identical `--gateway`/`--provider` argparse boilerplate out of all six example scripts into one helper

### Changed

- **`_calculate_costs` signature** extended with optional `cached_tokens` and `cache_creation_tokens` parameters (both default `0`), so the ~19 existing call sites compile unchanged and only cache-bearing providers pass the new arguments
- **`Usage.cache_creation_tokens`** is keyword-only with a `0` default, so existing positional/keyword `Usage` construction is unaffected

## [0.15.0] - 2026-07-22

### Fixed

- **Structured outputs no longer report an empty result as success (headline bug).** A forced tool call is mandatory to *invoke* but its arguments are ordinary generation, so Claude could return `{}` or an all-null object. The only validation gate (`canonicalize_json_schema_output`) ran `jsonschema.validate`, which *passes* an empty/all-null object whenever the schema's fields are optional/nullable — so the library returned a billed, schema-valid non-answer as a successful `LLMResponse`, indistinguishable from a real result. It now detects that case and raises

### Added

- **Configurable reasoning effort on Anthropic.** `reasoning_effort` (one of `low`/`medium`/`high`/`xhigh`/`max`) on the `Anthropic` provider and per-model in `llm_config.yaml`, applied as `output_config.effort` on every request (plain, streaming, native structured, and forced-tool paths). Previously every call ran at the API default (`high`); this lets callers dial down for routine extraction or up (`xhigh`) for agentic work. `None` (default) preserves today's behavior. Register the same SKU under multiple YAML keys via the `model` override to expose distinct effort profiles. Invalid values raise `ValueError`. Verified live against `claude-opus-4-8`
- **Configurable thinking on Anthropic.** `thinking` (`adaptive` or `disabled`) on the `Anthropic` provider and per-model in `llm_config.yaml`, applied as the `thinking.type` request field on every path. `None` (default) omits it, so the model runs without thinking (today's behavior). Pairs with `reasoning_effort`, which only meaningfully modulates depth when thinking is on. Thinking blocks are filtered from the returned content (answer text only). Invalid values raise `ValueError`. Verified live against `claude-opus-4-8` (adaptive thinking, plain + native). Caveats documented in the constructor: `disabled` is rejected on Fable 5 (thinking always on), and with thinking on the fixed `max_tokens` covers thinking + answer — raise it via a dedicated config entry if answers truncate
- **Three effort/thinking profiles of `claude-opus-4-8`** in `llm_config.yaml` (same SKU via the `model` override): `claude-opus-4-8-fast` (low effort, no thinking), `claude-opus-4-8-medium` (medium effort + adaptive thinking), and `claude-opus-4-8-deep` (`xhigh` effort + adaptive thinking). Verified live end-to-end
- **Current flagship Anthropic models** in `llm_config.yaml`: `claude-fable-5` ($10/$50), `claude-opus-4-8` ($5/$25), and `claude-sonnet-5` ($3/$15) — the library previously topped out at the now-legacy 4.6/4.7 tier, and Opus 4.8 existed only under `bedrock_mantle`. All three are `supports_temperature_top_p: false` (the 4.7+/5 generation rejects sampling params) and `supports_structured_outputs: true` (verified live, 10/10 conforming). Opus 4.8 and Sonnet 5 are flagged `supports_web_search: true`; Fable 5 is left without it pending verification. Existing models are retained as legacy (no removals)
- **`EmptyStructuredResponseError`** (subclass of `ResponseParsingError`) — raised when a structured result validates against the schema but is empty (`{}` or every top-level value `null`). It is **retryable**: the structured-output retry wrapper re-samples before it surfaces (wrapped in `tenacity.RetryError` on exhaustion, matching every other retryable error; `LLMCascade` unwraps it as usual)
- **`is_empty_structured_result(content)`** and a `reject_empty: bool = True` parameter on `canonicalize_json_schema_output` in `base.py`. All providers route their structured output through that gate, so the check protects every provider; a caller that genuinely wants an all-null object can pass `reject_empty=False`
- **Native structured outputs (constrained decoding) on Anthropic.** `supports_structured_outputs` per-model flag in `llm_config.yaml` (plumbed through `factory.py` and the `Anthropic`/`BedrockMantle` constructors). When set, `_get_json_schema_response` uses `client.messages.create(output_config={"format": {"type": "json_schema", "schema": …}})` — the model physically cannot emit malformed or missing-key output. Verified via a live N=10 harness (10/10 conforming, 0 empty) and enabled for `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-sonnet-4-5-20250929`, and `claude-haiku-4-5-20251001`. The wire schema is `strip_unsupported_schema_constraints(enforce_strict_object_schema(schema))` (the constrained decoder rejects `minimum`/`maximum`/`pattern`/`format`/array bounds and requires `additionalProperties: false` + full `required`); those keyword constraints are re-enforced post-hoc by validating the response against the caller's original schema. No `name` key is sent inside `format` (it 400s)

### Changed

- **Aliases repointed to current flagships:** `smart` → `claude-opus-4-8` (was `claude-opus-4-6`), `thinking` → `claude-sonnet-5` (was `claude-sonnet-4-6`). `fast` stays `claude-haiku-4-5-20251001`
- **Anthropic forced tool calling is now the fallback**, used only for models without `supports_structured_outputs` and for web-search requests (which can't combine with `output_config`). It now sends the **strict** schema (`enforce_strict_object_schema`, full `required`) instead of a relaxed one, so an empty `{}` fails schema validation loudly and an all-null result is caught by the empty-check — both then re-sampled by the retry policy. In the live harness this path was also 10/10 on `claude-opus-4-7`
- **Bedrock (Converse) structured output** gets the same treatment on its json-schema path: strict schema (`enforce_strict_object_schema`) instead of relaxed, and the empty-check via the shared gate
- **`anthropic` dependency floor raised `>=0.76.0` → `>=0.116.0`** so `output_config` is available on the stable `client.messages.create` (it is beta-only in 0.76). `BedrockMantle` inherits the Anthropic path; its models default to the forced-tool fallback until Mantle's `output_config` support is separately verified

### Removed

- **`relax_strict_object_schema` and `fill_strict_nullable_defaults`** (added in 0.14.0). Relaxing the schema before a forced tool call — emptying `required` for nullable properties — was itself a primary cause of the empty results (it turned a loud parse failure into a silently-valid all-null object). The 0.14.0 spec's approach is superseded by native constrained decoding plus the strict-schema fallback and the empty-check
- The dead `Anthropic._get_structured_response` override (and its web-search helpers) — the public Pydantic path already routes through `get_json_schema_response`, so dict-callers and Pydantic-callers now share one wire path

### Audit (unchanged)

- OpenAI, Gemini, Cohere, DeepSeek, Fireworks, and Together share the same "no emptiness check" gap but now inherit the shared `reject_empty` guard through `canonicalize_json_schema_output`. Their send-path schema handling (constrained decoding or prompt injection) is outside this change and was left as-is

## [0.14.0] - 2026-07-21

### Fixed

- **Strict-dialect JSON schemas no longer fail on forced-tool-call providers.** `get_json_schema_response` accepts an arbitrary JSON Schema, but the Anthropic and Converse-based Bedrock providers forwarded it verbatim as a tool `input_schema`. A schema written in OpenAI's strict dialect — every property in `required`, optionality spelled as `anyOf: [T, null]` with `default: null` — conflicts with Anthropic's tool-calling convention (omit keys whose value is null), so the model omitted the very keys the schema demanded and the call failed at validation with `{}` or a placeholder wrapper key. Both providers now translate the schema on the way out and reconstruct the OpenAI-equivalent shape on the way back

### Added

- **`relax_strict_object_schema(schema)`** in `base.py` — the inverse of `enforce_strict_object_schema`. Walks every object node and, for each property listed in `required` whose subschema is nullable (`anyOf`/`oneOf` containing `{"type": "null"}`, or a `type` array containing `"null"`), removes it from `required` and unwraps the subschema to its non-null branch (dropping the `default: null` that invites omission). Non-nullable required properties are left enforced; `additionalProperties` is left as found
- **`fill_strict_nullable_defaults(instance, schema)`** in `base.py` — after relaxation, populates any nullable-optional property the model omitted from its declared `default` (an explicit null in strict dialect), walking nested objects, arrays, and `anyOf`/`oneOf` branches. Keyed on the strict idiom, it is a no-op on plain non-strict schemas

### Changed

- **Anthropic** (`_get_json_schema_response` and the web-search helper) and **Bedrock** (`_get_json_schema_response`, Converse tool calling) now send `relax_strict_object_schema(response_schema)` as the tool `input_schema` and fill omitted nullable-optionals before validating against the caller's original schema. Validation still runs against the unmodified caller schema, matching the OpenAI path. `BedrockMantle` inherits the fix from `Anthropic`
- Structured responses from these providers for strict-dialect schemas now contain explicit nulls for properties the model omitted, identical in shape to what the OpenAI path produces for the same caller schema. Non-strict schemas are unaffected

### Note

- The Pydantic-model paths (`get_structured_json_response`) are unaffected — they emit non-strict schemas via `model_json_schema()`. Gemini, Cohere, Together, Fireworks and DeepSeek use constrained decoding or prompt injection rather than forced tool calls and are outside this bug class; they were audited and left unchanged

## [0.12.0] - 2026-06-13

### Added

- **`use_web_search` extended to OpenAI and Gemini.** Previously only Anthropic accepted the flag (and only on `claude-sonnet-4-5-20250929`). OpenAI now wires the Responses API `web_search_preview` tool; Gemini attaches the `google_search` grounding tool to its `GenerateContentConfig`. Bedrock continues to accept the flag as a no-op for interface parity
- **`get_llm_instance(..., use_web_search=...)` factory forwarding.** The flag is validated against a new `supports_web_search: true` config flag per model in `llm_config.yaml` and forwarded to capable providers (openai, anthropic, gemini, bedrock). Passing `use_web_search=True` for a model whose config does not declare the flag raises `ConfigurationError`. Providers without a web-search story (cohere, deepseek, fireworks, together, bedrock_mantle) silently ignore the flag
- **`tool_use_cost: float` on `Usage`** (kw-only, defaults to `0.0`). Anthropic populates it from `response.usage.server_tool_use.web_search_requests` at $0.01/request; Gemini populates it by counting candidates with `grounding_metadata` at $0.035/request. OpenAI bills web search through normal output tokens, so it remains `0.0` for that provider. The value is added into `total_cost`
- `examples/web_search_demo.py` — one supported model from each of Anthropic, OpenAI, and Gemini with `use_web_search=True`, printing the full cost breakdown (input + output + tool)

### Changed

- **Anthropic web-search gate relaxed from a single hard-coded model to a YAML capability flag.** The `self.model == "claude-sonnet-4-5-20250929"` check is gone; the factory's `supports_web_search` validation drives gating. Web search is now flagged on Claude 4.5 and 4.6 SKUs (opus-4-7, opus-4-6, sonnet-4-6, opus-4-5-20251101, sonnet-4-5-20250929, haiku-4-5-20251001)
- **Anthropic plain-text web-search tool fixed.** The plain-text response path was previously appending `{"type": "web_search_tool", "name": "web_search_20250305"}`, which is not a valid tool type. Now uses `WebSearchTool20250305Param(type="web_search_20250305", name="web_search")` to match the structured-output paths
- OpenAI `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5`, `gpt-5-mini`, `gpt-4.1`, `gpt-4.1-mini` flagged `supports_web_search: true`. Gemini `gemini-3.5-flash`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash` flagged the same

### Known limitations

- **Gemini cannot combine grounding with `response_schema` in a single request** — the API rejects the combination. The Gemini provider raises `ConfigurationError` from `_get_json_schema_response` (and any path that routes through it) when `use_web_search=True`. To use both features, construct one Gemini instance with `use_web_search=True` for grounded text/stream calls and a separate instance with `use_web_search=False` for structured calls

## [0.11.1] - 2026-06-12

### Changed

- **Gemini catalog refreshed** against https://ai.google.dev/gemini-api/docs/models — added new flagship `gemini-3.5-flash` ($1.50/$9.00, Stable), promoted `gemini-3.1-flash-lite-preview` to the stable name `gemini-3.1-flash-lite` (same pricing), and registered the deprecation mapping so existing callers auto-upgrade. `gemini-3.1-pro-preview` and `gemini-3-flash-preview` remain listed as preview. `examples/shared.py` and `examples/flagship_demo.py` Gemini entries point at `gemini-3.5-flash`; smoke-test `EXTRA_MODELS` swept to the new stable models

## [0.11.0] - 2026-06-01

### Added

- **Bedrock Mantle provider** (`BedrockMantle`) — Anthropic Claude served via AWS-native Anthropic Messages API at `https://bedrock-mantle.{region}.api.aws/anthropic`. Implemented as a thin subclass of `Anthropic`, so Claude's full feature set (structured outputs, prompt caching, extended thinking, tool use, streaming) works out of the box without Converse-shape gymnastics. Authenticates via `AWS_BEARER_TOKEN_BEDROCK` (same bearer token used for the legacy Bedrock Converse path). Region from `AWS_REGION` / `AWS_DEFAULT_REGION` / `region=` constructor arg
- 3 BedrockMantle SKUs in `llm_config.yaml`: Claude Opus 4.8, Opus 4.7, Haiku 4.5 (model IDs use the bare `anthropic.claude-<name>` format). Sonnet 4.6 is not yet hosted on Mantle (returns 404 not_found_error); will be added when AWS lists it

### Changed

- **Bedrock provider scope narrowed to non-Anthropic models.** Anthropic Claude entries removed from the `bedrock:` YAML block — those now live under `bedrock_mantle:`. The remaining Bedrock catalog covers Moonshot Kimi, NVIDIA Nemotron, Meta Llama 4, and DeepSeek-on-Bedrock
- **Removed the Bedrock native Structured Outputs path** (`outputConfig.textFormat.json_schema` via Converse). The supporting allowlist (`_BEDROCK_STRUCTURED_OUTPUTS_SUPPORTED`) and helper (`_bedrock_output_config`) are gone. Rationale: the only beneficiary was Anthropic Claude on Bedrock, which has moved to BedrockMantle where Claude's structured outputs are first-class. Non-Anthropic Bedrock models (Llama 4, Kimi, Nemotron, DeepSeek-on-Bedrock) keep the Converse tool-calling path, which is now the sole Bedrock structured-output mechanism. Eliminates the per-version Anthropic substring maintenance burden — newer Claude releases (Opus 4.8+) just work via BedrockMantle without any allowlist update

### Removed

- `enforce_strict_object_schema` and `strip_unsupported_schema_constraints` are no longer used by the Bedrock provider (they remain in `base.py` and continue to be used by OpenAI strict mode and Cohere respectively)
- `us.anthropic.claude-*` entries from the `bedrock:` YAML block. Migration: use `bedrock_mantle` with `anthropic.claude-*` model IDs (no `us.` prefix, no `-v1` suffix). No backward-compatible alias provided — no users on the previous Bedrock Claude path

### Known limitations

- **Bedrock Nemotron Nano structured output** is grammar-enforced via Bedrock Structured Outputs, but the model can produce malformed JSON on deeply nested or complex schemas (~3+ levels). Simpler schemas pass reliably. For high-reliability structured calls, cascade to a larger model (e.g. `nemotron → claude-haiku`)
- **Together / `json_schema` response format** is supported on a subset of hosted models. The Together provider sends the `json_schema` shape uniformly; models that reject it surface as `ProviderError`. Use the cross-vendor `deepseek-v4-pro` alias to fail over to Fireworks for structured calls on Together-only DeepSeek models

## [0.10.0] - 2026-05-31

### Fixed

- **OpenAI / strict schemas with enum fields**: `inline_schema_refs()` now correctly inlines `$ref` references that have sibling keys (e.g. Pydantic-generated `{"$ref": "...", "description": "..."}` for fields typed with an `Enum`). Previously the `len(obj) == 1` guard left the dangling reference in place after `$defs` was popped, producing `Invalid schema for response_format: reference to component '#/$defs/...' which was not found in the schema` from OpenAI. Field-level descriptions on enum fields are preserved in the inlined output
- **Cohere / strict schema validator** rejects standard JSON Schema constraints (`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`, `minItems`, `maxItems`, `uniqueItems`, `minLength`, `maxLength`, `pattern`, `format`). The Cohere provider now strips these recursively before sending so Pydantic models using `Field(ge=, le=, min_length=, ...)` work as-is. The stripping logic was promoted to a shared `strip_unsupported_schema_constraints` helper in `base.py` and is reused by the Bedrock Structured Outputs path
- **Bedrock / Llama 4 structured output**: `us.meta.llama4-*` models reject `toolConfig.toolChoice.tool` in the Converse API. The Bedrock provider now omits the `toolChoice` field for Llama 4 model IDs while still exposing the tool, relying on the system-prompt instruction to steer the model toward the tool call
- **Bedrock / Nemotron structured output (and grammar-enforced JSON for all supported models)**: previously, Bedrock structured output went exclusively through Converse tool calling, which produced opaque `InternalServerException` errors on NVIDIA Nemotron Nano. The Bedrock provider now uses native Bedrock Structured Outputs (`outputConfig.textFormat.json_schema`) for the supported model families — Anthropic Claude, NVIDIA Nemotron Nano, Qwen3, Google Gemma, Mistral — where Bedrock compiles the schema into a grammar and enforces it during generation. Tool calling remains the fallback path for models outside that allowlist (Llama 4, Moonshot Kimi K2.5, DeepSeek v3.2). Schemas are auto-normalized with `additionalProperties: false` and full `required` lists (the new shared `enforce_strict_object_schema` helper, previously `_enforce_openai_strict_schema`), and the same grammar-incompatible constraints stripped by Cohere (`minimum`, `maximum`, `minItems`, etc.) are stripped before being sent to Bedrock
- **DeepSeek / structured output uses correct response_format**: DeepSeek's API supports only `response_format={"type": "json_object"}` (per https://api-docs.deepseek.com/guides/json_mode); the previous `json_schema` request shape was rejected by every DeepSeek SKU with `"This response_format type is unavailable now"`. The DeepSeek provider now uses `json_object` mode and injects the schema into the system prompt via `build_schema_prompt()`, restoring structured output across `deepseek-chat`, `deepseek-v4-pro`, and `deepseek-v4-flash`
- **Fireworks / `reasoning_effort` + `thinking` conflict**: Fireworks rejects requests that specify both `reasoning_effort` and `thinking` (`cannot specify both 'thinking' and 'reasoning_effort'`), which broke the `deepseek-v4-pro-reasoning` and `deepseek-v4-pro-hard` profile aliases (both set both fields). The Fireworks provider now collapses the two fields: `thinking="disabled"` takes precedence (explicit opt-out wins); otherwise `reasoning_effort` is sent alone since it already implies thinking is on. Together still accepts both fields and is unaffected

### Added

- **Fireworks AI provider** (`Fireworks`) via the OpenAI-compatible `https://api.fireworks.ai/inference/v1` endpoint. Supports text, streaming, raw JSON-schema structured output, and Pydantic-validated structured output. Authenticates with `FIREWORKS_API_KEY`
- **Together AI provider** (`Together`) via the OpenAI-compatible `https://api.together.xyz/v1` endpoint. Same capability surface as Fireworks. Authenticates with `TOGETHER_API_KEY`. Note: Together's `json_schema` response format is supported on a subset of hosted models; the request uses the standard shape uniformly and surfaces model-side rejections as `ProviderError`
- 4 Fireworks serverless SKUs in `llm_config.yaml`: DeepSeek-V4-Pro, Kimi-K2.5, Kimi-K2.6, GLM-5.1
- 7 Together serverless SKUs in `llm_config.yaml`: DeepSeek-V4-Pro, Kimi-K2.6, Qwen3.6-Plus, Qwen3.5-9B, Qwen3-235B-A22B-fp8-tput, GLM-5.1, GLM-5
- `reasoning_effort` and `thinking` constructor kwargs on `Fireworks` and `Together`, mirroring the `DeepSeek` provider. Validated against the same effort/thinking value sets; forwarded via top-level `reasoning_effort` and `extra_body={"thinking": {"type": ...}}` respectively. Plumbed through `get_llm_instance()` from YAML attributes
- **Multi-profile model registration**: a `llm_config.yaml` model entry may declare a `model:` field that overrides the API model ID, decoupling it from the YAML key. Lets the same upstream SKU be registered under multiple profile names with different reasoning configs
- Three DeepSeek-V4-Pro reasoning profiles (`-reasoning`, `-hard`) registered under `deepseek`, `fireworks`, and `together` using the new `model:` override
- Cross-vendor cascade aliases — `deepseek-v4-pro`, `kimi-k2.6`, `glm-5.1` (Fireworks → Together), and `deepseek-v4-pro-reasoning` / `deepseek-v4-pro-hard` (Fireworks → Together → Anthropic Sonnet/Opus as quality safety net)
- `flagship_demo.py` expanded to compare closed-source frontiers (Opus 4.7, GPT-5.5, Gemini 3.1 Pro Preview, DeepSeek-Reasoner) side-by-side with DeepSeek-V4-Pro at three reasoning profiles across both Fireworks and Together

## [0.9.1] - 2026-05-18

### Fixed

- **OpenAI structured outputs** now normalize Pydantic-derived schemas to satisfy OpenAI's strict-mode requirements. Every object node in the schema gets `additionalProperties: false` and every property is added to `required`, with `$ref`/`$defs` inlined first. Previously, calling `get_structured_json_response()` against any current OpenAI model failed with `Invalid schema for response_format ...: 'additionalProperties' is required to be supplied and to be false`

## [0.7.0] - 2026-05-16

### Added

- **Amazon Bedrock provider** using the Converse API. Authenticates with long-term Bedrock API keys (`AWS_BEARER_TOKEN_BEDROCK`) and an AWS region (`AWS_REGION` / `AWS_DEFAULT_REGION` or `region=` constructor/factory kwarg). Supports text responses, streaming, raw JSON-schema structured output, and Pydantic-validated structured output via Converse tool calling
- 16 Bedrock models in `llm_config.yaml` (us-east-1 on-demand pricing): Claude 4.x family, Moonshot Kimi K2/K2.5, NVIDIA Nemotron Nano/Nano-3/Super-3, Meta Llama 4 Maverick/Scout, DeepSeek R1/v3.2
- `aioboto3` promoted to a core dependency
- **Deprecated model auto-replacement**: Passing a deprecated model to `get_llm_instance()` automatically resolves to the provider-recommended replacement with a logged warning
- `deprecated_models` section in `llm_config.yaml` mapping old model IDs to replacements for OpenAI, Anthropic, and Gemini
- `LLMResponse.deprecation_warning` field — set when a deprecated model was auto-replaced
- `LLM.requested_model` and `LLM.deprecation_warning` attributes for programmatic detection
- New OpenAI models: `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.4-pro`
- New Anthropic models: `claude-opus-4-6`, `claude-sonnet-4-6`
- New Gemini models: `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`
- `get_json_schema_response()` for raw JSON Schema structured outputs across providers and `LLMCascade`
- Canonical JSON serialization for raw schema responses so equivalent outputs are byte-comparable
- `StructuredOutputUnsupported` error for provider/model structured-output capability failures
- New Anthropic model: `claude-opus-4-7`

### Removed

- Deprecated OpenAI models removed from active config: `gpt-4o`, `gpt-4o-mini`, `gpt-5-pro`, `o1`, `o3-mini`
- Deprecated Anthropic models removed from active config: `claude-3-7-sonnet-20250219`, `claude-3-5-haiku-20241022`, `claude-3-haiku-20240307`
- Deprecated Gemini models removed from active config: `gemini-3-pro-preview`, `gemini-2.0-flash`, `gemini-2.0-flash-lite`

### Changed

- Updated aliases: `fast` → `claude-haiku-4-5-20251001`, `thinking` → `claude-sonnet-4-6`, `smart` → `claude-opus-4-6`, `resilient-sonnet` cascade uses `gpt-4.1` instead of `gpt-4o`
- `Gemini.__init__()` now accepts `supports_temperature_top_p` for constructor consistency across providers
- `get_llm_instance()` accepts a new `region` kwarg, forwarded to the Bedrock provider only
- `examples/shared.py` `PROVIDERS` list now requires a tuple of env vars per entry and includes Bedrock entries (one per upstream model family)

### Fixed

- DeepSeek v4 models (`deepseek-v4-flash`, `deepseek-v4-pro`) no longer send `thinking: disabled` alongside `reasoning_effort: medium`, which the DeepSeek API rejects as mutually exclusive

## [0.3.1] - 2026-02-19

### Added

- `api_key` parameter to `get_llm_instance()` and `LLMCascade` for passing API keys directly instead of relying on environment variables

## [0.2.0] - 2026-02-08

### Added

- Streaming responses via `get_response_stream()` for all providers (OpenAI, Anthropic, Gemini, DeepSeek, Cohere)
- `LLMStreamResponse` async-iterable wrapper with real-time chunk yielding, `.usage` property, and `.collect()` method
- Streaming support in `LLMCascade` with failover on stream creation errors
- Streaming support in `LoggingLLM` with fire-and-forget logging via callbacks
- Streaming demo (`examples/streaming_demo.py`) with real-time output and collect examples

### Fixed

- `claude-haiku-4-5-20251001` config missing `supports_temperature_top_p: false`, causing API errors when both temperature and top_p were sent

## [0.1.6] - 2025-01-31

### Added

- New OpenAI models: `gpt-4o-mini`, `gpt-5-pro`, `o1`, `o3`, `o3-mini`, `o4-mini`
- New Anthropic models: `claude-opus-4-5-20251101`, `claude-haiku-4-5-20251001`, `claude-3-haiku-20240307`
- New Gemini models: `gemini-2.5-pro`, `gemini-3-pro-preview`, `gemini-3-flash-preview`
- Documentation: Basic Usage recipe
- Documentation: Core Concepts section with Structured Outputs, Cost Tracking, and Cascade Failover guides
- Documentation: Expanded homepage with feature overview and quick example
- Documentation: Deprecation automation roadmap (`docs/roadmap/deprecation-automation.md`)

### Changed

- Fixed Anthropic model IDs to use dated snapshots (`claude-3-7-sonnet-20250219`, `claude-3-5-haiku-20241022`) instead of `-latest` aliases
- Organized `llm_config.yaml` with section comments for model families
- Added deprecation comments for Gemini 2.0 models (shutdown March 31, 2026)
- Updated Structured Outputs recipe with comprehensive examples (enums, nested models, constraints)

## [0.1.5] - 2025-01-26

### Added

- Structured response demo (`examples/structured_response_demo.py`) showcasing Pydantic models with enums, nested models, constrained fields, and complex lists
- `inline_schema_refs()` helper to flatten nested JSON schemas by inlining `$defs/$ref` references
- `resolve_api_key()` helper for DRY API key resolution across providers
- `build_schema_prompt()` helper for consistent schema prompt injection
- Shared utilities module (`examples/shared.py`) for common demo functionality

### Changed

- Improved Cohere structured output handling for nested models by flattening schemas
- Refactored provider implementations to use shared helper functions (DRY)
- Moved duplicate `get_json_response()` markdown stripping logic to base class

## [0.1.4] - 2025-01-26

### Added

- API key tracking: `api_key_hash` (SHA256 truncated to 16 chars) and optional `api_key_alias` fields in log entries
- `api_key_alias` parameter to all provider constructors for human-readable key identification
- SQLite adapter (`SqliteAdapter`) for lightweight local development logging
- File storage adapter (`FileStorageAdapter`) for local request/response body storage
- Demo application in `examples/` showcasing multi-provider usage with logging

### Changed

- Updated all database adapter schemas to include `api_key_hash` and `api_key_alias` columns

## [0.1.3] - 2025-01-25

### Added

- Async request logging with `LoggingLLM` wrapper
- PostgreSQL adapter (`PostgresAdapter`) for metrics storage
- MySQL adapter (`MySQLAdapter`) for metrics storage
- S3 adapter (`S3Adapter`) for request/response body storage
- Optional `logging` dependency group: `pip install majordomo-llm[logging]`

## [0.1.2] - 2025-01-25

### Added

- `LLMCascade` class for automatic failover between providers

## [0.1.1] - 2025-01-25

### Added

- DeepSeek provider support (deepseek-chat, deepseek-reasoner models)
- Cohere provider support (Command A, Command R+, Command R, Command R7B models)

### Changed

- Moved LLM configuration from Python dict to external YAML file (llm_config.yaml)
- Added pyyaml as a dependency

## [0.1.0] - 2025-01-25

### Added

- Initial release of majordomo-llm
- Unified interface for multiple LLM providers:
  - OpenAI (GPT-5, GPT-4.1, GPT-4o series)
  - Anthropic (Claude Opus 4, Sonnet 4, Haiku 3.5)
  - Google Gemini (2.0 and 2.5 Flash series)
- Automatic cost tracking for all requests (input/output tokens, USD costs)
- Three response modes:
  - `get_response()` - Plain text responses
  - `get_json_response()` - Parsed JSON responses
  - `get_structured_json_response()` - Pydantic model-validated responses
- Built-in retry logic with exponential backoff (via tenacity)
- Full async/await support for high-performance applications
- Type annotations and py.typed marker for IDE support
- Web search capability for Anthropic Claude models
- Custom exception hierarchy:
  - `MajordomoError` - Base exception
  - `ConfigurationError` - Invalid configuration
  - `ProviderError` - Provider API errors
  - `ResponseParsingError` - Response parsing failures

[Unreleased]: https://github.com/superset-studio/majordomo-llm/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/superset-studio/majordomo-llm/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/superset-studio/majordomo-llm/compare/v0.2.0...v0.3.1
[0.2.0]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/superset-studio/majordomo-llm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/superset-studio/majordomo-llm/releases/tag/v0.1.0
