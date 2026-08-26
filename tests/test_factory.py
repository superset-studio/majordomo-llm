"""Tests for the factory module."""

from unittest.mock import patch

import pytest

from majordomo_llm import (
    LLM_CONFIG,
    get_aliases,
    get_all_llm_instances,
    get_llm_instance,
    get_supported_providers,
)
from majordomo_llm.base import DEFAULT_STREAM_MAX_TOKENS
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers.anthropic import Anthropic
from majordomo_llm.providers.deepseek import DeepSeek
from majordomo_llm.providers.gemini import Gemini
from majordomo_llm.providers.openai import OpenAI


@pytest.fixture
def mock_all_clients():
    """Mock all provider API clients and environment variables."""
    env_vars = {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENAI_API_KEY": "test-key",
        "GEMINI_API_KEY": "test-key",
        "DEEPSEEK_API_KEY": "test-key",
        "CO_API_KEY": "test-key",
        "AWS_BEARER_TOKEN_BEDROCK": "test-key",
        "AWS_REGION": "us-east-1",
        "FIREWORKS_API_KEY": "test-key",
        "TOGETHER_API_KEY": "test-key",
        "BASETEN_API_KEY": "test-key",
        "NEBIUS_API_KEY": "test-key",
        "DEEPINFRA_API_KEY": "test-key",
        "MOONSHOT_API_KEY": "test-key",
        "NOVITA_API_KEY": "test-key",
    }
    with (
        patch.dict("os.environ", env_vars),
        patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"),
        patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"),
        patch("majordomo_llm.providers.gemini.genai.Client"),
        patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"),
        patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2"),
        patch("majordomo_llm.providers.fireworks.openai.AsyncOpenAI"),
        patch("majordomo_llm.providers.together.openai.AsyncOpenAI"),
        # Baseten, Nebius, DeepInfra, Moonshot and Novita all build their client
        # in the shared OpenAI-compatible base, so one patch covers them all.
        patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"),
    ):
        yield


class TestGetLLMInstance:
    """Tests for get_llm_instance factory function."""

    def test_creates_anthropic_provider(self, mock_all_clients):
        """Should create Anthropic instance for anthropic provider."""
        llm = get_llm_instance("anthropic", "claude-sonnet-5")

        assert isinstance(llm, Anthropic)
        assert llm.provider == "anthropic"
        assert llm.model == "claude-sonnet-5"

    def test_creates_openai_provider(self, mock_all_clients):
        """Should create OpenAI instance for openai provider."""
        llm = get_llm_instance("openai", "gpt-4.1")

        assert isinstance(llm, OpenAI)
        assert llm.provider == "openai"
        assert llm.model == "gpt-4.1"

    def test_creates_gemini_provider(self, mock_all_clients):
        """Should create Gemini instance for gemini provider."""
        llm = get_llm_instance("gemini", "gemini-2.5-flash")

        assert isinstance(llm, Gemini)
        assert llm.provider == "gemini"
        assert llm.model == "gemini-2.5-flash"

    def test_creates_deepseek_v4_with_reasoning_options(self, mock_all_clients):
        """Should create DeepSeek V4 models with configured reasoning options."""
        llm = get_llm_instance("deepseek", "deepseek-v4-pro")

        assert isinstance(llm, DeepSeek)
        assert llm.provider == "deepseek"
        assert llm.model == "deepseek-v4-pro"
        assert llm.input_cost == 0.435
        assert llm.output_cost == 0.87
        assert llm.reasoning_effort == "medium"
        assert llm.thinking is None

    def test_sets_correct_costs_from_config(self, mock_all_clients):
        """Should set input/output costs from LLM_CONFIG."""
        llm = get_llm_instance("anthropic", "claude-sonnet-5")

        expected_config = LLM_CONFIG["anthropic"]["models"]["claude-sonnet-5"]
        assert llm.input_cost == expected_config["input_cost"]
        assert llm.output_cost == expected_config["output_cost"]

    def test_yaml_model_override_resolves_to_upstream_id(self, mock_all_clients):
        """Profile entries with ``model:`` override should pass the upstream ID
        to the provider while keeping the YAML key as the lookup name."""
        llm = get_llm_instance("fireworks", "deepseek-v4-pro-reasoning")
        assert llm.model == "accounts/fireworks/models/deepseek-v4-pro"
        assert llm.reasoning_effort == "medium"
        assert llm.thinking == "enabled"

        llm = get_llm_instance("together", "deepseek-v4-pro-hard")
        assert llm.model == "deepseek-ai/DeepSeek-V4-Pro"
        assert llm.reasoning_effort == "high"
        assert llm.thinking == "enabled"

    def test_sets_supports_temperature_top_p_flag(self, mock_all_clients):
        """Should set supports_temperature_top_p from config."""
        # Model with flag set to False
        llm = get_llm_instance("anthropic", "claude-sonnet-4-5-20250929")
        assert llm.supports_temperature_top_p is False

        # Model without flag (defaults to True)
        llm = get_llm_instance("anthropic", "claude-opus-4-5-20251101")
        assert llm.supports_temperature_top_p is True

    def test_raises_for_unknown_provider(self):
        """Should raise ConfigurationError for unknown provider."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_instance("unknown_provider", "some-model")

        assert "Unknown LLM provider" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)

    def test_raises_for_unknown_model(self):
        """Should raise ConfigurationError for unknown model."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_instance("anthropic", "unknown-model")

        assert "Unknown model" in str(exc_info.value)
        assert "unknown-model" in str(exc_info.value)

    def test_replaces_deprecated_model_with_warning(self, mock_all_clients):
        """Should auto-replace a deprecated model and log a warning."""
        llm = get_llm_instance("openai", "gpt-4o")

        assert isinstance(llm, OpenAI)
        assert llm.model == "gpt-4.1"

    def test_replaces_deprecated_anthropic_model(self, mock_all_clients):
        """Should auto-replace deprecated Anthropic models."""
        llm = get_llm_instance("anthropic", "claude-3-5-haiku-20241022")

        assert isinstance(llm, Anthropic)
        assert llm.model == "claude-haiku-4-5-20251001"

    def test_deprecated_model_sets_warning_and_requested_model(self, mock_all_clients):
        """Should set deprecation_warning and requested_model on the LLM instance."""
        llm = get_llm_instance("openai", "gpt-4o")

        assert llm.requested_model == "gpt-4o"
        assert llm.deprecation_warning is not None
        assert "gpt-4o" in llm.deprecation_warning
        assert "gpt-4.1" in llm.deprecation_warning

    def test_non_deprecated_model_has_no_warning(self, mock_all_clients):
        """Should not set deprecation info for active models."""
        llm = get_llm_instance("openai", "gpt-4.1")

        assert llm.deprecation_warning is None
        assert llm.requested_model is None


class TestGetAllLLMInstances:
    """Tests for get_all_llm_instances function."""

    def test_yields_instances_for_all_configured_models(self, mock_all_clients):
        """Should yield an LLM instance for each directly-callable model."""
        instances = list(get_all_llm_instances())

        # Count expected models, excluding gateway-only providers (majordomo),
        # which are not directly instantiable and are skipped by enumeration.
        expected_count = sum(
            len(provider_config["models"])
            for provider, provider_config in LLM_CONFIG.items()
            if provider != "majordomo"
        )

        assert len(instances) == expected_count

    def test_yields_correct_provider_types(self, mock_all_clients):
        """Should yield correct provider types."""
        instances = list(get_all_llm_instances())

        providers = {llm.provider for llm in instances}
        assert providers == {
            "openai",
            "anthropic",
            "gemini",
            "deepseek",
            "cohere",
            "bedrock",
            "bedrock_mantle",
            "fireworks",
            "together",
            "baseten",
            "nebius",
            "deepinfra",
            "moonshot",
            "novita",
        }


class TestLLMConfig:
    """Tests for LLM_CONFIG structure."""

    def test_all_providers_have_models(self):
        """Each provider should have at least one model configured."""
        for provider, config in LLM_CONFIG.items():
            assert "models" in config, f"{provider} missing 'models' key"
            assert len(config["models"]) > 0, f"{provider} has no models"

    def test_all_models_have_required_costs(self):
        """Each model should have input_cost and output_cost.

        Gateway-only providers (majordomo) are exempt: their cost is resolved
        per request from the backend the gateway routes to, so their config
        entries intentionally carry no token costs.
        """
        for provider, config in LLM_CONFIG.items():
            if provider == "majordomo":
                continue
            for model, model_config in config["models"].items():
                assert "input_cost" in model_config, f"{provider}/{model} missing input_cost"
                assert "output_cost" in model_config, f"{provider}/{model} missing output_cost"
                assert model_config["input_cost"] >= 0, f"{provider}/{model} invalid input_cost"
                assert model_config["output_cost"] >= 0, f"{provider}/{model} invalid output_cost"


class TestUseWebSearchForwarding:
    """Tests for use_web_search forwarding from the factory."""

    def test_forwards_to_supported_provider(self, mock_all_clients):
        llm = get_llm_instance("anthropic", "claude-sonnet-4-6", use_web_search=True)
        assert llm.use_web_search is True

    def test_default_is_false(self, mock_all_clients):
        llm = get_llm_instance("anthropic", "claude-sonnet-4-6")
        assert llm.use_web_search is False

    def test_raises_on_unsupported_model(self, mock_all_clients):
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_instance("openai", "gpt-5-nano", use_web_search=True)
        assert "does not support web search" in str(exc_info.value)

    def test_silently_ignored_for_unsupported_provider(self, mock_all_clients):
        # Cohere does not implement web search. The factory should accept
        # use_web_search=True without raising and not forward the flag — the
        # resulting instance should still report use_web_search=False.
        llm = get_llm_instance("cohere", "command-a-03-2025", use_web_search=True)
        assert llm.use_web_search is False


class TestUsePromptCachingForwarding:
    """Tests for use_prompt_caching forwarding/override from the factory."""

    def test_default_is_true_for_anthropic(self, mock_all_clients):
        llm = get_llm_instance("anthropic", "claude-sonnet-4-6")
        assert llm.use_prompt_caching is True

    def test_override_disables_caching(self, mock_all_clients):
        llm = get_llm_instance(
            "anthropic", "claude-sonnet-4-6", use_prompt_caching=False
        )
        assert llm.use_prompt_caching is False

    def test_override_forwarded_to_bedrock_mantle(self, mock_all_clients):
        llm = get_llm_instance(
            "bedrock_mantle",
            "anthropic.claude-haiku-4-5",
            use_prompt_caching=False,
            region="us-east-1",
        )
        assert llm.use_prompt_caching is False

    def test_cache_costs_loaded_from_config(self, mock_all_clients):
        llm = get_llm_instance("anthropic", "claude-sonnet-4-6")
        assert llm.cached_input_cost == 0.30
        assert llm.cache_write_cost == 3.75


class TestStructuredOutputForwarding:
    """Tests for supports_structured_outputs on OpenAI-compatible providers."""

    def test_defaults_to_true(self, mock_all_clients):
        llm = get_llm_instance("nebius", "moonshotai/Kimi-K3")
        assert llm.supports_structured_outputs is True

    def test_config_can_opt_out(self, mock_all_clients):
        # Nebius serves Kimi-K2.6 without grammar-constrained decoding: it accepts
        # json_schema but only honors it intermittently.
        llm = get_llm_instance("nebius", "moonshotai/Kimi-K2.6")
        assert llm.supports_structured_outputs is False

    def test_other_providers_unaffected(self, mock_all_clients):
        for provider, model in (
            ("baseten", "moonshotai/Kimi-K2.6"),
            ("deepinfra", "moonshotai/Kimi-K2.6"),
            ("novita", "moonshotai/kimi-k2.6"),
            ("moonshot", "kimi-k2.6"),
        ):
            llm = get_llm_instance(provider, model)
            assert llm.supports_structured_outputs is True, provider


class TestGatewayProviderIsOptIn:
    """The majordomo provider must never be reached by default.

    It is a routing pseudo-provider: it names a canonical model and lets the
    gateway pick a backend, so it cannot run without a live Steward instance.
    Anything that enumerates providers has to skip it, and it must fail loudly
    rather than silently when reached without a gateway URL.

    This is distinct from routing a CONCRETE provider through the gateway for
    usage tracking, which is just a base_url plus headers and stays available.
    """

    def test_excluded_from_sweeps(self, mock_all_clients):
        providers = {llm.provider for llm in get_all_llm_instances()}
        assert "majordomo" not in providers

    def test_still_listed_as_supported(self):
        # Excluded from sweeps, but callable on purpose — it must stay
        # discoverable so an opt-in caller can find it.
        assert "majordomo" in get_supported_providers()

    def test_requires_explicit_gateway_url(self, mock_all_clients):
        with pytest.raises(ConfigurationError, match="requires base_url"):
            get_llm_instance("majordomo", "glm-5.2")

    def test_no_alias_resolves_to_it(self):
        for name, target in get_aliases().items():
            hops = target if isinstance(target, list) else [target]
            for provider, _ in hops:
                assert provider != "majordomo", (
                    f"alias {name!r} would route to the gateway provider by default"
                )


class TestMaxTokensForwarding:
    """max_tokens is pinned only where a model cannot take the library default.

    The defaults are 16000 non-streaming / 64000 streaming. A model whose real
    ceiling is at or above those inherits them and must NOT pin a value: pinning
    the vendor ceiling instead makes it the per-request default, and on Anthropic
    anything over 21333 is rejected by the SDK before the request is sent.
    """

    def test_anthropic_models_pin_nothing(self, mock_all_clients):
        """Every Anthropic ceiling is >= the streaming default, so none pins."""
        for model in LLM_CONFIG["anthropic"]["models"]:
            assert get_llm_instance("anthropic", model).max_tokens is None, model

    def test_bedrock_mantle_pins_nothing(self, mock_all_clients):
        for model in LLM_CONFIG["bedrock_mantle"]["models"]:
            llm = get_llm_instance("bedrock_mantle", model, region="us-east-1")
            assert llm.max_tokens is None, model

    def test_bedrock_pins_only_the_low_ceiling_models(self, mock_all_clients):
        """Llama 4 (8192) and DeepSeek-R1 (32768) sit below the 64000 stream default."""
        pinned = {
            m: a["max_tokens"]
            for m, a in LLM_CONFIG["bedrock"]["models"].items()
            if "max_tokens" in a
        }
        assert pinned == {
            "us.meta.llama4-maverick-17b-instruct-v1:0": 8192,
            "us.meta.llama4-scout-17b-instruct-v1:0": 8192,
            "us.deepseek.r1-v1:0": 32768,
        }

    def test_pinned_value_reaches_the_instance(self, mock_all_clients):
        llm = get_llm_instance("bedrock", "us.deepseek.r1-v1:0", region="us-east-1")
        assert llm.max_tokens == 32768

    def test_no_pin_exceeds_the_streaming_default(self):
        """A pin above the default is pointless; above 21333 it breaks Anthropic."""
        for provider in ("anthropic", "bedrock", "bedrock_mantle"):
            for model, attrs in LLM_CONFIG[provider]["models"].items():
                pinned = attrs.get("max_tokens")
                if pinned is not None:
                    assert pinned < DEFAULT_STREAM_MAX_TOKENS, f"{provider}/{model}"

    def test_not_forwarded_to_providers_that_ignore_it(self, mock_all_clients):
        """Providers that send no cap inherit the model default, not ours."""
        assert get_llm_instance("openai", "gpt-4.1").max_tokens is None
        assert get_llm_instance("gemini", "gemini-2.5-flash").max_tokens is None


class TestClaude4FamilyDeprecation:
    """The Claude 4 family 404s on the Messages API and was retired 2026-08-25."""

    RETIRED = {
        "claude-opus-4-1-20250805": "claude-opus-5",
        "claude-opus-4-20250514": "claude-opus-5",
        "claude-sonnet-4-20250514": "claude-sonnet-5",
    }

    def test_removed_from_the_models_block(self):
        """The factory only consults deprecated_models when a model is absent."""
        models = LLM_CONFIG["anthropic"]["models"]
        assert [m for m in self.RETIRED if m in models] == []

    @pytest.mark.parametrize("retired,replacement", sorted(RETIRED.items()))
    def test_resolves_to_replacement(self, retired, replacement, mock_all_clients):
        llm = get_llm_instance("anthropic", retired)
        assert llm.model == replacement
        assert llm.requested_model == retired
        assert llm.deprecation_warning is not None
        assert retired in llm.deprecation_warning

    @pytest.mark.parametrize("retired", sorted(RETIRED))
    def test_replacement_pins_nothing(self, retired, mock_all_clients):
        """The replacement inherits the library defaults, like every Anthropic model."""
        assert get_llm_instance("anthropic", retired).max_tokens is None

    def test_no_alias_still_points_at_a_retired_model(self):
        """Alias validation resolves against the models block only, not
        deprecated_models — a YAML alias naming a retired model raises
        ConfigurationError at import and takes the whole package down with it."""
        for name, target in get_aliases().items():
            members = target if isinstance(target, list) else [target]
            for _provider, model in members:
                assert model not in self.RETIRED, (
                    f"alias {name!r} still targets retired model {model!r}"
                )
