"""Tests for the alias feature."""

from unittest.mock import patch

import pytest

from majordomo_llm import (
    LLM_CONFIG,
    clear_aliases,
    get_aliases,
    get_llm_by_alias,
    register_alias,
    unregister_alias,
)
from majordomo_llm.cascade import LLMCascade
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.factory import _ALIAS_REGISTRY
from majordomo_llm.providers.anthropic import Anthropic
from majordomo_llm.providers.gemini import Gemini


@pytest.fixture
def mock_all_clients():
    """Mock all provider API clients and environment variables."""
    env_vars = {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENAI_API_KEY": "test-key",
        "GEMINI_API_KEY": "test-key",
        "DEEPSEEK_API_KEY": "test-key",
        "CO_API_KEY": "test-key",
    }
    with (
        patch.dict("os.environ", env_vars),
        patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"),
        patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"),
        patch("majordomo_llm.providers.gemini.genai.Client"),
        patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"),
        patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2"),
    ):
        yield


@pytest.fixture(autouse=True)
def restore_alias_registry():
    """Snapshot and restore the alias registry around each test."""
    original = dict(_ALIAS_REGISTRY)
    yield
    _ALIAS_REGISTRY.clear()
    _ALIAS_REGISTRY.update(original)


class TestGetLlmByAlias:
    """Tests for get_llm_by_alias function."""

    def test_returns_single_provider_instance(self, mock_all_clients):
        """Should return correct provider instance for a single alias."""
        llm = get_llm_by_alias("fast")

        assert isinstance(llm, Anthropic)
        assert llm.model == "claude-haiku-4-5-20251001"

    def test_returns_cascade_for_cascade_alias(self, mock_all_clients):
        """Should return LLMCascade for a cascade alias."""
        llm = get_llm_by_alias("resilient-sonnet")

        assert isinstance(llm, LLMCascade)
        assert len(llm.llms) == 3

    def test_raises_for_unknown_alias(self):
        """Should raise ConfigurationError for unknown alias."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_by_alias("nonexistent")

        assert "Unknown alias" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_passes_base_url_to_single_provider(self, mock_all_clients):
        """Should forward base_url to the created instance."""
        llm = get_llm_by_alias("cheap", base_url="https://proxy.example.com")

        assert isinstance(llm, Gemini)
        assert llm.base_url == "https://proxy.example.com"

    def test_passes_default_headers_to_single_provider(self, mock_all_clients):
        """Should forward default_headers to the created instance."""
        headers = {"X-Custom": "value"}
        llm = get_llm_by_alias("fast", default_headers=headers)

        assert isinstance(llm, Anthropic)
        assert llm.default_headers == headers

    def test_resolves_programmatic_alias(self, mock_all_clients):
        """Should resolve aliases registered via register_alias."""
        register_alias("custom", ("gemini", "gemini-2.5-flash"))
        llm = get_llm_by_alias("custom")

        assert isinstance(llm, Gemini)
        assert llm.model == "gemini-2.5-flash"

    def test_available_aliases_listed_on_error(self):
        """Error message should list available aliases."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_by_alias("nope")

        error_msg = str(exc_info.value)
        assert "fast" in error_msg
        assert "thinking" in error_msg

    def test_empty_registry_shows_none(self, mock_all_clients):
        """Should show '(none)' when no aliases are registered."""
        clear_aliases()
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm_by_alias("anything")

        assert "(none)" in str(exc_info.value)


class TestRegisterAlias:
    """Tests for register_alias function."""

    def test_registers_single_provider(self, mock_all_clients):
        """Should register a single provider alias."""
        register_alias("myalias", ("anthropic", "claude-sonnet-5"))

        llm = get_llm_by_alias("myalias")
        assert isinstance(llm, Anthropic)
        assert llm.model == "claude-sonnet-5"

    def test_registers_cascade(self, mock_all_clients):
        """Should register a cascade alias."""
        register_alias("myfallback", [
            ("anthropic", "claude-sonnet-5"),
            ("openai", "gpt-4.1"),
        ])

        llm = get_llm_by_alias("myfallback")
        assert isinstance(llm, LLMCascade)
        assert len(llm.llms) == 2

    def test_overrides_existing_alias(self, mock_all_clients):
        """Should replace an existing alias when re-registered."""
        register_alias("fast", ("gemini", "gemini-2.5-flash"))

        llm = get_llm_by_alias("fast")
        assert isinstance(llm, Gemini)
        assert llm.model == "gemini-2.5-flash"

    def test_raises_for_unknown_provider(self):
        """Should raise ConfigurationError for unknown provider."""
        with pytest.raises(ConfigurationError) as exc_info:
            register_alias("bad", ("fake_provider", "some-model"))

        assert "unknown provider" in str(exc_info.value)

    def test_raises_for_unknown_model(self):
        """Should raise ConfigurationError for unknown model."""
        with pytest.raises(ConfigurationError) as exc_info:
            register_alias("bad", ("anthropic", "fake-model"))

        assert "unknown model" in str(exc_info.value)

    def test_raises_for_cascade_with_one_provider(self):
        """Cascade must have at least 2 providers."""
        with pytest.raises(ConfigurationError) as exc_info:
            register_alias("bad", [("anthropic", "claude-sonnet-5")])

        assert "at least 2" in str(exc_info.value)

    def test_raises_for_invalid_target_type(self):
        """Should raise ConfigurationError for invalid target type."""
        with pytest.raises(ConfigurationError):
            register_alias("bad", "not-a-tuple")  # type: ignore[arg-type]


class TestUnregisterAlias:
    """Tests for unregister_alias function."""

    def test_removes_existing_alias(self, mock_all_clients):
        """Should remove the alias so it can no longer be resolved."""
        unregister_alias("fast")

        with pytest.raises(ConfigurationError):
            get_llm_by_alias("fast")

    def test_raises_for_nonexistent_alias(self):
        """Should raise ConfigurationError for unknown alias."""
        with pytest.raises(ConfigurationError) as exc_info:
            unregister_alias("nonexistent")

        assert "Unknown alias" in str(exc_info.value)


class TestClearAliases:
    """Tests for clear_aliases function."""

    def test_removes_all_aliases(self):
        """Should empty the entire alias registry."""
        clear_aliases()

        assert get_aliases() == {}

    def test_clear_is_idempotent(self):
        """Calling clear_aliases twice should not raise."""
        clear_aliases()
        clear_aliases()
        assert get_aliases() == {}


class TestGetAliases:
    """Tests for get_aliases function."""

    def test_returns_yaml_loaded_aliases(self):
        """Should include aliases loaded from YAML config."""
        aliases = get_aliases()

        assert "fast" in aliases
        assert "thinking" in aliases
        assert "smart" in aliases
        assert "cheap" in aliases
        assert "resilient-sonnet" in aliases

    def test_includes_programmatic_aliases(self):
        """Should include aliases added via register_alias."""
        register_alias("custom", ("anthropic", "claude-sonnet-5"))

        aliases = get_aliases()
        assert "custom" in aliases
        assert aliases["custom"] == ("anthropic", "claude-sonnet-5")

    def test_returns_copy_not_reference(self):
        """Modifying the returned dict should not affect the registry."""
        aliases = get_aliases()
        aliases["injected"] = ("openai", "gpt-4.1")

        assert "injected" not in get_aliases()


class TestAliasYamlLoading:
    """Tests for YAML alias loading at module init."""

    def test_aliases_not_in_llm_config(self):
        """aliases key should be popped from LLM_CONFIG after loading."""
        assert "aliases" not in LLM_CONFIG

    def test_deprecated_models_not_in_llm_config(self):
        """deprecated_models key should be popped from LLM_CONFIG after loading."""
        assert "deprecated_models" not in LLM_CONFIG

    def test_llm_config_still_has_providers(self):
        """LLM_CONFIG should still contain all provider entries."""
        assert "openai" in LLM_CONFIG
        assert "anthropic" in LLM_CONFIG
        assert "gemini" in LLM_CONFIG
        assert "deepseek" in LLM_CONFIG
        assert "cohere" in LLM_CONFIG

    def test_single_alias_target_is_tuple(self):
        """Single-provider aliases should be stored as tuples."""
        aliases = get_aliases()
        assert aliases["fast"] == ("anthropic", "claude-haiku-4-5-20251001")

    def test_cascade_alias_target_is_list_of_tuples(self):
        """Cascade aliases should be stored as lists of tuples."""
        aliases = get_aliases()
        target = aliases["resilient-sonnet"]
        assert isinstance(target, list)
        assert target == [
            ("anthropic", "claude-sonnet-5"),
            ("openai", "gpt-5.6-terra"),
            ("gemini", "gemini-2.5-pro"),
        ]
