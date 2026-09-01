"""Tests for image-generation factory discovery."""

from unittest.mock import patch

import pytest

from majordomo_llm import (
    LLM_CONFIG,
    get_image_instance,
    get_llm_instance,
    get_supported_image_models,
    get_supported_image_providers,
)
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.hooks import ImageHookPipeline, ImageRequestLimitsHook
from majordomo_llm.providers.gemini_image import GeminiImage
from majordomo_llm.providers.openai_image import OpenAIImage


def test_image_discovery_is_separate_from_text_models():
    assert get_supported_image_providers() == ["openai", "gemini"]
    assert get_supported_image_models("openai") == ["gpt-image-2"]
    assert "gpt-image-2" not in LLM_CONFIG["openai"]["models"]


def test_creates_openai_image_model_with_configured_pricing():
    with patch("majordomo_llm.providers.openai_image.openai.AsyncOpenAI"):
        model = get_image_instance(
            "openai", "gpt-image-2", api_key="test-key", api_key_alias="primary"
        )
    assert isinstance(model, OpenAIImage)
    assert model.image_output_cost == 30.0
    assert model.api_key_hash is not None
    assert model.api_key_alias == "primary"


def test_creates_gemini_image_model():
    with patch("majordomo_llm.providers.gemini_image.genai.Client"):
        model = get_image_instance("gemini", "gemini-3.1-flash-image", api_key="test-key")
    assert isinstance(model, GeminiImage)
    assert model.image_output_cost == 60.0


def test_rejects_unknown_image_provider_and_model():
    with pytest.raises(ConfigurationError, match="Unknown image provider"):
        get_image_instance("anthropic", "claude-image", api_key="test-key")
    with pytest.raises(ConfigurationError, match="Unknown image model"):
        get_image_instance("openai", "missing", api_key="test-key")


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_image_input_capability_is_explicit_on_supported_models(provider):
    provider_config = LLM_CONFIG[provider]
    assert "supports_image_input" not in provider_config
    assert all(
        model_config.get("supports_image_input") is True
        for model_config in provider_config["models"].values()
    )


def test_image_input_capability_defaults_to_false_when_model_flag_is_absent(monkeypatch):
    model_config = LLM_CONFIG["openai"]["models"]["gpt-4.1"]
    monkeypatch.delitem(model_config, "supports_image_input")

    with patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"):
        model = get_llm_instance("openai", "gpt-4.1", api_key="test-key")

    assert model.supports_image_input is False


def test_factory_forwards_true_model_image_input_capability():
    with patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"):
        model = get_llm_instance("openai", "gpt-4.1", api_key="test-key")

    assert model.supports_image_input is True


def test_image_factory_forwards_hook_pipeline():
    pipeline = ImageHookPipeline([ImageRequestLimitsHook("limits")])
    with patch("majordomo_llm.providers.openai_image.openai.AsyncOpenAI"):
        model = get_image_instance(
            "openai", "gpt-image-2", api_key="test-key", hook_pipeline=pipeline
        )

    assert model.hook_pipeline is pipeline
