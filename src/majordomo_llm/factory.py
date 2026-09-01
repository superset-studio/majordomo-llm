"""Factory functions for creating LLM instances."""

from __future__ import annotations

import importlib.resources
import logging
from collections.abc import Iterator
from typing import Any, NamedTuple, cast

import yaml

from majordomo_llm.base import LLM
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.hooks.image_pipeline import ImageHookPipeline
from majordomo_llm.image import ImageModel
from majordomo_llm.providers.anthropic import Anthropic
from majordomo_llm.providers.baseten import Baseten
from majordomo_llm.providers.bedrock import Bedrock
from majordomo_llm.providers.bedrock_mantle import BedrockMantle
from majordomo_llm.providers.cohere import Cohere
from majordomo_llm.providers.deepinfra import DeepInfra
from majordomo_llm.providers.deepseek import DeepSeek
from majordomo_llm.providers.fireworks import Fireworks
from majordomo_llm.providers.gemini import Gemini
from majordomo_llm.providers.gemini_image import GeminiImage
from majordomo_llm.providers.majordomo import Majordomo
from majordomo_llm.providers.moonshot import Moonshot
from majordomo_llm.providers.nebius import Nebius
from majordomo_llm.providers.novita import Novita
from majordomo_llm.providers.openai import OpenAI
from majordomo_llm.providers.openai_image import OpenAIImage
from majordomo_llm.providers.together import Together

logger = logging.getLogger(__name__)

#: Type for alias targets: single (provider, model) or cascade list.
AliasTarget = tuple[str, str] | list[tuple[str, str]]

#: Mapping of provider name to its implementing class. Module-level so pricing
#: lookups (:func:`get_model_pricing`) can read a provider's cache-accounting
#: mode without instantiating a client.
_PROVIDER_CLASSES: dict[str, type[LLM]] = {
    "openai": OpenAI,
    "anthropic": Anthropic,
    "gemini": Gemini,
    "deepseek": DeepSeek,
    "cohere": Cohere,
    "bedrock": Bedrock,
    "bedrock_mantle": BedrockMantle,
    "fireworks": Fireworks,
    "together": Together,
    "baseten": Baseten,
    "nebius": Nebius,
    "deepinfra": DeepInfra,
    "moonshot": Moonshot,
    "novita": Novita,
    "majordomo": Majordomo,
}

_IMAGE_PROVIDER_CLASSES: dict[str, type[ImageModel]] = {
    "openai": OpenAIImage,
    "gemini": GeminiImage,
}

#: Gateway-routed pseudo-providers. These require a live gateway (``base_url`` +
#: gateway key) and cannot be instantiated standalone, so :func:`get_llm_instance`
#: treats their per-model token costs as optional (cost is resolved per request
#: from the backend the gateway selects) and :func:`get_all_llm_instances` skips
#: them entirely.
_GATEWAY_PROVIDERS = frozenset({"majordomo"})


def _load_llm_config() -> dict[str, dict[str, Any]]:
    """Load LLM configuration from the bundled YAML file."""
    config_file = importlib.resources.files("majordomo_llm").joinpath("llm_config.yaml")
    with config_file.open("r") as f:
        return cast(dict[str, dict[str, Any]], yaml.safe_load(f))


#: Configuration mapping for all supported providers and models.
#: Costs are specified in USD per million tokens.
LLM_CONFIG: dict[str, dict[str, Any]] = _load_llm_config()

#: Image-generation configuration is intentionally removed from LLM_CONFIG so
#: text-provider enumeration and alias validation remain backward compatible.
IMAGE_CONFIG: dict[str, dict[str, Any]] = LLM_CONFIG.pop("image_generation", {})

#: Mapping of deprecated model names to their recommended replacements,
#: keyed by provider.
_DEPRECATED_MODELS: dict[str, dict[str, str]] = LLM_CONFIG.pop("deprecated_models", {})


def _validate_provider_model(provider: str, model: str, alias_name: str) -> None:
    """Validate that a provider/model pair exists in LLM_CONFIG."""
    provider_config = LLM_CONFIG.get(provider)
    if provider_config is None:
        available = ", ".join(LLM_CONFIG.keys())
        raise ConfigurationError(
            f"Alias '{alias_name}' references unknown provider '{provider}'. Available: {available}"
        )
    if model not in provider_config.get("models", {}):
        available = ", ".join(provider_config["models"].keys())
        raise ConfigurationError(
            f"Alias '{alias_name}' references unknown model '{model}' for provider "
            f"'{provider}'. Available: {available}"
        )


def _load_aliases_from_config() -> dict[str, AliasTarget]:
    """Load aliases from the 'aliases' section of LLM_CONFIG.

    Pops the aliases key from LLM_CONFIG so provider iteration stays clean.
    """
    raw_aliases = LLM_CONFIG.pop("aliases", {})
    if not raw_aliases:
        return {}

    registry: dict[str, AliasTarget] = {}
    for alias_name, alias_def in raw_aliases.items():
        if "cascade" in alias_def:
            targets: list[tuple[str, str]] = []
            for entry in alias_def["cascade"]:
                _validate_provider_model(entry["provider"], entry["model"], alias_name)
                targets.append((entry["provider"], entry["model"]))
            if len(targets) < 2:
                raise ConfigurationError(
                    f"Alias '{alias_name}' cascade must have at least 2 providers"
                )
            registry[alias_name] = targets
        elif "provider" in alias_def and "model" in alias_def:
            _validate_provider_model(alias_def["provider"], alias_def["model"], alias_name)
            registry[alias_name] = (alias_def["provider"], alias_def["model"])
        else:
            raise ConfigurationError(
                f"Alias '{alias_name}' must have either 'provider'+'model' or 'cascade' key"
            )
    return registry


_ALIAS_REGISTRY: dict[str, AliasTarget] = _load_aliases_from_config()


def get_supported_providers() -> list[str]:
    """Return a list of all supported provider names.

    Returns:
        A list of provider name strings (e.g., ["openai", "anthropic", ...]).

    Example:
        >>> providers = get_supported_providers()
        >>> "anthropic" in providers
        True
    """
    return list(LLM_CONFIG.keys())


def get_supported_models(provider: str) -> list[str]:
    """Return a list of all supported model names for the given provider.

    Args:
        provider: The LLM provider name (e.g., "openai", "anthropic").

    Returns:
        A list of model identifier strings.

    Raises:
        ConfigurationError: If the provider is not recognized.

    Example:
        >>> models = get_supported_models("anthropic")
        >>> "claude-sonnet-5" in models
        True
    """
    provider_config = LLM_CONFIG.get(provider)
    if provider_config is None:
        available = ", ".join(LLM_CONFIG.keys())
        raise ConfigurationError(f"Unknown LLM provider '{provider}'. Available: {available}")
    return list(provider_config.get("models", {}).keys())


def get_supported_image_providers() -> list[str]:
    """Return providers with configured image-generation models."""
    return list(IMAGE_CONFIG.keys())


def get_supported_image_models(provider: str) -> list[str]:
    """Return image-generation models configured for a provider."""
    provider_config = IMAGE_CONFIG.get(provider)
    if provider_config is None:
        available = ", ".join(IMAGE_CONFIG.keys())
        raise ConfigurationError(f"Unknown image provider '{provider}'. Available: {available}")
    return list(provider_config.get("models", {}).keys())


def get_image_instance(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    api_key_alias: str | None = None,
    hook_pipeline: ImageHookPipeline | None = None,
    base_url: str | None = None,
    default_headers: dict[str, str] | None = None,
) -> ImageModel:
    """Create an image-generation model from bundled configuration."""
    provider_config = IMAGE_CONFIG.get(provider)
    if provider_config is None:
        available = ", ".join(IMAGE_CONFIG.keys())
        raise ConfigurationError(f"Unknown image provider '{provider}'. Available: {available}")
    attributes = provider_config.get("models", {}).get(model)
    if attributes is None:
        available = ", ".join(provider_config.get("models", {}).keys())
        raise ConfigurationError(
            f"Unknown image model '{model}' for provider '{provider}'. Available: {available}"
        )
    cls = _IMAGE_PROVIDER_CLASSES.get(provider)
    if cls is None:
        raise ConfigurationError(f"Image provider '{provider}' has no implementation")

    required_costs = (
        "text_input_cost",
        "image_input_cost",
        "text_output_cost",
        "image_output_cost",
    )
    missing = [name for name in required_costs if name not in attributes]
    if missing:
        raise ConfigurationError(
            f"Image model '{provider}/{model}' is missing pricing fields: {', '.join(missing)}"
        )
    image_cls: Any = cls
    return cast(
        ImageModel,
        image_cls(
            model=model,
            text_input_cost=attributes["text_input_cost"],
            image_input_cost=attributes["image_input_cost"],
            text_output_cost=attributes["text_output_cost"],
            image_output_cost=attributes["image_output_cost"],
            api_key=api_key,
            api_key_alias=api_key_alias,
            hook_pipeline=hook_pipeline,
            base_url=base_url,
            default_headers=default_headers,
        ),
    )


def get_all_image_instances() -> Iterator[ImageModel]:
    """Yield every configured image-generation model."""
    for provider, provider_config in IMAGE_CONFIG.items():
        for model in provider_config.get("models", {}):
            yield get_image_instance(provider, model)


class ModelPricing(NamedTuple):
    """Resolved per-million token rates for a (provider, model) pair.

    Returned by :func:`get_model_pricing` so a request can be priced against a
    backend other than the calling instance — used by the Majordomo provider to
    price a call using the rates of whichever backend the gateway routed to.
    """

    input_cost: float
    output_cost: float
    cached_input_cost: float | None
    cache_write_cost: float | None
    cache_accounting: str


def get_model_pricing(provider: str, model: str) -> ModelPricing | None:
    """Resolve the token pricing for a concrete (provider, model) pair.

    Looks the pair up in ``llm_config.yaml`` and pairs its rates with the
    provider's cache-accounting mode (read from the provider class without
    instantiating a client). Intended for pricing a call after the fact — e.g.
    the Majordomo gateway reports which backend it routed to, and the caller
    prices the usage against that backend's published rates.

    Args:
        provider: The concrete backend provider name (e.g. "fireworks").
        model: The backend's native model identifier (e.g.
            "accounts/fireworks/models/kimi-k3").

    Returns:
        A :class:`ModelPricing` for the pair, or ``None`` if the provider or
        model is not present in the configuration.
    """
    provider_config = LLM_CONFIG.get(provider)
    if provider_config is None:
        return None
    model_attributes = provider_config.get("models", {}).get(model)
    if model_attributes is None:
        return None
    cls = _PROVIDER_CLASSES.get(provider)
    cache_accounting = getattr(cls, "_cache_accounting", "subset") if cls is not None else "subset"
    return ModelPricing(
        input_cost=model_attributes["input_cost"],
        output_cost=model_attributes["output_cost"],
        cached_input_cost=model_attributes.get("cached_input_cost"),
        cache_write_cost=model_attributes.get("cache_write_cost"),
        cache_accounting=cache_accounting,
    )


def get_llm_instance(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    default_headers: dict[str, str] | None = None,
    region: str | None = None,
    use_web_search: bool = False,
    use_prompt_caching: bool | None = None,
) -> LLM:
    """Create an LLM instance for the specified provider and model.

    This is the primary factory function for creating LLM instances. It handles
    provider-specific initialization and configuration lookup.

    Args:
        provider: The LLM provider name. One of: "openai", "anthropic", "gemini",
            "deepseek", "cohere".
        model: The model identifier (e.g., "gpt-4o", "claude-sonnet-5").
        api_key: Optional API key. If not provided, the provider will fall back
            to its respective environment variable.
        base_url: Optional custom base URL for routing through a proxy.
        default_headers: Optional headers sent with every request.
        region: AWS region for the Bedrock provider (e.g., "us-east-1").
            Ignored by other providers. Defaults to ``AWS_REGION`` /
            ``AWS_DEFAULT_REGION`` env vars when not specified.
        use_web_search: Enable the provider's server-side web search tool.
            Validated against the model's ``supports_web_search`` flag in
            ``llm_config.yaml``. Silently ignored for providers that do not
            implement web search (cohere, deepseek, fireworks, together,
            baseten, nebius, deepinfra, moonshot, novita, bedrock_mantle).
        use_prompt_caching: Override the model's ``use_prompt_caching`` config
            default for providers with an explicit cache breakpoint (Anthropic,
            Bedrock Mantle). ``None`` (default) keeps the config value (which
            itself defaults to ``True``); ``True``/``False`` force caching on or
            off. Ignored by providers without explicit cache control.

    Returns:
        An LLM instance configured for the specified provider and model.

    Raises:
        ConfigurationError: If the provider or model is not recognized, or if
            ``use_web_search`` is set on a model whose config does not declare
            ``supports_web_search: true``.

    Example:
        >>> llm = get_llm_instance("anthropic", "claude-sonnet-5")
        >>> response = await llm.get_response("Hello!")
    """
    llm_config_entry = LLM_CONFIG.get(provider)
    if llm_config_entry is None:
        available = ", ".join(LLM_CONFIG.keys())
        raise ConfigurationError(f"Unknown LLM provider '{provider}'. Available: {available}")

    llm_models = llm_config_entry["models"]
    model_attributes = llm_models.get(model)

    # Check if the requested model is deprecated and resolve to its replacement.
    deprecation_warning = None
    requested_model = None
    if model_attributes is None:
        provider_deprecated = _DEPRECATED_MODELS.get(provider, {})
        replacement = provider_deprecated.get(model)
        if replacement is not None:
            deprecation_warning = (
                f"Model '{model}' for provider '{provider}' is deprecated. "
                f"Automatically replaced with '{replacement}'."
            )
            logger.warning(deprecation_warning)
            requested_model = model
            model = replacement
            model_attributes = llm_models.get(model)

    if model_attributes is None:
        available = ", ".join(llm_models.keys())
        raise ConfigurationError(
            f"Unknown model '{model}' for provider '{provider}'. Available: {available}"
        )

    _WEB_SEARCH_PROVIDERS = ("openai", "anthropic", "gemini", "bedrock")
    if (
        use_web_search
        and provider in _WEB_SEARCH_PROVIDERS
        and not model_attributes.get("supports_web_search", False)
    ):
        raise ConfigurationError(
            f"Model '{model}' for provider '{provider}' does not support web search."
        )

    cls = _PROVIDER_CLASSES.get(provider)
    if cls is None:
        raise ConfigurationError(f"Unknown LLM provider '{provider}'")

    provider_kwargs: dict[str, Any] = {}
    if provider in (
        "deepseek",
        "fireworks",
        "together",
        "baseten",
        "nebius",
        "deepinfra",
        "moonshot",
        "novita",
    ):
        provider_kwargs = {
            "reasoning_effort": model_attributes.get("reasoning_effort"),
            "thinking": model_attributes.get("thinking"),
        }
    elif provider in ("bedrock", "bedrock_mantle"):
        provider_kwargs = {"region": region}

    if provider in ("openai", "anthropic", "gemini", "bedrock"):
        provider_kwargs["use_web_search"] = use_web_search

    if provider in ("openai", "anthropic", "gemini"):
        provider_kwargs["supports_image_input"] = model_attributes.get(
            "supports_image_input", False
        )

    # Providers on the shared OpenAI-compatible base default to supporting strict
    # json_schema; a model opts out in config when its deployment accepts the
    # parameter without enforcing it. Fireworks and Together predate the base and
    # do not accept this kwarg, so they are deliberately excluded.
    if provider in ("baseten", "nebius", "deepinfra", "moonshot", "novita"):
        provider_kwargs["supports_structured_outputs"] = model_attributes.get(
            "supports_structured_outputs", True
        )

    if provider in ("anthropic", "bedrock_mantle"):
        provider_kwargs["supports_structured_outputs"] = model_attributes.get(
            "supports_structured_outputs", False
        )
        provider_kwargs["use_prompt_caching"] = (
            use_prompt_caching
            if use_prompt_caching is not None
            else model_attributes.get("use_prompt_caching", True)
        )

    if provider == "anthropic":
        provider_kwargs["reasoning_effort"] = model_attributes.get("reasoning_effort")
        provider_kwargs["thinking"] = model_attributes.get("thinking")

    # Only providers whose API requires an output cap on every request read this.
    # Everywhere else the model's own default applies and the key is meaningless,
    # so it is not forwarded rather than silently accepted.
    if provider in ("anthropic", "bedrock", "bedrock_mantle"):
        provider_kwargs["max_tokens"] = model_attributes.get("max_tokens")

    # An entry may override its API model ID via the ``model`` attribute. This
    # lets the same underlying model be registered under multiple YAML keys —
    # e.g. distinct "reasoning effort" profiles that share one upstream SKU.
    api_model = model_attributes.get("model", model)

    # Gateway-routed providers (majordomo) price each call from the backend the
    # gateway actually selects, so their config entries carry no token costs;
    # the placeholder rates below are never used to price a request.
    if provider in _GATEWAY_PROVIDERS:
        input_cost = model_attributes.get("input_cost", 0.0)
        output_cost = model_attributes.get("output_cost", 0.0)
    else:
        input_cost = model_attributes["input_cost"]
        output_cost = model_attributes["output_cost"]

    llm = cls(
        model=api_model,
        input_cost=input_cost,
        output_cost=output_cost,
        cached_input_cost=model_attributes.get("cached_input_cost"),
        cache_write_cost=model_attributes.get("cache_write_cost"),
        supports_temperature_top_p=model_attributes.get("supports_temperature_top_p", True),
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        **provider_kwargs,
    )

    if deprecation_warning:
        llm.deprecation_warning = deprecation_warning
        llm.requested_model = requested_model

    return llm


def get_all_llm_instances() -> Iterator[LLM]:
    """Create LLM instances for all configured providers and models.

    Yields LLM instances one at a time, which is useful for initialization
    or testing all available models. Gateway-routed pseudo-providers
    (:data:`_GATEWAY_PROVIDERS`) are skipped, as they cannot be instantiated
    without a live gateway ``base_url``.

    Yields:
        LLM instances for each directly-callable provider/model combination.

    Example:
        >>> for llm in get_all_llm_instances():
        ...     print(llm.get_full_model_name())
    """
    for provider, provider_config in LLM_CONFIG.items():
        if provider in _GATEWAY_PROVIDERS:
            continue
        models = provider_config.get("models", {})
        for model in models:
            logger.debug("Creating LLM instance: %s/%s", provider, model)
            yield get_llm_instance(provider, model)


# ---------------------------------------------------------------------------
# Alias functions
# ---------------------------------------------------------------------------


def get_llm_by_alias(
    alias: str,
    *,
    base_url: str | None = None,
    default_headers: dict[str, str] | None = None,
) -> LLM:
    """Create an LLM instance (or cascade) from a registered alias.

    Args:
        alias: The alias name (e.g., "fast", "thinking").
        base_url: Optional custom base URL for routing through a proxy.
        default_headers: Optional headers sent with every request.

    Returns:
        An LLM instance for single-provider aliases, or an LLMCascade for
        cascade aliases.

    Raises:
        ConfigurationError: If the alias is not found.

    Example:
        >>> llm = get_llm_by_alias("fast")
        >>> response = await llm.get_response("Hello!")
    """
    target = _ALIAS_REGISTRY.get(alias)
    if target is None:
        available = ", ".join(sorted(_ALIAS_REGISTRY.keys())) or "(none)"
        raise ConfigurationError(f"Unknown alias '{alias}'. Available: {available}")

    if isinstance(target, list):
        # Lazy import to avoid circular dependency (cascade.py imports from factory.py)
        from majordomo_llm.cascade import LLMCascade

        return LLMCascade(
            target,
            base_url=base_url,
            default_headers=default_headers,
        )
    else:
        provider, model = target
        return get_llm_instance(provider, model, base_url=base_url, default_headers=default_headers)


def register_alias(name: str, target: AliasTarget) -> None:
    """Register or override a model alias.

    Args:
        name: The alias name (e.g., "fast", "thinking").
        target: Either a (provider, model) tuple for a single model,
            or a list of (provider, model) tuples for a cascade.

    Raises:
        ConfigurationError: If a referenced provider or model is not in LLM_CONFIG,
            or if the target format is invalid.

    Example:
        >>> register_alias("fast", ("anthropic", "claude-3-5-haiku-20241022"))
        >>> register_alias("resilient", [
        ...     ("anthropic", "claude-sonnet-5"),
        ...     ("openai", "gpt-4o"),
        ... ])
    """
    if isinstance(target, list):
        if len(target) < 2:
            raise ConfigurationError(f"Cascade alias '{name}' must have at least 2 providers")
        for provider, model in target:
            _validate_provider_model(provider, model, name)
    elif isinstance(target, tuple) and len(target) == 2:
        _validate_provider_model(target[0], target[1], name)
    else:
        raise ConfigurationError(
            f"Alias target must be a (provider, model) tuple or list of tuples, "
            f"got {type(target).__name__}"
        )
    _ALIAS_REGISTRY[name] = target


def unregister_alias(name: str) -> None:
    """Remove a previously registered alias.

    Args:
        name: The alias name to remove.

    Raises:
        ConfigurationError: If the alias does not exist.
    """
    if name not in _ALIAS_REGISTRY:
        available = ", ".join(sorted(_ALIAS_REGISTRY.keys())) or "(none)"
        raise ConfigurationError(f"Unknown alias '{name}'. Available: {available}")
    del _ALIAS_REGISTRY[name]


def clear_aliases() -> None:
    """Remove all registered aliases (both YAML-loaded and programmatic)."""
    _ALIAS_REGISTRY.clear()


def get_aliases() -> dict[str, AliasTarget]:
    """Return a copy of all currently registered aliases."""
    return dict(_ALIAS_REGISTRY)
