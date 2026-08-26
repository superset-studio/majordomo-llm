"""Unified Python interface for multiple LLM providers with cost tracking.

majordomo-llm provides a consistent API for interacting with OpenAI, Anthropic,
and Google Gemini models, with automatic retry logic, cost calculation, and
support for structured outputs via Pydantic models.

Example:
    >>> from majordomo_llm import get_llm_instance
    >>> llm = get_llm_instance("anthropic", "claude-sonnet-5")
    >>> response = await llm.get_response("What is the capital of France?")
    >>> print(response.content)
    Paris is the capital of France.
    >>> print(f"Cost: ${response.total_cost:.6f}")
    Cost: $0.000045
"""

from majordomo_llm.base import (
    LLM,
    LLMJSONResponse,
    LLMResponse,
    LLMStreamResponse,
    LLMStructuredResponse,
    Usage,
)
from majordomo_llm.cascade import LLMCascade
from majordomo_llm.exceptions import (
    ConfigurationError,
    EmptyStructuredResponseError,
    MajordomoError,
    ProviderError,
    ResponseParsingError,
    ResponseTruncatedError,
    StructuredOutputUnsupported,
)
from majordomo_llm.factory import (
    LLM_CONFIG,
    ModelPricing,
    clear_aliases,
    get_aliases,
    get_all_llm_instances,
    get_llm_by_alias,
    get_llm_instance,
    get_model_pricing,
    get_supported_models,
    get_supported_providers,
    register_alias,
    unregister_alias,
)
from majordomo_llm.hooks import (
    HookBlocked,
    HookContext,
    HookOutcome,
    HookPipeline,
    HookVerdict,
    LLMHook,
    LLMJudgeHook,
    RegexHook,
)
from majordomo_llm.providers.anthropic import Anthropic
from majordomo_llm.providers.baseten import Baseten
from majordomo_llm.providers.bedrock import Bedrock
from majordomo_llm.providers.bedrock_mantle import BedrockMantle
from majordomo_llm.providers.cohere import Cohere
from majordomo_llm.providers.deepinfra import DeepInfra
from majordomo_llm.providers.deepseek import DeepSeek
from majordomo_llm.providers.fireworks import Fireworks
from majordomo_llm.providers.gemini import Gemini
from majordomo_llm.providers.majordomo import Majordomo
from majordomo_llm.providers.moonshot import Moonshot
from majordomo_llm.providers.nebius import Nebius
from majordomo_llm.providers.novita import Novita
from majordomo_llm.providers.openai import OpenAI
from majordomo_llm.providers.together import Together

__version__ = "0.22.1"

__all__ = [
    # Base classes and types
    "LLM",
    "LLMResponse",
    "LLMStreamResponse",
    "LLMJSONResponse",
    "LLMStructuredResponse",
    "Usage",
    # Exceptions
    "MajordomoError",
    "ConfigurationError",
    "HookBlocked",
    "ProviderError",
    "ResponseParsingError",
    "EmptyStructuredResponseError",
    "ResponseTruncatedError",
    "StructuredOutputUnsupported",
    # Hooks
    "HookContext",
    "HookOutcome",
    "HookPipeline",
    "HookVerdict",
    "LLMHook",
    "LLMJudgeHook",
    "RegexHook",
    # Factory functions
    "get_llm_instance",
    "get_all_llm_instances",
    "get_supported_providers",
    "get_supported_models",
    "get_model_pricing",
    "ModelPricing",
    "LLM_CONFIG",
    # Alias functions
    "get_llm_by_alias",
    "register_alias",
    "unregister_alias",
    "clear_aliases",
    "get_aliases",
    # Cascade
    "LLMCascade",
    # Provider implementations
    "Anthropic",
    "Baseten",
    "Bedrock",
    "BedrockMantle",
    "Cohere",
    "DeepInfra",
    "DeepSeek",
    "Fireworks",
    "Gemini",
    "Majordomo",
    "Moonshot",
    "Nebius",
    "Novita",
    "OpenAI",
    "Together",
    # Version
    "__version__",
]
