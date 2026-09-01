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
    ImageInput,
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
    ImageOptionUnsupported,
    InputModalityUnsupported,
    MajordomoError,
    ProviderError,
    ResponseParsingError,
    ResponseTruncatedError,
    StructuredOutputUnsupported,
)
from majordomo_llm.factory import (
    IMAGE_CONFIG,
    LLM_CONFIG,
    ModelPricing,
    clear_aliases,
    get_aliases,
    get_all_image_instances,
    get_all_llm_instances,
    get_image_instance,
    get_llm_by_alias,
    get_llm_instance,
    get_model_pricing,
    get_supported_image_models,
    get_supported_image_providers,
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
from majordomo_llm.hooks.image_builtin import (
    ImageIntegrityHook,
    ImagePromptRegexHook,
    ImageRequestLimitsHook,
)
from majordomo_llm.hooks.image_pipeline import ImageHookPipeline, ImageHookState
from majordomo_llm.hooks.image_protocol import (
    ImageHook,
    ImageHookOutcome,
    ImageHookRetryRequested,
)
from majordomo_llm.image import (
    GeneratedImage,
    ImageHookRequest,
    ImageModel,
    ImageResponse,
    ImageUsage,
)
from majordomo_llm.image_cascade import ImageCascade
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

__version__ = "0.23.0"

__all__ = [
    # Base classes and types
    "LLM",
    "ImageInput",
    "ImageModel",
    "GeneratedImage",
    "ImageResponse",
    "ImageUsage",
    "ImageHookRequest",
    "ImageCascade",
    "LLMResponse",
    "LLMStreamResponse",
    "LLMJSONResponse",
    "LLMStructuredResponse",
    "Usage",
    # Exceptions
    "MajordomoError",
    "ConfigurationError",
    "InputModalityUnsupported",
    "ImageOptionUnsupported",
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
    "ImageHook",
    "ImageHookOutcome",
    "ImageHookPipeline",
    "ImageHookRetryRequested",
    "ImageHookState",
    "ImageIntegrityHook",
    "ImagePromptRegexHook",
    "ImageRequestLimitsHook",
    # Factory functions
    "get_llm_instance",
    "get_image_instance",
    "get_all_image_instances",
    "get_supported_image_providers",
    "get_supported_image_models",
    "get_all_llm_instances",
    "get_supported_providers",
    "get_supported_models",
    "get_model_pricing",
    "ModelPricing",
    "LLM_CONFIG",
    "IMAGE_CONFIG",
    # Alias functions
    "get_llm_by_alias",
    "register_alias",
    "unregister_alias",
    "clear_aliases",
    "get_aliases",
    # Cascade
    "LLMCascade",
    "ImageCascade",
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
    "GeminiImage",
    "Majordomo",
    "Moonshot",
    "Nebius",
    "Novita",
    "OpenAI",
    "OpenAIImage",
    "Together",
    # Version
    "__version__",
]
