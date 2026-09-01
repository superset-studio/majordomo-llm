"""LLM provider implementations.

This module exports all available provider classes for direct instantiation.

Example:
    >>> from majordomo_llm.providers import Anthropic, Cohere, OpenAI, Gemini, DeepSeek
    >>> llm = Anthropic(model="claude-sonnet-5", input_cost=3.0, output_cost=15.0)
"""

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

__all__ = [
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
]
