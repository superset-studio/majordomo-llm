"""Tests for image inputs on the existing LLM response APIs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from majordomo_llm import ImageInput
from majordomo_llm.base import LLM, LLMResponse
from majordomo_llm.cascade import LLMCascade
from majordomo_llm.exceptions import InputModalityUnsupported
from majordomo_llm.logging.wrapper import _image_metadata
from majordomo_llm.providers.anthropic import Anthropic
from majordomo_llm.providers.gemini import Gemini
from majordomo_llm.providers.openai import OpenAI


class TextOnlyLLM(LLM):
    async def _get_response_impl(self, *args, **kwargs):
        raise AssertionError("provider should not be called")

    async def _get_response_stream_impl(self, *args, **kwargs):
        raise AssertionError("provider should not be called")


class VisionLLM(TextOnlyLLM):
    def __init__(self):
        super().__init__(
            provider="vision",
            model="vision-model",
            input_cost=0,
            output_cost=0,
            supports_image_input=True,
        )

    async def _get_response_with_images_impl(self, *args, **kwargs):
        return LLMResponse(
            content="vision response",
            input_tokens=1,
            output_tokens=1,
            cached_tokens=0,
            input_cost=0,
            output_cost=0,
            total_cost=0,
            response_time=0,
        )


def test_image_input_is_rejected_before_call_for_text_only_model():
    llm = TextOnlyLLM(provider="test", model="text-only", input_cost=0, output_cost=0)
    image = ImageInput(data=b"image", media_type="image/png")

    with pytest.raises(InputModalityUnsupported, match="text-only"):
        llm._validate_images((image,))


async def test_cascade_skips_text_only_member_for_image_request():
    text_only = TextOnlyLLM(provider="text", model="text-model", input_cost=0, output_cost=0)
    vision = VisionLLM()
    with patch("majordomo_llm.cascade.get_llm_instance", side_effect=[text_only, vision]):
        cascade = LLMCascade([("text", "text-model"), ("vision", "vision-model")])

    response = await cascade.get_response("Describe", images=(ImageInput(b"image", "image/png"),))

    assert response.content == "vision response"


def test_logging_records_image_metadata_without_bytes():
    metadata = _image_metadata((ImageInput(b"private-image", "image/png"),))

    assert metadata[0]["media_type"] == "image/png"
    assert metadata[0]["size_bytes"] == len(b"private-image")
    assert metadata[0]["sha256"] != "private-image"
    assert b"private-image" not in repr(metadata).encode()


async def test_openai_sends_data_url_image_block():
    with patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"):
        llm = OpenAI(
            model="gpt-4.1",
            input_cost=2,
            output_cost=8,
            api_key="test-key",
            supports_image_input=True,
        )
    response = MagicMock()
    response.output_text = "A lighthouse"
    response.usage.input_tokens = 20
    response.usage.output_tokens = 3
    response.usage.input_tokens_details.cached_tokens = 0
    llm.client.responses.create = AsyncMock(return_value=response)

    await llm.get_response("What is shown?", images=(ImageInput(b"image", "image/png"),))

    request_input = llm.client.responses.create.call_args.kwargs["input"]
    content = request_input[0]["content"]
    assert content[0]["type"] == "input_image"
    assert content[0]["image_url"].startswith("data:image/png;base64,")
    assert content[-1] == {"type": "input_text", "text": "What is shown?"}


async def test_openai_structured_response_keeps_image_input():
    with patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"):
        llm = OpenAI(
            model="gpt-4.1",
            input_cost=2,
            output_cost=8,
            api_key="test-key",
            supports_image_input=True,
        )
    response = MagicMock()
    response.output_text = '{"label":"lighthouse"}'
    response.usage.input_tokens = 20
    response.usage.output_tokens = 5
    response.usage.input_tokens_details.cached_tokens = 0
    llm.client.responses.create = AsyncMock(return_value=response)

    result = await llm.get_json_schema_response(
        "Classify this image",
        {"type": "object", "properties": {"label": {"type": "string"}}},
        images=(ImageInput(b"image", "image/png"),),
    )

    assert result.content == '{"label":"lighthouse"}'
    request_input = llm.client.responses.create.call_args.kwargs["input"]
    assert request_input[0]["content"][0]["type"] == "input_image"


async def test_anthropic_sends_image_before_text_block():
    with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        llm = Anthropic(
            model="claude-sonnet-5",
            input_cost=3,
            output_cost=15,
            api_key="test-key",
            supports_image_input=True,
        )
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="A lighthouse")],
        usage=SimpleNamespace(
            input_tokens=20,
            output_tokens=3,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        stop_reason="end_turn",
    )
    llm.client.messages.create = AsyncMock(return_value=response)

    await llm.get_response("What is shown?", images=(ImageInput(b"image", "image/jpeg"),))

    content = llm.client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[-1] == {"type": "text", "text": "What is shown?"}


async def test_gemini_sends_inline_image_part():
    with patch("majordomo_llm.providers.gemini.genai.Client"):
        llm = Gemini(
            model="gemini-2.5-flash",
            input_cost=0.3,
            output_cost=2.5,
            api_key="test-key",
            supports_image_input=True,
        )
    response = SimpleNamespace(
        text="A lighthouse",
        usage_metadata=SimpleNamespace(
            prompt_token_count=20,
            candidates_token_count=3,
            cached_content_token_count=0,
        ),
        candidates=[],
    )
    llm.client.aio.models.generate_content = AsyncMock(return_value=response)

    await llm.get_response("What is shown?", images=(ImageInput(b"image", "image/webp"),))

    contents = llm.client.aio.models.generate_content.call_args.kwargs["contents"]
    assert contents[0].inline_data.data == b"image"
    assert contents[0].inline_data.mime_type == "image/webp"
    assert contents[-1] == "What is shown?"
