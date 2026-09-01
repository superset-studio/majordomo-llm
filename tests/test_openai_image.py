"""Tests for OpenAI image generation and editing."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from majordomo_llm import ImageInput
from majordomo_llm.exceptions import ImageOptionUnsupported
from majordomo_llm.providers.openai_image import OpenAIImage


@pytest.fixture
def model():
    with patch("majordomo_llm.providers.openai_image.openai.AsyncOpenAI"):
        return OpenAIImage(
            model="gpt-image-2",
            text_input_cost=5.0,
            image_input_cost=8.0,
            text_output_cost=10.0,
            image_output_cost=30.0,
            api_key="test-key",
        )


def image_response():
    response = MagicMock()
    item = MagicMock()
    item.b64_json = base64.b64encode(b"generated-image").decode()
    item.revised_prompt = "revised"
    response.data = [item]
    response.output_format = "jpeg"
    response.usage.input_tokens_details.text_tokens = 10
    response.usage.input_tokens_details.image_tokens = 0
    response.usage.output_tokens_details.text_tokens = 0
    response.usage.output_tokens_details.image_tokens = 100
    response.usage.output_tokens = 100
    return response


async def test_generate_returns_decoded_image_and_usage(model):
    model.client.images.generate = AsyncMock(return_value=image_response())

    response = await model.generate("A lighthouse")

    assert response.images[0].data == b"generated-image"
    assert response.images[0].media_type == "image/jpeg"
    assert response.usage.image_output_tokens == 100
    assert response.total_cost > 0
    assert "response_format" not in model.client.images.generate.call_args.kwargs


async def test_edit_sends_reference_image_without_mask_none(model):
    model.client.images.edit = AsyncMock(return_value=image_response())
    reference = ImageInput(data=b"reference", media_type="image/png")

    await model.edit("Make it blue", (reference,))

    kwargs = model.client.images.edit.call_args.kwargs
    assert kwargs["image"][0][1] == b"reference"
    assert "mask" not in kwargs
    assert "response_format" not in kwargs


async def test_rejects_unrepresentable_openai_size(model):
    with pytest.raises(ImageOptionUnsupported, match="image_size=2K"):
        await model.generate("A lighthouse", image_size="2K")
