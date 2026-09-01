"""Tests for Gemini image generation and editing."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from majordomo_llm import ImageInput
from majordomo_llm.exceptions import ImageOptionUnsupported, ProviderError
from majordomo_llm.providers.gemini_image import GeminiImage


@pytest.fixture
def model():
    with patch("majordomo_llm.providers.gemini_image.genai.Client"):
        return GeminiImage(
            model="gemini-3.1-flash-image",
            text_input_cost=0.5,
            image_input_cost=0.5,
            text_output_cost=3.0,
            image_output_cost=60.0,
            api_key="test-key",
        )


def interaction_response():
    image = SimpleNamespace(
        type="image",
        data=base64.b64encode(b"gemini-image").decode(),
        mime_type="image/jpeg",
    )
    usage = SimpleNamespace(
        input_tokens_by_modality=[SimpleNamespace(modality="text", tokens=20)],
        output_tokens_by_modality=[SimpleNamespace(modality="image", tokens=1120)],
    )
    return SimpleNamespace(
        steps=[SimpleNamespace(type="model_output", content=[image])],
        output_image=None,
        usage=usage,
    )


async def test_generate_uses_interactions_and_decodes_image(model):
    model.client.aio.interactions.create = AsyncMock(return_value=interaction_response())

    response = await model.generate("A lighthouse", aspect_ratio="16:9", image_size="2K")

    assert response.images[0].data == b"gemini-image"
    assert response.usage.image_output_tokens == 1120
    kwargs = model.client.aio.interactions.create.call_args.kwargs
    assert kwargs["response_format"]["aspect_ratio"] == "16:9"
    assert kwargs["response_format"]["image_size"] == "2K"


async def test_edit_encodes_reference_images(model):
    model.client.aio.interactions.create = AsyncMock(return_value=interaction_response())
    reference = ImageInput(data=b"reference", media_type="image/png")

    await model.edit("Make it blue", (reference,))

    request_input = model.client.aio.interactions.create.call_args.kwargs["input"]
    assert base64.b64decode(request_input[0]["data"]) == b"reference"
    assert request_input[-1] == {"type": "text", "text": "Make it blue"}


async def test_rejects_unsupported_mask_and_count(model):
    image = ImageInput(data=b"image", media_type="image/png")
    with pytest.raises(ImageOptionUnsupported, match="mask=provided"):
        await model.edit("Edit", (image,), mask=image)
    with pytest.raises(ImageOptionUnsupported, match="count=2"):
        await model.generate("Generate", count=2)


async def test_rejects_unsupported_output_format(model):
    with pytest.raises(ImageOptionUnsupported, match="output_format=png"):
        await model.generate("Generate", output_format="png")


async def test_normalizes_interactions_api_errors(model, monkeypatch):
    class InteractionAPIError(Exception):
        pass

    monkeypatch.setattr(
        "majordomo_llm.providers.gemini_image.InteractionsAPIError",
        InteractionAPIError,
    )
    model.client.aio.interactions.create = AsyncMock(
        side_effect=InteractionAPIError("request blocked")
    )

    with pytest.raises(ProviderError, match="Gemini image API error") as exc_info:
        await model.generate("Generate")

    assert isinstance(exc_info.value.original_error, InteractionAPIError)
