"""Tests for image-generation cascade failover."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import Future, RetryError

from majordomo_llm import ImageCascade, ImageInput
from majordomo_llm.exceptions import (
    ImageOptionUnsupported,
    ProviderError,
    ResponseParsingError,
)
from majordomo_llm.image import ImageModel, ImageResponse


def _retry_error_with_exception(exc: BaseException) -> RetryError:
    return RetryError(Future.construct(attempt_number=3, value=exc, has_exception=True))


def _mock_model(provider: str, model: str) -> MagicMock:
    image_model = MagicMock(spec=ImageModel)
    image_model.provider = provider
    image_model.model = model
    image_model.text_input_cost = 1.0
    image_model.image_input_cost = 2.0
    image_model.text_output_cost = 3.0
    image_model.image_output_cost = 4.0
    image_model.api_key_hash = "key-hash"
    image_model.api_key_alias = "primary"
    image_model.generate = AsyncMock()
    image_model.edit = AsyncMock()
    return image_model


@pytest.fixture
def children() -> tuple[MagicMock, MagicMock]:
    return _mock_model("openai", "gpt-image-2"), _mock_model("gemini", "gemini-3.1-flash-image")


@pytest.fixture
def cascade(children: tuple[MagicMock, MagicMock]) -> ImageCascade:
    with patch("majordomo_llm.image_cascade.get_image_instance", side_effect=children):
        return ImageCascade([("openai", "gpt-image-2"), ("gemini", "gemini-3.1-flash-image")])


def test_requires_at_least_one_provider():
    with pytest.raises(ValueError, match="at least one provider"):
        ImageCascade([])


def test_builds_children_through_factory(children: tuple[MagicMock, MagicMock]):
    with patch("majordomo_llm.image_cascade.get_image_instance", side_effect=children) as factory:
        result = ImageCascade(
            [("openai", "gpt-image-2"), ("gemini", "gemini-3.1-flash-image")],
            api_key="test-key",
            api_key_alias="primary",
            base_url="https://gateway.example/v1",
            default_headers={"X-Test": "value"},
        )

    assert result.models == list(children)
    assert result.provider == "cascade"
    assert factory.call_count == 2
    factory.assert_any_call(
        "openai",
        "gpt-image-2",
        api_key="test-key",
        api_key_alias="primary",
        base_url="https://gateway.example/v1",
        default_headers={"X-Test": "value"},
    )
    assert result.api_key_hash == "key-hash"
    assert result.api_key_alias == "primary"


async def test_returns_primary_response(cascade: ImageCascade, children):
    response = MagicMock(spec=ImageResponse)
    children[0].generate.return_value = response

    assert await cascade.generate("A lighthouse") is response
    children[1].generate.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [
        ProviderError("unavailable", provider="openai"),
        ResponseParsingError("missing image"),
        ImageOptionUnsupported("openai", "gpt-image-2", "image_size", "2K"),
    ],
)
async def test_falls_back_on_provider_response_and_capability_errors(
    cascade: ImageCascade,
    children,
    error: Exception,
):
    response = MagicMock(spec=ImageResponse)
    children[0].generate.side_effect = error
    children[1].generate.return_value = response

    assert await cascade.generate("A lighthouse", image_size="2K") is response
    children[1].generate.assert_awaited_once()


async def test_falls_back_when_retries_exhaust(cascade: ImageCascade, children):
    response = MagicMock(spec=ImageResponse)
    failure = ProviderError("unavailable", provider="openai")
    children[0].generate.side_effect = _retry_error_with_exception(failure)
    children[1].generate.return_value = response

    assert await cascade.generate("A lighthouse") is response


async def test_does_not_swallow_caller_error(cascade: ImageCascade, children):
    children[0].generate.side_effect = ValueError("count must be between 1 and 10")

    with pytest.raises(ValueError, match="count must be"):
        await cascade.generate("A lighthouse", count=0)

    children[1].generate.assert_not_awaited()


async def test_raises_cascade_error_when_all_models_fail(cascade: ImageCascade, children):
    first = ProviderError("unavailable", provider="openai")
    last = ResponseParsingError("missing image")
    children[0].generate.side_effect = first
    children[1].generate.side_effect = last

    with pytest.raises(ProviderError, match="All image providers") as exc_info:
        await cascade.generate("A lighthouse")

    assert exc_info.value.provider == "cascade"
    assert exc_info.value.original_error is last


async def test_forwards_all_edit_arguments(cascade: ImageCascade, children):
    response = MagicMock(spec=ImageResponse)
    children[0].edit.return_value = response
    image = ImageInput(b"image", "image/png")
    mask = ImageInput(b"mask", "image/png")

    result = await cascade.edit(
        "Make it moonlit",
        (image,),
        mask=mask,
        count=2,
        aspect_ratio="3:2",
        image_size="1K",
        output_format="png",
        quality="high",
        background="transparent",
        extra_headers={"X-Test": "value"},
        caller_metadata=None,
    )

    assert result is response
    children[0].edit.assert_awaited_once_with(
        prompt="Make it moonlit",
        images=(image,),
        mask=mask,
        count=2,
        aspect_ratio="3:2",
        image_size="1K",
        output_format="png",
        quality="high",
        background="transparent",
        extra_headers={"X-Test": "value"},
        caller_metadata=None,
    )
