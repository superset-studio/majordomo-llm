"""Tests for provider-neutral image contracts."""

import pytest

from majordomo_llm import ImageInput
from majordomo_llm.image import ImageModel, ImageUsage


class DummyImageModel(ImageModel):
    async def _generate_impl(self, prompt, **kwargs):
        raise NotImplementedError

    async def _edit_impl(self, prompt, images, **kwargs):
        raise NotImplementedError


def test_image_input_validates_data_and_media_type():
    image = ImageInput(data=b"image", media_type="image/png")
    assert image.data == b"image"

    with pytest.raises(ValueError, match="must not be empty"):
        ImageInput(data=b"", media_type="image/png")
    with pytest.raises(ValueError, match="Unsupported image media type"):
        ImageInput(data=b"image", media_type="image/bmp")


def test_image_model_calculates_modality_specific_costs():
    model = DummyImageModel(
        provider="test",
        model="image-test",
        text_input_cost=5.0,
        image_input_cost=8.0,
        text_output_cost=10.0,
        image_output_cost=30.0,
    )
    usage = ImageUsage(
        text_input_tokens=100,
        image_input_tokens=200,
        text_output_tokens=10,
        image_output_tokens=1000,
    )

    input_cost, output_cost, total_cost = model._calculate_costs(usage)

    assert input_cost == pytest.approx((100 * 5 + 200 * 8) / 1_000_000)
    assert output_cost == pytest.approx((10 * 10 + 1000 * 30) / 1_000_000)
    assert total_cost == pytest.approx(input_cost + output_cost)
