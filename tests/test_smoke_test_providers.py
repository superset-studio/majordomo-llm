"""Tests for provider smoke-test capability helpers."""

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from scripts.smoke_test_providers import FAIL, OK, SKIP, _run_cell, _run_image


def _vision_model(content: str) -> SimpleNamespace:
    response = SimpleNamespace(
        content=content,
        input_tokens=12,
        output_tokens=1,
        total_cost=0.0001,
    )
    return SimpleNamespace(
        supports_image_input=True,
        get_response=AsyncMock(return_value=response),
    )


async def test_run_image_submits_valid_png_and_records_usage() -> None:
    llm = _vision_model("BLUE")

    result = await _run_image(llm, None)

    assert result.status == OK
    assert result.input_tokens == 12
    assert result.output_tokens == 1
    assert result.total_cost == pytest.approx(0.0001)
    image_input = llm.get_response.call_args.kwargs["images"][0]
    assert image_input.media_type == "image/png"
    with Image.open(io.BytesIO(image_input.data)) as decoded:
        decoded.load()
        assert decoded.size == (32, 32)


@pytest.mark.parametrize("content", ["", "The image is red."])
async def test_run_image_rejects_empty_or_wrong_answers(content: str) -> None:
    result = await _run_image(_vision_model(content), None)

    assert result.status == FAIL
    assert "blue" in result.error


async def test_run_cell_skips_models_without_image_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = SimpleNamespace(
        supports_image_input=False,
        get_response=AsyncMock(),
    )
    monkeypatch.setattr("scripts.smoke_test_providers._build_llm", lambda *args, **kwargs: llm)

    result = await _run_cell(
        "openai",
        "text-only",
        "image",
        "capability-matrix",
        via_steward=False,
        gateway_url="http://localhost:7680",
        gateway_key=None,
        run_id="test-run",
    )

    assert result.status == SKIP
    llm.get_response.assert_not_awaited()
