"""Tests for the live image smoke-test harness helpers."""

import io
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from majordomo_llm.exceptions import ProviderError
from majordomo_llm.image import GeneratedImage, ImageResponse, ImageUsage
from scripts.smoke_test_images import (
    _INVALID_API_KEY,
    _run_cascade_exhausted,
    _run_cascade_fallback,
    _run_invalid_auth,
    _select_models,
    _temporary_provider_keys,
    _validate_auth_failure,
    _validate_response,
)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def _response(
    *,
    data: bytes | None = None,
    media_type: str = "image/png",
    provider: str = "openai",
    model: str = "gpt-image-2",
) -> ImageResponse:
    return ImageResponse(
        images=(GeneratedImage(data=data or _png_bytes(), media_type=media_type),),
        usage=ImageUsage(text_input_tokens=1, image_output_tokens=2),
        input_cost=0.01,
        output_cost=0.02,
        total_cost=0.03,
        response_time=0.5,
        provider=provider,
        model=model,
    )


def test_validate_response_returns_decoded_metadata() -> None:
    metadata = _validate_response(
        _response(),
        expected_provider="openai",
        expected_model="gpt-image-2",
    )

    assert len(metadata) == 1
    assert metadata[0].media_type == "image/png"
    assert metadata[0].byte_length == len(_png_bytes())
    assert (metadata[0].width, metadata[0].height) == (8, 6)


def test_validate_response_rejects_media_type_mismatch() -> None:
    with pytest.raises(ValueError, match="media type mismatch"):
        _validate_response(
            _response(media_type="image/jpeg"),
            expected_provider="openai",
            expected_model="gpt-image-2",
        )


def test_validate_response_rejects_invalid_image_bytes() -> None:
    with pytest.raises(ValueError, match="invalid image bytes"):
        _validate_response(
            _response(data=b"not-an-image"),
            expected_provider="openai",
            expected_model="gpt-image-2",
        )


def test_select_models_uses_first_model_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.smoke_test_images.get_supported_image_models",
        lambda provider: [f"{provider}-primary", f"{provider}-secondary"],
    )

    assert _select_models(["openai", "gemini"], all_models=False) == [
        ("openai", "openai-primary"),
        ("gemini", "gemini-primary"),
    ]


def test_select_models_can_include_every_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.smoke_test_images.get_supported_image_models",
        lambda provider: [f"{provider}-primary", f"{provider}-secondary"],
    )

    assert _select_models(["gemini"], all_models=True) == [
        ("gemini", "gemini-primary"),
        ("gemini", "gemini-secondary"),
    ]


def test_select_models_can_target_exact_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.smoke_test_images.get_supported_image_models",
        lambda provider: [f"{provider}-primary", f"{provider}-secondary"],
    )

    assert _select_models(
        ["gemini"],
        all_models=False,
        requested_models=["gemini-secondary", "gemini-secondary"],
    ) == [("gemini", "gemini-secondary")]


def test_select_models_rejects_unknown_exact_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.smoke_test_images.get_supported_image_models",
        lambda provider: [f"{provider}-primary"],
    )

    with pytest.raises(ValueError, match="unknown image model"):
        _select_models(
            ["gemini"],
            all_models=False,
            requested_models=["missing-model"],
        )


def test_temporary_provider_keys_restores_existing_and_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with _temporary_provider_keys(
        {"openai": _INVALID_API_KEY, "gemini": _INVALID_API_KEY}
    ):
        assert os.environ["OPENAI_API_KEY"] == _INVALID_API_KEY
        assert os.environ["GEMINI_API_KEY"] == _INVALID_API_KEY

    assert os.environ["OPENAI_API_KEY"] == "real-openai-key"
    assert "GEMINI_API_KEY" not in os.environ


async def test_invalid_auth_requires_normalized_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_error = SimpleNamespace(status_code=401)
    model = SimpleNamespace(
        generate=AsyncMock(
            side_effect=ProviderError(
                "unauthorized",
                provider="openai",
                original_error=auth_error,
            )
        )
    )
    monkeypatch.setattr("scripts.smoke_test_images.get_image_instance", lambda *_: model)

    result = await _run_invalid_auth("openai", "gpt-image-2")

    assert result.status == "pass"
    assert "HTTP 401" in result.detail


def test_gemini_auth_accepts_only_structured_invalid_key_reason() -> None:
    valid_error = SimpleNamespace(
        status_code=400,
        body=[{"error": {"details": [{"reason": "API_KEY_INVALID"}]}}],
    )
    valid = ProviderError("invalid key", provider="gemini", original_error=valid_error)

    assert _validate_auth_failure(valid, expected_provider="gemini") == 400

    unrelated_error = SimpleNamespace(
        status_code=400,
        body={"error": {"details": [{"reason": "INVALID_ARGUMENT"}]}},
    )
    unrelated = ProviderError(
        "bad request",
        provider="gemini",
        original_error=unrelated_error,
    )
    with pytest.raises(ValueError, match="API_KEY_INVALID"):
        _validate_auth_failure(unrelated, expected_provider="gemini")


async def test_cascade_fallback_preserves_order_and_restores_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    captured: dict[str, object] = {}

    class FakeCascade:
        def __init__(self, providers):
            captured["providers"] = providers
            captured["primary_key"] = os.environ["OPENAI_API_KEY"]

        async def generate(self, prompt):
            return _response(provider="gemini", model="gemini-image")

    monkeypatch.setattr("scripts.smoke_test_images.ImageCascade", FakeCascade)
    models = {"openai": "openai-image", "gemini": "gemini-image"}

    result = await _run_cascade_fallback("openai", "gemini", models)

    assert result.status == "pass"
    assert captured["providers"] == [
        ("openai", "openai-image"),
        ("gemini", "gemini-image"),
    ]
    assert captured["primary_key"] == _INVALID_API_KEY
    assert os.environ["OPENAI_API_KEY"] == "real-openai-key"


async def test_cascade_exhausted_requires_preserved_final_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_error = SimpleNamespace(status_code=403)
    final_error = ProviderError(
        "forbidden",
        provider="gemini",
        original_error=auth_error,
    )

    class FakeCascade:
        def __init__(self, providers):
            assert providers[-1] == ("gemini", "gemini-image")

        async def generate(self, prompt):
            raise ProviderError(
                "all providers failed",
                provider="cascade",
                original_error=final_error,
            )

    monkeypatch.setattr("scripts.smoke_test_images.ImageCascade", FakeCascade)
    models = {"openai": "openai-image", "gemini": "gemini-image"}

    result = await _run_cascade_exhausted(models)

    assert result.status == "pass"
    assert "gemini HTTP 403" in result.detail
