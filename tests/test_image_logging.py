"""Tests for asynchronous image-generation logging."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from majordomo_llm import GeneratedImage, ImageInput, ImageResponse, ImageUsage
from majordomo_llm.exceptions import ProviderError
from majordomo_llm.image import ImageModel
from majordomo_llm.logging import LoggingImageModel
from majordomo_llm.logging.interfaces import DatabaseAdapter, StorageAdapter


@pytest.fixture
def wrapped_model() -> MagicMock:
    model = MagicMock(spec=ImageModel)
    model.provider = "cascade"
    model.model = "gpt-image-2"
    model.text_input_cost = 5.0
    model.image_input_cost = 8.0
    model.text_output_cost = 10.0
    model.image_output_cost = 30.0
    model.api_key_hash = "key-hash"
    model.api_key_alias = "primary"
    model.generate = AsyncMock()
    model.edit = AsyncMock()
    return model


@pytest.fixture
def database() -> AsyncMock:
    return AsyncMock(spec=DatabaseAdapter)


@pytest.fixture
def storage() -> AsyncMock:
    adapter = AsyncMock(spec=StorageAdapter)
    adapter.upload.return_value = ("request.json", "response.json")
    return adapter


@pytest.fixture
def response() -> ImageResponse:
    return ImageResponse(
        images=(GeneratedImage(b"generated-secret", "image/png", "revised"),),
        usage=ImageUsage(
            text_input_tokens=10,
            image_input_tokens=20,
            text_output_tokens=3,
            image_output_tokens=100,
        ),
        input_cost=0.01,
        output_cost=0.02,
        total_cost=0.03,
        response_time=1.5,
        provider="gemini",
        model="gemini-3.1-flash-image",
    )


async def test_generate_logs_metrics_and_safe_response_metadata(
    wrapped_model, database, storage, response
):
    wrapped_model.generate.return_value = response
    logged = LoggingImageModel(wrapped_model, database, storage)

    assert await logged.generate("A lighthouse", image_size="2K") is response
    await logged.flush()

    entry = database.insert.await_args.args[0]
    assert entry.provider == "gemini"
    assert entry.model == "gemini-3.1-flash-image"
    assert entry.input_tokens == 30
    assert entry.output_tokens == 103
    assert entry.total_cost == 0.03
    assert entry.api_key_hash == "key-hash"
    assert entry.api_key_alias == "primary"

    _, request_body, response_body = storage.upload.await_args.args
    assert request_body["operation"] == "generate"
    assert request_body["image_size"] == "2K"
    assert response_body["usage"]["image_output_tokens"] == 100
    assert response_body["images"][0]["size_bytes"] == len(b"generated-secret")
    assert b"generated-secret" not in repr(response_body).encode()


async def test_edit_logs_reference_and_mask_metadata_without_bytes(
    wrapped_model, database, storage, response
):
    wrapped_model.edit.return_value = response
    logged = LoggingImageModel(wrapped_model, database, storage)
    reference = ImageInput(b"reference-secret", "image/png")
    mask = ImageInput(b"mask-secret", "image/png")

    await logged.edit("Make it blue", (reference,), mask=mask)
    await logged.flush()

    _, request_body, _ = storage.upload.await_args.args
    assert request_body["images"][0]["size_bytes"] == len(b"reference-secret")
    assert request_body["mask"]["size_bytes"] == len(b"mask-secret")
    serialized = repr(request_body).encode()
    assert b"reference-secret" not in serialized
    assert b"mask-secret" not in serialized


async def test_failure_is_logged_and_reraised(wrapped_model, database, storage):
    failure = ProviderError("unavailable", provider="openai")
    wrapped_model.generate.side_effect = failure
    logged = LoggingImageModel(wrapped_model, database, storage)

    with pytest.raises(ProviderError) as exc_info:
        await logged.generate("A lighthouse")
    await logged.flush()

    assert exc_info.value is failure
    entry = database.insert.await_args.args[0]
    assert entry.status == "error"
    assert entry.error_message == "unavailable"
    assert entry.input_tokens is None
    assert storage.upload.await_args.args[2] is None


async def test_logging_without_storage_still_inserts_database_row(
    wrapped_model, database, response
):
    wrapped_model.generate.return_value = response
    logged = LoggingImageModel(wrapped_model, database)

    await logged.generate("A lighthouse")
    await logged.flush()

    database.insert.assert_awaited_once()


async def test_close_flushes_and_closes_adapters(wrapped_model, database, storage, response):
    wrapped_model.generate.return_value = response
    logged = LoggingImageModel(wrapped_model, database, storage)

    await logged.generate("A lighthouse")
    await logged.close()

    database.insert.assert_awaited_once()
    database.close.assert_awaited_once()
    storage.close.assert_awaited_once()
