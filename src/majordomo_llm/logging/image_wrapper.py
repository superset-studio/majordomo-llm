"""Asynchronous metadata logging for image generation and editing."""

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from majordomo_llm.base import ImageInput
from majordomo_llm.image import (
    GeneratedImage,
    ImageAspectRatio,
    ImageBackground,
    ImageFormat,
    ImageModel,
    ImageQuality,
    ImageResponse,
    ImageSize,
)
from majordomo_llm.logging.interfaces import DatabaseAdapter, StorageAdapter
from majordomo_llm.logging.models import LogEntry
from majordomo_llm.logging.wrapper import _image_metadata


class LoggingImageModel(ImageModel):
    """Wrap an :class:`ImageModel` with non-blocking metadata logging.

    Raw reference, mask, and generated image bytes are never written to the
    database or storage adapter. Logs contain only media type, byte length,
    SHA-256, and the provider's revised prompt when present.
    """

    def __init__(
        self,
        model: ImageModel,
        database: DatabaseAdapter,
        storage: StorageAdapter | None = None,
    ) -> None:
        super().__init__(
            provider=model.provider,
            model=model.model,
            text_input_cost=model.text_input_cost,
            image_input_cost=model.image_input_cost,
            text_output_cost=model.text_output_cost,
            image_output_cost=model.image_output_cost,
        )
        self.api_key_hash = model.api_key_hash
        self.api_key_alias = model.api_key_alias
        self._wrapped_model = model
        self._database = database
        self._storage = storage
        self._pending_tasks: set[asyncio.Task[None]] = set()

    async def _generate_impl(
        self,
        prompt: str,
        *,
        count: int = 1,
        aspect_ratio: ImageAspectRatio = "1:1",
        image_size: ImageSize = "1K",
        output_format: ImageFormat = "jpeg",
        quality: ImageQuality = "auto",
        background: ImageBackground = "auto",
        extra_headers: dict[str, str] | None = None,
        caller_metadata: dict[str, Any] | None = None,
    ) -> ImageResponse:
        request_body = _image_request_body(
            operation="generate",
            prompt=prompt,
            images=(),
            mask=None,
            count=count,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            output_format=output_format,
            quality=quality,
            background=background,
        )
        try:
            response = await self._wrapped_model.generate(
                prompt,
                count=count,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                output_format=output_format,
                quality=quality,
                background=background,
                extra_headers=extra_headers,
                caller_metadata=caller_metadata,
            )
        except Exception as exc:
            self._fire_and_forget(request_body, None, "error", str(exc))
            raise

        self._fire_and_forget(request_body, response, "success", None)
        return response

    async def _edit_impl(
        self,
        prompt: str,
        images: tuple[ImageInput, ...],
        *,
        mask: ImageInput | None = None,
        count: int = 1,
        aspect_ratio: ImageAspectRatio = "1:1",
        image_size: ImageSize = "1K",
        output_format: ImageFormat = "jpeg",
        quality: ImageQuality = "auto",
        background: ImageBackground = "auto",
        extra_headers: dict[str, str] | None = None,
        caller_metadata: dict[str, Any] | None = None,
    ) -> ImageResponse:
        request_body = _image_request_body(
            operation="edit",
            prompt=prompt,
            images=images,
            mask=mask,
            count=count,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            output_format=output_format,
            quality=quality,
            background=background,
        )
        try:
            response = await self._wrapped_model.edit(
                prompt,
                images,
                mask=mask,
                count=count,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                output_format=output_format,
                quality=quality,
                background=background,
                extra_headers=extra_headers,
                caller_metadata=caller_metadata,
            )
        except Exception as exc:
            self._fire_and_forget(request_body, None, "error", str(exc))
            raise

        self._fire_and_forget(request_body, response, "success", None)
        return response

    async def _log_request(
        self,
        request_body: dict[str, Any],
        response: ImageResponse | None,
        status: str,
        error_message: str | None,
    ) -> None:
        request_id = uuid4()
        response_body = _image_response_body(response) if response is not None else None
        request_key: str | None = None
        response_key: str | None = None
        if self._storage is not None:
            request_key, response_key = await self._storage.upload(
                request_id, request_body, response_body
            )

        usage = response.usage if response is not None else None
        entry = LogEntry(
            request_id=request_id,
            provider=response.provider if response is not None else self.provider,
            model=response.model if response is not None else self.model,
            timestamp=datetime.now(UTC),
            response_time=response.response_time if response is not None else None,
            input_tokens=(
                usage.text_input_tokens + usage.image_input_tokens if usage is not None else None
            ),
            output_tokens=(
                usage.text_output_tokens + usage.image_output_tokens if usage is not None else None
            ),
            cached_tokens=None,
            cache_creation_tokens=None,
            input_cost=response.input_cost if response is not None else None,
            output_cost=response.output_cost if response is not None else None,
            total_cost=response.total_cost if response is not None else None,
            s3_request_key=request_key,
            s3_response_key=response_key,
            status=status,
            error_message=error_message,
            api_key_hash=self.api_key_hash,
            api_key_alias=self.api_key_alias,
        )
        await self._database.insert(entry)

    def _fire_and_forget(
        self,
        request_body: dict[str, Any],
        response: ImageResponse | None,
        status: str,
        error_message: str | None,
    ) -> None:
        task = asyncio.create_task(self._log_request(request_body, response, status, error_message))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def flush(self) -> None:
        """Wait for currently pending log writes to finish."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)

    async def close(self) -> None:
        """Flush pending writes and close owned logging adapters."""
        await self.flush()
        await self._database.close()
        if self._storage is not None:
            await self._storage.close()


def _image_request_body(
    *,
    operation: str,
    prompt: str,
    images: tuple[ImageInput, ...],
    mask: ImageInput | None,
    count: int,
    aspect_ratio: ImageAspectRatio,
    image_size: ImageSize,
    output_format: ImageFormat,
    quality: ImageQuality,
    background: ImageBackground,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "prompt": prompt,
        "images": _image_metadata(images),
        "mask": _image_metadata((mask,))[0] if mask is not None else None,
        "count": count,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
        "output_format": output_format,
        "quality": quality,
        "background": background,
    }


def _generated_image_metadata(image: GeneratedImage) -> dict[str, str | int | None]:
    import hashlib

    return {
        "media_type": image.media_type,
        "size_bytes": len(image.data),
        "sha256": hashlib.sha256(image.data).hexdigest(),
        "revised_prompt": image.revised_prompt,
    }


def _image_response_body(response: ImageResponse) -> dict[str, Any]:
    return {
        "provider": response.provider,
        "model": response.model,
        "images": [_generated_image_metadata(image) for image in response.images],
        "usage": asdict(response.usage),
        "input_cost": response.input_cost,
        "output_cost": response.output_cost,
        "total_cost": response.total_cost,
        "response_time": response.response_time,
    }
