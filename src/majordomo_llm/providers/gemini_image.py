"""Gemini image generation and editing provider."""

import base64
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from google.genai._gaos.lib.compat_errors import APIError as InteractionsAPIError

from majordomo_llm.base import ImageInput, resolve_api_key
from majordomo_llm.exceptions import ImageOptionUnsupported, ProviderError, ResponseParsingError
from majordomo_llm.hooks.image_pipeline import ImageHookPipeline
from majordomo_llm.image import (
    GeneratedImage,
    ImageAspectRatio,
    ImageBackground,
    ImageFormat,
    ImageModel,
    ImageQuality,
    ImageResponse,
    ImageSize,
    ImageUsage,
    validate_image_count,
)
from majordomo_llm.retry import retry_provider_call


class GeminiImage(ImageModel):
    """Generate and edit images through Gemini Interactions."""

    def __init__(
        self,
        model: str,
        text_input_cost: float,
        image_input_cost: float,
        text_output_cost: float,
        image_output_cost: float,
        *,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        hook_pipeline: ImageHookPipeline | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        resolved_api_key = resolve_api_key(api_key, "GEMINI_API_KEY", "Gemini")
        super().__init__(
            provider="gemini",
            model=model,
            text_input_cost=text_input_cost,
            image_input_cost=image_input_cost,
            text_output_cost=text_output_cost,
            image_output_cost=image_output_cost,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
            hook_pipeline=hook_pipeline,
        )
        http_options = None
        if base_url or default_headers:
            http_options = types.HttpOptions(base_url=base_url, headers=default_headers)
        self.client = genai.Client(api_key=resolved_api_key, http_options=http_options)

    @retry_provider_call
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
        del caller_metadata
        self._validate_options(count, quality, background, output_format)
        return await self._create(
            input_content=prompt,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            output_format=output_format,
            extra_headers=extra_headers,
        )

    @retry_provider_call
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
        del caller_metadata
        if not images:
            raise ValueError("images must not be empty for an edit request")
        if mask is not None:
            raise ImageOptionUnsupported(self.provider, self.model, "mask", "provided")
        self._validate_options(count, quality, background, output_format)
        input_content: list[dict[str, str]] = [
            {
                "type": "image",
                "data": base64.b64encode(image.data).decode("ascii"),
                "mime_type": image.media_type,
            }
            for image in images
        ]
        input_content.append({"type": "text", "text": prompt})
        return await self._create(
            input_content=input_content,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            output_format=output_format,
            extra_headers=extra_headers,
        )

    def _validate_options(
        self,
        count: int,
        quality: ImageQuality,
        background: ImageBackground,
        output_format: ImageFormat,
    ) -> None:
        validate_image_count(count)
        if count != 1:
            raise ImageOptionUnsupported(self.provider, self.model, "count", count, supported="1")
        if quality != "auto":
            raise ImageOptionUnsupported(
                self.provider, self.model, "quality", quality, supported="auto"
            )
        if background != "auto":
            raise ImageOptionUnsupported(
                self.provider, self.model, "background", background, supported="auto"
            )
        if output_format != "jpeg":
            raise ImageOptionUnsupported(
                self.provider, self.model, "output_format", output_format, supported="jpeg"
            )

    async def _create(
        self,
        *,
        input_content: str | list[dict[str, str]],
        aspect_ratio: ImageAspectRatio,
        image_size: ImageSize,
        output_format: ImageFormat,
        extra_headers: dict[str, str] | None,
    ) -> ImageResponse:
        response_format = {
            "type": "image",
            "mime_type": f"image/{output_format}",
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
        }
        start = time.time()
        try:
            response = await self.client.aio.interactions.create(
                model=self.model,
                input=input_content,
                response_format=response_format,
                extra_headers=extra_headers,
            )
        # The Interactions client currently uses its generated SDK's compatibility
        # error hierarchy rather than google.genai.errors.APIError.
        except (genai_errors.APIError, InteractionsAPIError) as e:
            raise ProviderError(
                f"Gemini image API error: {e}", provider="gemini", original_error=e
            ) from e
        return self._build_response(response, time.time() - start)

    def _build_response(self, response: Any, response_time: float) -> ImageResponse:
        images: list[GeneratedImage] = []
        for step in response.steps or []:
            if getattr(step, "type", None) != "model_output":
                continue
            for block in step.content or []:
                if getattr(block, "type", None) != "image" or not block.data:
                    continue
                try:
                    data = base64.b64decode(block.data, validate=True)
                except ValueError as e:
                    raise ResponseParsingError("Gemini returned invalid base64 image data") from e
                images.append(
                    GeneratedImage(
                        data=data,
                        media_type=block.mime_type or "image/jpeg",
                    )
                )
        if not images and response.output_image is not None and response.output_image.data:
            try:
                data = base64.b64decode(response.output_image.data, validate=True)
            except ValueError as e:
                raise ResponseParsingError("Gemini returned invalid base64 image data") from e
            images.append(
                GeneratedImage(
                    data=data,
                    media_type=response.output_image.mime_type or "image/jpeg",
                )
            )
        if not images:
            raise ResponseParsingError("Gemini image response did not contain an image")

        usage = _gemini_image_usage(response.usage)
        input_cost, output_cost, total_cost = self._calculate_costs(usage)
        return ImageResponse(
            images=tuple(images),
            usage=usage,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=response_time,
            provider=self.provider,
            model=self.model,
        )


def _gemini_image_usage(raw_usage: Any) -> ImageUsage:
    if raw_usage is None:
        return ImageUsage()

    def tokens_by_modality(items: Any, modality: str) -> int:
        return sum(
            int(item.tokens or 0)
            for item in (items or [])
            if getattr(item, "modality", None) == modality
        )

    return ImageUsage(
        text_input_tokens=tokens_by_modality(raw_usage.input_tokens_by_modality, "text"),
        image_input_tokens=tokens_by_modality(raw_usage.input_tokens_by_modality, "image"),
        text_output_tokens=tokens_by_modality(raw_usage.output_tokens_by_modality, "text"),
        image_output_tokens=tokens_by_modality(raw_usage.output_tokens_by_modality, "image"),
    )
