"""OpenAI image generation and editing provider."""

import base64
import time
from typing import Any

import openai

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

_OPENAI_SIZES: dict[ImageAspectRatio, str] = {
    "1:1": "1024x1024",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
}


class OpenAIImage(ImageModel):
    """Generate and edit images through OpenAI's Images API."""

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
        resolved_api_key = resolve_api_key(api_key, "OPENAI_API_KEY", "OpenAI")
        super().__init__(
            provider="openai",
            model=model,
            text_input_cost=text_input_cost,
            image_input_cost=image_input_cost,
            text_output_cost=text_output_cost,
            image_output_cost=image_output_cost,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
            hook_pipeline=hook_pipeline,
        )
        self.client = openai.AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
            default_headers=default_headers,
        )

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
        validate_image_count(count)
        size = self._resolve_size(aspect_ratio, image_size)
        start = time.time()
        try:
            response = await self.client.images.generate(
                model=self.model,
                prompt=prompt,
                n=count,
                size=size,
                output_format=output_format,
                quality=quality,
                background=background,
                extra_headers=extra_headers,
            )
        except openai.APIError as e:
            raise ProviderError(
                f"OpenAI image API error: {e}", provider="openai", original_error=e
            ) from e
        return self._build_response(response, time.time() - start, output_format)

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
        validate_image_count(count)
        size = self._resolve_size(aspect_ratio, image_size)
        image_files = [self._file_tuple(image, index) for index, image in enumerate(images)]
        mask_file = self._file_tuple(mask, 0, prefix="mask") if mask is not None else None
        start = time.time()
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "image": image_files,
            "n": count,
            "size": size,
            "output_format": output_format,
            "quality": quality,
            "background": background,
            "extra_headers": extra_headers,
        }
        if mask_file is not None:
            request_kwargs["mask"] = mask_file
        try:
            response = await self.client.images.edit(**request_kwargs)
        except openai.APIError as e:
            raise ProviderError(
                f"OpenAI image API error: {e}", provider="openai", original_error=e
            ) from e
        return self._build_response(response, time.time() - start, output_format)

    def _resolve_size(self, aspect_ratio: ImageAspectRatio, image_size: ImageSize) -> str:
        if image_size != "1K":
            raise ImageOptionUnsupported(
                self.provider,
                self.model,
                "image_size",
                image_size,
                supported="1K",
            )
        try:
            return _OPENAI_SIZES[aspect_ratio]
        except KeyError as e:
            supported = ", ".join(_OPENAI_SIZES)
            raise ImageOptionUnsupported(
                self.provider,
                self.model,
                "aspect_ratio",
                aspect_ratio,
                supported=supported,
            ) from e

    @staticmethod
    def _file_tuple(
        image: ImageInput, index: int, *, prefix: str = "image"
    ) -> tuple[str, bytes, str]:
        extension = "jpg" if image.media_type == "image/jpeg" else image.media_type.split("/")[1]
        return (f"{prefix}-{index}.{extension}", image.data, image.media_type)

    def _build_response(
        self, response: Any, response_time: float, requested_format: ImageFormat
    ) -> ImageResponse:
        data = response.data or []
        images: list[GeneratedImage] = []
        media_type = f"image/{response.output_format or requested_format}"
        for item in data:
            if not item.b64_json:
                raise ResponseParsingError("OpenAI image response did not contain base64 data")
            try:
                decoded = base64.b64decode(item.b64_json, validate=True)
            except ValueError as e:
                raise ResponseParsingError("OpenAI returned invalid base64 image data") from e
            images.append(
                GeneratedImage(
                    data=decoded,
                    media_type=media_type,
                    revised_prompt=item.revised_prompt,
                )
            )
        if not images:
            raise ResponseParsingError("OpenAI image response did not contain an image")

        raw_usage = response.usage
        input_details = getattr(raw_usage, "input_tokens_details", None)
        output_details = getattr(raw_usage, "output_tokens_details", None)
        usage = ImageUsage(
            text_input_tokens=int(getattr(input_details, "text_tokens", 0) or 0),
            image_input_tokens=int(getattr(input_details, "image_tokens", 0) or 0),
            text_output_tokens=int(getattr(output_details, "text_tokens", 0) or 0),
            image_output_tokens=int(
                getattr(output_details, "image_tokens", 0)
                or getattr(raw_usage, "output_tokens", 0)
                or 0
            ),
        )
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
