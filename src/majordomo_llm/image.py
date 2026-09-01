"""Provider-neutral contracts for image generation and editing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from majordomo_llm.base import TOKENS_PER_MILLION, ImageInput, _hash_api_key

if TYPE_CHECKING:
    from majordomo_llm.hooks.image_pipeline import ImageHookPipeline

ImageAspectRatio = Literal["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
ImageSize = Literal["512", "1K", "2K", "4K"]
ImageFormat = Literal["png", "jpeg", "webp"]
ImageQuality = Literal["auto", "low", "medium", "high"]
ImageBackground = Literal["auto", "opaque", "transparent"]


@dataclass(frozen=True)
class GeneratedImage:
    """One generated image returned as decoded bytes."""

    data: bytes
    media_type: str
    revised_prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("GeneratedImage.data must not be empty")


@dataclass(frozen=True)
class ImageUsage:
    """Modality-specific usage reported by an image provider."""

    text_input_tokens: int = 0
    image_input_tokens: int = 0
    text_output_tokens: int = 0
    image_output_tokens: int = 0


@dataclass(frozen=True)
class ImageResponse:
    """Generated images with usage, pricing, and latency metadata."""

    images: tuple[GeneratedImage, ...]
    usage: ImageUsage
    input_cost: float
    output_cost: float
    total_cost: float
    response_time: float
    provider: str
    model: str

    def __post_init__(self) -> None:
        if not self.images:
            raise ValueError("ImageResponse.images must not be empty")


@dataclass(frozen=True)
class ImageHookRequest:
    """Immutable image operation supplied to before and after hooks."""

    operation: Literal["generate", "edit"]
    prompt: str
    images: tuple[ImageInput, ...] = ()
    mask: ImageInput | None = None
    count: int = 1
    aspect_ratio: ImageAspectRatio = "1:1"
    image_size: ImageSize = "1K"
    output_format: ImageFormat = "jpeg"
    quality: ImageQuality = "auto"
    background: ImageBackground = "auto"


class ImageModel(ABC):
    """Common async interface for image generation and editing providers."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        text_input_cost: float,
        image_input_cost: float,
        text_output_cost: float,
        image_output_cost: float,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        hook_pipeline: "ImageHookPipeline | None" = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.text_input_cost = text_input_cost
        self.image_input_cost = image_input_cost
        self.text_output_cost = text_output_cost
        self.image_output_cost = image_output_cost
        self.api_key_hash = _hash_api_key(api_key) if api_key else None
        self.api_key_alias = api_key_alias
        self.hook_pipeline = hook_pipeline

    def get_full_model_name(self) -> str:
        return f"{self.provider}:{self.model}"

    def _calculate_costs(self, usage: ImageUsage) -> tuple[float, float, float]:
        input_cost = (
            usage.text_input_tokens * self.text_input_cost
            + usage.image_input_tokens * self.image_input_cost
        ) / TOKENS_PER_MILLION
        output_cost = (
            usage.text_output_tokens * self.text_output_cost
            + usage.image_output_tokens * self.image_output_cost
        ) / TOKENS_PER_MILLION
        return input_cost, output_cost, input_cost + output_cost

    async def generate(
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
        """Generate one or more images from a text prompt."""
        request = ImageHookRequest(
            operation="generate",
            prompt=prompt,
            count=count,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            output_format=output_format,
            quality=quality,
            background=background,
        )
        return await self._execute_request(request, extra_headers, caller_metadata)

    async def edit(
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
        """Edit one or more reference images using a text prompt."""
        request = ImageHookRequest(
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
        return await self._execute_request(request, extra_headers, caller_metadata)

    async def _execute_request(
        self,
        request: ImageHookRequest,
        extra_headers: dict[str, str] | None,
        caller_metadata: dict[str, Any] | None,
    ) -> ImageResponse:
        if self.hook_pipeline is None:
            return await self._dispatch_request(request, extra_headers, caller_metadata)

        async def call(modified: ImageHookRequest) -> ImageResponse:
            return await self._dispatch_request(modified, extra_headers, caller_metadata)

        return await self.hook_pipeline.run(request, call, caller_metadata=caller_metadata)

    async def _dispatch_request(
        self,
        request: ImageHookRequest,
        extra_headers: dict[str, str] | None,
        caller_metadata: dict[str, Any] | None,
    ) -> ImageResponse:
        if request.operation == "generate":
            return await self._generate_impl(
                request.prompt,
                count=request.count,
                aspect_ratio=request.aspect_ratio,
                image_size=request.image_size,
                output_format=request.output_format,
                quality=request.quality,
                background=request.background,
                extra_headers=extra_headers,
                caller_metadata=caller_metadata,
            )
        return await self._edit_impl(
            request.prompt,
            request.images,
            mask=request.mask,
            count=request.count,
            aspect_ratio=request.aspect_ratio,
            image_size=request.image_size,
            output_format=request.output_format,
            quality=request.quality,
            background=request.background,
            extra_headers=extra_headers,
            caller_metadata=caller_metadata,
        )

    @abstractmethod
    async def _generate_impl(
        self,
        prompt: str,
        *,
        count: int,
        aspect_ratio: ImageAspectRatio,
        image_size: ImageSize,
        output_format: ImageFormat,
        quality: ImageQuality,
        background: ImageBackground,
        extra_headers: dict[str, str] | None,
        caller_metadata: dict[str, Any] | None,
    ) -> ImageResponse: ...

    @abstractmethod
    async def _edit_impl(
        self,
        prompt: str,
        images: tuple[ImageInput, ...],
        *,
        mask: ImageInput | None,
        count: int,
        aspect_ratio: ImageAspectRatio,
        image_size: ImageSize,
        output_format: ImageFormat,
        quality: ImageQuality,
        background: ImageBackground,
        extra_headers: dict[str, str] | None,
        caller_metadata: dict[str, Any] | None,
    ) -> ImageResponse: ...


def validate_image_count(count: int) -> None:
    """Validate the provider-neutral output-count range."""
    if not 1 <= count <= 10:
        raise ValueError(f"count must be between 1 and 10, got {count}")
