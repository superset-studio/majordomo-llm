"""Provider-neutral built-in hooks for image operations."""

import io
import re
from dataclasses import dataclass, field, replace
from typing import Literal

from PIL import Image, UnidentifiedImageError

from majordomo_llm.base import ImageInput
from majordomo_llm.hooks.image_protocol import ImageHookOutcome
from majordomo_llm.hooks.protocol import HookContext
from majordomo_llm.image import ImageHookRequest, ImageResponse

_FORMAT_MEDIA_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


@dataclass
class ImagePromptRegexHook:
    """Apply block, warn, or redact behavior to an image prompt before generation."""

    name: str
    pattern: str
    flags: int = 0
    action: Literal["block", "warn", "redact"] = "warn"
    redaction: str = "[REDACTED]"
    _compiled: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern, self.flags)

    async def before_call(self, request: ImageHookRequest, ctx: HookContext) -> ImageHookOutcome:
        del ctx
        if self._compiled.search(request.prompt) is None:
            return ImageHookOutcome.pass_through(self.name)
        reason = f"prompt matched pattern {self.pattern!r}"
        if self.action == "block":
            return ImageHookOutcome.block(self.name, reason)
        if self.action == "warn":
            return ImageHookOutcome.warn(self.name, reason)
        modified = replace(request, prompt=self._compiled.sub(self.redaction, request.prompt))
        return ImageHookOutcome.modify_request(self.name, modified, reason)

    async def after_call(
        self,
        request: ImageHookRequest,
        response: ImageResponse,
        ctx: HookContext,
    ) -> ImageHookOutcome:
        del request, response, ctx
        return ImageHookOutcome.pass_through(self.name)


@dataclass
class ImageRequestLimitsHook:
    """Block image requests that exceed configured cost or payload limits."""

    name: str
    max_count: int = 1
    max_reference_images: int = 10
    max_total_input_bytes: int = 20 * 1024 * 1024
    allowed_image_sizes: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.max_count < 1:
            raise ValueError("max_count must be at least 1")
        if self.max_reference_images < 0:
            raise ValueError("max_reference_images must not be negative")
        if self.max_total_input_bytes < 1:
            raise ValueError("max_total_input_bytes must be at least 1")

    async def before_call(self, request: ImageHookRequest, ctx: HookContext) -> ImageHookOutcome:
        del ctx
        reason = self._violation(request)
        if reason is not None:
            return ImageHookOutcome.block(self.name, reason)
        return ImageHookOutcome.pass_through(self.name)

    async def after_call(
        self,
        request: ImageHookRequest,
        response: ImageResponse,
        ctx: HookContext,
    ) -> ImageHookOutcome:
        del request, response, ctx
        return ImageHookOutcome.pass_through(self.name)

    def _violation(self, request: ImageHookRequest) -> str | None:
        if request.count > self.max_count:
            return f"requested count {request.count} exceeds maximum {self.max_count}"
        if len(request.images) > self.max_reference_images:
            return (
                f"reference image count {len(request.images)} exceeds maximum "
                f"{self.max_reference_images}"
            )
        total_bytes = sum(len(image.data) for image in request.images)
        if request.mask is not None:
            total_bytes += len(request.mask.data)
        if total_bytes > self.max_total_input_bytes:
            return f"input image bytes {total_bytes} exceed maximum {self.max_total_input_bytes}"
        if (
            self.allowed_image_sizes is not None
            and request.image_size not in self.allowed_image_sizes
        ):
            allowed = ", ".join(sorted(self.allowed_image_sizes))
            return f"image size {request.image_size} is not allowed; allowed: {allowed}"
        return None


@dataclass
class ImageIntegrityHook:
    """Verify that input and generated image bytes decode and match their MIME type."""

    name: str
    max_pixels: int = 40_000_000

    def __post_init__(self) -> None:
        if self.max_pixels < 1:
            raise ValueError("max_pixels must be at least 1")

    async def before_call(self, request: ImageHookRequest, ctx: HookContext) -> ImageHookOutcome:
        del ctx
        inputs = list(request.images)
        if request.mask is not None:
            inputs.append(request.mask)
        for index, image in enumerate(inputs):
            problem = self._validate(image)
            if problem is not None:
                return ImageHookOutcome.block(
                    self.name, f"input image {index} failed integrity validation: {problem}"
                )
        return ImageHookOutcome.pass_through(self.name)

    async def after_call(
        self,
        request: ImageHookRequest,
        response: ImageResponse,
        ctx: HookContext,
    ) -> ImageHookOutcome:
        del request, ctx
        for index, generated in enumerate(response.images):
            problem: str | None
            try:
                image = ImageInput(generated.data, generated.media_type)
            except ValueError as exc:
                problem = str(exc)
            else:
                problem = self._validate(image)
            if problem is not None:
                return ImageHookOutcome.retry(
                    self.name,
                    f"generated image {index} failed integrity validation: {problem}",
                )
        return ImageHookOutcome.pass_through(self.name)

    def _validate(self, image: ImageInput) -> str | None:
        try:
            with Image.open(io.BytesIO(image.data)) as decoded:
                actual_media_type = _FORMAT_MEDIA_TYPES.get(decoded.format or "")
                if actual_media_type != image.media_type:
                    return (
                        f"declared MIME type {image.media_type} does not match "
                        f"decoded format {decoded.format}"
                    )
                if decoded.width * decoded.height > self.max_pixels:
                    return (
                        f"pixel count {decoded.width * decoded.height} exceeds "
                        f"maximum {self.max_pixels}"
                    )
                decoded.verify()
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
            return f"image could not be decoded ({exc})"
        return None
