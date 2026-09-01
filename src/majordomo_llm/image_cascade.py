"""Automatic failover across image-generation providers."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from tenacity import RetryError

from majordomo_llm.base import ImageInput
from majordomo_llm.exceptions import (
    ImageOptionUnsupported,
    ProviderError,
    ResponseParsingError,
)
from majordomo_llm.factory import get_image_instance
from majordomo_llm.hooks.image_pipeline import ImageHookPipeline, ImageHookState
from majordomo_llm.hooks.image_protocol import ImageHookRetryRequested
from majordomo_llm.image import (
    ImageAspectRatio,
    ImageBackground,
    ImageFormat,
    ImageHookRequest,
    ImageModel,
    ImageQuality,
    ImageResponse,
    ImageSize,
)

logger = logging.getLogger(__name__)

_FAILOVER_EXCEPTIONS = (ProviderError, ResponseParsingError, ImageOptionUnsupported)


class ImageCascade(ImageModel):
    """Try image-generation models in priority order until one succeeds."""

    def __init__(
        self,
        providers: list[tuple[str, str]],
        *,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        hook_pipeline: ImageHookPipeline | None = None,
    ) -> None:
        if not providers:
            raise ValueError("ImageCascade requires at least one provider")

        self.models = [
            get_image_instance(
                provider,
                model,
                api_key=api_key,
                api_key_alias=api_key_alias,
                base_url=base_url,
                default_headers=default_headers,
            )
            for provider, model in providers
        ]
        primary = self.models[0]
        super().__init__(
            provider="cascade",
            model=primary.model,
            text_input_cost=primary.text_input_cost,
            image_input_cost=primary.image_input_cost,
            text_output_cost=primary.text_output_cost,
            image_output_cost=primary.image_output_cost,
            hook_pipeline=hook_pipeline,
        )
        self.api_key_hash = primary.api_key_hash
        self.api_key_alias = primary.api_key_alias

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
        """Generate with the first model that accepts the options and succeeds."""
        return await self._cascade_call(
            "generate",
            prompt=prompt,
            count=count,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            output_format=output_format,
            quality=quality,
            background=background,
            extra_headers=extra_headers,
            caller_metadata=caller_metadata,
        )

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
        """Edit with the first model that accepts the options and succeeds."""
        return await self._cascade_call(
            "edit",
            prompt=prompt,
            images=images,
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

    async def _execute_request(
        self,
        request: ImageHookRequest,
        extra_headers: dict[str, str] | None,
        caller_metadata: dict[str, Any] | None,
    ) -> ImageResponse:
        if self.hook_pipeline is None:
            return await super()._execute_request(request, extra_headers, caller_metadata)

        state = await self.hook_pipeline.run_before(request, caller_metadata=caller_metadata)
        kwargs: dict[str, Any] = {
            "prompt": state.request.prompt,
            "count": state.request.count,
            "aspect_ratio": state.request.aspect_ratio,
            "image_size": state.request.image_size,
            "output_format": state.request.output_format,
            "quality": state.request.quality,
            "background": state.request.background,
            "extra_headers": extra_headers,
            "caller_metadata": caller_metadata,
            "hook_state": state,
        }
        if state.request.operation == "edit":
            kwargs["images"] = state.request.images
            kwargs["mask"] = state.request.mask
        return await self._cascade_call(state.request.operation, **kwargs)

    async def _cascade_call(
        self,
        method_name: Literal["generate", "edit"],
        hook_state: ImageHookState | None = None,
        **kwargs: Any,
    ) -> ImageResponse:
        last_error: Exception | None = None
        for model in self.models:
            try:
                method = cast(
                    Callable[..., Awaitable[ImageResponse]],
                    getattr(model, method_name),
                )
                response = await method(**kwargs)
                if self.hook_pipeline is not None and hook_state is not None:
                    try:
                        return await self.hook_pipeline.run_after(
                            hook_state, response, emit_on_retry=False
                        )
                    except ImageHookRetryRequested as exc:
                        self._log_provider_failure(model, exc)
                        last_error = exc
                        continue
                return response
            except _FAILOVER_EXCEPTIONS as exc:
                self._log_provider_failure(model, exc)
                last_error = exc
            except RetryError as exc:
                cause = exc.last_attempt.exception()
                if not isinstance(cause, _FAILOVER_EXCEPTIONS):
                    raise
                self._log_provider_failure(model, cause)
                last_error = cause

        if self.hook_pipeline is not None and hook_state is not None:
            await self.hook_pipeline.emit(hook_state)
        raise ProviderError(
            f"All image providers in cascade failed. Last error: {last_error}",
            provider="cascade",
            original_error=last_error,
        )

    @staticmethod
    def _log_provider_failure(model: ImageModel, exc: Exception) -> None:
        logger.warning(
            "Image provider %s/%s failed: %s. Trying next provider.",
            model.provider,
            model.model,
            exc,
        )
