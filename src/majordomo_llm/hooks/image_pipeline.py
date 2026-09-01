"""Ordered hook pipeline for image generation and editing."""

import inspect
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from majordomo_llm.hooks.image_protocol import (
    ImageHook,
    ImageHookOutcome,
    ImageHookRetryRequested,
)
from majordomo_llm.hooks.pipeline import OnVerdicts
from majordomo_llm.hooks.protocol import HookBlocked, HookContext, HookVerdict
from majordomo_llm.image import ImageHookRequest, ImageResponse

logger = logging.getLogger(__name__)


@dataclass
class ImageHookState:
    """Mutable state shared across one before/after pipeline run."""

    request_id: uuid.UUID
    request: ImageHookRequest
    caller_metadata: dict[str, Any]
    verdicts: list[HookVerdict]


class ImageHookPipeline:
    """Run typed image hooks in order around an image-model call."""

    def __init__(
        self,
        hooks: list[ImageHook],
        *,
        on_verdicts: OnVerdicts | None = None,
    ) -> None:
        self._hooks = list(hooks)
        self._on_verdicts = on_verdicts

    async def run(
        self,
        request: ImageHookRequest,
        call: Callable[[ImageHookRequest], Awaitable[ImageResponse]],
        *,
        caller_metadata: dict[str, Any] | None = None,
    ) -> ImageResponse:
        state = await self.run_before(request, caller_metadata=caller_metadata)
        response = await call(state.request)
        return await self.run_after(state, response)

    async def run_before(
        self,
        request: ImageHookRequest,
        *,
        caller_metadata: dict[str, Any] | None = None,
    ) -> ImageHookState:
        state = ImageHookState(
            request_id=uuid.uuid4(),
            request=request,
            caller_metadata=caller_metadata if caller_metadata is not None else {},
            verdicts=[],
        )
        ctx = HookContext(state.request_id, "before", state.caller_metadata)
        for hook in self._hooks:
            outcome = await self._run_hook(
                hook.before_call(state.request, ctx), hook_name=hook.name
            )
            state.verdicts.append(outcome.verdict)
            if outcome.modified_request is not None:
                state.request = outcome.modified_request
            if outcome.modified_response is not None or outcome.retry_next_provider:
                logger.warning(
                    "Image hook %s returned an after-only outcome during before_call; ignoring",
                    hook.name,
                )
            if outcome.blocked:
                await self.emit(state)
                raise HookBlocked(state.verdicts)
        return state

    async def run_after(
        self,
        state: ImageHookState,
        response: ImageResponse,
        *,
        emit_on_retry: bool = True,
    ) -> ImageResponse:
        ctx = HookContext(state.request_id, "after", state.caller_metadata)
        current_response = response
        for hook in self._hooks:
            outcome = await self._run_hook(
                hook.after_call(state.request, current_response, ctx),
                hook_name=hook.name,
            )
            state.verdicts.append(outcome.verdict)
            if outcome.modified_response is not None:
                current_response = outcome.modified_response
            if outcome.modified_request is not None:
                logger.warning(
                    "Image hook %s returned a before-only outcome during after_call; ignoring",
                    hook.name,
                )
            if outcome.blocked:
                await self.emit(state)
                raise HookBlocked(state.verdicts)
            if outcome.retry_next_provider:
                if emit_on_retry:
                    await self.emit(state)
                raise ImageHookRetryRequested(state.verdicts)
        await self.emit(state)
        return current_response

    async def _run_hook(
        self,
        hook_coro: Awaitable[ImageHookOutcome],
        *,
        hook_name: str,
    ) -> ImageHookOutcome:
        start = time.monotonic()
        try:
            outcome = await hook_coro
        except HookBlocked:
            raise
        except Exception:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.exception("Image hook %s raised; treating as pass", hook_name)
            return ImageHookOutcome(
                HookVerdict(
                    hook_name,
                    "pass",
                    "pass",
                    reason="hook raised exception",
                    latency_ms=elapsed_ms,
                )
            )
        if outcome.verdict.latency_ms is not None:
            return outcome
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ImageHookOutcome(
            HookVerdict(
                outcome.verdict.hook_name,
                outcome.verdict.verdict,
                outcome.verdict.action_taken,
                reason=outcome.verdict.reason,
                latency_ms=elapsed_ms,
            ),
            modified_request=outcome.modified_request,
            modified_response=outcome.modified_response,
            blocked=outcome.blocked,
            retry_next_provider=outcome.retry_next_provider,
        )

    async def emit(self, state: ImageHookState) -> None:
        if self._on_verdicts is None:
            return
        try:
            result = self._on_verdicts(state.request_id, state.verdicts)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Image hook on_verdicts callback raised; swallowing")
