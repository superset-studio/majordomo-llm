"""Typed hook contracts for image generation and editing."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from majordomo_llm.exceptions import MajordomoError
from majordomo_llm.hooks.protocol import HookContext, HookVerdict
from majordomo_llm.image import ImageHookRequest, ImageResponse


@dataclass(frozen=True)
class ImageHookOutcome:
    """Result of an image hook evaluation."""

    verdict: HookVerdict
    modified_request: ImageHookRequest | None = None
    modified_response: ImageResponse | None = None
    blocked: bool = False
    retry_next_provider: bool = False

    @staticmethod
    def pass_through(hook_name: str, reason: str | None = None) -> "ImageHookOutcome":
        return ImageHookOutcome(HookVerdict(hook_name, "pass", "pass", reason=reason))

    @staticmethod
    def warn(hook_name: str, reason: str) -> "ImageHookOutcome":
        return ImageHookOutcome(HookVerdict(hook_name, "warn", "warn", reason=reason))

    @staticmethod
    def block(hook_name: str, reason: str) -> "ImageHookOutcome":
        return ImageHookOutcome(
            HookVerdict(hook_name, "fail", "block", reason=reason),
            blocked=True,
        )

    @staticmethod
    def modify_request(
        hook_name: str, request: ImageHookRequest, reason: str
    ) -> "ImageHookOutcome":
        return ImageHookOutcome(
            HookVerdict(hook_name, "fail", "modify", reason=reason),
            modified_request=request,
        )

    @staticmethod
    def modify_response(hook_name: str, response: ImageResponse, reason: str) -> "ImageHookOutcome":
        return ImageHookOutcome(
            HookVerdict(hook_name, "fail", "modify", reason=reason),
            modified_response=response,
        )

    @staticmethod
    def retry(hook_name: str, reason: str) -> "ImageHookOutcome":
        return ImageHookOutcome(
            HookVerdict(hook_name, "fail", "retry", reason=reason),
            retry_next_provider=True,
        )


class ImageHookRetryRequested(MajordomoError):
    """Raised when an after hook rejects a result and requests another provider."""

    def __init__(self, verdicts: list[HookVerdict]) -> None:
        self.verdicts = verdicts
        retry = next((v for v in reversed(verdicts) if v.action_taken == "retry"), None)
        reason = retry.reason if retry is not None else "image hook rejected provider output"
        super().__init__(f"Image hook requested the next provider: {reason}")


@runtime_checkable
class ImageHook(Protocol):
    """Protocol implemented by typed image request/response hooks."""

    name: str

    async def before_call(
        self, request: ImageHookRequest, ctx: HookContext
    ) -> ImageHookOutcome: ...

    async def after_call(
        self,
        request: ImageHookRequest,
        response: ImageResponse,
        ctx: HookContext,
    ) -> ImageHookOutcome: ...
