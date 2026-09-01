"""Hook protocol and value types for LLM call interception."""

import uuid
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from majordomo_llm.exceptions import MajordomoError


@dataclass(frozen=True)
class HookContext:
    """Per-call metadata passed to every hook.

    Populated by :class:`HookPipeline` for each invocation. Hooks read it
    to make context-aware decisions (e.g. dispatch on caller-supplied
    metadata) and never mutate it.

    Attributes:
        request_id: Unique id generated for this pipeline run; correlates
            ``before_call`` and ``after_call`` verdicts.
        phase: Which phase the hook is currently executing in.
        caller_metadata: Free-form dict supplied by the caller via
            ``caller_metadata`` kwarg on the LLM method. The library does
            not interpret it.
    """

    request_id: uuid.UUID
    phase: Literal["before", "after"]
    caller_metadata: dict[str, Any]


@dataclass(frozen=True)
class HookVerdict:
    """The recorded outcome of a single hook evaluation."""

    hook_name: str
    verdict: Literal["pass", "fail", "warn"]
    action_taken: Literal["pass", "block", "warn", "redact", "modify", "retry"]
    reason: str | None = None
    latency_ms: int | None = None


@dataclass(frozen=True)
class HookOutcome:
    """What a hook returns from ``before_call`` or ``after_call``.

    Attributes:
        verdict: Structured record of the evaluation.
        modified_text: When not None, replaces the prompt (in ``before_call``)
            or the response (in ``after_call``) for downstream hooks and
            the final caller.
        blocked: When True, the pipeline raises :class:`HookBlocked` after
            recording the verdict.
    """

    verdict: HookVerdict
    modified_text: str | None = None
    blocked: bool = False

    @staticmethod
    def pass_through(hook_name: str, reason: str | None = None) -> "HookOutcome":
        """Build a pass-through outcome (no change, not blocked)."""
        return HookOutcome(
            verdict=HookVerdict(
                hook_name=hook_name,
                verdict="pass",
                action_taken="pass",
                reason=reason,
            )
        )

    @staticmethod
    def block(hook_name: str, reason: str) -> "HookOutcome":
        """Build a blocking outcome; the pipeline will raise ``HookBlocked``."""
        return HookOutcome(
            verdict=HookVerdict(
                hook_name=hook_name,
                verdict="fail",
                action_taken="block",
                reason=reason,
            ),
            blocked=True,
        )

    @staticmethod
    def redact(hook_name: str, modified: str, reason: str) -> "HookOutcome":
        """Build a redact outcome that replaces the text with ``modified``."""
        return HookOutcome(
            verdict=HookVerdict(
                hook_name=hook_name,
                verdict="fail",
                action_taken="redact",
                reason=reason,
            ),
            modified_text=modified,
        )

    @staticmethod
    def warn(hook_name: str, reason: str) -> "HookOutcome":
        """Build a warning outcome (text unchanged, not blocked)."""
        return HookOutcome(
            verdict=HookVerdict(
                hook_name=hook_name,
                verdict="warn",
                action_taken="warn",
                reason=reason,
            )
        )


class HookBlocked(MajordomoError):
    """Raised when a hook in a :class:`HookPipeline` blocks an LLM call.

    Carries the full list of verdicts produced up to and including the
    blocking hook so the caller can introspect which hook fired and why.

    Attributes:
        verdicts: All verdicts collected before the block, in order.
    """

    def __init__(self, verdicts: list[HookVerdict]) -> None:
        self.verdicts = verdicts
        blocking = next(
            (v for v in verdicts if v.action_taken == "block"),
            None,
        )
        if blocking is not None:
            message = f"Call blocked by hook {blocking.hook_name!r}: {blocking.reason}"
        else:
            message = "Call blocked by hook pipeline"
        super().__init__(message)


@runtime_checkable
class LLMHook(Protocol):
    """Protocol every hook implementation must satisfy.

    Either method may be a no-op by returning
    :meth:`HookOutcome.pass_through`. Both methods must be async even when
    they do no I/O — the pipeline always awaits them.
    """

    name: str

    async def before_call(self, prompt: str, ctx: HookContext) -> HookOutcome: ...

    async def after_call(
        self, prompt: str, response: str, ctx: HookContext
    ) -> HookOutcome: ...
