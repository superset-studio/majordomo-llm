"""Hook primitives for intercepting LLM calls.

Hooks attach to any :class:`~majordomo_llm.LLM` (provider or cascade) via
the ``hook_pipeline`` constructor kwarg. The pipeline runs ``before_call``
hooks on the prompt, invokes the LLM, and runs ``after_call`` hooks on
the response. Hooks can pass, warn, redact, or block.

Example:
    >>> from majordomo_llm import OpenAI
    >>> from majordomo_llm.hooks import HookPipeline, RegexHook
    >>> pipeline = HookPipeline([
    ...     RegexHook(name="ssn", pattern=r"\\d{3}-\\d{2}-\\d{4}", action="redact"),
    ... ])
    >>> llm = OpenAI(model="gpt-5", input_cost=..., output_cost=..., hook_pipeline=pipeline)
"""

from majordomo_llm.hooks.image_builtin import (
    ImageIntegrityHook,
    ImagePromptRegexHook,
    ImageRequestLimitsHook,
)
from majordomo_llm.hooks.image_pipeline import ImageHookPipeline, ImageHookState
from majordomo_llm.hooks.image_protocol import (
    ImageHook,
    ImageHookOutcome,
    ImageHookRetryRequested,
)
from majordomo_llm.hooks.llm_judge_hook import LLMJudgeHook
from majordomo_llm.hooks.pipeline import HookPipeline, OnVerdicts
from majordomo_llm.hooks.protocol import (
    HookBlocked,
    HookContext,
    HookOutcome,
    HookVerdict,
    LLMHook,
)
from majordomo_llm.hooks.regex_hook import RegexHook

__all__ = [
    "HookBlocked",
    "HookContext",
    "HookOutcome",
    "HookPipeline",
    "HookVerdict",
    "ImageHook",
    "ImageHookOutcome",
    "ImageHookPipeline",
    "ImageHookRetryRequested",
    "ImageHookState",
    "ImageIntegrityHook",
    "ImagePromptRegexHook",
    "ImageRequestLimitsHook",
    "LLMHook",
    "LLMJudgeHook",
    "OnVerdicts",
    "RegexHook",
]
