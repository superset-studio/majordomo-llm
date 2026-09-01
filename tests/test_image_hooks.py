"""Tests for typed image hooks, built-ins, and cascade quality failover."""

import io
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from majordomo_llm import (
    GeneratedImage,
    HookBlocked,
    ImageCascade,
    ImageHookOutcome,
    ImageHookPipeline,
    ImageHookRetryRequested,
    ImageInput,
    ImageIntegrityHook,
    ImageModel,
    ImagePromptRegexHook,
    ImageRequestLimitsHook,
    ImageResponse,
    ImageUsage,
)


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "blue").save(output, format="PNG")
    return output.getvalue()


def _response(data: bytes, *, provider: str = "openai") -> ImageResponse:
    return ImageResponse(
        images=(GeneratedImage(data, "image/png"),),
        usage=ImageUsage(image_output_tokens=10),
        input_cost=0.0,
        output_cost=0.01,
        total_cost=0.01,
        response_time=0.1,
        provider=provider,
        model=f"{provider}-image",
    )


class _StubImageModel(ImageModel):
    def __init__(self, response: ImageResponse, hook_pipeline=None):
        super().__init__(
            provider="stub",
            model="stub-image",
            text_input_cost=0,
            image_input_cost=0,
            text_output_cost=0,
            image_output_cost=0,
            hook_pipeline=hook_pipeline,
        )
        self.response = response
        self.calls: list[tuple[str, dict | None]] = []

    async def _generate_impl(self, prompt: str, **kwargs) -> ImageResponse:
        self.calls.append((prompt, kwargs["caller_metadata"]))
        return self.response

    async def _edit_impl(self, prompt: str, images, **kwargs) -> ImageResponse:
        self.calls.append((prompt, kwargs["caller_metadata"]))
        return self.response


class _Hook:
    def __init__(self, name="hook", before=None, after=None):
        self.name = name
        self._before = before
        self._after = after
        self.contexts = []

    async def before_call(self, request, ctx):
        self.contexts.append(ctx)
        if self._before is None:
            return ImageHookOutcome.pass_through(self.name)
        return self._before(request, ctx)

    async def after_call(self, request, response, ctx):
        self.contexts.append(ctx)
        if self._after is None:
            return ImageHookOutcome.pass_through(self.name)
        return self._after(request, response, ctx)


async def test_before_hook_can_modify_prompt_and_receives_metadata():
    hook = _Hook(
        before=lambda request, _: ImageHookOutcome.modify_request(
            "hook", replace(request, prompt="safe prompt"), "normalized"
        )
    )
    model = _StubImageModel(_response(_png_bytes()), ImageHookPipeline([hook]))

    await model.generate("original", caller_metadata={"tenant": "acme"})

    assert model.calls == [("safe prompt", {"tenant": "acme"})]
    assert [context.phase for context in hook.contexts] == ["before", "after"]
    assert hook.contexts[0].request_id == hook.contexts[1].request_id


async def test_before_block_prevents_provider_call():
    hook = _Hook(before=lambda request, _: ImageHookOutcome.block("hook", "not allowed"))
    model = _StubImageModel(_response(_png_bytes()), ImageHookPipeline([hook]))

    with pytest.raises(HookBlocked, match="not allowed"):
        await model.generate("blocked")

    assert model.calls == []


async def test_after_hook_can_replace_response():
    replacement = _response(_png_bytes(), provider="filtered")
    hook = _Hook(
        after=lambda request, response, _: ImageHookOutcome.modify_response(
            "hook", replacement, "post-processed"
        )
    )
    model = _StubImageModel(_response(_png_bytes()), ImageHookPipeline([hook]))

    assert await model.generate("prompt") is replacement


async def test_retry_outcome_surfaces_on_direct_model():
    hook = _Hook(after=lambda request, response, _: ImageHookOutcome.retry("hook", "bad image"))
    model = _StubImageModel(_response(_png_bytes()), ImageHookPipeline([hook]))

    with pytest.raises(ImageHookRetryRequested, match="bad image"):
        await model.generate("prompt")


async def test_hook_exception_is_recorded_as_pass():
    recorded = []

    def explode(request, ctx):
        raise RuntimeError("bug")

    hook = _Hook(before=explode)
    pipeline = ImageHookPipeline(
        [hook], on_verdicts=lambda request_id, verdicts: recorded.extend(verdicts)
    )
    model = _StubImageModel(_response(_png_bytes()), pipeline)

    await model.generate("prompt")

    assert recorded[0].action_taken == "pass"
    assert recorded[0].reason == "hook raised exception"


async def test_prompt_regex_redacts_before_provider():
    hook = ImagePromptRegexHook(name="secret", pattern="secret", action="redact", redaction="safe")
    model = _StubImageModel(_response(_png_bytes()), ImageHookPipeline([hook]))

    await model.generate("draw a secret document")

    assert model.calls[0][0] == "draw a safe document"


@pytest.mark.parametrize(
    ("hook", "kwargs", "message"),
    [
        (ImageRequestLimitsHook("limits", max_count=1), {"count": 2}, "count 2"),
        (
            ImageRequestLimitsHook("limits", allowed_image_sizes=frozenset({"1K"})),
            {"image_size": "2K"},
            "size 2K",
        ),
    ],
)
async def test_request_limits_block_before_spend(hook, kwargs, message):
    model = _StubImageModel(_response(_png_bytes()), ImageHookPipeline([hook]))

    with pytest.raises(HookBlocked, match=message):
        await model.generate("prompt", **kwargs)

    assert model.calls == []


async def test_integrity_hook_blocks_invalid_reference_image():
    hook = ImageIntegrityHook("integrity")
    model = _StubImageModel(_response(_png_bytes()), ImageHookPipeline([hook]))

    with pytest.raises(HookBlocked, match="could not be decoded"):
        await model.edit("edit", (ImageInput(b"not-an-image", "image/png"),))


async def test_integrity_hook_accepts_valid_images():
    hook = ImageIntegrityHook("integrity")
    model = _StubImageModel(_response(_png_bytes()), ImageHookPipeline([hook]))

    response = await model.edit("edit", (ImageInput(_png_bytes(), "image/png"),))

    assert response.images[0].data == _png_bytes()


async def test_integrity_rejection_advances_image_cascade():
    first = MagicMock(spec=ImageModel)
    second = MagicMock(spec=ImageModel)
    for provider, child in (("openai", first), ("gemini", second)):
        child.provider = provider
        child.model = f"{provider}-image"
        child.text_input_cost = 0.0
        child.image_input_cost = 0.0
        child.text_output_cost = 0.0
        child.image_output_cost = 0.0
        child.api_key_hash = None
        child.api_key_alias = None
        child.generate = AsyncMock()
    first.generate.return_value = _response(b"invalid", provider="openai")
    second.generate.return_value = _response(_png_bytes(), provider="gemini")
    pipeline = ImageHookPipeline([ImageIntegrityHook("integrity")])

    with patch(
        "majordomo_llm.image_cascade.get_image_instance",
        side_effect=[first, second],
    ):
        cascade = ImageCascade(
            [("openai", "openai-image"), ("gemini", "gemini-image")],
            hook_pipeline=pipeline,
        )

    response = await cascade.generate("prompt")

    assert response.provider == "gemini"
    first.generate.assert_awaited_once()
    second.generate.assert_awaited_once()
