"""Run billable image-generation and editing smoke tests against live providers.

The script loads API keys from ``.env`` through python-dotenv, but never prints
credentials, prompts, or image bytes. Output is limited to operational metadata.

Usage:
  uv run python scripts/smoke_test_images.py
  uv run python scripts/smoke_test_images.py --provider openai
  uv run python scripts/smoke_test_images.py --operation generate
  uv run python scripts/smoke_test_images.py --provider gemini --model MODEL
  uv run python scripts/smoke_test_images.py --all-models
  uv run python scripts/smoke_test_images.py --failure-tests
  uv run python scripts/smoke_test_images.py --failure-tests --failure-case gemini-auth
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, cast

from dotenv import load_dotenv
from PIL import Image

from majordomo_llm import (
    ImageCascade,
    ImageInput,
    ImageResponse,
    get_image_instance,
    get_supported_image_models,
    get_supported_image_providers,
)
from majordomo_llm.exceptions import ProviderError

Operation = Literal["generate", "edit"]
FailureCase = Literal[
    "openai-auth",
    "gemini-auth",
    "openai-to-gemini",
    "gemini-to-openai",
    "exhausted",
]

PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_FAILURE_TEST_PROVIDERS = ("openai", "gemini")
_FAILURE_CASES: tuple[FailureCase, ...] = (
    "openai-auth",
    "gemini-auth",
    "openai-to-gemini",
    "gemini-to-openai",
    "exhausted",
)
_INVALID_API_KEY = "majordomo-smoke-test-invalid-key"
_AUTH_FAILURE_STATUS_CODES = frozenset({401, 403})

_PROMPTS: dict[Operation, str] = {
    "generate": "A simple blue circle centered on a plain white background.",
    "edit": "Add a thin green border around this simple geometric test image.",
}

_PIL_FORMAT_MEDIA_TYPES: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


@dataclass(frozen=True)
class ImageMetadata:
    media_type: str
    byte_length: int
    width: int
    height: int


@dataclass(frozen=True)
class SmokeResult:
    provider: str
    model: str
    operation: Operation
    status: Literal["pass", "fail"]
    elapsed: float
    image_count: int = 0
    total_bytes: int = 0
    total_cost: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class FailureTestResult:
    name: str
    status: Literal["pass", "fail"]
    elapsed: float
    detail: str
    total_cost: float = 0.0


@contextmanager
def _temporary_provider_keys(overrides: dict[str, str]) -> Iterator[None]:
    """Temporarily replace provider keys and restore the exact prior environment."""
    env_overrides = {PROVIDER_API_KEY_ENV[provider]: value for provider, value in overrides.items()}
    previous = {name: os.environ.get(name) for name in env_overrides}
    os.environ.update(env_overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _status_code(exc: BaseException | None) -> int | None:
    """Read status codes from both standard and Interactions provider errors."""
    if exc is None:
        return None
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _sanitized_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".replace(_INVALID_API_KEY, "[redacted]").replace(
        "\n", " "
    )[:500]


def _contains_error_reason(value: object, expected_reason: str) -> bool:
    """Search a provider JSON error body for a specific structured reason."""
    if isinstance(value, dict):
        if value.get("reason") == expected_reason:
            return True
        return any(_contains_error_reason(item, expected_reason) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_error_reason(item, expected_reason) for item in value)
    return False


def _validate_auth_failure(exc: ProviderError, *, expected_provider: str) -> int:
    if exc.provider != expected_provider:
        raise ValueError(
            f"provider mismatch: expected {expected_provider!r}, got {exc.provider!r}"
        )
    status_code = _status_code(exc.original_error)
    standard_auth_failure = status_code in _AUTH_FAILURE_STATUS_CODES
    gemini_invalid_key = (
        expected_provider == "gemini"
        and status_code == 400
        and _contains_error_reason(
            getattr(exc.original_error, "body", None),
            "API_KEY_INVALID",
        )
    )
    if not standard_auth_failure and not gemini_invalid_key:
        raise ValueError(
            "expected HTTP 401/403 or Gemini HTTP 400 with API_KEY_INVALID, "
            f"got {status_code!r}"
        )
    assert status_code is not None
    return status_code


def _edit_fixture() -> ImageInput:
    """Create a small, deterministic edit input without a committed binary fixture."""
    image = Image.new("RGB", (256, 256), "white")
    for x in range(80, 176):
        for y in range(80, 176):
            image.putpixel((x, y), (0, 102, 204))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return ImageInput(data=buffer.getvalue(), media_type="image/png")


def _validate_response(
    response: ImageResponse,
    *,
    expected_provider: str,
    expected_model: str,
) -> tuple[ImageMetadata, ...]:
    """Validate response metadata and decode every returned image with Pillow."""
    if response.provider != expected_provider:
        raise ValueError(
            f"provider mismatch: expected {expected_provider!r}, got {response.provider!r}"
        )
    if response.model != expected_model:
        raise ValueError(f"model mismatch: expected {expected_model!r}, got {response.model!r}")
    if response.response_time < 0:
        raise ValueError("response_time must not be negative")
    if min(response.input_cost, response.output_cost, response.total_cost) < 0:
        raise ValueError("cost values must not be negative")

    usage_values = (
        response.usage.text_input_tokens,
        response.usage.image_input_tokens,
        response.usage.text_output_tokens,
        response.usage.image_output_tokens,
    )
    if min(usage_values) < 0:
        raise ValueError("usage token counts must not be negative")

    metadata: list[ImageMetadata] = []
    for generated in response.images:
        try:
            with Image.open(io.BytesIO(generated.data)) as decoded:
                decoded.load()
                image_format = decoded.format
                width, height = decoded.size
        except (OSError, ValueError) as exc:
            raise ValueError("provider returned invalid image bytes") from exc

        if not image_format or image_format not in _PIL_FORMAT_MEDIA_TYPES:
            raise ValueError(f"unsupported decoded image format: {image_format!r}")
        decoded_media_type = _PIL_FORMAT_MEDIA_TYPES[image_format]
        if generated.media_type != decoded_media_type:
            raise ValueError(
                f"media type mismatch: response declared {generated.media_type!r}, "
                f"decoded as {decoded_media_type!r}"
            )
        if width <= 0 or height <= 0:
            raise ValueError("generated image dimensions must be positive")
        metadata.append(
            ImageMetadata(
                media_type=generated.media_type,
                byte_length=len(generated.data),
                width=width,
                height=height,
            )
        )
    return tuple(metadata)


def _select_models(
    providers: list[str],
    *,
    all_models: bool,
    requested_models: list[str] | None = None,
) -> list[tuple[str, str]]:
    if requested_models is not None:
        if len(providers) != 1:
            raise ValueError("exact model selection requires exactly one provider")
        provider = providers[0]
        available = get_supported_image_models(provider)
        unknown = [model for model in requested_models if model not in available]
        if unknown:
            raise ValueError(
                f"unknown image model for {provider}: {', '.join(unknown)}. "
                f"Available: {', '.join(available)}"
            )
        return [(provider, model) for model in dict.fromkeys(requested_models)]

    selected: list[tuple[str, str]] = []
    for provider in providers:
        models = get_supported_image_models(provider)
        if not models:
            raise ValueError(f"provider {provider!r} has no configured image models")
        selected.extend((provider, model) for model in (models if all_models else models[:1]))
    return selected


async def _run_operation(provider: str, model: str, operation: Operation) -> SmokeResult:
    start = time.monotonic()
    try:
        image_model = get_image_instance(provider, model)
        if operation == "generate":
            response = await image_model.generate(_PROMPTS[operation])
        else:
            response = await image_model.edit(
                _PROMPTS[operation],
                images=(_edit_fixture(),),
            )
        metadata = _validate_response(
            response,
            expected_provider=provider,
            expected_model=model,
        )
        return SmokeResult(
            provider=provider,
            model=model,
            operation=operation,
            status="pass",
            elapsed=time.monotonic() - start,
            image_count=len(metadata),
            total_bytes=sum(item.byte_length for item in metadata),
            total_cost=response.total_cost,
        )
    except Exception as exc:  # noqa: BLE001 - a smoke matrix must continue after one failure
        return SmokeResult(
            provider=provider,
            model=model,
            operation=operation,
            status="fail",
            elapsed=time.monotonic() - start,
            error=f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500],
        )


async def _run_all(
    models: list[tuple[str, str]], operations: list[Operation]
) -> list[SmokeResult]:
    results: list[SmokeResult] = []
    for provider, model in models:
        for operation in operations:
            print(f"[run ] {provider}/{model} {operation}", flush=True)
            result = await _run_operation(provider, model, operation)
            details = (
                f"images={result.image_count} bytes={result.total_bytes} "
                f"cost=${result.total_cost:.6f}"
                if result.status == "pass"
                else result.error
            )
            print(
                f"[{result.status:4}] {provider}/{model} {operation} "
                f"({result.elapsed:.1f}s) {details}",
                flush=True,
            )
            results.append(result)
    return results


async def _run_invalid_auth(provider: str, model: str) -> FailureTestResult:
    name = f"invalid-auth/{provider}"
    start = time.monotonic()
    try:
        with _temporary_provider_keys({provider: _INVALID_API_KEY}):
            image_model = get_image_instance(provider, model)
            await image_model.generate(_PROMPTS["generate"])
    except ProviderError as exc:
        try:
            status_code = _validate_auth_failure(exc, expected_provider=provider)
        except ValueError as validation_error:
            return FailureTestResult(
                name=name,
                status="fail",
                elapsed=time.monotonic() - start,
                detail=str(validation_error),
            )
        return FailureTestResult(
            name=name,
            status="pass",
            elapsed=time.monotonic() - start,
            detail=f"received normalized HTTP {status_code}",
        )
    except Exception as exc:  # noqa: BLE001 - report unexpected SDK exception types
        return FailureTestResult(
            name=name,
            status="fail",
            elapsed=time.monotonic() - start,
            detail=f"unwrapped error: {_sanitized_error(exc)}",
        )
    return FailureTestResult(
        name=name,
        status="fail",
        elapsed=time.monotonic() - start,
        detail="request unexpectedly succeeded with an invalid credential",
    )


async def _run_cascade_fallback(
    primary: str,
    fallback: str,
    models: dict[str, str],
) -> FailureTestResult:
    name = f"cascade/{primary}-to-{fallback}"
    start = time.monotonic()
    try:
        with _temporary_provider_keys({primary: _INVALID_API_KEY}):
            cascade = ImageCascade(
                [(primary, models[primary]), (fallback, models[fallback])]
            )
            response = await cascade.generate(_PROMPTS["generate"])
        metadata = _validate_response(
            response,
            expected_provider=fallback,
            expected_model=models[fallback],
        )
        return FailureTestResult(
            name=name,
            status="pass",
            elapsed=time.monotonic() - start,
            detail=f"failed over to {fallback}; images={len(metadata)}",
            total_cost=response.total_cost,
        )
    except Exception as exc:  # noqa: BLE001 - a failure test must report the actual outcome
        return FailureTestResult(
            name=name,
            status="fail",
            elapsed=time.monotonic() - start,
            detail=_sanitized_error(exc),
        )


async def _run_cascade_exhausted(models: dict[str, str]) -> FailureTestResult:
    name = "cascade/all-providers-fail"
    start = time.monotonic()
    providers = list(_FAILURE_TEST_PROVIDERS)
    try:
        with _temporary_provider_keys(
            {provider: _INVALID_API_KEY for provider in providers}
        ):
            cascade = ImageCascade([(provider, models[provider]) for provider in providers])
            await cascade.generate(_PROMPTS["generate"])
    except ProviderError as exc:
        if exc.provider != "cascade":
            return FailureTestResult(
                name=name,
                status="fail",
                elapsed=time.monotonic() - start,
                detail=f"expected cascade error, got provider {exc.provider!r}",
            )
        if not isinstance(exc.original_error, ProviderError):
            return FailureTestResult(
                name=name,
                status="fail",
                elapsed=time.monotonic() - start,
                detail="cascade error did not preserve the final ProviderError",
            )
        try:
            status_code = _validate_auth_failure(
                exc.original_error,
                expected_provider=providers[-1],
            )
        except ValueError as validation_error:
            return FailureTestResult(
                name=name,
                status="fail",
                elapsed=time.monotonic() - start,
                detail=str(validation_error),
            )
        return FailureTestResult(
            name=name,
            status="pass",
            elapsed=time.monotonic() - start,
            detail=f"preserved final {providers[-1]} HTTP {status_code} error",
        )
    except Exception as exc:  # noqa: BLE001 - report unexpected SDK exception types
        return FailureTestResult(
            name=name,
            status="fail",
            elapsed=time.monotonic() - start,
            detail=f"unwrapped error: {_sanitized_error(exc)}",
        )
    return FailureTestResult(
        name=name,
        status="fail",
        elapsed=time.monotonic() - start,
        detail="cascade unexpectedly succeeded when every credential was invalid",
    )


async def _run_failure_tests(
    models: dict[str, str],
    cases: list[FailureCase] | None = None,
) -> list[FailureTestResult]:
    selected_cases = list(_FAILURE_CASES) if cases is None else list(dict.fromkeys(cases))
    results: list[FailureTestResult] = []
    for case in selected_cases:
        if case == "openai-auth":
            result = await _run_invalid_auth("openai", models["openai"])
        elif case == "gemini-auth":
            result = await _run_invalid_auth("gemini", models["gemini"])
        elif case == "openai-to-gemini":
            result = await _run_cascade_fallback("openai", "gemini", models)
        elif case == "gemini-to-openai":
            result = await _run_cascade_fallback("gemini", "openai", models)
        else:
            result = await _run_cascade_exhausted(models)
        cost = f" cost=${result.total_cost:.6f}" if result.total_cost else ""
        print(
            f"[{result.status:4}] {result.name} ({result.elapsed:.1f}s) "
            f"{result.detail}{cost}",
            flush=True,
        )
        results.append(result)
    return results


def _parse_args() -> argparse.Namespace:
    providers = get_supported_image_providers()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider",
        action="append",
        choices=providers,
        help="Restrict to one provider (repeatable).",
    )
    parser.add_argument(
        "--operation",
        action="append",
        choices=("generate", "edit"),
        help="Restrict to one operation (repeatable).",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Test an exact model for the single selected provider (repeatable).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Test every configured model instead of the first model per provider.",
    )
    parser.add_argument(
        "--failure-tests",
        action="store_true",
        help="Run deterministic live authentication and cascade failure tests.",
    )
    parser.add_argument(
        "--failure-case",
        action="append",
        choices=_FAILURE_CASES,
        help="Restrict --failure-tests to one case (repeatable).",
    )
    args = parser.parse_args()
    if args.failure_tests and (args.provider or args.operation or args.model or args.all_models):
        parser.error(
            "--failure-tests cannot be combined with --provider, --operation, --model, "
            "or --all-models"
        )
    if args.failure_case and not args.failure_tests:
        parser.error("--failure-case requires --failure-tests")
    if args.model and (not args.provider or len(args.provider) != 1):
        parser.error("--model requires exactly one --provider")
    if args.model and args.all_models:
        parser.error("--model cannot be combined with --all-models")
    return args


def main() -> int:
    args = _parse_args()
    load_dotenv()

    providers: list[str] = (
        list(_FAILURE_TEST_PROVIDERS)
        if args.failure_tests
        else args.provider or get_supported_image_providers()
    )
    operations: list[Operation] = args.operation or ["generate", "edit"]
    for provider in providers:
        env_var = PROVIDER_API_KEY_ENV.get(provider)
        if env_var is None:
            print(f"ERROR: no API-key mapping for provider {provider!r}.", file=sys.stderr)
            return 2
        if not os.environ.get(env_var):
            print(f"ERROR: {env_var} is not set for provider {provider!r}.", file=sys.stderr)
            return 2

    try:
        models = _select_models(
            providers,
            all_models=args.all_models,
            requested_models=args.model,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.failure_tests:
        model_by_provider = dict(models)
        print("Failure tests use temporary invalid credentials and restore the environment.")
        print("Credentials, prompts, and image bytes are not printed or persisted.")
        print()
        failure_cases = cast(list[FailureCase] | None, args.failure_case)
        failure_results = asyncio.run(
            _run_failure_tests(model_by_provider, cases=failure_cases)
        )
        failure_test_failures = [
            result for result in failure_results if result.status == "fail"
        ]
        total_cost = sum(result.total_cost for result in failure_results)
        print()
        print(
            f"Summary: {len(failure_results) - len(failure_test_failures)} passed, "
            f"{len(failure_test_failures)} failed, "
            f"successful fallback cost=${total_cost:.6f}"
        )
        return 1 if failure_test_failures else 0

    print(f"Models: {', '.join(f'{provider}/{model}' for provider, model in models)}")
    print(f"Operations: {', '.join(operations)}")
    print("Prompts and image bytes are not printed or persisted.")
    print()

    results = asyncio.run(_run_all(models, operations))
    failures = [result for result in results if result.status == "fail"]
    print()
    print(f"Summary: {len(results) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
