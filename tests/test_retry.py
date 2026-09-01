"""Tests for provider retry classification."""

import anthropic
import httpx
import openai
import pytest
from cohere.core.api_error import ApiError as CohereApiError
from google.genai import errors as genai_errors
from google.genai._gaos.lib.compat_errors import APIError as InteractionsAPIError

from majordomo_llm.exceptions import ProviderError
from majordomo_llm.retry import (
    get_provider_error_status_code,
    is_retryable_exception,
    is_retryable_provider_error,
    retry_provider_call,
)


def _httpx_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.example.test")
    return httpx.Response(status_code, request=request, json={"error": "test"})


def _interactions_error(status_code: int) -> InteractionsAPIError:
    return InteractionsAPIError.generate(
        status_code=status_code,
        body={"error": {"message": "test"}},
        message="test",
        response=_httpx_response(status_code),
    )


def test_extracts_status_codes_from_supported_sdk_errors():
    """Should extract status codes from each provider SDK's documented shape."""
    anthropic_error = anthropic.APIStatusError(
        "overloaded",
        response=_httpx_response(529),
        body={"error": "overloaded"},
    )
    openai_error = openai.APIStatusError(
        "server error",
        response=_httpx_response(500),
        body={"error": "server error"},
    )
    gemini_error = genai_errors.APIError(
        408,
        {"error": {"message": "timeout", "status": "DEADLINE_EXCEEDED"}},
    )
    interactions_error = _interactions_error(503)
    cohere_error = CohereApiError(status_code=503, body={"message": "unavailable"})

    assert get_provider_error_status_code(anthropic_error) == 529
    assert get_provider_error_status_code(openai_error) == 500
    assert get_provider_error_status_code(gemini_error) == 408
    assert get_provider_error_status_code(interactions_error) == 503
    assert get_provider_error_status_code(cohere_error) == 503


@pytest.mark.parametrize("status_code", [408, 500, 502, 503, 504])
def test_retries_transient_status_codes(status_code):
    """Should retry transient provider status codes."""
    original_error = openai.APIStatusError(
        "transient error",
        response=_httpx_response(status_code),
        body={"error": "transient"},
    )
    error = ProviderError("OpenAI failed", provider="openai", original_error=original_error)

    assert is_retryable_provider_error(error)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422, 429, 529])
def test_does_not_retry_non_transient_or_availability_first_status_codes(status_code):
    """Should not retry permanent, rate-limited, or overloaded provider statuses."""
    original_error = anthropic.APIStatusError(
        "non-retryable error",
        response=_httpx_response(status_code),
        body={"error": "non-retryable"},
    )
    error = ProviderError("Anthropic failed", provider="anthropic", original_error=original_error)

    assert not is_retryable_provider_error(error)


@pytest.mark.parametrize("status_code", [408, 500, 502, 503, 504])
def test_retries_transient_gemini_interactions_status_codes(status_code):
    """Should classify transient Interactions failures using their status_code."""
    error = ProviderError(
        "Gemini Interactions failed",
        provider="gemini",
        original_error=_interactions_error(status_code),
    )

    assert is_retryable_provider_error(error)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422, 429, 529])
def test_does_not_retry_permanent_gemini_interactions_status_codes(status_code):
    """Should preserve the shared fail-fast policy for Interactions failures."""
    error = ProviderError(
        "Gemini Interactions failed",
        provider="gemini",
        original_error=_interactions_error(status_code),
    )

    assert not is_retryable_provider_error(error)


@pytest.mark.parametrize(
    "original_error",
    [
        anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.example.test")),
        anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.example.test")),
        openai.APIConnectionError(request=httpx.Request("POST", "https://api.example.test")),
        openai.APITimeoutError(request=httpx.Request("POST", "https://api.example.test")),
        httpx.ConnectError("connect failed"),
        httpx.TimeoutException("timed out"),
        TimeoutError(),
    ],
)
def test_retries_explicit_transport_errors(original_error):
    """Should retry only explicit timeout and connection failure types."""
    assert is_retryable_exception(original_error)


def test_does_not_retry_unknown_exception():
    """Should not retry unknown exceptions without a provider status or transport type."""
    assert not is_retryable_exception(RuntimeError("bug"))


async def test_decorated_function_propagates_unknown_exception_without_retry():
    """Should guard against accidentally widening retries to all exceptions."""
    calls = 0

    @retry_provider_call
    async def raises_unknown_error() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        await raises_unknown_error()

    assert calls == 1


async def test_decorated_function_does_not_retry_529():
    """Should let overloaded providers fail fast so cascade can fall back."""
    calls = 0
    original_error = anthropic.APIStatusError(
        "overloaded",
        response=_httpx_response(529),
        body={"error": "overloaded"},
    )

    @retry_provider_call
    async def raises_overloaded_error() -> None:
        nonlocal calls
        calls += 1
        raise ProviderError(
            "Anthropic overloaded",
            provider="anthropic",
            original_error=original_error,
        )

    with pytest.raises(ProviderError, match="Anthropic overloaded"):
        await raises_overloaded_error()

    assert calls == 1
