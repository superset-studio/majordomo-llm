"""Shared retry policy for provider API calls."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import anthropic
import httpx
import openai
from cohere.core.api_error import ApiError as CohereApiError
from google.genai import errors as genai_errors
from google.genai._gaos.lib.compat_errors import APIError as InteractionsAPIError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from majordomo_llm.exceptions import EmptyStructuredResponseError, ProviderError

RETRYABLE_STATUS_CODES = frozenset({408, 500, 502, 503, 504})


def retry_provider_call[**P, R](
    func: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Retry provider calls only for transient provider and transport failures."""
    decorated = retry(
        retry=retry_if_exception(is_retryable_exception),
        wait=wait_random_exponential(min=0.2, max=1),
        stop=stop_after_attempt(3),
    )(func)
    return decorated


def is_retryable_exception(exc: BaseException) -> bool:
    """Return whether an exception should be retried by provider wrappers."""
    if isinstance(exc, EmptyStructuredResponseError):
        # A schema-valid but empty structured result is a model punt (only
        # possible on the forced-tool fallback path — constrained decoding
        # cannot produce it). Re-sampling usually recovers a real answer.
        return True
    if isinstance(exc, ProviderError):
        return is_retryable_provider_error(exc)
    return is_retryable_transport_error(exc)


def is_retryable_provider_error(exc: ProviderError) -> bool:
    """Return whether a ProviderError wraps a transient provider failure."""
    original_error = exc.original_error
    if original_error is None:
        return False

    status_code = get_provider_error_status_code(original_error)
    if status_code is not None:
        return status_code in RETRYABLE_STATUS_CODES

    return is_retryable_transport_error(original_error)


def get_provider_error_status_code(exc: BaseException) -> int | None:
    """Extract HTTP status from supported provider SDK exception types."""
    if isinstance(exc, anthropic.APIStatusError | openai.APIStatusError | CohereApiError):
        status_code = exc.status_code
        return status_code if isinstance(status_code, int) else None

    if isinstance(exc, genai_errors.APIError):
        code = exc.code
        return code if isinstance(code, int) else None

    if isinstance(exc, InteractionsAPIError):
        status_code = exc.status_code
        return status_code if isinstance(status_code, int) else None

    return None


def is_retryable_transport_error(exc: BaseException) -> bool:
    """Return whether an exception is an explicit transient transport failure."""
    return isinstance(
        exc,
        (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            httpx.ConnectError,
            httpx.TimeoutException,
            asyncio.TimeoutError,
        ),
    )
