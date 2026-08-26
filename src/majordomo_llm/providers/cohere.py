"""Cohere LLM provider implementation."""

import time
from collections.abc import AsyncIterator
from typing import Any

import cohere
from cohere import JsonObjectResponseFormatV2, SystemChatMessageV2, UserChatMessageV2
from cohere.core.request_options import RequestOptions

from majordomo_llm.base import (
    LLM,
    LLMResponse,
    LLMStreamResponse,
    _StreamState,
    canonicalize_json_schema_output,
    inline_schema_refs,
    resolve_api_key,
    strip_unsupported_schema_constraints,
)
from majordomo_llm.exceptions import ProviderError
from majordomo_llm.retry import retry_provider_call

# Alias kept for backward compatibility — tests and downstream callers may
# import this name. Delegates to the shared helper in base.py since Bedrock
# Structured Outputs rejects the same constraint set.
_strip_cohere_unsupported_constraints = strip_unsupported_schema_constraints


class Cohere(LLM):
    """Cohere LLM provider.

    Implements the LLM interface for Cohere's models using the V2 API.
    Supports Command A, Command R+, Command R, and Command R7B models.

    The API key is read from the ``CO_API_KEY`` environment variable.

    Attributes:
        client: The async Cohere client instance.

    Example:
        >>> llm = Cohere(
        ...     model="command-a-03-2025",
        ...     input_cost=2.50,
        ...     output_cost=10.00,
        ... )
        >>> response = await llm.get_response("Hello, Cohere!")
    """

    def __init__(
        self,
        model: str,
        input_cost: float,
        output_cost: float,
        supports_temperature_top_p: bool = True,
        *,
        cached_input_cost: float | None = None,
        cache_write_cost: float | None = None,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the Cohere provider.

        Args:
            model: The Cohere model identifier (e.g., "command-a-03-2025").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            cached_input_cost: Unused by Cohere (no prompt caching); accepted for
                a uniform factory signature.
            cache_write_cost: Unused by Cohere; accepted for a uniform factory
                signature.
            api_key: Optional API key. Defaults to ``CO_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom base URL for routing through a proxy.
            default_headers: Optional headers sent with every request.

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
        """
        resolved_api_key = resolve_api_key(api_key, "CO_API_KEY", "Cohere")
        super().__init__(
            provider="cohere",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            cached_input_cost=cached_input_cost,
            cache_write_cost=cache_write_cost,
            supports_temperature_top_p=supports_temperature_top_p,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
            base_url=base_url,
            default_headers=default_headers,
        )
        self.client = cohere.AsyncClientV2(
            api_key=resolved_api_key,
            base_url=self.base_url,
        )

    def _cohere_request_options(
        self, extra_headers: dict[str, str] | None
    ) -> RequestOptions | None:
        """Build request_options with merged default + extra headers."""
        merged = dict(self.default_headers or {})
        if extra_headers:
            merged.update(extra_headers)
        if not merged:
            return None
        return RequestOptions(additional_headers=merged)

    def _cohere_sampling(
        self, temperature: float | None, top_p: float | None
    ) -> dict[str, Any]:
        """Translate the shared sampling params to Cohere's names.

        Cohere calls nucleus sampling ``p``, not ``top_p``.
        """
        params = self._sampling_params(temperature, top_p)
        if "top_p" in params:
            params["p"] = params.pop("top_p")
        return params

    @retry_provider_call
    async def _get_response_impl(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Get a plain text response from Cohere."""
        return await self._get_response(
            user_prompt, system_prompt, temperature, top_p, extra_headers=extra_headers
        )

    async def _get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Internal method to get a response from Cohere."""
        messages: list[Any] = []
        if system_prompt:
            messages.append(SystemChatMessageV2(content=system_prompt))
        messages.append(UserChatMessageV2(content=user_prompt))

        request_options = self._cohere_request_options(extra_headers)

        start_time = time.time()
        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                request_options=request_options,
                **self._cohere_sampling(temperature, top_p),
            )
        except cohere.core.api_error.ApiError as e:
            raise ProviderError(
                f"Cohere API error: {e}",
                provider="cohere",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        input_tokens, output_tokens = _cohere_token_counts(response)
        cached_tokens = 0
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMResponse(
            content=_cohere_text_content(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
            deprecation_warning=self.deprecation_warning,
        )

    async def _get_response_stream_impl(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMStreamResponse:
        """Get a streaming text response from Cohere."""
        messages: list[Any] = []
        if system_prompt:
            messages.append(SystemChatMessageV2(content=system_prompt))
        messages.append(UserChatMessageV2(content=user_prompt))

        state = _StreamState()
        request_options = self._cohere_request_options(extra_headers)

        try:
            response = self.client.chat_stream(
                model=self.model,
                messages=messages,
                request_options=request_options,
                **self._cohere_sampling(temperature, top_p),
            )
        except cohere.core.api_error.ApiError as e:
            raise ProviderError(
                f"Cohere API error: {e}",
                provider="cohere",
                original_error=e,
            ) from e

        async def generator() -> AsyncIterator[str]:
            try:
                async for event in response:
                    event_any: Any = event
                    if event_any.type == "content-delta":
                        yield event_any.delta.message.content.text or ""
                    elif event_any.type == "message-end":
                        state.input_tokens, state.output_tokens = _cohere_usage_tokens(
                            event_any.delta.usage
                        )
            except cohere.core.api_error.ApiError as e:
                raise ProviderError(
                    f"Cohere API error: {e}",
                    provider="cohere",
                    original_error=e,
                ) from e

        return LLMStreamResponse(stream=generator(), state=state, llm=self)

    async def _get_json_schema_response(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Cohere-specific implementation using native JSON schema response format."""
        schema = _strip_cohere_unsupported_constraints(inline_schema_refs(response_schema))
        messages: list[Any] = []
        if system_prompt is not None:
            messages.append(SystemChatMessageV2(content=system_prompt))
        messages.append(UserChatMessageV2(content=user_prompt))

        request_options = self._cohere_request_options(extra_headers)

        start_time = time.time()
        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                response_format=JsonObjectResponseFormatV2(json_schema=schema),
                request_options=request_options,
                **self._cohere_sampling(temperature, top_p),
            )
        except cohere.core.api_error.ApiError as e:
            raise ProviderError(
                f"Cohere API error: {e}",
                provider="cohere",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time

        input_tokens, output_tokens = _cohere_token_counts(response)
        cached_tokens = 0
        input_cost, output_cost, total_cost = self._calculate_costs(input_tokens, output_tokens)

        return LLMResponse(
            content=canonicalize_json_schema_output(
                _cohere_text_content(response),
                response_schema,
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )


def _cohere_token_counts(response: Any) -> tuple[int, int]:
    """Extract Cohere token counts from a chat response."""
    usage = response.usage
    assert usage is not None
    return _cohere_usage_tokens(usage)


def _cohere_usage_tokens(usage: Any) -> tuple[int, int]:
    """Extract (input, output) token counts from a Cohere ``Usage`` object.

    Cohere reports two figures: ``billed_units`` — what the account is actually
    charged for — and ``tokens`` — total tokens processed, which for Command
    models includes a large built-in system preamble (hundreds of tokens the
    caller never sent). We report ``billed_units`` so cost matches Cohere's
    billing, falling back to ``tokens`` only when billed units are unavailable.
    """
    billed = getattr(usage, "billed_units", None)
    if billed is not None:
        input_tokens = billed.input_tokens
        output_tokens = billed.output_tokens
        if input_tokens is not None and output_tokens is not None:
            return int(input_tokens), int(output_tokens)
    tokens = usage.tokens
    assert tokens is not None
    return int(tokens.input_tokens or 0), int(tokens.output_tokens or 0)


def _cohere_text_content(response: Any) -> str:
    """Extract the first Cohere text content block."""
    content = response.message.content
    assert content is not None
    text = content[0].text
    return text or ""
