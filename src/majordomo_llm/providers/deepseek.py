"""DeepSeek LLM provider implementation."""

import time
from collections.abc import AsyncIterator
from typing import Any

import openai

from majordomo_llm.base import (
    LLM,
    LLMResponse,
    LLMStreamResponse,
    _StreamState,
    build_schema_prompt,
    canonicalize_json_schema_output,
    resolve_api_key,
)
from majordomo_llm.exceptions import ProviderError
from majordomo_llm.retry import retry_provider_call


class DeepSeek(LLM):
    """DeepSeek LLM provider.

    Implements the LLM interface for DeepSeek's models using the OpenAI-compatible
    API. Supports both DeepSeek-V3 (chat) and DeepSeek-R1 (reasoner) models.

    The API key is read from the ``DEEPSEEK_API_KEY`` environment variable.

    Attributes:
        client: The async OpenAI client instance configured for DeepSeek.

    Example:
        >>> llm = DeepSeek(
        ...     model="deepseek-chat",
        ...     input_cost=0.28,
        ...     output_cost=0.42,
        ... )
        >>> response = await llm.get_response("Hello, DeepSeek!")
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})
    THINKING_MODES = frozenset({"enabled", "disabled"})

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
        reasoning_effort: str | None = None,
        thinking: str | None = None,
    ) -> None:
        """Initialize the DeepSeek provider.

        Args:
            model: The DeepSeek model identifier (e.g., "deepseek-chat", "deepseek-reasoner").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            cached_input_cost: Cost per million cache-hit (cache-read) tokens in
                USD; a subset of input tokens re-priced below ``input_cost``.
            cache_write_cost: Unused by DeepSeek; accepted for a uniform factory
                signature.
            api_key: Optional API key. Defaults to ``DEEPSEEK_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom base URL. Overrides DEEPSEEK_BASE_URL when set.
            default_headers: Optional headers sent with every request.
            reasoning_effort: Optional reasoning effort for supported DeepSeek models.
            thinking: Optional thinking mode ("enabled" or "disabled") for supported models.

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
            ValueError: If reasoning_effort or thinking is invalid.
        """
        if reasoning_effort is not None and reasoning_effort not in self.REASONING_EFFORTS:
            valid = ", ".join(sorted(self.REASONING_EFFORTS))
            raise ValueError(
                f"Invalid DeepSeek reasoning_effort '{reasoning_effort}'. Valid: {valid}"
            )
        if thinking is not None and thinking not in self.THINKING_MODES:
            valid = ", ".join(sorted(self.THINKING_MODES))
            raise ValueError(f"Invalid DeepSeek thinking mode '{thinking}'. Valid: {valid}")

        resolved_api_key = resolve_api_key(api_key, "DEEPSEEK_API_KEY", "DeepSeek")

        # When routing through a proxy (e.g. Majordomo Steward), auto-inject
        # ``x-majordomo-provider: deepseek`` so the gateway can disambiguate
        # DeepSeek traffic from vanilla OpenAI (both speak the same wire
        # shape). Caller-supplied default_headers win on key collision.
        if base_url is not None:
            merged_headers: dict[str, str] = {"x-majordomo-provider": "deepseek"}
            if default_headers:
                merged_headers.update(default_headers)
            default_headers = merged_headers

        super().__init__(
            provider="deepseek",
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
        self.client = openai.AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=self.base_url or self.DEEPSEEK_BASE_URL,
            default_headers=self.default_headers,
        )
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking

    def _deepseek_request_kwargs(self) -> dict[str, Any]:
        """Build DeepSeek-specific request options for supported models."""
        kwargs: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.thinking is not None:
            kwargs["extra_body"] = {"thinking": {"type": self.thinking}}
        return kwargs

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
        """Get a plain text response from DeepSeek."""
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
        """Internal method to get a response from DeepSeek."""
        messages: list[Any] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        start_time = time.time()
        request_kwargs = self._deepseek_request_kwargs()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_headers=extra_headers,
                **request_kwargs,
                **self._sampling_params(temperature, top_p),
            )
        except openai.APIError as e:
            raise ProviderError(
                f"DeepSeek API error: {e}",
                provider="deepseek",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        assert response.usage is not None
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = (
            getattr(
                getattr(response.usage, "prompt_tokens_details", None),
                "cached_tokens",
                0,
            )
            or 0
        )
        input_cost, output_cost, total_cost = self._calculate_costs(
            input_tokens, output_tokens, cached_tokens
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
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
        """Get a streaming text response from DeepSeek."""
        messages: list[Any] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        state = _StreamState()
        request_kwargs = self._deepseek_request_kwargs()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                extra_headers=extra_headers,
                **request_kwargs,
                **self._sampling_params(temperature, top_p),
            )
        except openai.APIError as e:
            raise ProviderError(
                f"DeepSeek API error: {e}",
                provider="deepseek",
                original_error=e,
            ) from e

        async def generator() -> AsyncIterator[str]:
            try:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    if chunk.usage:
                        state.input_tokens = chunk.usage.prompt_tokens
                        state.output_tokens = chunk.usage.completion_tokens
                        state.cached_tokens = (
                            getattr(
                                getattr(chunk.usage, "prompt_tokens_details", None),
                                "cached_tokens",
                                0,
                            )
                            or 0
                        )
            except openai.APIError as e:
                raise ProviderError(
                    f"DeepSeek API error: {e}",
                    provider="deepseek",
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
        """DeepSeek-specific implementation using json_object mode.

        DeepSeek's API supports only ``response_format={'type': 'json_object'}``
        (see https://api-docs.deepseek.com/guides/json_mode); ``json_schema`` is
        rejected with ``"This response_format type is unavailable now"``. The
        schema is therefore injected into the system prompt so the model knows
        the expected shape, and ``json_object`` mode constrains the output to
        valid JSON. The response is still canonicalized against the schema so
        callers receive a deterministic, byte-comparable string.
        """
        effective_system_prompt = build_schema_prompt(response_schema, system_prompt)

        messages: list[Any] = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_format: Any = {"type": "json_object"}

        start_time = time.time()
        request_kwargs = self._deepseek_request_kwargs()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                extra_headers=extra_headers,
                **request_kwargs,
                **self._sampling_params(temperature, top_p),
            )
        except openai.APIError as e:
            raise ProviderError(
                f"DeepSeek API error: {e}",
                provider="deepseek",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time

        assert response.usage is not None
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = (
            getattr(
                getattr(response.usage, "prompt_tokens_details", None),
                "cached_tokens",
                0,
            )
            or 0
        )
        input_cost, output_cost, total_cost = self._calculate_costs(
            input_tokens, output_tokens, cached_tokens
        )

        return LLMResponse(
            content=canonicalize_json_schema_output(
                response.choices[0].message.content or "",
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
