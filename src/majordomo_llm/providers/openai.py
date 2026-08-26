"""OpenAI LLM provider implementation."""

import time
from collections.abc import AsyncIterator
from typing import Any

import openai

from majordomo_llm.base import (
    LLM,
    LLMResponse,
    LLMStreamResponse,
    _StreamState,
    canonicalize_json_schema_output,
    enforce_strict_object_schema,
    resolve_api_key,
)
from majordomo_llm.exceptions import ProviderError
from majordomo_llm.retry import retry_provider_call

# Kept as a module-level alias for backward compatibility — tests and downstream
# callers may import it. Delegates to the shared helper in base.py since the
# normalization is identical between OpenAI strict mode and Bedrock Structured
# Outputs.
_enforce_openai_strict_schema = enforce_strict_object_schema


class OpenAI(LLM):
    """OpenAI LLM provider.

    Implements the LLM interface for OpenAI's GPT models, including
    support for structured outputs via the responses.parse API.

    The API key is read from the ``OPENAI_API_KEY`` environment variable.

    Attributes:
        client: The async OpenAI client instance.

    Example:
        >>> llm = OpenAI(
        ...     model="gpt-4o",
        ...     input_cost=2.5,
        ...     output_cost=10.0,
        ... )
        >>> response = await llm.get_response("Hello, GPT!")
    """

    def __init__(
        self,
        model: str,
        input_cost: float,
        output_cost: float,
        supports_temperature_top_p: bool = True,
        use_web_search: bool = False,
        *,
        cached_input_cost: float | None = None,
        cache_write_cost: float | None = None,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            model: The GPT model identifier (e.g., "gpt-4o", "gpt-5").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            use_web_search: Enable Responses API ``web_search_preview`` tool.
            cached_input_cost: Cost per million cache-read tokens in USD. OpenAI
                reports cached tokens as a subset of input tokens, so this
                re-prices them below ``input_cost``.
            cache_write_cost: Unused by OpenAI (no distinct cache-write rate);
                accepted for a uniform factory signature.
            api_key: Optional API key. Defaults to ``OPENAI_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom base URL for routing through a proxy.
            default_headers: Optional headers sent with every request.

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
        """
        resolved_api_key = resolve_api_key(api_key, "OPENAI_API_KEY", "OpenAI")
        super().__init__(
            provider="openai",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            cached_input_cost=cached_input_cost,
            cache_write_cost=cache_write_cost,
            supports_temperature_top_p=supports_temperature_top_p,
            use_web_search=use_web_search,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
            base_url=base_url,
            default_headers=default_headers,
        )
        self.client = openai.AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=self.base_url,
            default_headers=self.default_headers,
        )

    def _web_search_kwargs(self) -> dict[str, Any]:
        """Return ``tools=`` kwarg for the Responses API when web search is on.

        OpenAI bills the web_search_preview tool's tokens through normal output
        tokens, so no separate ``tool_use_cost`` is added.
        """
        if not self.use_web_search:
            return {}
        return {"tools": [{"type": "web_search_preview"}]}

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
        """Get a plain text response from OpenAI."""
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
        """Internal method to get a response from OpenAI."""
        start_time = time.time()
        web_search_kwargs = self._web_search_kwargs()
        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_prompt,
                extra_headers=extra_headers,
                **web_search_kwargs,
                **self._sampling_params(temperature, top_p),
            )
        except openai.APIError as e:
            raise ProviderError(
                f"OpenAI API error: {e}",
                provider="openai",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        assert response.usage is not None
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cached_tokens = response.usage.input_tokens_details.cached_tokens
        input_cost, output_cost, total_cost = self._calculate_costs(
            input_tokens, output_tokens, cached_tokens
        )

        return LLMResponse(
            content=response.output_text,
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
        """Get a streaming text response from OpenAI."""
        state = _StreamState()
        web_search_kwargs = self._web_search_kwargs()

        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_prompt,
                stream=True,
                extra_headers=extra_headers,
                **web_search_kwargs,
                **self._sampling_params(temperature, top_p),
            )
        except openai.APIError as e:
            raise ProviderError(
                f"OpenAI API error: {e}",
                provider="openai",
                original_error=e,
            ) from e

        async def generator() -> AsyncIterator[str]:
            try:
                async for event in response:
                    if event.type == "response.output_text.delta":
                        yield event.delta
                    elif event.type == "response.completed":
                        assert event.response.usage is not None
                        state.input_tokens = event.response.usage.input_tokens
                        state.output_tokens = event.response.usage.output_tokens
                        cached = event.response.usage.input_tokens_details
                        state.cached_tokens = cached.cached_tokens
            except openai.APIError as e:
                raise ProviderError(
                    f"OpenAI API error: {e}",
                    provider="openai",
                    original_error=e,
                ) from e

        return LLMStreamResponse(stream=generator(), state=state, llm=self)

    async def _get_json_schema_response(
        self,
        user_prompt: str,
        response_schema: dict[str, object],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """OpenAI-specific implementation using structured outputs with JSON Schema."""
        start_time = time.time()
        strict_schema = _enforce_openai_strict_schema(response_schema)
        response_format: dict[str, object] = {
            "type": "json_schema",
            "name": schema_name,
            "schema": strict_schema,
            "strict": True,
        }
        if schema_description is not None:
            response_format["description"] = schema_description
        text_config: Any = {"format": response_format}
        web_search_kwargs = self._web_search_kwargs()

        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_prompt,
                text=text_config,
                extra_headers=extra_headers,
                **web_search_kwargs,
                **self._sampling_params(temperature, top_p),
            )
        except openai.APIError as e:
            raise ProviderError(
                f"OpenAI API error: {e}",
                provider="openai",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        assert response.usage is not None
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cached_tokens = getattr(
            getattr(response.usage, "input_tokens_details", None),
            "cached_tokens",
            0,
        ) or 0
        input_cost, output_cost, total_cost = self._calculate_costs(
            input_tokens, output_tokens, cached_tokens
        )

        return LLMResponse(
            content=canonicalize_json_schema_output(response.output_text, response_schema),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )
