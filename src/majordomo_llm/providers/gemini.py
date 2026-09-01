"""Google Gemini LLM provider implementation."""

import time
from collections.abc import AsyncIterator
from typing import Any, cast

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from majordomo_llm.base import (
    LLM,
    ImageInput,
    LLMResponse,
    LLMStreamResponse,
    _StreamState,
    canonicalize_json_schema_output,
    inline_schema_refs,
    resolve_api_key,
)
from majordomo_llm.exceptions import ConfigurationError, ProviderError
from majordomo_llm.retry import retry_provider_call


class Gemini(LLM):
    """Google Gemini LLM provider.

    Implements the LLM interface for Google's Gemini models, including
    support for structured outputs via response schemas.

    The API key is read from the ``GEMINI_API_KEY`` environment variable.

    Attributes:
        client: The Google GenAI client instance.

    Example:
        >>> llm = Gemini(
        ...     model="gemini-2.5-flash",
        ...     input_cost=0.30,
        ...     output_cost=2.50,
        ... )
        >>> response = await llm.get_response("Hello, Gemini!")
    """

    def __init__(
        self,
        model: str,
        input_cost: float,
        output_cost: float,
        *,
        supports_temperature_top_p: bool = True,
        use_web_search: bool = False,
        supports_image_input: bool = False,
        cached_input_cost: float | None = None,
        cache_write_cost: float | None = None,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the Gemini provider.

        Args:
            model: The Gemini model identifier (e.g., "gemini-2.5-flash").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether the model supports temperature/top_p.
            use_web_search: Enable the Google Search grounding tool.
            supports_image_input: Whether this model accepts image inputs.
            cached_input_cost: Cost per million cached-content (cache-read) tokens
                in USD. Gemini reports cached content as a subset of prompt
                tokens, so this re-prices them below ``input_cost``.
            cache_write_cost: Unused by Gemini (implicit caching has no per-token
                write fee); accepted for a uniform factory signature.
            api_key: Optional API key. Defaults to ``GEMINI_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom base URL for routing through a proxy.
            default_headers: Optional headers sent with every request.

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
        """
        resolved_api_key = resolve_api_key(api_key, "GEMINI_API_KEY", "Gemini")
        super().__init__(
            provider="gemini",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            cached_input_cost=cached_input_cost,
            cache_write_cost=cache_write_cost,
            supports_temperature_top_p=True,
            use_web_search=use_web_search,
            supports_image_input=supports_image_input,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
            base_url=base_url,
            default_headers=default_headers,
        )
        http_options = None
        if self.base_url or self.default_headers:
            http_options = types.HttpOptions(
                base_url=self.base_url,
                headers=self.default_headers,
            )
        self.client = genai.Client(api_key=resolved_api_key, http_options=http_options)

    # Gemini bills grounded queries at $35 per 1,000 requests.
    _GROUNDED_QUERY_COST = 0.035

    def _apply_web_search(self, config_kwargs: dict[str, Any]) -> None:
        """Attach the Google Search tool to a request config when enabled."""
        if not self.use_web_search:
            return
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    def _supports_search_with_structured_output(self) -> bool:
        """Whether this model can combine a grounding tool with a response schema.

        Grounded structured outputs are a Gemini 3 series preview feature; 2.5
        and earlier reject a request that sets both a grounding tool and a
        response schema.
        """
        return (
            self.model.startswith("gemini-3.")
            or self.model.startswith("gemini-3-")
            or self.model == "gemini-3"
        )

    def _compute_web_search_cost(self, response: Any) -> float:
        """Return the per-call grounded-query fee charged by Gemini.

        Counts response candidates that carry ``grounding_metadata`` — the
        only signal the API surfaces when a grounded query was actually
        performed.
        """
        candidates = getattr(response, "candidates", None) or []
        grounded = sum(1 for c in candidates if getattr(c, "grounding_metadata", None) is not None)
        return grounded * self._GROUNDED_QUERY_COST

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
        """Get a plain text response from Gemini."""
        return await self._get_response(
            user_prompt, system_prompt, temperature, top_p, extra_headers=extra_headers
        )

    @retry_provider_call
    async def _get_response_with_images_impl(
        self,
        user_prompt: str,
        images: tuple[ImageInput, ...],
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return await self._get_response(
            user_prompt,
            system_prompt,
            temperature,
            top_p,
            extra_headers=extra_headers,
            images=images,
        )

    async def _get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        images: tuple[ImageInput, ...] = (),
    ) -> LLMResponse:
        """Internal method to get a response from Gemini."""
        start_time = time.time()
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            **self._sampling_params(temperature, top_p),
        }
        if extra_headers:
            config_kwargs["http_options"] = types.HttpOptions(headers=extra_headers)
        self._apply_web_search(config_kwargs)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(**config_kwargs),
                contents=_gemini_contents(user_prompt, images),
            )
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Gemini API error: {e}",
                provider="gemini",
                original_error=e,
            ) from e
        execution_time = time.time() - start_time

        input_tokens, output_tokens, cached_tokens = _gemini_token_counts(response)
        input_cost, output_cost, total_cost = self._calculate_costs(
            input_tokens, output_tokens, cached_tokens
        )
        tool_use_cost = self._compute_web_search_cost(response)
        total_cost += tool_use_cost

        return LLMResponse(
            content=response.text or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
            tool_use_cost=tool_use_cost,
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
        """Get a streaming text response from Gemini."""
        state = _StreamState()
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            **self._sampling_params(temperature, top_p),
        }
        if extra_headers:
            config_kwargs["http_options"] = types.HttpOptions(headers=extra_headers)
        self._apply_web_search(config_kwargs)

        try:
            response = await self.client.aio.models.generate_content_stream(
                model=self.model,
                config=types.GenerateContentConfig(**config_kwargs),
                contents=user_prompt,
            )
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Gemini API error: {e}",
                provider="gemini",
                original_error=e,
            ) from e

        async def generator() -> AsyncIterator[str]:
            try:
                async for chunk in response:
                    if chunk.text:
                        yield chunk.text
                    if chunk.usage_metadata:
                        state.input_tokens = chunk.usage_metadata.prompt_token_count or 0
                        state.output_tokens = chunk.usage_metadata.candidates_token_count or 0
                        state.cached_tokens = chunk.usage_metadata.cached_content_token_count or 0
            except genai_errors.APIError as e:
                raise ProviderError(
                    f"Gemini API error: {e}",
                    provider="gemini",
                    original_error=e,
                ) from e

        return LLMStreamResponse(stream=generator(), state=state, llm=self)

    async def _get_response_stream_with_images_impl(
        self,
        user_prompt: str,
        images: tuple[ImageInput, ...],
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMStreamResponse:
        state = _StreamState()
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            **self._sampling_params(temperature, top_p),
        }
        if extra_headers:
            config_kwargs["http_options"] = types.HttpOptions(headers=extra_headers)
        self._apply_web_search(config_kwargs)
        try:
            response = await self.client.aio.models.generate_content_stream(
                model=self.model,
                config=types.GenerateContentConfig(**config_kwargs),
                contents=_gemini_contents(user_prompt, images),
            )
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Gemini API error: {e}", provider="gemini", original_error=e
            ) from e

        async def generator() -> AsyncIterator[str]:
            try:
                async for chunk in response:
                    if chunk.text:
                        yield chunk.text
                    if chunk.usage_metadata:
                        counts = _gemini_token_counts(chunk)
                        state.input_tokens, state.output_tokens, state.cached_tokens = counts
            except genai_errors.APIError as e:
                raise ProviderError(
                    f"Gemini API error: {e}", provider="gemini", original_error=e
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
        """Gemini-specific implementation using response schema for structured outputs."""
        return await self._get_json_schema_response_common(
            user_prompt=user_prompt,
            images=(),
            response_schema=response_schema,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            extra_headers=extra_headers,
        )

    async def _get_json_schema_response_with_images(
        self,
        user_prompt: str,
        images: tuple[ImageInput, ...],
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return await self._get_json_schema_response_common(
            user_prompt=user_prompt,
            images=images,
            response_schema=response_schema,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            extra_headers=extra_headers,
        )

    async def _get_json_schema_response_common(
        self,
        *,
        user_prompt: str,
        images: tuple[ImageInput, ...],
        response_schema: dict[str, Any],
        system_prompt: str | None,
        temperature: float | None,
        top_p: float | None,
        extra_headers: dict[str, str] | None,
    ) -> LLMResponse:
        if self.use_web_search and not self._supports_search_with_structured_output():
            raise ConfigurationError(
                f"Gemini model '{self.model}' does not support combining grounded "
                "web search with response_schema in the same request. Only Gemini 3 "
                "series models support grounded structured outputs. Use a separate "
                "Gemini instance with use_web_search=False for structured calls."
            )
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            **self._sampling_params(temperature, top_p),
            "response_schema": _gemini_schema(response_schema),
            "response_mime_type": "application/json",
        }
        if extra_headers:
            config_kwargs["http_options"] = types.HttpOptions(headers=extra_headers)
        self._apply_web_search(config_kwargs)

        start_time = time.time()
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(**config_kwargs),
                contents=_gemini_contents(user_prompt, images),
            )
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Gemini API error: {e}",
                provider="gemini",
                original_error=e,
            ) from e
        execution_time = time.time() - start_time

        input_tokens, output_tokens, cached_tokens = _gemini_token_counts(response)
        input_cost, output_cost, total_cost = self._calculate_costs(
            input_tokens, output_tokens, cached_tokens
        )
        tool_use_cost = self._compute_web_search_cost(response)
        total_cost += tool_use_cost

        return LLMResponse(
            content=canonicalize_json_schema_output(response.text or "", response_schema),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
            tool_use_cost=tool_use_cost,
        )


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a Gemini-compatible copy of a JSON schema.

    Nested Pydantic models emit their sub-models under ``$defs`` and reference
    them with ``$ref`` pointers. Gemini's ``generateContent`` schema compiler
    does not resolve named ``$ref``s, so we inline them first (via
    :func:`inline_schema_refs`) and then strip the remaining keywords Gemini
    rejects.
    """
    schema = inline_schema_refs(schema)
    unsupported_keywords = {"$schema", "$id", "additionalProperties"}

    def strip_unsupported(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip_unsupported(nested_value)
                for key, nested_value in value.items()
                if key not in unsupported_keywords
            }
        if isinstance(value, list):
            return [strip_unsupported(item) for item in value]
        return value

    return cast(dict[str, Any], strip_unsupported(schema))


def _gemini_contents(user_prompt: str, images: tuple[ImageInput, ...]) -> Any:
    """Build Gemini contents while preserving the text-only fast path."""
    if not images:
        return user_prompt
    parts: list[Any] = [
        types.Part.from_bytes(data=image.data, mime_type=image.media_type) for image in images
    ]
    parts.append(user_prompt)
    return parts


def _gemini_token_counts(response: Any) -> tuple[int, int, int]:
    """Extract Gemini token counts with typed non-None defaults.

    Returns ``(input_tokens, output_tokens, cached_tokens)``. Gemini's
    ``prompt_token_count`` is the full prompt size and already includes
    ``cached_content_token_count`` (cache reads), so the cached count is a subset
    of the input count.
    """
    usage_metadata = response.usage_metadata
    assert usage_metadata is not None
    return (
        int(usage_metadata.prompt_token_count or 0),
        int(usage_metadata.candidates_token_count or 0),
        int(usage_metadata.cached_content_token_count or 0),
    )
