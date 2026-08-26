"""Shared implementation for OpenAI-compatible LLM providers.

Several inference platforms (Baseten, Nebius Token Factory, DeepInfra, Moonshot)
expose the OpenAI chat-completions wire protocol verbatim and differ only in the
endpoint, the API key environment variable, and the model-ID namespace. This
module holds the request/response, streaming, and JSON-schema machinery once;
each provider subclasses :class:`OpenAICompatibleLLM` and sets four class
attributes.

Example:
    >>> class Example(OpenAICompatibleLLM):
    ...     PROVIDER_NAME = "example"
    ...     DISPLAY_NAME = "Example"
    ...     DEFAULT_BASE_URL = "https://api.example.com/v1"
    ...     API_KEY_ENV = "EXAMPLE_API_KEY"
"""

import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import openai

from majordomo_llm.base import (
    LLM,
    LLMResponse,
    LLMStreamResponse,
    _StreamState,
    canonicalize_json_schema_output,
    resolve_api_key,
)
from majordomo_llm.exceptions import ProviderError, StructuredOutputUnsupported
from majordomo_llm.retry import retry_provider_call


def _cached_tokens(usage: Any) -> int:
    """Read cache-read tokens from an OpenAI-shaped usage object.

    Providers that do not report a prompt-cache breakdown omit
    ``prompt_tokens_details`` entirely (or set it to ``None``), in which case
    cached tokens are reported as 0 and the whole prompt bills at ``input_cost``.
    """
    return getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0


class OpenAICompatibleLLM(LLM):
    """Base class for providers speaking the OpenAI chat-completions protocol.

    Subclasses set :attr:`PROVIDER_NAME`, :attr:`DISPLAY_NAME`,
    :attr:`DEFAULT_BASE_URL`, and :attr:`API_KEY_ENV`. Everything else — text,
    streaming, and structured JSON-schema output, plus cost accounting and error
    wrapping — is inherited.

    Cached prompt tokens follow the default ``"subset"`` accounting mode: the
    provider counts them inside ``prompt_tokens``, so ``cached_input_cost``
    re-prices that subset rather than adding to it.

    Attributes:
        client: The async OpenAI client instance configured for the provider.
    """

    #: Factory key for the provider (e.g. ``"baseten"``). Also the value sent as
    #: ``x-majordomo-provider`` when routing through a gateway.
    PROVIDER_NAME: ClassVar[str]
    #: Human-readable provider name, used in error messages.
    DISPLAY_NAME: ClassVar[str]
    #: Endpoint used when the caller does not pass ``base_url``.
    DEFAULT_BASE_URL: ClassVar[str]
    #: Environment variable consulted when the caller does not pass ``api_key``.
    API_KEY_ENV: ClassVar[str]

    REASONING_EFFORTS: ClassVar[frozenset[str]] = frozenset({"minimal", "low", "medium", "high"})
    THINKING_MODES: ClassVar[frozenset[str]] = frozenset({"enabled", "disabled"})

    def __init__(
        self,
        model: str,
        input_cost: float,
        output_cost: float,
        supports_temperature_top_p: bool = True,
        *,
        supports_structured_outputs: bool = True,
        cached_input_cost: float | None = None,
        cache_write_cost: float | None = None,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
        thinking: str | None = None,
    ) -> None:
        """Initialize an OpenAI-compatible provider.

        Args:
            model: The provider's model ID, passed through to the API verbatim.
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            supports_structured_outputs: Whether the endpoint honors a strict
                ``json_schema`` response format. Defaults to ``True``; set it
                ``False`` for a deployment that accepts the parameter without
                grammar-constrained decoding, so structured calls fail fast
                instead of returning prose some fraction of the time.
            cached_input_cost: Cost per million cache-read tokens in USD (a subset
                of input tokens). Left unset for providers with no discounted
                cache tier, in which case cached tokens bill at ``input_cost``.
            cache_write_cost: Unused; accepted for a uniform factory signature.
            api_key: Optional API key. Defaults to :attr:`API_KEY_ENV`.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom base URL. Overrides :attr:`DEFAULT_BASE_URL`.
            default_headers: Optional headers sent with every request.
            reasoning_effort: Optional reasoning effort level for models that
                support it, forwarded as the OpenAI-standard field.
            thinking: Optional thinking mode ("enabled" or "disabled") for models
                that support it. Forwarded via ``extra_body``.

        Raises:
            ConfigurationError: If no API key is provided and the env var is unset.
            ValueError: If reasoning_effort or thinking is invalid.
        """
        if reasoning_effort is not None and reasoning_effort not in self.REASONING_EFFORTS:
            valid = ", ".join(sorted(self.REASONING_EFFORTS))
            raise ValueError(
                f"Invalid {self.DISPLAY_NAME} reasoning_effort '{reasoning_effort}'. "
                f"Valid: {valid}"
            )
        if thinking is not None and thinking not in self.THINKING_MODES:
            valid = ", ".join(sorted(self.THINKING_MODES))
            raise ValueError(
                f"Invalid {self.DISPLAY_NAME} thinking mode '{thinking}'. Valid: {valid}"
            )

        resolved_api_key = resolve_api_key(api_key, self.API_KEY_ENV, self.DISPLAY_NAME)

        # When routing through a proxy (e.g. Majordomo Steward), auto-inject
        # ``x-majordomo-provider`` so the gateway can tell this backend apart from
        # vanilla OpenAI — they share a wire shape. Caller-supplied
        # default_headers win on key collision.
        if base_url is not None:
            merged_headers: dict[str, str] = {"x-majordomo-provider": self.PROVIDER_NAME}
            if default_headers:
                merged_headers.update(default_headers)
            default_headers = merged_headers

        super().__init__(
            provider=self.PROVIDER_NAME,
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
            base_url=self.base_url or self.DEFAULT_BASE_URL,
            default_headers=self.default_headers,
        )
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.supports_structured_outputs = supports_structured_outputs

    def _provider_request_kwargs(self) -> dict[str, Any]:
        """Build provider-specific request options for reasoning-capable models.

        The default forwards ``reasoning_effort`` as the OpenAI-standard field and
        ``thinking`` via ``extra_body``. Both are omitted unless the model's config
        sets them. Providers that reject the two together (Fireworks) override this.
        """
        kwargs: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.thinking is not None:
            kwargs["extra_body"] = {"thinking": {"type": self.thinking}}
        return kwargs

    def _build_messages(
        self, user_prompt: str, system_prompt: str | None
    ) -> list[dict[str, str]]:
        """Assemble the chat-completions messages array."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _sampling_kwargs(
        self, temperature: float | None, top_p: float | None
    ) -> dict[str, Any]:
        """Return only the sampling parameters that should be sent."""
        return self._sampling_params(temperature, top_p)

    def _provider_error(self, error: openai.APIError) -> ProviderError:
        """Wrap an OpenAI-client error as a :class:`ProviderError`."""
        return ProviderError(
            f"{self.DISPLAY_NAME} API error: {error}",
            provider=self.PROVIDER_NAME,
            original_error=error,
        )

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
        """Get a plain text response."""
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
        """Internal method to get a response from the provider."""
        start_time = time.time()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(user_prompt, system_prompt),  # type: ignore[arg-type]
                extra_headers=extra_headers,
                **self._sampling_kwargs(temperature, top_p),
                **self._provider_request_kwargs(),
            )
        except openai.APIError as e:
            raise self._provider_error(e) from e

        execution_time = time.time() - start_time
        assert response.usage is not None
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = _cached_tokens(response.usage)
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
        """Get a streaming text response."""
        state = _StreamState()

        try:
            # ``**`` unpacking erases the literal type of ``stream``, so mypy
            # cannot pick the streaming overload; the runtime call is correct.
            response = await self.client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=self._build_messages(user_prompt, system_prompt),
                stream=True,
                stream_options={"include_usage": True},
                extra_headers=extra_headers,
                **self._sampling_kwargs(temperature, top_p),
                **self._provider_request_kwargs(),
            )
        except openai.APIError as e:
            raise self._provider_error(e) from e

        async def generator() -> AsyncIterator[str]:
            try:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    if chunk.usage:
                        state.input_tokens = chunk.usage.prompt_tokens
                        state.output_tokens = chunk.usage.completion_tokens
                        state.cached_tokens = _cached_tokens(chunk.usage)
            except openai.APIError as e:
                raise self._provider_error(e) from e

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
        """Structured output via the OpenAI-compatible JSON Schema response format.

        Raises:
            StructuredOutputUnsupported: If the model is configured as not
                honoring strict ``json_schema``. Raised before the request is
                built, and not retried (it carries no ``original_error``).
        """
        if not self.supports_structured_outputs:
            raise StructuredOutputUnsupported(self.PROVIDER_NAME, self.model)

        json_schema_payload: dict[str, object] = {
            "name": schema_name,
            "schema": response_schema,
            "strict": True,
        }
        if schema_description is not None:
            json_schema_payload["description"] = schema_description

        response_format: Any = {
            "type": "json_schema",
            "json_schema": json_schema_payload,
        }

        start_time = time.time()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(user_prompt, system_prompt),  # type: ignore[arg-type]
                response_format=response_format,
                extra_headers=extra_headers,
                **self._sampling_kwargs(temperature, top_p),
                **self._provider_request_kwargs(),
            )
        except openai.APIError as e:
            raise self._provider_error(e) from e

        execution_time = time.time() - start_time

        assert response.usage is not None
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = _cached_tokens(response.usage)
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
