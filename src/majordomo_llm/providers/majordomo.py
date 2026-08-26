"""Majordomo gateway provider — optimal routing across open-weight backends."""

import logging
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
    compute_costs,
    resolve_api_key,
)
from majordomo_llm.exceptions import ConfigurationError, ProviderError
from majordomo_llm.retry import retry_provider_call

logger = logging.getLogger(__name__)


class Majordomo(LLM):
    """Majordomo gateway provider with server-side optimal routing.

    Unlike the concrete providers, ``majordomo`` does not name a backend. The
    caller names a canonical open-weight model (e.g. ``"kimi-k3"``) and Majordomo
    Steward selects the optimal backend (Fireworks, Together, …) for that model
    at request time. Because the backend — and therefore its published rates —
    is only known *after* the call, this provider does not price from its own
    config entry. Instead it reads the gateway's ``X-Majordomo-Routed-Provider``
    and ``X-Majordomo-Routed-Model`` response headers and prices the usage
    against that pair's rates in ``llm_config.yaml`` (:func:`get_model_pricing`).

    Routing is signalled to the gateway with ``x-majordomo-provider: majordomo``.
    This provider only makes sense behind Steward, so ``base_url`` is required
    and ``MAJORDOMO_API_KEY`` must be set — the key is injected automatically as
    the ``X-Majordomo-Key`` header on every request.

    The wire protocol is OpenAI-compatible chat completions, matching the
    open-weight backends the gateway routes to.

    Attributes:
        client: The async OpenAI client instance pointed at the gateway.

    Example:
        >>> llm = Majordomo(
        ...     model="kimi-k3",
        ...     base_url=os.environ["MAJORDOMO_GATEWAY_URL"],
        ... )
        >>> response = await llm.get_response("Hello!")
        >>> print(response.routed_provider, response.routed_model)
    """

    ROUTED_PROVIDER_HEADER = "X-Majordomo-Routed-Provider"
    ROUTED_MODEL_HEADER = "X-Majordomo-Routed-Model"

    def __init__(
        self,
        model: str,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        supports_temperature_top_p: bool = True,
        *,
        cached_input_cost: float | None = None,
        cache_write_cost: float | None = None,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the Majordomo gateway provider.

        Args:
            model: The canonical model name the gateway routes on (e.g. "kimi-k3").
            input_cost: Ignored — cost is resolved per request from the routed
                backend. Accepted for a uniform factory signature.
            output_cost: Ignored — see ``input_cost``.
            supports_temperature_top_p: Whether temperature/top_p are forwarded.
            cached_input_cost: Ignored — see ``input_cost``.
            cache_write_cost: Ignored — see ``input_cost``.
            api_key: Optional Majordomo API key. Defaults to the
                ``MAJORDOMO_API_KEY`` env var. Sent as the ``X-Majordomo-Key``
                header; the gateway injects the backend provider's own key.
            api_key_alias: Optional human-readable name for the API key.
            base_url: The Majordomo gateway URL. **Required** — this provider
                only operates behind the gateway.
            default_headers: Optional headers sent with every request. Caller
                values win over the auto-injected routing/auth headers.

        Raises:
            ConfigurationError: If ``base_url`` is not provided, or if no
                ``MAJORDOMO_API_KEY`` is available.
        """
        if base_url is None:
            raise ConfigurationError(
                "The 'majordomo' provider routes through the Majordomo gateway and "
                "requires base_url (set it to your MAJORDOMO_GATEWAY_URL)."
            )

        resolved_api_key = resolve_api_key(api_key, "MAJORDOMO_API_KEY", "Majordomo")

        # Signal optimal routing and authenticate to the gateway. The gateway
        # routes on x-majordomo-model (the canonical model name), not the wire
        # body's "model" field, so it is sent as a header too. Caller-supplied
        # default_headers win on key collision.
        merged_headers: dict[str, str] = {
            "x-majordomo-provider": "majordomo",
            "x-majordomo-model": model,
            "X-Majordomo-Key": resolved_api_key,
        }
        if default_headers:
            merged_headers.update(default_headers)
        default_headers = merged_headers

        super().__init__(
            provider="majordomo",
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
            base_url=self.base_url,
            default_headers=self.default_headers,
        )

    def _price_routed(
        self,
        routed_provider: str | None,
        routed_model: str | None,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        cache_creation_tokens: int = 0,
    ) -> tuple[float, float, float]:
        """Price a call using the rates of the backend the gateway routed to.

        Falls back to zero cost (with a warning) when the gateway did not report
        a routed pair, or the reported pair has no configured pricing — usage
        counts still stand, but the cost is unknown to this client.
        """
        if not routed_provider or not routed_model:
            logger.warning(
                "Majordomo: gateway response missing %s/%s headers; reporting cost 0.0",
                self.ROUTED_PROVIDER_HEADER,
                self.ROUTED_MODEL_HEADER,
            )
            return 0.0, 0.0, 0.0

        # Lazy import to avoid a circular import (factory imports this module).
        from majordomo_llm.factory import get_model_pricing

        pricing = get_model_pricing(routed_provider, routed_model)
        if pricing is None:
            logger.warning(
                "Majordomo: no pricing configured for routed pair %s/%s; reporting cost 0.0",
                routed_provider,
                routed_model,
            )
            return 0.0, 0.0, 0.0

        return compute_costs(
            input_tokens,
            output_tokens,
            cached_tokens,
            cache_creation_tokens,
            input_cost=pricing.input_cost,
            output_cost=pricing.output_cost,
            cached_input_cost=pricing.cached_input_cost,
            cache_write_cost=pricing.cache_write_cost,
            cache_accounting=pricing.cache_accounting,
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
        """Get a plain text response routed through the Majordomo gateway."""
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
        """Internal method to get a routed response from the gateway."""
        messages: list[Any] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        start_time = time.time()
        try:
            raw = await self.client.chat.completions.with_raw_response.create(
                **self._request_kwargs(messages, temperature, top_p, extra_headers)
            )
        except openai.APIError as e:
            raise ProviderError(
                f"Majordomo gateway error: {e}",
                provider="majordomo",
                original_error=e,
            ) from e

        routed_provider = raw.headers.get(self.ROUTED_PROVIDER_HEADER)
        routed_model = raw.headers.get(self.ROUTED_MODEL_HEADER)
        response = raw.parse()

        execution_time = time.time() - start_time
        assert response.usage is not None
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = _cached_tokens(response.usage)
        input_cost, output_cost, total_cost = self._price_routed(
            routed_provider, routed_model, input_tokens, output_tokens, cached_tokens
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
            routed_provider=routed_provider,
            routed_model=routed_model,
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
        """Get a streaming text response routed through the Majordomo gateway."""
        messages: list[Any] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        state = _StreamState()
        try:
            raw = await self.client.chat.completions.with_raw_response.create(
                **self._request_kwargs(messages, temperature, top_p, extra_headers),
                stream=True,
                stream_options={"include_usage": True},
            )
        except openai.APIError as e:
            raise ProviderError(
                f"Majordomo gateway error: {e}",
                provider="majordomo",
                original_error=e,
            ) from e

        # Routing headers are known at stream start; price the final usage
        # against the routed backend rather than this provider's (empty) rates.
        routed_provider = raw.headers.get(self.ROUTED_PROVIDER_HEADER)
        routed_model = raw.headers.get(self.ROUTED_MODEL_HEADER)
        state.routed_provider = routed_provider
        state.routed_model = routed_model
        state.price_override = (
            lambda i, o, c, w: self._price_routed(routed_provider, routed_model, i, o, c, w)
        )
        response = raw.parse()

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
                raise ProviderError(
                    f"Majordomo gateway error: {e}",
                    provider="majordomo",
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
        """Gateway-routed structured output via OpenAI-compatible JSON Schema."""
        messages: list[Any] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

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
            raw = await self.client.chat.completions.with_raw_response.create(
                **self._request_kwargs(messages, temperature, top_p, extra_headers),
                response_format=response_format,
            )
        except openai.APIError as e:
            raise ProviderError(
                f"Majordomo gateway error: {e}",
                provider="majordomo",
                original_error=e,
            ) from e

        routed_provider = raw.headers.get(self.ROUTED_PROVIDER_HEADER)
        routed_model = raw.headers.get(self.ROUTED_MODEL_HEADER)
        response = raw.parse()

        execution_time = time.time() - start_time
        assert response.usage is not None
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = _cached_tokens(response.usage)
        input_cost, output_cost, total_cost = self._price_routed(
            routed_provider, routed_model, input_tokens, output_tokens, cached_tokens
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
            routed_provider=routed_provider,
            routed_model=routed_model,
        )

    def _request_kwargs(
        self,
        messages: list[Any],
        temperature: float | None,
        top_p: float | None,
        extra_headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Assemble the shared chat-completion request keyword arguments."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "extra_headers": extra_headers,
        }
        kwargs.update(self._sampling_params(temperature, top_p))
        return kwargs


def _cached_tokens(usage: Any) -> int:
    """Extract cache-read token count from an OpenAI-compatible usage object."""
    return (
        getattr(
            getattr(usage, "prompt_tokens_details", None),
            "cached_tokens",
            0,
        )
        or 0
    )
