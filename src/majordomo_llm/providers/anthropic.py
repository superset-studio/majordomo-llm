"""Anthropic (Claude) LLM provider implementation."""

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlockParam,
    ToolChoiceAutoParam,
    ToolChoiceToolParam,
    ToolParam,
    WebSearchTool20250305Param,
)

from majordomo_llm.base import (
    DEFAULT_MAX_TOKENS,
    LLM,
    LLMResponse,
    LLMStreamResponse,
    _StreamState,
    canonicalize_json_schema_output,
    enforce_strict_object_schema,
    resolve_api_key,
    strip_unsupported_schema_constraints,
)
from majordomo_llm.exceptions import ProviderError, ResponseParsingError
from majordomo_llm.retry import retry_provider_call

logger = logging.getLogger(__name__)

#: Valid ``output_config.effort`` levels (Claude 4.7+/5 generation).
_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})

#: The Anthropic SDK refuses a non-streaming request whose ``max_tokens`` implies
#: more than its 10-minute timeout, computed as ``3600 * max_tokens / 128_000``.
#: That puts the hard limit at 21_333. We check it ourselves so the failure names
#: our parameter and the remedy, instead of surfacing the SDK's message from
#: several frames down inside ``messages.create``.
MAX_NONSTREAMING_TOKENS = 21_333

#: Valid ``thinking.type`` modes. ``adaptive`` is the on-mode for the 4.6+/5
#: generation (Claude decides depth); ``disabled`` turns thinking off (rejected
#: on Fable 5, where thinking is always on). Legacy ``enabled`` + ``budget_tokens``
#: is intentionally not exposed here.
_THINKING_MODES = frozenset({"adaptive", "disabled"})


class Anthropic(LLM):
    """Anthropic (Claude) LLM provider.

    Implements the LLM interface for Anthropic's Claude models, including
    support for tool calling for structured outputs and optional web search.

    The API key is read from the ``ANTHROPIC_API_KEY`` environment variable.

    Attributes:
        client: The async Anthropic client instance.

    Example:
        >>> llm = Anthropic(
        ...     model="claude-sonnet-5",
        ...     input_cost=3.0,
        ...     output_cost=15.0,
        ... )
        >>> response = await llm.get_response("Hello, Claude!")
    """

    #: Anthropic reports cache-read/cache-write tokens separately from
    #: ``input_tokens``, so cache cost is added on top of the uncached input.
    _cache_accounting = "additive"

    def __init__(
        self,
        model: str,
        input_cost: float,
        output_cost: float,
        supports_temperature_top_p: bool = True,
        use_web_search: bool = False,
        supports_structured_outputs: bool = False,
        reasoning_effort: str | None = None,
        thinking: str | None = None,
        *,
        cached_input_cost: float | None = None,
        cache_write_cost: float | None = None,
        use_prompt_caching: bool = True,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            model: The Claude model identifier (e.g., "claude-sonnet-5").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            use_web_search: Enable web search (requires claude-sonnet-4-5-20250929).
            supports_structured_outputs: Whether the model supports native
                structured outputs (constrained decoding via
                ``output_config.format``). When False, structured JSON requests
                fall back to forced tool calling. Defaults to False.
            reasoning_effort: Optional ``output_config.effort`` level applied to
                every request — one of ``low``, ``medium``, ``high``, ``xhigh``,
                ``max``. Controls thinking depth and overall token spend on the
                4.7+/5 generation. ``None`` (default) sends no effort, so the
                API default (``high``) applies. Register the same SKU under
                multiple YAML keys (via the ``model`` override) to expose
                distinct effort profiles.
            thinking: Optional ``thinking.type`` mode applied to every request —
                ``adaptive`` (Claude decides how much to think; the on-mode for
                the 4.6+/5 generation) or ``disabled``. ``None`` (default) omits
                the field, so the model runs without thinking. Effort only
                meaningfully modulates depth when thinking is on, so pair the two.
                Note: ``disabled`` is rejected on Fable 5 (thinking is always on),
                and with thinking on ``max_tokens`` covers thinking + answer, so
                a deep-thinking profile needs headroom for both. Raise it with the
                ``max_tokens`` key in ``llm_config.yaml`` or the per-request
                ``max_tokens`` argument; a response that hits the cap raises
                :class:`~majordomo_llm.exceptions.ResponseTruncatedError` rather
                than returning silently truncated content.
            cached_input_cost: Cost per million cache-read tokens in USD
                (``cache_read_input_tokens``), billed on top of uncached input.
            cache_write_cost: Cost per million cache-creation tokens in USD
                (``cache_creation_input_tokens``), billed on top of uncached input.
            use_prompt_caching: When ``True`` (default), the system prompt is sent
                with an ephemeral ``cache_control`` breakpoint so Anthropic caches
                it. Set ``False`` to disable prompt caching (e.g. for short,
                non-reused system prompts where the cache-write premium is wasted).
            api_key: Optional API key. Defaults to ``ANTHROPIC_API_KEY`` env var.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom base URL for routing through a proxy.
            default_headers: Optional headers sent with every request.
            max_tokens: Default output cap for this model. ``None`` uses the
                library defaults (16000 non-streaming, 64000 streaming).

        Raises:
            ConfigurationError: If no API key is provided and env var is not set.
            ValueError: If ``reasoning_effort`` or ``thinking`` is invalid.
        """
        if reasoning_effort is not None and reasoning_effort not in _EFFORT_LEVELS:
            valid = ", ".join(sorted(_EFFORT_LEVELS))
            raise ValueError(
                f"Invalid Anthropic reasoning_effort '{reasoning_effort}'. Valid: {valid}"
            )
        if thinking is not None and thinking not in _THINKING_MODES:
            valid = ", ".join(sorted(_THINKING_MODES))
            raise ValueError(f"Invalid Anthropic thinking mode '{thinking}'. Valid: {valid}")
        resolved_api_key = resolve_api_key(api_key, "ANTHROPIC_API_KEY", "Anthropic")
        self.supports_structured_outputs = supports_structured_outputs
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        super().__init__(
            provider="anthropic",
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            cached_input_cost=cached_input_cost,
            cache_write_cost=cache_write_cost,
            use_prompt_caching=use_prompt_caching,
            supports_temperature_top_p=supports_temperature_top_p,
            use_web_search=use_web_search,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
            base_url=base_url,
            default_headers=default_headers,
            max_tokens=max_tokens,
        )
        self.client = anthropic.AsyncAnthropic(
            api_key=resolved_api_key,
            base_url=self.base_url,
            default_headers=self.default_headers,
        )

    def _resolve_nonstreaming_max_tokens(self, max_tokens: int | None) -> int:
        """Resolve the cap for a non-streaming call, rejecting one the SDK cannot send.

        Args:
            max_tokens: Per-request override, or None.

        Returns:
            The cap to send.

        Raises:
            ValueError: If the resolved cap exceeds :data:`MAX_NONSTREAMING_TOKENS`.
        """
        resolved = self._resolve_max_tokens(max_tokens)
        if resolved > MAX_NONSTREAMING_TOKENS:
            raise ValueError(
                f"max_tokens={resolved} exceeds the {MAX_NONSTREAMING_TOKENS} limit the "
                f"Anthropic SDK allows on a non-streaming request. Lower it, or use "
                f"get_response_stream(), which has no such limit."
            )
        return resolved

    def _config_create_kwargs(self, fmt: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build config-derived ``messages.create`` kwargs, for splatting.

        Combines the optional structured-output ``format`` with the configured
        ``reasoning_effort`` (both under ``output_config``) and the configured
        ``thinking`` mode. Returns ``{}`` (a no-op splat) when none are present,
        so callers can uniformly write ``**self._config_create_kwargs()`` without
        a conditional.
        """
        kwargs: dict[str, Any] = {}
        output_config: dict[str, Any] = {}
        if fmt is not None:
            output_config["format"] = fmt
        if self.reasoning_effort is not None:
            output_config["effort"] = self.reasoning_effort
        if output_config:
            kwargs["output_config"] = output_config
        if self.thinking is not None:
            kwargs["thinking"] = {"type": self.thinking}
        return kwargs

    # Anthropic bills server-side web search at $10 per 1,000 requests.
    _WEB_SEARCH_COST_PER_REQUEST = 0.01

    def _compute_web_search_cost(self, response: Any) -> float:
        """Return the per-call web-search fee charged by Anthropic.

        Reads ``response.usage.server_tool_use.web_search_requests`` which is
        populated only when the web_search tool was actually invoked.
        """
        server_tool_use = getattr(response.usage, "server_tool_use", None)
        if server_tool_use is None:
            return 0.0
        requests = getattr(server_tool_use, "web_search_requests", 0) or 0
        return requests * self._WEB_SEARCH_COST_PER_REQUEST

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
        """Get a plain text response from Anthropic."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant"
        start_time = time.time()

        messages = _anthropic_user_message(user_prompt)
        system_message = _anthropic_system_prompt(system_prompt, self.use_prompt_caching)

        tools: list[Any] = []
        if self.use_web_search:
            tools.append(
                WebSearchTool20250305Param(type="web_search_20250305", name="web_search")
            )

        resolved_max_tokens = self._resolve_nonstreaming_max_tokens(max_tokens)

        try:
            response_message = await self.client.messages.create(
                model=self.model,
                max_tokens=resolved_max_tokens,
                system=system_message,
                messages=messages,
                tools=tools,
                tool_choice=ToolChoiceAutoParam(type="auto"),
                **self._config_create_kwargs(),
                extra_headers=extra_headers,
                **self._sampling_params(temperature, top_p),
            )
        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        final_response = [c.text for c in response_message.content if c.type == "text"]

        input_tokens = response_message.usage.input_tokens
        output_tokens = response_message.usage.output_tokens
        self._check_truncation(
            response_message.stop_reason,
            resolved_max_tokens,
            output_tokens,
            "\n".join(final_response),
        )
        cached_tokens = response_message.usage.cache_read_input_tokens or 0
        cache_creation_tokens = response_message.usage.cache_creation_input_tokens or 0
        input_cost, output_cost, total_cost = self._calculate_costs(
            input_tokens, output_tokens, cached_tokens, cache_creation_tokens
        )
        tool_use_cost = self._compute_web_search_cost(response_message)
        total_cost += tool_use_cost

        return LLMResponse(
            content="\n".join(final_response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
            tool_use_cost=tool_use_cost,
            deprecation_warning=self.deprecation_warning,
            stop_reason=response_message.stop_reason,
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
        """Get a streaming text response from Anthropic."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant"

        state = _StreamState()
        resolved_max_tokens = self._resolve_max_tokens(max_tokens, streaming=True)
        messages = _anthropic_user_message(user_prompt)
        system_message = _anthropic_system_prompt(system_prompt, self.use_prompt_caching)

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=resolved_max_tokens,
                system=system_message,
                messages=messages,
                stream=True,
                **self._config_create_kwargs(),
                extra_headers=extra_headers,
                **self._sampling_params(temperature, top_p),
            )
        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                original_error=e,
            ) from e

        async def generator() -> AsyncIterator[str]:
            try:
                async for event in response:
                    if event.type == "message_start":
                        state.input_tokens = event.message.usage.input_tokens
                        state.cached_tokens = event.message.usage.cache_read_input_tokens or 0
                        state.cache_creation_tokens = (
                            event.message.usage.cache_creation_input_tokens or 0
                        )
                    elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield event.delta.text
                    elif event.type == "message_delta":
                        state.output_tokens = event.usage.output_tokens
                        state.stop_reason = event.delta.stop_reason
            except anthropic.APIError as e:
                raise ProviderError(
                    f"Anthropic API error: {e}",
                    provider="anthropic",
                    original_error=e,
                ) from e
            # Raised after the last chunk is yielded, so a truncated stream fails
            # the same way a truncated non-streaming call does rather than ending
            # quietly mid-sentence.
            self._check_truncation(
                state.stop_reason, resolved_max_tokens, state.output_tokens, ""
            )

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
        """Anthropic structured JSON output.

        Uses native structured outputs (constrained decoding via
        ``output_config.format``) on models that support it — the model
        physically cannot emit malformed or missing-key output. Falls back to
        forced tool calling on models without native support, and always uses
        the forced-tool path when web search is enabled (structured outputs and
        the ``web_search`` tool cannot be combined). Every path validates
        against the caller's original schema and rejects an empty/all-null
        result via :func:`canonicalize_json_schema_output`.
        """
        resolved_max_tokens = self._resolve_nonstreaming_max_tokens(max_tokens)

        if self.use_web_search:
            response, execution_time = await self._json_schema_response_with_web_search_helper(
                user_prompt=user_prompt,
                response_schema=response_schema,
                system_prompt=system_prompt,
                schema_name=schema_name,
                schema_description=schema_description,
                extra_headers=extra_headers,
                max_tokens=resolved_max_tokens,
            )
            content = _extract_tool_use_content(response.content, schema_name)
            return self._finalize_json_schema_response(
                content, response, execution_time, response_schema, resolved_max_tokens
            )

        if self.supports_structured_outputs:
            return await self._native_json_schema_response(
                user_prompt=user_prompt,
                response_schema=response_schema,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
                max_tokens=resolved_max_tokens,
            )

        return await self._forced_tool_json_schema_response(
            user_prompt=user_prompt,
            response_schema=response_schema,
            system_prompt=system_prompt,
            schema_name=schema_name,
            schema_description=schema_description,
            temperature=temperature,
            top_p=top_p,
            extra_headers=extra_headers,
            max_tokens=resolved_max_tokens,
        )

    async def _native_json_schema_response(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None,
        temperature: float | None,
        top_p: float | None,
        extra_headers: dict[str, str] | None,
        max_tokens: int,
    ) -> LLMResponse:
        """Native structured outputs via ``output_config.format`` (constrained decoding).

        The constrained decoder requires strict object schemas
        (``additionalProperties: false`` + full ``required``) and rejects a set
        of validation keywords (numeric/string/array bounds, ``pattern``,
        ``format``). Those are stripped from the wire schema and re-enforced
        post-hoc by validating the response against the original schema.
        """
        sent_schema = strip_unsupported_schema_constraints(
            enforce_strict_object_schema(response_schema)
        )
        output_config = self._config_create_kwargs(
            fmt={"type": "json_schema", "schema": sent_schema}
        )

        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        messages = _anthropic_user_message(user_prompt)
        system_message = _anthropic_system_prompt(system_prompt, self.use_prompt_caching)

        start_time = time.time()
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_message,
                messages=messages,
                **output_config,
                extra_headers=extra_headers,
                **self._sampling_params(temperature, top_p),
            )
        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        if response.stop_reason == "refusal":
            raise ResponseParsingError(
                "Anthropic refused the structured-output request.",
                raw_content=str(response.content),
            )
        content = _extract_structured_text(response.content)
        return self._finalize_json_schema_response(
            content, response, execution_time, response_schema, max_tokens
        )

    async def _forced_tool_json_schema_response(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None,
        schema_name: str,
        schema_description: str | None,
        temperature: float | None,
        top_p: float | None,
        extra_headers: dict[str, str] | None,
        max_tokens: int,
    ) -> LLMResponse:
        """Forced-tool fallback for models without native structured outputs.

        Sends the schema in strict form (full ``required``) so an empty ``{}``
        fails schema validation loudly; an all-null result is caught by the
        emptiness check in :func:`canonicalize_json_schema_output`. Both surface
        as :class:`EmptyStructuredResponseError`, which
        :func:`~majordomo_llm.retry.retry_provider_call` re-samples before it
        propagates.
        """
        tool_instruction = f"Use the {schema_name} tool to provide your answer."
        if system_prompt is None:
            system_prompt = f"You are a helpful assistant. {tool_instruction}"
        else:
            system_prompt = f"{system_prompt}\n\n{tool_instruction}"

        messages = _anthropic_user_message(user_prompt)
        system_message = _anthropic_system_prompt(system_prompt, self.use_prompt_caching)
        tools = [
            ToolParam(
                name=schema_name,
                description=schema_description
                or f"Provide a structured response using the {schema_name} JSON schema",
                input_schema=enforce_strict_object_schema(response_schema),
            )
        ]

        start_time = time.time()
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_message,
                messages=messages,
                tools=tools,
                tool_choice=ToolChoiceToolParam(type="tool", name=schema_name),
                **self._config_create_kwargs(),
                extra_headers=extra_headers,
                **self._sampling_params(temperature, top_p),
            )
        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        content = _extract_tool_use_content(response.content, schema_name)
        return self._finalize_json_schema_response(
            content, response, execution_time, response_schema, max_tokens
        )

    def _finalize_json_schema_response(
        self,
        content: Any,
        response: Any,
        execution_time: float,
        response_schema: dict[str, Any],
        max_tokens: int,
    ) -> LLMResponse:
        """Compute usage/cost and validate structured content into an ``LLMResponse``.

        Raises :class:`ResponseTruncatedError` before parsing: a structured
        response cut off at the cap is malformed JSON, and reporting the cause
        beats a downstream parse error that names the symptom.
        """
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        self._check_truncation(response.stop_reason, max_tokens, output_tokens, str(content))
        cached_tokens = response.usage.cache_read_input_tokens or 0
        cache_creation_tokens = response.usage.cache_creation_input_tokens or 0
        input_cost, output_cost, total_cost = self._calculate_costs(
            input_tokens, output_tokens, cached_tokens, cache_creation_tokens
        )
        tool_use_cost = self._compute_web_search_cost(response)
        total_cost += tool_use_cost

        return LLMResponse(
            content=canonicalize_json_schema_output(content, response_schema),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
            tool_use_cost=tool_use_cost,
            stop_reason=response.stop_reason,
        )

    async def _json_schema_response_with_web_search_helper(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[Any, float]:
        """Helper for web search with raw JSON-schema structured response."""
        structured_response_tool = ToolParam(
            name=schema_name,
            description=schema_description
            or f"Provide a structured response using the {schema_name} JSON schema",
            input_schema=enforce_strict_object_schema(response_schema),
        )
        web_search_tool = WebSearchTool20250305Param(
            name="web_search",
            type="web_search_20250305",
        )
        tools: list[Any] = [structured_response_tool, web_search_tool]

        tool_instruction = f"Use the {schema_name} tool to provide your answer."
        if system_prompt is None:
            system_prompt = f"You are a helpful assistant. {tool_instruction}"
        else:
            system_prompt = f"{system_prompt}\n\n{tool_instruction}"

        messages = _anthropic_user_message(user_prompt)
        system_message = _anthropic_system_prompt(system_prompt, self.use_prompt_caching)

        start_time = time.time()
        current_messages = messages.copy()
        search_count = 0

        try:
            while search_count < 3:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_message,
                    messages=current_messages,
                    tools=tools,
                    tool_choice=ToolChoiceAutoParam(type="auto"),
                    **self._config_create_kwargs(),
                    extra_headers=extra_headers,
                )

                if response.stop_reason == "tool_use":
                    tool_uses = [block for block in response.content if block.type == "tool_use"]
                    if any(tool_use.name == schema_name for tool_use in tool_uses):
                        execution_time = time.time() - start_time
                        return response, execution_time

                    if any(tool_use.name == "web_search" for tool_use in tool_uses):
                        logger.info("Web search initiated (turn %d)", search_count + 1)
                        search_count += 1
                        current_messages.append({"role": "assistant", "content": response.content})
                        current_messages.append({
                            "role": "user",
                            "content": (
                                "Continue with your analysis. Use the structured response "
                                "tool when ready to generate the final output."
                            ),
                        })
                        continue
                break

            final_response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=_anthropic_system_prompt(system_prompt, self.use_prompt_caching),
                messages=current_messages,
                tools=[structured_response_tool],
                tool_choice=ToolChoiceToolParam(type="tool", name=schema_name),
                **self._config_create_kwargs(),
                extra_headers=extra_headers,
            )
        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        return final_response, execution_time


def _extract_tool_use_content(content_blocks: list[Any], tool_name: str) -> Any:
    """Extract a named Anthropic tool_use input from response content blocks."""
    for block in content_blocks:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input
    raise ResponseParsingError(
        f"No {tool_name} tool use found in Anthropic response",
        raw_content=str(content_blocks),
    )


def _extract_structured_text(content_blocks: list[Any]) -> str:
    """Extract the JSON text from a native structured-output (``output_config``) response."""
    parts = [block.text for block in content_blocks if block.type == "text"]
    if not parts:
        raise ResponseParsingError(
            "No text content in Anthropic structured-output response",
            raw_content=str(content_blocks),
        )
    return "\n".join(parts)


def _anthropic_system_prompt(
    system_prompt: str, use_prompt_caching: bool = True
) -> list[TextBlockParam]:
    """Create an Anthropic system prompt block.

    When ``use_prompt_caching`` is ``True`` the block carries an ephemeral
    ``cache_control`` breakpoint so Anthropic caches the system prompt; when
    ``False`` the breakpoint is omitted and no cache is created.
    """
    if use_prompt_caching:
        return [
            TextBlockParam(
                type="text",
                text=system_prompt,
                cache_control=CacheControlEphemeralParam(type="ephemeral"),
            )
        ]
    return [TextBlockParam(type="text", text=system_prompt)]


def _anthropic_user_message(user_prompt: str) -> list[MessageParam]:
    """Create Anthropic user message."""
    return [
        MessageParam(
            role="user",
            content=user_prompt,
        )
    ]
