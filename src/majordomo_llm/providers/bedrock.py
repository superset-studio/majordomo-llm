"""Amazon Bedrock LLM provider implementation (Converse API)."""

import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import aiobotocore.session
from botocore.exceptions import BotoCoreError, ClientError

from majordomo_llm.base import (
    LLM,
    LLMJSONResponse,
    LLMResponse,
    LLMStreamResponse,
    T,
    _StreamState,
    canonicalize_json_schema_output,
    enforce_strict_object_schema,
    resolve_api_key,
)
from majordomo_llm.exceptions import ConfigurationError, ProviderError, ResponseParsingError
from majordomo_llm.retry import retry_provider_call

logger = logging.getLogger(__name__)


class Bedrock(LLM):
    """Amazon Bedrock LLM provider using the Converse API.

    Authenticates with a long-term Amazon Bedrock API key passed via the
    ``AWS_BEARER_TOKEN_BEDROCK`` environment variable (or the ``api_key``
    constructor argument). The AWS region must be supplied via ``region``
    or the ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` environment variables.

    Routes calls through ``bedrock-runtime`` ``converse`` / ``converse_stream``.

    Example:
        >>> llm = Bedrock(
        ...     model="us.anthropic.claude-sonnet-4-5-v1:0",
        ...     input_cost=3.0,
        ...     output_cost=15.0,
        ...     region="us-east-1",
        ... )
        >>> response = await llm.get_response("Hello!")
    """

    #: Bedrock Converse reports ``cacheReadInputTokens`` /
    #: ``cacheWriteInputTokens`` separately from ``inputTokens``, so cache cost is
    #: added on top of the uncached input.
    _cache_accounting = "additive"

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
        region: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the Bedrock provider.

        Args:
            model: The Bedrock model identifier or inference profile ID
                (e.g., "us.anthropic.claude-sonnet-4-5-v1:0").
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            cached_input_cost: Cost per million cache-read tokens
                (``cacheReadInputTokens``) in USD, billed on top of input.
            cache_write_cost: Cost per million cache-creation tokens
                (``cacheWriteInputTokens``) in USD, billed on top of input.
            use_web_search: Accepted for interface parity; Bedrock Converse has
                no native web search and this flag is ignored.
            api_key: Optional Bedrock API key. Defaults to
                ``AWS_BEARER_TOKEN_BEDROCK`` env var.
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom endpoint URL. When set (e.g. a Majordomo
                Steward gateway), enables the proxy header-injection hook
                described below.
            default_headers: When ``base_url`` is set, these headers are
                injected on every outbound request via a botocore ``before-send``
                hook (aiobotocore exposes no native ``default_headers`` kwarg).
                Used to pass ``X-Majordomo-Key`` and other gateway metadata
                through to Steward. Ignored when ``base_url`` is not set, since
                direct AWS callers must not receive proxy auth headers.
            region: AWS region (e.g., "us-east-1"). Defaults to ``AWS_REGION``
                or ``AWS_DEFAULT_REGION`` env vars.
            max_tokens: Default output cap for this model. ``None`` uses the
                library defaults (16000 non-streaming, 64000 streaming).

        Raises:
            ConfigurationError: If no API key or region can be resolved.
        """
        resolved_api_key = resolve_api_key(
            api_key, "AWS_BEARER_TOKEN_BEDROCK", "Amazon Bedrock"
        )
        resolved_region = (
            region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        )
        if not resolved_region:
            raise ConfigurationError(
                "Amazon Bedrock region not found. Pass region= to the constructor "
                "or set the AWS_REGION (or AWS_DEFAULT_REGION) environment variable."
            )

        super().__init__(
            provider="bedrock",
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
            max_tokens=max_tokens,
        )
        self.region = resolved_region
        # botocore reads the bearer token from this env var when signing
        # bedrock-runtime requests. Set it so explicit constructor keys work
        # even when the env var was not pre-set.
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = resolved_api_key
        self._session = aiobotocore.session.AioSession()

        # When routing through a proxy (e.g. Majordomo Steward), aiobotocore
        # offers no ``default_headers`` kwarg the way the OpenAI SDK does, so we
        # inject headers via botocore's event system. Three things land on the
        # wire:
        #
        # 1. ``Host`` is overridden to the proxy hostname. Without this,
        #    aiobotocore sets it to ``bedrock-runtime.<region>.amazonaws.com``
        #    and intermediate sidecars (istio, envoy, etc.) 404 because Host
        #    doesn't match the proxy's listener.
        # 2. Every entry in ``default_headers`` is copied onto the request —
        #    this is how ``X-Majordomo-Key`` reaches Steward for auth.
        # 3. ``X-Majordomo-Bedrock-Region`` carries the AWS region so Steward
        #    knows which upstream region to forward to.
        #
        # The hook is only registered when ``base_url`` is set; direct AWS
        # callers must not receive proxy headers.
        if self.base_url:
            self._register_proxy_header_hook()

    def _register_proxy_header_hook(self) -> None:
        """Register the before-send hook that injects Majordomo proxy headers."""
        proxy_host = urlparse(self.base_url or "").netloc
        region = self.region
        proxy_headers = dict(self.default_headers or {})

        def _inject(request: Any, **_kwargs: Any) -> None:
            request.headers["Host"] = proxy_host
            for key, value in proxy_headers.items():
                request.headers[key] = value
            request.headers["X-Majordomo-Bedrock-Region"] = region

        # ``before-send.bedrock-runtime`` fires for all bedrock-runtime
        # operations (Converse, ConverseStream, etc.) — session-level events
        # propagate to every client created from the session.
        self._session.register("before-send.bedrock-runtime", _inject)

    def _client(self) -> Any:
        """Open an async bedrock-runtime client context manager."""
        kwargs: dict[str, Any] = {"region_name": self.region}
        if self.base_url:
            kwargs["endpoint_url"] = self.base_url
        return self._session.create_client("bedrock-runtime", **kwargs)

    def _inference_config(
        self, temperature: float | None, top_p: float | None, max_tokens: int
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {"maxTokens": max_tokens}
        # Converse spells nucleus sampling "topP"; _sampling_params returns the
        # OpenAI-style names, so translate rather than re-implementing the rule.
        params = self._sampling_params(temperature, top_p)
        if "temperature" in params:
            cfg["temperature"] = params["temperature"]
        if "top_p" in params:
            cfg["topP"] = params["top_p"]
        return cfg

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
        """Get a plain text response from Bedrock via Converse."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant"
        if extra_headers:
            logger.debug("extra_headers ignored by Bedrock provider")

        resolved_max_tokens = self._resolve_max_tokens(max_tokens)

        start_time = time.time()
        try:
            async with self._client() as client:
                response = await client.converse(
                    modelId=self.model,
                    messages=_bedrock_user_message(user_prompt),
                    system=_bedrock_system_prompt(system_prompt),
                    inferenceConfig=self._inference_config(
                        temperature, top_p, resolved_max_tokens
                    ),
                )
        except (ClientError, BotoCoreError) as e:
            raise ProviderError(
                f"Bedrock API error: {e}",
                provider="bedrock",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        content = _extract_text_content(response)
        input_tokens, output_tokens, cached_tokens, cache_creation_tokens = _extract_usage(response)
        # Converse spells the stop reason "stopReason" but uses the same
        # "max_tokens" value as the Messages API, so no normalization is needed.
        stop_reason = response.get("stopReason")
        self._check_truncation(stop_reason, resolved_max_tokens, output_tokens, content)
        input_cost, output_cost, total_cost = self._calculate_costs(
            input_tokens, output_tokens, cached_tokens, cache_creation_tokens
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
            deprecation_warning=self.deprecation_warning,
            stop_reason=stop_reason,
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
        """Get a streaming text response from Bedrock via Converse Stream."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant"
        if extra_headers:
            logger.debug("extra_headers ignored by Bedrock provider")

        state = _StreamState()
        resolved_max_tokens = self._resolve_max_tokens(max_tokens, streaming=True)

        async def generator() -> AsyncIterator[str]:
            try:
                async with self._client() as client:
                    response = await client.converse_stream(
                        modelId=self.model,
                        messages=_bedrock_user_message(user_prompt),
                        system=_bedrock_system_prompt(system_prompt),
                        inferenceConfig=self._inference_config(
                            temperature, top_p, resolved_max_tokens
                        ),
                    )
                    async for event in response["stream"]:
                        if "contentBlockDelta" in event:
                            delta = event["contentBlockDelta"].get("delta", {})
                            text = delta.get("text")
                            if text:
                                yield text
                        elif "messageStop" in event:
                            state.stop_reason = event["messageStop"].get("stopReason")
                        elif "metadata" in event:
                            usage = event["metadata"].get("usage", {})
                            state.input_tokens = usage.get("inputTokens", state.input_tokens)
                            state.output_tokens = usage.get("outputTokens", state.output_tokens)
                            state.cached_tokens = usage.get(
                                "cacheReadInputTokens", state.cached_tokens
                            )
                            state.cache_creation_tokens = usage.get(
                                "cacheWriteInputTokens", state.cache_creation_tokens
                            )
            except (ClientError, BotoCoreError) as e:
                raise ProviderError(
                    f"Bedrock API error: {e}",
                    provider="bedrock",
                    original_error=e,
                ) from e
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
        """Bedrock structured output via Converse tool calling.

        Anthropic Claude models are served via the BedrockMantle provider, which
        uses the AWS-native Anthropic Messages API and supports first-class
        structured outputs. This (Converse-based) Bedrock provider is for
        non-Anthropic models — Llama 4, Kimi, Nemotron, DeepSeek-on-Bedrock —
        where tool calling is the only reliable structured-output mechanism.
        """
        if extra_headers:
            logger.debug("extra_headers ignored by Bedrock provider")

        description = schema_description or (
            f"Provide a structured response using the {schema_name} JSON schema"
        )

        tool_instruction = f"Use the {schema_name} tool to provide your answer."
        if system_prompt is None:
            system_prompt = f"You are a helpful assistant. {tool_instruction}"
        else:
            system_prompt = f"{system_prompt}\n\n{tool_instruction}"

        resolved_max_tokens = self._resolve_max_tokens(max_tokens)
        tool_config = _bedrock_tool_config(
            name=schema_name,
            description=description,
            schema=enforce_strict_object_schema(response_schema),
            model_id=self.model,
        )

        start_time = time.time()
        try:
            async with self._client() as client:
                response = await client.converse(
                    modelId=self.model,
                    messages=_bedrock_user_message(user_prompt),
                    system=_bedrock_system_prompt(system_prompt),
                    inferenceConfig=self._inference_config(
                        temperature, top_p, resolved_max_tokens
                    ),
                    toolConfig=tool_config,
                )
        except (ClientError, BotoCoreError) as e:
            raise ProviderError(
                f"Bedrock API error: {e}",
                provider="bedrock",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        input_tokens, output_tokens, cached_tokens, cache_creation_tokens = _extract_usage(response)
        stop_reason = response.get("stopReason")
        # Guard before extraction: a truncated turn has no complete toolUse block,
        # so this would otherwise surface as a missing-tool parse error.
        self._check_truncation(stop_reason, resolved_max_tokens, output_tokens, "")
        content = _extract_tool_use_input(response, schema_name)
        input_cost, output_cost, total_cost = self._calculate_costs(
            input_tokens, output_tokens, cached_tokens, cache_creation_tokens
        )

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
            stop_reason=stop_reason,
        )

    @retry_provider_call
    async def _get_structured_response(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMJSONResponse:
        """Bedrock structured output via Converse forced tool use.

        See ``_get_json_schema_response`` for the rationale on why this provider
        always uses tool calling rather than native Bedrock Structured Outputs.
        """
        if extra_headers:
            logger.debug("extra_headers ignored by Bedrock provider")

        schema = response_model.model_json_schema()
        description = f"Provide a structured response using the {response_model.__name__} format"

        tool_instruction = "Use the structured_response tool to provide your answer."
        if system_prompt is None:
            system_prompt = f"You are a helpful assistant. {tool_instruction}"
        else:
            system_prompt = f"{system_prompt}\n\n{tool_instruction}"

        resolved_max_tokens = self._resolve_max_tokens(max_tokens)
        tool_config = _bedrock_tool_config(
            name="structured_response",
            description=description,
            schema=schema,
            model_id=self.model,
        )

        start_time = time.time()
        try:
            async with self._client() as client:
                response = await client.converse(
                    modelId=self.model,
                    messages=_bedrock_user_message(user_prompt),
                    system=_bedrock_system_prompt(system_prompt),
                    inferenceConfig=self._inference_config(
                        temperature, top_p, resolved_max_tokens
                    ),
                    toolConfig=tool_config,
                )
        except (ClientError, BotoCoreError) as e:
            raise ProviderError(
                f"Bedrock API error: {e}",
                provider="bedrock",
                original_error=e,
            ) from e

        execution_time = time.time() - start_time
        input_tokens, output_tokens, cached_tokens, cache_creation_tokens = _extract_usage(response)
        # Guard before extraction: a truncated turn has no complete toolUse block.
        self._check_truncation(
            response.get("stopReason"), resolved_max_tokens, output_tokens, ""
        )
        content = _extract_tool_use_input(response, "structured_response")
        input_cost, output_cost, total_cost = self._calculate_costs(
            input_tokens, output_tokens, cached_tokens, cache_creation_tokens
        )

        return LLMJSONResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=execution_time,
        )


def _bedrock_user_message(user_prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"text": user_prompt}]}]


def _bedrock_system_prompt(system_prompt: str) -> list[dict[str, Any]]:
    return [{"text": system_prompt}]


# Bedrock model substrings that reject ``toolConfig.toolChoice.tool`` in the
# Converse API. The tool itself is still exposed via ``tools``; the model is
# expected to invoke it because the system prompt instructs it to. Add a new
# substring whenever a model surfaces "toolChoice.tool field" validation errors.
_BEDROCK_MODELS_WITHOUT_FORCED_TOOL_CHOICE = frozenset(
    {
        "llama4",
    }
)

def _supports_forced_tool_choice(model_id: str) -> bool:
    return not any(token in model_id for token in _BEDROCK_MODELS_WITHOUT_FORCED_TOOL_CHOICE)


def _bedrock_tool_config(
    name: str, description: str, schema: dict[str, Any], *, model_id: str
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "tools": [
            {
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"json": schema},
                }
            }
        ],
    }
    if _supports_forced_tool_choice(model_id):
        config["toolChoice"] = {"tool": {"name": name}}
    return config


def _extract_text_content(response: dict[str, Any]) -> str:
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    parts = [block["text"] for block in blocks if "text" in block]
    return "\n".join(parts)


def _extract_tool_use_input(response: dict[str, Any], tool_name: str) -> Any:
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    for block in blocks:
        tool_use = block.get("toolUse")
        if tool_use and tool_use.get("name") == tool_name:
            return tool_use.get("input")
    raise ResponseParsingError(
        f"No {tool_name} tool use found in Bedrock response",
        raw_content=str(blocks),
    )


def _extract_usage(response: dict[str, Any]) -> tuple[int, int, int, int]:
    usage = response.get("usage", {}) or {}
    return (
        int(usage.get("inputTokens", 0)),
        int(usage.get("outputTokens", 0)),
        int(usage.get("cacheReadInputTokens", 0) or 0),
        int(usage.get("cacheWriteInputTokens", 0) or 0),
    )
