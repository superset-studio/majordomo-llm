"""Base classes and types for the majordomo-llm library."""

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate as validate_json_schema
from pydantic import BaseModel

from majordomo_llm.exceptions import (
    ConfigurationError,
    EmptyStructuredResponseError,
    InputModalityUnsupported,
    ResponseParsingError,
    ResponseTruncatedError,
    StructuredOutputUnsupported,
)
from majordomo_llm.retry import retry_provider_call

if TYPE_CHECKING:
    from majordomo_llm.hooks.pipeline import HookPipeline

#: Output cap applied when neither the caller nor ``llm_config.yaml`` sets one,
#: for providers that require ``max_tokens`` on every request (Anthropic,
#: Bedrock). Sized to keep a non-streaming response inside the provider SDKs'
#: default HTTP timeout — larger ceilings need the streaming path.
DEFAULT_MAX_TOKENS = 16_000

#: Output cap for streaming requests, where the SDK read timeout does not
#: apply between chunks, so the model gets substantially more room.
DEFAULT_STREAM_MAX_TOKENS = 64_000

SUPPORTED_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


@dataclass(frozen=True)
class ImageInput:
    """Validated in-memory image supplied to a multimodal LLM request.

    Image bytes are deliberately kept in memory and are never fetched from a
    URL by the library. This gives every provider the same input contract and
    avoids surprising network access or provider-specific URL semantics.
    """

    data: bytes
    media_type: str

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("ImageInput.data must not be empty")
        if self.media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_MEDIA_TYPES))
            raise ValueError(
                f"Unsupported image media type '{self.media_type}'. Supported: {supported}"
            )


def _hash_api_key(api_key: str) -> str:
    """Compute a truncated SHA256 hash of an API key.

    Returns the first 16 characters of the hex digest, which is enough
    to identify keys without being reversible.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def resolve_api_key(api_key: str | None, env_var: str, provider_name: str) -> str:
    """Resolve an API key from parameter or environment variable.

    Args:
        api_key: Optional API key passed directly.
        env_var: Environment variable name to check if api_key is None.
        provider_name: Provider name for error message (e.g., "OpenAI", "Anthropic").

    Returns:
        The resolved API key.

    Raises:
        ConfigurationError: If no API key is found.
    """
    resolved = api_key or os.environ.get(env_var)
    if not resolved:
        raise ConfigurationError(
            f"{provider_name} API key not found. Set the {env_var} environment "
            "variable or pass api_key to the constructor."
        )
    return resolved


def inline_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline all $ref references in a JSON schema, removing $defs.

    This flattens nested model definitions so the schema is self-contained
    without JSON Schema $ref pointers, which some LLMs handle poorly.

    Args:
        schema: The JSON schema dict (from Pydantic's model_json_schema()).

    Returns:
        A new schema dict with all $ref replaced by their definitions.
    """
    import copy

    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve_refs(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                # Pydantic emits enum/model refs as either a bare {"$ref": ...}
                # or as {"$ref": ..., "description": ...} when the property has
                # its own metadata. Inline the ref in both cases and merge any
                # sibling keys on top so user-provided descriptions win.
                ref_path = obj["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path[len("#/$defs/") :]
                    if def_name in defs:
                        inlined = resolve_refs(copy.deepcopy(defs[def_name]))
                        siblings = {k: resolve_refs(v) for k, v in obj.items() if k != "$ref"}
                        if isinstance(inlined, dict):
                            return {**inlined, **siblings}
                        return inlined
                return obj
            return {k: resolve_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        return obj

    return cast(dict[str, Any], resolve_refs(schema))


# JSON Schema constraints that grammar-enforced structured-output backends
# (Cohere, Bedrock Structured Outputs) reject. The set is empirically derived
# from the providers' validation errors — both report "constraint not supported
# for type X" for the same keywords. Add a key whenever a new provider surfaces
# the same class of rejection.
_UNSUPPORTED_SCHEMA_CONSTRAINTS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "pattern",
        "format",
    }
)


def strip_unsupported_schema_constraints(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove schema keywords that strict-grammar backends reject.

    Used by Cohere and Bedrock Structured Outputs, whose schema compilers do not
    support the full JSON Schema vocabulary. Removed keys include numeric bounds
    (``minimum``/``maximum``/``multipleOf``), array bounds (``minItems``/
    ``maxItems``/``uniqueItems``), and string constraints (``minLength``/
    ``maxLength``/``pattern``/``format``).
    """

    def strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k not in _UNSUPPORTED_SCHEMA_CONSTRAINTS}
        if isinstance(obj, list):
            return [strip(item) for item in obj]
        return obj

    return cast(dict[str, Any], strip(schema))


def enforce_strict_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON schema for strict structured-output modes.

    Both OpenAI's ``response_format`` strict mode and Bedrock Structured Outputs
    require every object node to declare ``additionalProperties: false`` and to
    list every defined property in ``required``. Pydantic's
    ``model_json_schema()`` emits neither, so structured-output calls with raw
    Pydantic schemas are rejected by both providers.

    Inlines ``$ref``/``$defs`` first so the walker does not need to resolve
    references itself, then mutates a deep copy of the schema in place.
    """
    schema = inline_schema_refs(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return schema


def build_schema_prompt(schema: dict[str, Any], system_prompt: str | None = None) -> str:
    """Build a system prompt that includes a JSON schema instruction.

    Args:
        schema: The JSON schema dict (from Pydantic's model_json_schema()).
        system_prompt: Optional existing system prompt to prepend.

    Returns:
        Combined system prompt with schema instructions.
    """
    schema_instruction = f"""You must respond with valid JSON that matches this exact schema:
{json.dumps(schema, indent=2)}

Important: Return only the JSON object, no additional text or markdown formatting."""

    if system_prompt:
        return f"{system_prompt}\n\n{schema_instruction}"
    return schema_instruction


def canonical_json_dumps(content: Any) -> str:
    """Serialize JSON content in canonical byte-comparable form."""
    return json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _strip_markdown_fences(content: str) -> str:
    """Strip markdown code fences from provider output."""
    stripped = content.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _first_balanced_json_value(content: str) -> str | None:
    """Return the first balanced JSON object or array from text."""
    start_index = None
    opening_char = ""
    closing_char = ""

    for index, char in enumerate(content):
        if char == "{":
            start_index = index
            opening_char = "{"
            closing_char = "}"
            break
        if char == "[":
            start_index = index
            opening_char = "["
            closing_char = "]"
            break

    if start_index is None:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start_index, len(content)):
        char = content[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == opening_char:
            depth += 1
        elif char == closing_char:
            depth -= 1
            if depth == 0:
                return content[start_index : index + 1]

    return None


def _json_parse_candidates(raw_content: str) -> list[tuple[str, str]]:
    """Build one normal parse candidate and repair candidates for provider output."""
    candidates = [("raw", raw_content.strip())]

    fenced = _strip_markdown_fences(raw_content)
    if fenced != candidates[0][1]:
        candidates.append(("markdown-fence-stripped", fenced))

    balanced = _first_balanced_json_value(raw_content)
    if balanced is not None and all(balanced != candidate for _, candidate in candidates):
        candidates.append(("first-balanced-json", balanced))

    return candidates


def is_empty_structured_result(content: Any) -> bool:
    """Return True if a parsed structured result carries no information.

    True for an empty object ``{}`` and for an object whose every top-level
    value is ``null`` — both are the signature of a model that invoked the
    structured-output tool without producing an answer. Non-dict content
    (arrays, scalars) is never treated as empty here.
    """
    if not isinstance(content, dict):
        return False
    if not content:
        return True
    return all(value is None for value in content.values())


def canonicalize_json_schema_output(
    content: Any, response_schema: dict[str, Any], *, reject_empty: bool = True
) -> str:
    """Validate provider output against a JSON schema and serialize canonically.

    String outputs are parsed directly first, then repaired once by stripping markdown
    fences or extracting the first balanced JSON object/array. The original raw output
    is included in parsing errors for debugging.

    When ``reject_empty`` is true (the default for structured outputs), a
    schema-valid but empty result — ``{}`` or an all-null object — raises
    :class:`EmptyStructuredResponseError` instead of being returned as a
    successful response. This distinguishes a real answer from a model that
    invoked the tool without populating it. See :func:`is_empty_structured_result`.
    """
    if not isinstance(content, str):
        try:
            validate_json_schema(instance=content, schema=response_schema)
        except JSONSchemaValidationError as e:
            raise ResponseParsingError(
                f"JSON response did not validate against schema: {e.message}",
                raw_content=canonical_json_dumps(content),
            ) from e
        except (TypeError, ValueError) as e:
            raise ResponseParsingError(
                f"Failed to serialize JSON response: {e}",
                raw_content=str(content),
            ) from e
        if reject_empty and is_empty_structured_result(content):
            raise EmptyStructuredResponseError(
                "Structured response validated against the schema but was empty "
                "(no fields populated).",
                raw_content=canonical_json_dumps(content),
            )
        return canonical_json_dumps(content)

    raw_content = content
    last_error: Exception | None = None

    for _, candidate in _json_parse_candidates(raw_content):
        try:
            parsed_content = json.loads(candidate)
            validate_json_schema(instance=parsed_content, schema=response_schema)
        except (json.JSONDecodeError, JSONSchemaValidationError) as e:
            last_error = e
            continue
        if reject_empty and is_empty_structured_result(parsed_content):
            raise EmptyStructuredResponseError(
                "Structured response validated against the schema but was empty "
                "(no fields populated).",
                raw_content=raw_content,
            )
        return canonical_json_dumps(parsed_content)

    if isinstance(last_error, JSONSchemaValidationError):
        raise ResponseParsingError(
            f"JSON response did not validate against schema: {last_error.message}",
            raw_content=raw_content,
        ) from last_error

    raise ResponseParsingError(
        f"Failed to parse JSON schema response: {last_error}",
        raw_content=raw_content,
    ) from last_error


def ensure_no_unexpected_kwargs(kwargs: dict[str, Any]) -> None:
    """Reject unknown per-call keyword arguments consistently."""
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")


#: Type variable for Pydantic model types used in structured responses.
T = TypeVar("T", bound=BaseModel)

#: Number of tokens per million (used for cost calculations).
logger = logging.getLogger(__name__)

TOKENS_PER_MILLION = 1_000_000


def compute_costs(
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
    *,
    input_cost: float,
    output_cost: float,
    cached_input_cost: float | None = None,
    cache_write_cost: float | None = None,
    cache_accounting: str = "subset",
) -> tuple[float, float, float]:
    """Price a request from explicit per-million rates.

    This is the stateless core of :meth:`LLM._calculate_costs`, extracted so a
    request can be priced against rates that are not the calling instance's own
    — e.g. the Majordomo provider prices a call using the rates of whichever
    backend the gateway actually routed to. See :meth:`LLM._calculate_costs`
    for the meaning of ``cache_accounting`` ("subset" vs "additive").

    Args:
        input_tokens: Number of input tokens (provider-reported).
        output_tokens: Number of output tokens.
        cached_tokens: Number of cache-read prompt tokens.
        cache_creation_tokens: Number of cache-write prompt tokens.
        input_cost: Cost per million input tokens in USD.
        output_cost: Cost per million output tokens in USD.
        cached_input_cost: Cost per million cache-read tokens in USD, or ``None``.
        cache_write_cost: Cost per million cache-write tokens in USD, or ``None``.
        cache_accounting: ``"subset"`` or ``"additive"`` (see
            :meth:`LLM._calculate_costs`).

    Returns:
        Tuple of (input_cost, output_cost, total_cost) in USD.
    """
    if cache_accounting == "additive":
        read_rate = cached_input_cost if cached_input_cost is not None else 0.0
        write_rate = cache_write_cost if cache_write_cost is not None else 0.0
        in_cost = (
            input_tokens * input_cost
            + cached_tokens * read_rate
            + cache_creation_tokens * write_rate
        ) / TOKENS_PER_MILLION
    else:
        cached_rate = cached_input_cost if cached_input_cost is not None else input_cost
        uncached_tokens = max(input_tokens - cached_tokens, 0)
        in_cost = (uncached_tokens * input_cost + cached_tokens * cached_rate) / TOKENS_PER_MILLION
    out_cost = (output_tokens * output_cost) / TOKENS_PER_MILLION
    return in_cost, out_cost, in_cost + out_cost


@dataclass
class Usage:
    """Token usage and cost metrics for an LLM request.

    Attributes:
        input_tokens: Number of tokens in the input/prompt.
        output_tokens: Number of tokens in the response.
        cached_tokens: Number of prompt tokens served from cache (cache reads,
            provider-specific). Its relationship to ``input_tokens`` differs by
            provider: for OpenAI-family providers cached tokens are a subset of
            ``input_tokens``; for Anthropic/Bedrock they are reported separately
            and excluded from ``input_tokens``.
        cache_creation_tokens: Number of prompt tokens written to the cache
            (cache-creation/cache-write). Only populated by providers that bill a
            distinct cache-write rate (Anthropic, Bedrock); ``0`` elsewhere.
        input_cost: Prompt-side cost in USD (uncached input plus any cache
            read/write cost) after cache pricing is applied.
        output_cost: Cost for output tokens in USD.
        total_cost: Total cost (input + output + tool use) in USD.
        response_time: Time taken for the request in seconds.
        tool_use_cost: Cost for provider-side tool calls (e.g. web search) in USD.
    """

    input_tokens: int
    output_tokens: int
    cached_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    response_time: float
    cache_creation_tokens: int = field(default=0, kw_only=True)
    tool_use_cost: float = field(default=0.0, kw_only=True)


@dataclass
class LLMResponse(Usage):
    """Response from an LLM containing plain text content.

    Inherits all usage metrics from :class:`Usage`.

    Attributes:
        content: The text content of the LLM response.
        deprecation_warning: Warning if a deprecated model was auto-replaced.
        routed_provider: When routed through the Majordomo gateway's optimal
            router, the concrete backend the gateway selected (e.g. "fireworks");
            ``None`` for direct provider calls.
        routed_model: When routed through the Majordomo gateway, the backend's
            native model identifier the call actually ran on; ``None`` otherwise.
        stop_reason: Why the provider stopped generating, verbatim from the
            provider (e.g. ``"end_turn"``, ``"tool_use"``, ``"max_tokens"``).
            ``None`` for providers that do not report one.
    """

    content: str
    deprecation_warning: str | None = None
    routed_provider: str | None = None
    routed_model: str | None = None
    stop_reason: str | None = None


@dataclass
class LLMJSONResponse(Usage):
    """Response from an LLM containing parsed JSON content.

    Inherits all usage metrics from :class:`Usage`.

    Attributes:
        content: The parsed JSON content as a Python dict.
    """

    content: dict[str, Any]


@dataclass
class LLMStructuredResponse(Usage):
    """Response from an LLM containing a validated Pydantic model.

    Inherits all usage metrics from :class:`Usage`.

    Attributes:
        content: The validated Pydantic model instance.
    """

    content: BaseModel


@dataclass
class _StreamState:
    """Internal mutable state populated by provider stream generators.

    Provider async generators update these fields as streaming events arrive.
    After the stream completes, ``LLMStreamResponse._finalize()`` uses this
    data to compute final :class:`Usage` metrics.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    #: Optional pricing override applied at finalization instead of the owning
    #: LLM's own rates. Set by providers (e.g. Majordomo) that only learn the
    #: authoritative per-million rates mid-stream — from the routed backend the
    #: gateway selected. Called as ``price_override(input_tokens, output_tokens,
    #: cached_tokens, cache_creation_tokens) -> (input_cost, output_cost,
    #: total_cost)``. ``None`` uses ``LLM._calculate_costs`` (default behaviour).
    price_override: Callable[[int, int, int, int], tuple[float, float, float]] | None = None
    #: Concrete backend a gateway routed this call to, learned mid-stream from
    #: response headers. Mirrors LLMResponse.routed_provider/routed_model so the
    #: streaming path reports the same identity as the non-streaming one.
    routed_provider: str | None = None
    routed_model: str | None = None
    #: Why the provider stopped generating, captured from the terminal stream
    #: event. Mirrors ``LLMResponse.stop_reason`` so the streaming path reports
    #: the same signal as the non-streaming one.
    stop_reason: str | None = None


class LLMStreamResponse:
    """Async-iterable wrapper around a streaming LLM response.

    Yields text chunks as they arrive. After iteration completes, usage
    and cost data is available via the :attr:`usage` property.

    Example:
        >>> stream = await llm.get_response_stream("Hello")
        >>> async for chunk in stream:
        ...     print(chunk, end="")
        >>> print(stream.usage.total_cost)
    """

    def __init__(
        self,
        stream: AsyncIterator[str],
        state: _StreamState,
        llm: "LLM",
    ) -> None:
        self._stream = stream
        self._state = state
        self._llm = llm
        self._chunks: list[str] = []
        self._consumed = False
        self._usage: Usage | None = None
        self._on_complete: Callable[[Usage, str], None] | None = None
        self._on_error: Callable[[Exception], None] | None = None

    def __aiter__(self) -> "LLMStreamResponse":
        return self

    async def __anext__(self) -> str:
        try:
            chunk = await self._stream.__anext__()
            self._chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as e:
            if self._on_error:
                self._on_error(e)
            raise

    def _finalize(self) -> None:
        if self._consumed:
            return
        self._consumed = True
        response_time = time.time() - self._state.start_time
        if self._state.price_override is not None:
            input_cost, output_cost, total_cost = self._state.price_override(
                self._state.input_tokens,
                self._state.output_tokens,
                self._state.cached_tokens,
                self._state.cache_creation_tokens,
            )
        else:
            input_cost, output_cost, total_cost = self._llm._calculate_costs(
                self._state.input_tokens,
                self._state.output_tokens,
                self._state.cached_tokens,
                self._state.cache_creation_tokens,
            )
        self._usage = Usage(
            input_tokens=self._state.input_tokens,
            output_tokens=self._state.output_tokens,
            cached_tokens=self._state.cached_tokens,
            cache_creation_tokens=self._state.cache_creation_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=response_time,
        )
        if self._on_complete:
            self._on_complete(self._usage, "".join(self._chunks))

    @property
    def usage(self) -> Usage | None:
        """Usage metrics, available after the stream is fully consumed."""
        return self._usage

    @property
    def routed_provider(self) -> str | None:
        """Backend a gateway routed to, or None on a direct provider call."""
        return self._state.routed_provider

    @property
    def routed_model(self) -> str | None:
        """Routed backend's native model id, or None on a direct provider call."""
        return self._state.routed_model

    @property
    def stop_reason(self) -> str | None:
        """Why generation stopped, available once the stream is consumed."""
        return self._state.stop_reason

    async def collect(self) -> LLMResponse:
        """Consume the entire stream and return an :class:`LLMResponse`."""
        chunks: list[str] = []
        async for chunk in self:
            chunks.append(chunk)
        assert self._usage is not None
        return LLMResponse(
            content="".join(self._chunks),
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
            cached_tokens=self._usage.cached_tokens,
            cache_creation_tokens=self._usage.cache_creation_tokens,
            input_cost=self._usage.input_cost,
            output_cost=self._usage.output_cost,
            total_cost=self._usage.total_cost,
            response_time=self._usage.response_time,
            deprecation_warning=self._llm.deprecation_warning,
            routed_provider=self._state.routed_provider,
            routed_model=self._state.routed_model,
            stop_reason=self._state.stop_reason,
        )


class LLM(ABC):
    """Abstract base class for LLM provider implementations.

    Provides a unified interface for interacting with different LLM providers
    (OpenAI, Anthropic, Gemini) with automatic retry logic and cost tracking.

    Subclasses must implement the :meth:`get_response` method. Other methods
    have default implementations that can be overridden for provider-specific
    optimizations.

    Attributes:
        provider: The LLM provider name (e.g., "openai", "anthropic", "gemini").
        model: The specific model identifier (e.g., "gpt-4o", "claude-sonnet-5").
        input_cost: Cost per million input tokens in USD.
        output_cost: Cost per million output tokens in USD.
        supports_temperature_top_p: Whether the model supports temperature/top_p params.
        use_web_search: Whether to enable web search (Anthropic only).
        api_key_hash: Truncated SHA256 hash of the API key (for logging).
        api_key_alias: Optional human-readable name for the API key.

    Example:
        >>> from majordomo_llm import get_llm_instance
        >>> llm = get_llm_instance("anthropic", "claude-sonnet-5")
        >>> response = await llm.get_response("What is 2+2?")
        >>> print(response.content)
        4
        >>> print(f"Cost: ${response.total_cost:.6f}")
    """

    #: How the provider accounts for cached prompt tokens, which determines the
    #: cache cost formula in :meth:`_calculate_costs`:
    #:
    #: - ``"subset"`` (default): ``cached_tokens`` are already counted in
    #:   ``input_tokens`` (OpenAI, Gemini, DeepSeek, Fireworks, Together). Cost
    #:   re-prices those tokens down from ``input_cost`` to ``cached_input_cost``.
    #: - ``"additive"``: ``cached_tokens`` / ``cache_creation_tokens`` are
    #:   reported separately and excluded from ``input_tokens`` (Anthropic,
    #:   Bedrock). Cost adds cache read/write on top of the uncached input.
    #:
    #: Providers whose accounting is "additive" override this class attribute.
    _cache_accounting: str = "subset"

    def __init__(
        self,
        provider: str,
        model: str,
        input_cost: float,
        output_cost: float,
        supports_temperature_top_p: bool = True,
        use_web_search: bool = False,
        api_key: str | None = None,
        api_key_alias: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        hook_pipeline: "HookPipeline | None" = None,
        cached_input_cost: float | None = None,
        cache_write_cost: float | None = None,
        use_prompt_caching: bool = True,
        max_tokens: int | None = None,
        supports_image_input: bool = False,
    ) -> None:
        """Initialize the LLM instance.

        Args:
            provider: The LLM provider name.
            model: The model identifier.
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
            use_web_search: Enable web search capability (Anthropic only).
            api_key: The API key (used to compute hash for logging).
            api_key_alias: Optional human-readable name for the API key.
            base_url: Optional custom base URL for routing through a proxy.
            default_headers: Optional headers sent with every request.
            hook_pipeline: Optional :class:`HookPipeline` that wraps every
                text-producing call. ``get_response_stream`` does not run
                hooks; streaming-chunk interception is deferred.
            cached_input_cost: Cost per million cache-read tokens in USD. When
                ``None``, no cache-read discount is applied (see
                :meth:`_calculate_costs`).
            cache_write_cost: Cost per million cache-creation tokens in USD, for
                providers with a distinct cache-write rate (Anthropic, Bedrock).
                When ``None``, cache writes are not billed.
            use_prompt_caching: Whether to request prompt caching on providers
                that support explicit cache breakpoints (Anthropic). Defaults to
                ``True``. Ignored by providers without explicit cache control.
            max_tokens: Default output cap for this model, from the ``max_tokens``
                key in ``llm_config.yaml``. ``None`` falls back to
                :data:`DEFAULT_MAX_TOKENS` / :data:`DEFAULT_STREAM_MAX_TOKENS`.
                Only providers whose API requires an output cap send it.
            supports_image_input: Whether the model accepts image inputs.
        """
        self.provider = provider
        self.model = model
        self.input_cost = input_cost
        self.output_cost = output_cost
        self.cached_input_cost = cached_input_cost
        self.cache_write_cost = cache_write_cost
        self.use_prompt_caching = use_prompt_caching
        self.max_tokens = max_tokens
        self.supports_image_input = supports_image_input
        self.supports_temperature_top_p = supports_temperature_top_p
        self.use_web_search = use_web_search
        self.api_key_hash = _hash_api_key(api_key) if api_key else None
        self.api_key_alias = api_key_alias
        self.base_url = base_url
        self.default_headers = default_headers
        self.hook_pipeline = hook_pipeline
        self.deprecation_warning: str | None = None
        self.requested_model: str | None = None

    def _validate_images(self, images: tuple[ImageInput, ...]) -> None:
        """Reject image inputs before a provider call when unsupported."""
        if images and not self.supports_image_input:
            raise InputModalityUnsupported(self.provider, self.model, "image")

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
        """Provider implementation for image understanding."""
        raise InputModalityUnsupported(self.provider, self.model, "image")

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
        """Provider streaming implementation for image understanding."""
        raise InputModalityUnsupported(self.provider, self.model, "image")

    def get_full_model_name(self) -> str:
        """Get the fully qualified model name.

        Returns:
            Model name in the format "provider:model" (e.g., "anthropic:claude-sonnet-5").
        """
        return f"{self.provider}:{self.model}"

    def _resolve_max_tokens(self, override: int | None, *, streaming: bool = False) -> int:
        """Resolve the output cap to send on a request.

        Precedence: an explicit per-request ``override``, then this model's
        ``max_tokens`` from ``llm_config.yaml``, then the library default. The
        streaming default is larger because the SDK read timeout does not apply
        between chunks.

        This is the single place the number is decided; providers call it rather
        than choosing a value of their own.

        Args:
            override: Per-request ``max_tokens``, or None if the caller passed none.
            streaming: Whether the request uses the streaming path.

        Returns:
            The output cap to send.

        Raises:
            ValueError: If ``override`` is not a positive integer.
        """
        if override is not None:
            if override < 1:
                raise ValueError(f"max_tokens must be a positive integer, got {override}")
            return override
        if self.max_tokens is not None:
            return self.max_tokens
        return DEFAULT_STREAM_MAX_TOKENS if streaming else DEFAULT_MAX_TOKENS

    def _check_truncation(
        self, stop_reason: str | None, max_tokens: int, output_tokens: int, content: str
    ) -> None:
        """Raise when the provider reports the response was cut off by the cap.

        Args:
            stop_reason: The provider's stop reason, verbatim.
            max_tokens: The cap that was sent on the request.
            output_tokens: Tokens the model emitted.
            content: Whatever content arrived before the cut.

        Raises:
            ResponseTruncatedError: If ``stop_reason`` indicates truncation.
        """
        if stop_reason != "max_tokens":
            return
        raise ResponseTruncatedError(
            max_tokens=max_tokens,
            output_tokens=output_tokens,
            partial_content=content,
            provider=self.provider,
        )

    def _sampling_params(self, temperature: float | None, top_p: float | None) -> dict[str, Any]:
        """Resolve which sampling parameters to send on a request.

        The library does not impose a sampling policy: a parameter is sent only
        when the caller asked for it. Omitting them lets each provider apply its
        own documented default rather than a value this library invented.

        A model whose deployment rejects these parameters
        (``supports_temperature_top_p`` is False) never receives them. That
        covers every current OpenAI and Anthropic flagship, plus deployments
        that pin their sampling values — Moonshot's Kimi SKUs require
        ``temperature=1`` / ``top_p=0.95``, and sending anything else is a 400.
        Caller-supplied values are silently dropped in that case rather than
        failing the call: an LLMCascade or alias chain can legitimately mix
        members that do and do not accept sampling parameters.

        Args:
            temperature: Caller-supplied temperature, or None if unset.
            top_p: Caller-supplied nucleus sampling value, or None if unset.

        Returns:
            A kwargs dict holding only the parameters that should be sent,
            empty when nothing should be. Typed ``dict[str, Any]`` rather than
            ``dict[str, float]`` because ``**`` unpacking is checked against
            every parameter of the target signature, not just the keys present,
            so a narrower value type fails against non-float parameters.
        """
        if not self.supports_temperature_top_p:
            return {}

        params: dict[str, Any] = {}
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        return params

    def _calculate_costs(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> tuple[float, float, float]:
        """Calculate costs for a request, accounting for prompt caching.

        The returned ``input_cost`` is the full prompt-side cost: uncached input
        plus any cache read/write cost. How cached tokens fold in depends on the
        provider's :attr:`_cache_accounting` mode:

        - ``"subset"``: ``cached_tokens`` are part of ``input_tokens`` already.
          They are re-priced from ``input_cost`` down to ``cached_input_cost``
          (falling back to ``input_cost`` — i.e. no discount — when unset).
        - ``"additive"``: ``cached_tokens`` (reads) and ``cache_creation_tokens``
          (writes) are separate from ``input_tokens`` and are added on top at
          ``cached_input_cost`` / ``cache_write_cost`` (each contributing ``0``
          when its rate is unset, matching prior un-modelled behaviour).

        Args:
            input_tokens: Number of input tokens (provider-reported).
            output_tokens: Number of output tokens.
            cached_tokens: Number of cache-read prompt tokens.
            cache_creation_tokens: Number of cache-write prompt tokens.

        Returns:
            Tuple of (input_cost, output_cost, total_cost) in USD.
        """
        return compute_costs(
            input_tokens,
            output_tokens,
            cached_tokens,
            cache_creation_tokens,
            input_cost=self.input_cost,
            output_cost=self.output_cost,
            cached_input_cost=self.cached_input_cost,
            cache_write_cost=self.cache_write_cost,
            cache_accounting=self._cache_accounting,
        )

    @abstractmethod
    async def _get_response_impl(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Provider-specific implementation of ``get_response``.

        Providers apply ``@retry_provider_call`` here. The public
        :meth:`get_response` wraps this with the optional hook pipeline.
        """
        raise NotImplementedError()

    @abstractmethod
    async def _get_response_stream_impl(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMStreamResponse:
        """Provider-specific implementation of ``get_response_stream``."""
        raise NotImplementedError()

    async def get_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        *,
        max_tokens: int | None = None,
        caller_metadata: dict[str, Any] | None = None,
        images: tuple[ImageInput, ...] = (),
    ) -> LLMResponse:
        """Get a plain text response from the LLM.

        Runs the optional :attr:`hook_pipeline` around the provider call.
        Hooks see the prompt before the call and the response text after.

        Args:
            user_prompt: The user's input prompt.
            system_prompt: Optional system prompt to set context/behavior.
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            top_p: Nucleus sampling parameter (0.0-1.0).
            extra_headers: Optional per-request headers merged with default_headers.
            max_tokens: Optional output cap for this request, overriding the
                model's ``max_tokens`` config value. Only providers whose API
                requires an output cap (Anthropic, Bedrock) act on it.
            caller_metadata: Free-form dict forwarded to every hook via
                :class:`HookContext`. Unused when no pipeline is configured.
            images: Validated in-memory images to analyze. Supported providers
                translate these into native multimodal content blocks.

        Returns:
            LLMResponse containing the text content and usage metrics.

        Raises:
            HookBlocked: If a hook in the pipeline blocks the call.
            ResponseTruncatedError: If the response hit the output cap.
            Exception: If the API request fails after retries.
        """

        async def impl(prompt: str) -> LLMResponse:
            self._validate_images(images)
            if images:
                return await self._get_response_with_images_impl(
                    prompt,
                    images,
                    system_prompt,
                    temperature,
                    top_p,
                    extra_headers=extra_headers,
                    max_tokens=max_tokens,
                )
            return await self._get_response_impl(
                prompt,
                system_prompt,
                temperature,
                top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
            )

        return await self._run_hooks_returning_response(user_prompt, caller_metadata, impl)

    async def get_response_stream(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        *,
        max_tokens: int | None = None,
        caller_metadata: dict[str, Any] | None = None,
        images: tuple[ImageInput, ...] = (),
    ) -> LLMStreamResponse:
        """Get a streaming text response from the LLM.

        Hooks do not run on streaming responses; ``caller_metadata`` is
        accepted for API symmetry and ignored.

        ``max_tokens`` overrides the model's configured output cap for this
        request; streaming defaults to :data:`DEFAULT_STREAM_MAX_TOKENS`.
        ``images`` supplies in-memory image inputs on vision-capable models.
        """
        del caller_metadata
        self._validate_images(images)
        if images:
            return await self._get_response_stream_with_images_impl(
                user_prompt,
                images,
                system_prompt,
                temperature,
                top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
            )
        return await self._get_response_stream_impl(
            user_prompt,
            system_prompt,
            temperature,
            top_p,
            extra_headers=extra_headers,
            max_tokens=max_tokens,
        )

    async def _run_hooks_returning_response(
        self,
        prompt: str,
        caller_metadata: dict[str, Any] | None,
        impl: Callable[[str], Awaitable[LLMResponse]],
    ) -> LLMResponse:
        """Run the configured hook pipeline around an LLMResponse-returning call.

        Hooks operate on text. We capture the underlying ``LLMResponse`` so
        usage metrics survive even when the pipeline rewrites the content.
        """
        if self.hook_pipeline is None:
            return await impl(prompt)

        captured: LLMResponse | None = None

        async def call(modified_prompt: str) -> str:
            nonlocal captured
            captured = await impl(modified_prompt)
            return captured.content

        final_text = await self.hook_pipeline.run(prompt, call, caller_metadata=caller_metadata)
        assert captured is not None
        if final_text == captured.content:
            return captured
        return LLMResponse(
            content=final_text,
            input_tokens=captured.input_tokens,
            output_tokens=captured.output_tokens,
            cached_tokens=captured.cached_tokens,
            cache_creation_tokens=captured.cache_creation_tokens,
            input_cost=captured.input_cost,
            output_cost=captured.output_cost,
            total_cost=captured.total_cost,
            response_time=captured.response_time,
            deprecation_warning=captured.deprecation_warning,
            routed_provider=captured.routed_provider,
            routed_model=captured.routed_model,
            stop_reason=captured.stop_reason,
        )

    async def get_json_response(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        *,
        max_tokens: int | None = None,
        caller_metadata: dict[str, Any] | None = None,
        images: tuple[ImageInput, ...] = (),
    ) -> LLMJSONResponse:
        """Get a JSON response from the LLM.

        Automatically parses the LLM's text response as JSON.

        Args:
            user_prompt: The user's input prompt.
            system_prompt: Optional system prompt to set context/behavior.
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            top_p: Nucleus sampling parameter (0.0-1.0).
            extra_headers: Optional per-request headers merged with default_headers.
            images: Validated in-memory images to analyze.

        Returns:
            LLMJSONResponse containing the parsed JSON dict and usage metrics.

        Raises:
            HookBlocked: If a hook in the pipeline blocks the call.
            ResponseParsingError: If the response cannot be parsed as JSON.
            Exception: If the API request fails after retries.
        """
        response = await self.get_response(
            user_prompt,
            system_prompt,
            temperature,
            top_p,
            extra_headers=extra_headers,
            max_tokens=max_tokens,
            caller_metadata=caller_metadata,
            images=images,
        )
        # Strip markdown code fencing if present
        content = response.content.replace("```json", "").replace("```", "").strip()
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError as e:
            raise ResponseParsingError(
                f"Failed to parse JSON response: {e}",
                raw_content=response.content,
            ) from e
        return LLMJSONResponse(
            content=parsed_content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cached_tokens,
            cache_creation_tokens=response.cache_creation_tokens,
            input_cost=response.input_cost,
            output_cost=response.output_cost,
            total_cost=response.total_cost,
            response_time=response.response_time,
        )

    async def get_structured_json_response(
        self,
        response_model: type[T],
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        *,
        max_tokens: int | None = None,
        caller_metadata: dict[str, Any] | None = None,
        images: tuple[ImageInput, ...] = (),
    ) -> LLMStructuredResponse:
        """Get a structured response validated against a Pydantic model.

        Uses provider-specific mechanisms (tool calling, response schemas) to
        ensure the response conforms to the specified Pydantic model schema.

        Args:
            response_model: Pydantic model class defining the expected structure.
            user_prompt: The user's input prompt.
            system_prompt: Optional system prompt to set context/behavior.
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            top_p: Nucleus sampling parameter (0.0-1.0).
            images: Validated in-memory images to analyze.

        Returns:
            LLMStructuredResponse containing the validated Pydantic model instance.

        Raises:
            pydantic.ValidationError: If the response doesn't match the model schema.
            Exception: If the API request fails after retries.

        Example:
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> response = await llm.get_structured_json_response(
            ...     response_model=Person,
            ...     user_prompt="Extract: John is 30 years old",
            ... )
            >>> print(response.content.name)
            John
        """
        response = await self.get_json_schema_response(
            user_prompt=user_prompt,
            response_schema=response_model.model_json_schema(),
            system_prompt=system_prompt,
            schema_name=response_model.__name__,
            schema_description=(
                f"Provide a structured response using the {response_model.__name__} schema"
            ),
            temperature=temperature,
            top_p=top_p,
            extra_headers=extra_headers,
            max_tokens=max_tokens,
            caller_metadata=caller_metadata,
            images=images,
        )
        parsed_content = response_model.model_validate_json(response.content)

        return LLMStructuredResponse(
            content=parsed_content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cached_tokens,
            cache_creation_tokens=response.cache_creation_tokens,
            input_cost=response.input_cost,
            output_cost=response.output_cost,
            total_cost=response.total_cost,
            response_time=response.response_time,
        )

    async def get_json_schema_response(
        self,
        user_prompt: str,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
        schema_name: str = "Response",
        schema_description: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        *,
        max_tokens: int | None = None,
        caller_metadata: dict[str, Any] | None = None,
        images: tuple[ImageInput, ...] = (),
        **kwargs: Any,
    ) -> LLMResponse:
        """Get a structured JSON response validated against a raw JSON schema.

        Runs the optional :attr:`hook_pipeline` around the provider call.
        Hooks see the raw provider JSON text in ``after_call`` before
        downstream pydantic/JSON-schema parsing.

        Args:
            user_prompt: The user's input prompt.
            response_schema: Raw JSON schema dict defining the expected response.
            system_prompt: Optional system prompt to set context/behavior.
            schema_name: Provider-facing schema/tool name.
            schema_description: Optional provider-facing schema/tool description.
            temperature: Sampling temperature (0.0-2.0).
            top_p: Nucleus sampling parameter (0.0-1.0).
            extra_headers: Optional per-request headers merged with default_headers.
            caller_metadata: Free-form dict forwarded to every hook.
            images: Validated in-memory images to analyze.
            **kwargs: Reserved for future provider-specific passthrough arguments.

        Returns:
            LLMResponse whose content is canonical JSON with sorted keys and no extra whitespace.

        Raises:
            HookBlocked: If a hook in the pipeline blocks the call.
        """
        ensure_no_unexpected_kwargs(kwargs)

        async def impl(prompt: str) -> LLMResponse:
            self._validate_images(images)
            if images:
                return await self._get_json_schema_response_with_images_retried(
                    user_prompt=prompt,
                    images=images,
                    response_schema=response_schema,
                    system_prompt=system_prompt,
                    schema_name=schema_name,
                    schema_description=schema_description,
                    temperature=temperature,
                    top_p=top_p,
                    extra_headers=extra_headers,
                    max_tokens=max_tokens,
                )
            return await self._get_json_schema_response_retried(
                user_prompt=prompt,
                response_schema=response_schema,
                system_prompt=system_prompt,
                schema_name=schema_name,
                schema_description=schema_description,
                temperature=temperature,
                top_p=top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
            )

        return await self._run_hooks_returning_response(user_prompt, caller_metadata, impl)

    @retry_provider_call
    async def _get_json_schema_response_with_images_retried(
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
        return await self._get_json_schema_response_with_images(
            user_prompt=user_prompt,
            images=images,
            response_schema=response_schema,
            system_prompt=system_prompt,
            schema_name=schema_name,
            schema_description=schema_description,
            temperature=temperature,
            top_p=top_p,
            extra_headers=extra_headers,
            max_tokens=max_tokens,
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
        """Provider implementation for structured image-understanding calls."""
        raise InputModalityUnsupported(self.provider, self.model, "image")

    @retry_provider_call
    async def _get_json_schema_response_retried(
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
        """Retry-wrapped delegate to the provider override.

        Sits inside the hook boundary so retries do not re-fire hooks.
        """
        return await self._get_json_schema_response(
            user_prompt=user_prompt,
            response_schema=response_schema,
            system_prompt=system_prompt,
            schema_name=schema_name,
            schema_description=schema_description,
            temperature=temperature,
            top_p=top_p,
            extra_headers=extra_headers,
            max_tokens=max_tokens,
        )

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
        """Provider-specific implementation for raw JSON-schema structured responses."""
        raise StructuredOutputUnsupported(self.provider, self.model)

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
        """Provider-specific implementation for structured responses.

        Default implementation injects the JSON schema into the system prompt.
        Providers should override this to use native structured output features.

        Args:
            response_model: Pydantic model class defining the expected structure.
            user_prompt: The user's input prompt.
            system_prompt: Optional system prompt to set context/behavior.
            temperature: Sampling temperature (0.0-2.0).
            top_p: Nucleus sampling parameter (0.0-1.0).
            extra_headers: Optional per-request headers merged with default_headers.

        Returns:
            LLMJSONResponse containing the parsed JSON content.
        """
        response = await self.get_json_schema_response(
            user_prompt=user_prompt,
            response_schema=response_model.model_json_schema(),
            system_prompt=system_prompt,
            schema_name=response_model.__name__,
            temperature=temperature,
            top_p=top_p,
            extra_headers=extra_headers,
            max_tokens=max_tokens,
        )
        return LLMJSONResponse(
            content=json.loads(response.content),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cached_tokens,
            cache_creation_tokens=response.cache_creation_tokens,
            input_cost=response.input_cost,
            output_cost=response.output_cost,
            total_cost=response.total_cost,
            response_time=response.response_time,
        )
