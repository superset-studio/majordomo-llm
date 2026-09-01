"""Custom exceptions for majordomo-llm."""


class MajordomoError(Exception):
    """Base exception for all majordomo-llm errors.

    All custom exceptions in this library inherit from this class,
    allowing users to catch all library-specific errors with a single
    except clause.

    Example:
        >>> try:
        ...     response = await llm.get_response("Hello")
        ... except MajordomoError as e:
        ...     print(f"LLM error: {e}")
    """


class ConfigurationError(MajordomoError):
    """Raised when configuration is invalid or missing.

    This includes missing API keys, invalid provider/model combinations,
    and other configuration-related issues.

    Example:
        >>> # Missing API key
        >>> llm = get_llm_instance("openai", "gpt-4o")
        ConfigurationError: Missing OPENAI_API_KEY environment variable.
    """


class InputModalityUnsupported(ConfigurationError):
    """Raised when a model cannot accept a requested input modality."""

    def __init__(self, provider: str, model: str, modality: str):
        super().__init__(
            f"Input modality '{modality}' is not supported by provider "
            f"'{provider}' model '{model}'."
        )
        self.provider = provider
        self.model = model
        self.modality = modality


class ImageOptionUnsupported(ConfigurationError):
    """Raised when an image model cannot represent a requested option.

    This is narrower than :class:`ConfigurationError` so an image cascade can
    try another model without swallowing missing credentials, unknown models,
    or other invalid configuration.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        option: str,
        value: object,
        *,
        supported: str | None = None,
    ) -> None:
        supported_message = f" Supported: {supported}." if supported else ""
        super().__init__(
            f"Image option '{option}={value}' is not supported by provider "
            f"'{provider}' model '{model}'.{supported_message}"
        )
        self.provider = provider
        self.model = model
        self.option = option
        self.value = value
        self.supported = supported


class ProviderError(MajordomoError):
    """Raised when an LLM provider returns an error.

    This wraps errors from the underlying provider SDKs (OpenAI, Anthropic,
    Google) to provide a consistent interface.

    Attributes:
        provider: The provider that raised the error.
        original_error: The original exception from the provider SDK.
    """

    def __init__(self, message: str, provider: str, original_error: Exception | None = None):
        """Initialize the provider error.

        Args:
            message: Human-readable error description.
            provider: The provider name (e.g., "openai", "anthropic").
            original_error: The original exception from the provider SDK.
        """
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error


class StructuredOutputUnsupported(ProviderError):
    """Raised when a provider/model does not support structured outputs.

    Attributes:
        provider: The provider that does not support structured outputs.
        model: The model that does not support structured outputs.
    """

    def __init__(self, provider: str, model: str):
        """Initialize the unsupported structured output error.

        Args:
            provider: The provider name (e.g., "openai", "anthropic").
            model: The provider model identifier.
        """
        super().__init__(
            f"Structured outputs are not supported by provider '{provider}' model '{model}'.",
            provider=provider,
        )
        self.model = model


class ResponseParsingError(MajordomoError):
    """Raised when response parsing fails.

    This is raised when the LLM response cannot be parsed as expected,
    such as invalid JSON or missing structured output fields.

    Attributes:
        raw_content: The raw response content that failed to parse.
    """

    def __init__(self, message: str, raw_content: str | None = None):
        """Initialize the parsing error.

        Args:
            message: Human-readable error description.
            raw_content: The raw response content that failed to parse.
        """
        super().__init__(message)
        self.raw_content = raw_content


class EmptyStructuredResponseError(ResponseParsingError):
    """Raised when a structured response is schema-valid but empty.

    A forced tool call is mandatory to *invoke* but its arguments are ordinary
    generation, so a model can return ``{}`` or an object whose every field is
    ``null`` — a schema-valid non-answer. Rather than report that as a
    successful response, the library raises this so callers (and
    :class:`~majordomo_llm.cascade.LLMCascade` / the retry policy) can react. It
    is retryable: the structured-output retry wrapper re-samples before it
    surfaces.

    Subclasses :class:`ResponseParsingError`; a caller that genuinely wants to
    accept an all-null object can catch this type and read ``raw_content``.
    """


class ResponseTruncatedError(MajordomoError):
    """Raised when a response was cut short by the ``max_tokens`` output cap.

    Providers that require an output cap (Anthropic's Messages API, Bedrock's
    Converse API) report this as ``stop_reason == "max_tokens"``. Without this
    error a truncated call returns successfully with partial — or, when the cut
    lands before any text block is emitted, empty — content, which is
    indistinguishable from a model that had nothing to say.

    Deliberately inherits :class:`MajordomoError` rather than
    :class:`ProviderError`, so that:

    - :func:`~majordomo_llm.retry.is_retryable_exception` does not re-sample it.
      Retrying a truncation spends the same budget on the same ceiling.
    - :class:`~majordomo_llm.cascade.LLMCascade` does not fail over on it. The
      cap is a configuration choice, not a provider outage; the next provider
      in the chain would truncate identically.

    Raise the ceiling with the ``max_tokens`` key in ``llm_config.yaml`` or the
    per-request ``max_tokens`` argument.

    Attributes:
        max_tokens: The output cap that was hit.
        output_tokens: Tokens the model actually emitted before being cut off.
        partial_content: Whatever content arrived before truncation. May be an
            empty string when the cut preceded the first text block.
    """

    def __init__(
        self,
        max_tokens: int,
        output_tokens: int,
        partial_content: str = "",
        *,
        provider: str | None = None,
    ):
        """Initialize the truncation error.

        Args:
            max_tokens: The output cap that was hit.
            output_tokens: Tokens emitted before the cut.
            partial_content: Content received before truncation.
            provider: Optional provider name, included in the message.
        """
        where = f" from provider '{provider}'" if provider else ""
        super().__init__(
            f"Response{where} was truncated at the max_tokens limit of {max_tokens} "
            f"({output_tokens} output tokens emitted). Raise it via the 'max_tokens' "
            f"key in llm_config.yaml or the per-request max_tokens argument."
        )
        self.max_tokens = max_tokens
        self.output_tokens = output_tokens
        self.partial_content = partial_content
        self.provider = provider
