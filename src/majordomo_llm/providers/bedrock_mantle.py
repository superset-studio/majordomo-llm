"""Bedrock Mantle provider — Claude via AWS-native Anthropic Messages API.

Bedrock Mantle is the modern Bedrock integration path for Anthropic Claude
models. It exposes the same Anthropic Messages API shape served at
``https://api.anthropic.com``, but on an AWS-managed endpoint
(``https://bedrock-mantle.{region}.api.aws/anthropic``) with AWS bearer-token
auth and AWS-side data residency.

Why a separate provider class instead of routing through the existing
``Bedrock`` (Converse) provider:

- Claude's feature set lands first-class on Bedrock Mantle (structured outputs,
  prompt caching, extended thinking, tool use) because the wire shape is the
  Anthropic Messages API — no Converse-style translation layer.
- Avoids the per-model capability allowlist that Converse-based Bedrock needs
  for native Structured Outputs (which fails on Opus 4.7+ because AWS reuses
  ``output_config`` for adaptive-thinking control on those models).

This class is implemented as a thin subclass of ``Anthropic``: same client
(``anthropic.AsyncAnthropic``), same methods, same retry policy. Only the
endpoint, the API key resolution, and the ``provider`` identity change.

References:
- https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock
- https://builder.aws.com/content/3Cl90CMMnqzCrkk6mXcmnGo1WTG/
"""

import os

from majordomo_llm.base import resolve_api_key
from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers.anthropic import Anthropic


class BedrockMantle(Anthropic):
    """Anthropic Claude served via the Bedrock Mantle endpoint.

    Authenticates with an Amazon Bedrock long-term bearer token
    (``AWS_BEARER_TOKEN_BEDROCK`` env var) or short-term minted token. The AWS
    region must be supplied via ``region`` or the ``AWS_REGION`` /
    ``AWS_DEFAULT_REGION`` environment variables.

    Model IDs use the bare Anthropic format with no ``us.`` prefix and no
    ``-v1`` suffix — e.g. ``anthropic.claude-opus-4-7``.

    Example:
        >>> llm = BedrockMantle(
        ...     model="anthropic.claude-opus-4-7",
        ...     input_cost=5.0,
        ...     output_cost=25.0,
        ...     region="us-east-1",
        ... )
        >>> response = await llm.get_response("Hello, Claude!")
    """

    MANTLE_ENDPOINT_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/anthropic"

    def __init__(
        self,
        model: str,
        input_cost: float,
        output_cost: float,
        supports_temperature_top_p: bool = True,
        use_web_search: bool = False,
        supports_structured_outputs: bool = False,
        use_prompt_caching: bool = True,
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
        """Initialize the Bedrock Mantle provider.

        Args:
            model: The Bedrock Mantle model ID (e.g., ``anthropic.claude-opus-4-7``).
            input_cost: Cost per million input tokens in USD.
            output_cost: Cost per million output tokens in USD.
            supports_temperature_top_p: Whether temperature/top_p are supported.
                Opus 4.7+ deprecates these — set to ``False`` for those models.
            use_web_search: Forwarded to the Anthropic provider.
            use_prompt_caching: Forwarded to the Anthropic provider; controls the
                ephemeral ``cache_control`` breakpoint on the system prompt.
            cached_input_cost: Cost per million cache-read tokens in USD.
            cache_write_cost: Cost per million cache-creation tokens in USD.
            api_key: Optional Bedrock bearer token. Defaults to
                ``AWS_BEARER_TOKEN_BEDROCK`` env var.
            api_key_alias: Optional human-readable name for the key.
            base_url: Optional custom endpoint URL. When unset, derived from
                ``region`` using ``MANTLE_ENDPOINT_TEMPLATE``.
            default_headers: Optional headers sent with every request.
            region: AWS region (e.g., ``us-east-1``). Defaults to ``AWS_REGION``
                or ``AWS_DEFAULT_REGION`` env vars. Required unless ``base_url``
                is explicitly provided.
            max_tokens: Default output cap for this model, forwarded to the
                Anthropic provider.

        Raises:
            ConfigurationError: If no bearer token or region can be resolved.
        """
        resolved_api_key = resolve_api_key(
            api_key, "AWS_BEARER_TOKEN_BEDROCK", "Bedrock Mantle"
        )

        resolved_region = (
            region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        )

        proxying = base_url is not None
        if proxying:
            # User pointed us at a proxy (e.g. Majordomo Steward). Auto-inject
            # the metadata Steward needs to route the request correctly:
            #
            # - ``x-majordomo-provider: bedrock-mantle`` disambiguates Mantle
            #   traffic from vanilla Anthropic traffic (both speak the
            #   Messages API shape; Steward routes on this header, not on the
            #   request path).
            # - ``X-Majordomo-Bedrock-Region`` tells Steward which AWS region
            #   to forward the request to (Mantle is region-pinned upstream).
            #
            # Caller-supplied default_headers win on key collision, so users
            # who need to override either value can.
            resolved_base_url = base_url
            merged_headers: dict[str, str] = {
                "x-majordomo-provider": "bedrock-mantle",
            }
            if resolved_region:
                merged_headers["X-Majordomo-Bedrock-Region"] = resolved_region
            if default_headers:
                merged_headers.update(default_headers)
            default_headers = merged_headers
        else:
            if not resolved_region:
                raise ConfigurationError(
                    "Bedrock Mantle region not found. Pass region= to the constructor "
                    "or set the AWS_REGION (or AWS_DEFAULT_REGION) environment variable."
                )
            resolved_base_url = self.MANTLE_ENDPOINT_TEMPLATE.format(region=resolved_region)

        super().__init__(
            model=model,
            input_cost=input_cost,
            output_cost=output_cost,
            cached_input_cost=cached_input_cost,
            cache_write_cost=cache_write_cost,
            use_prompt_caching=use_prompt_caching,
            supports_temperature_top_p=supports_temperature_top_p,
            use_web_search=use_web_search,
            supports_structured_outputs=supports_structured_outputs,
            api_key=resolved_api_key,
            api_key_alias=api_key_alias,
            base_url=resolved_base_url,
            default_headers=default_headers,
            max_tokens=max_tokens,
        )
        # Override the provider identity set by Anthropic.__init__ so cost
        # tracking, logging, and cascade dispatch see Bedrock Mantle as its own
        # provider rather than vanilla Anthropic.
        self.provider = "bedrock_mantle"
        self.region = resolved_region
