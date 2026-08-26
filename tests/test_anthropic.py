"""Tests for the Anthropic provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from tenacity import RetryError

from majordomo_llm.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_STREAM_MAX_TOKENS,
    TOKENS_PER_MILLION,
)
from majordomo_llm.exceptions import (
    ConfigurationError,
    EmptyStructuredResponseError,
    ResponseParsingError,
    ResponseTruncatedError,
)
from majordomo_llm.providers import Anthropic
from majordomo_llm.providers.anthropic import MAX_NONSTREAMING_TOKENS


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


COUNTRY_SCHEMA = CountryInfo.model_json_schema()

# A schema whose only field is nullable-optional: an empty/all-null result is
# schema-valid, so it exercises the emptiness check rather than a plain
# validation failure.
NULLABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
    },
}


def _usage_mock() -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = 25
    usage.output_tokens = 10
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0
    usage.server_tool_use = None
    return usage


def _tool_use_response(name: str, value: dict, stop_reason: str = "tool_use") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = value
    response = MagicMock()
    response.content = [block]
    response.stop_reason = stop_reason
    response.usage = _usage_mock()
    return response


def _text_response(text: str, stop_reason: str = "end_turn") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = stop_reason
    response.usage = _usage_mock()
    return response


class TestAnthropicGetResponse:
    """Tests for Anthropic.get_response method."""

    @pytest.fixture
    def anthropic_llm(self):
        """Create Anthropic instance with mocked client."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-5",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )
            return llm

    async def test_returns_text_content(self, anthropic_llm, mock_anthropic_text_response):
        """Should extract text content from response."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        response = await anthropic_llm.get_response("What is the capital of France?")

        assert response.content == "Paris is the capital of France."

    async def test_returns_correct_token_counts(self, anthropic_llm, mock_anthropic_text_response):
        """Should return correct token counts."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        response = await anthropic_llm.get_response("Test prompt")

        assert response.input_tokens == 25
        assert response.output_tokens == 10
        assert response.cached_tokens == 0

    async def test_calculates_costs_correctly(self, anthropic_llm, mock_anthropic_text_response):
        """Should calculate costs based on token counts and rates."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        response = await anthropic_llm.get_response("Test prompt")

        expected_input_cost = 25 * 3.0 / TOKENS_PER_MILLION
        expected_output_cost = 10 * 15.0 / TOKENS_PER_MILLION

        assert response.input_cost == expected_input_cost
        assert response.output_cost == expected_output_cost
        assert response.total_cost == expected_input_cost + expected_output_cost

    async def test_includes_response_time(self, anthropic_llm, mock_anthropic_text_response):
        """Should include response time measurement."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        response = await anthropic_llm.get_response("Test prompt")

        assert response.response_time >= 0

    async def test_passes_temperature_and_top_p(self, anthropic_llm, mock_anthropic_text_response):
        """Should pass temperature and top_p to API."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        await anthropic_llm.get_response(
            "Test prompt",
            temperature=0.7,
            top_p=0.9,
        )

        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9

    async def test_uses_default_system_prompt(self, anthropic_llm, mock_anthropic_text_response):
        """Should use default system prompt when none provided."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        await anthropic_llm.get_response("Test prompt")

        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        system_text = call_kwargs["system"][0]["text"]
        assert "helpful assistant" in system_text


class TestAnthropicStructuredResponse:
    """Tests for Anthropic structured response methods."""

    @pytest.fixture
    def anthropic_llm(self):
        """Anthropic instance without native structured outputs (forced-tool fallback)."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-5",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )
            return llm

    @pytest.fixture
    def native_llm(self):
        """Anthropic instance with native structured outputs (constrained decoding)."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            return Anthropic(
                model="claude-opus-4-7",
                input_cost=5.0,
                output_cost=25.0,
                supports_temperature_top_p=False,
                supports_structured_outputs=True,
                api_key="test-key",
            )

    async def test_extracts_tool_use_content(self, anthropic_llm, mock_anthropic_tool_response):
        """Should extract content from tool_use block."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_tool_response)

        response = await anthropic_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        assert response.content.name == "France"
        assert response.content.capital == "Paris"
        assert response.content.population == 67000000

    async def test_returns_validated_pydantic_model(
        self, anthropic_llm, mock_anthropic_tool_response
    ):
        """Should return a validated Pydantic model instance."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_tool_response)

        response = await anthropic_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        assert isinstance(response.content, CountryInfo)

    async def test_forces_tool_choice(self, anthropic_llm, mock_anthropic_tool_response):
        """Should force tool choice to the schema name."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_tool_response)

        await anthropic_llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"]["type"] == "tool"
        assert call_kwargs["tool_choice"]["name"] == "CountryInfo"

    async def test_forced_tool_fallback_sends_strict_schema(
        self, anthropic_llm, mock_anthropic_tool_response
    ):
        """A model without native structured outputs uses forced tool calling
        with a strict (enforced) input schema — not a relaxed one — and no
        output_config."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_tool_response)

        response = await anthropic_llm.get_json_schema_response(
            user_prompt="Tell me about France",
            response_schema=COUNTRY_SCHEMA,
            schema_name="CountryInfo",
        )

        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        sent = call_kwargs["tools"][0]["input_schema"]
        assert call_kwargs["tools"][0]["name"] == "CountryInfo"
        assert call_kwargs["tool_choice"]["name"] == "CountryInfo"
        assert "output_config" not in call_kwargs
        assert sent["additionalProperties"] is False
        assert set(sent["required"]) == {"name", "capital", "population"}
        assert response.content == '{"capital":"Paris","name":"France","population":67000000}'

    async def test_native_structured_output_uses_output_config(self, native_llm):
        """A model with native support uses output_config.format (constrained
        decoding), not a tool call, and never sends a name key inside format."""
        native_llm.client.messages.create = AsyncMock(
            return_value=_text_response(
                '{"name":"France","capital":"Paris","population":67000000}'
            )
        )

        response = await native_llm.get_json_schema_response(
            user_prompt="Tell me about France",
            response_schema=COUNTRY_SCHEMA,
            schema_name="CountryInfo",
        )

        call_kwargs = native_llm.client.messages.create.call_args.kwargs
        fmt = call_kwargs["output_config"]["format"]
        assert "tools" not in call_kwargs
        assert fmt["type"] == "json_schema"
        assert "name" not in fmt  # a name key inside format is rejected with 400
        assert fmt["schema"]["additionalProperties"] is False
        assert set(fmt["schema"]["required"]) == {"name", "capital", "population"}
        assert response.content == '{"capital":"Paris","name":"France","population":67000000}'

    async def test_native_strips_unsupported_keywords(self, native_llm):
        """Constrained-decoder-unsupported keywords are stripped from the wire
        schema but still enforced post-hoc against the original schema."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["n"],
            "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 10}},
        }
        native_llm.client.messages.create = AsyncMock(return_value=_text_response('{"n":5}'))

        await native_llm.get_json_schema_response(user_prompt="x", response_schema=schema)

        sent = native_llm.client.messages.create.call_args.kwargs["output_config"]["format"][
            "schema"
        ]
        assert "minimum" not in sent["properties"]["n"]
        assert "maximum" not in sent["properties"]["n"]

    async def test_native_refusal_raises(self, native_llm):
        """A refusal stop_reason surfaces as an error rather than empty text."""
        native_llm.client.messages.create = AsyncMock(
            return_value=_text_response("", stop_reason="refusal")
        )

        with pytest.raises(ResponseParsingError):
            await native_llm.get_json_schema_response(
                user_prompt="x", response_schema=COUNTRY_SCHEMA
            )

    async def test_empty_forced_tool_result_raises(self, anthropic_llm):
        """An all-null (schema-valid) forced-tool result surfaces as an error,
        not a silent empty success. It is re-sampled three times, then surfaces
        as an EmptyStructuredResponseError (wrapped in RetryError on exhaustion,
        the library's convention for every retryable error)."""
        anthropic_llm.client.messages.create = AsyncMock(
            return_value=_tool_use_response("Answer", {"note": None})
        )

        with pytest.raises(RetryError) as exc_info:
            await anthropic_llm.get_json_schema_response(
                user_prompt="x", response_schema=NULLABLE_SCHEMA, schema_name="Answer"
            )

        assert isinstance(exc_info.value.last_attempt.exception(), EmptyStructuredResponseError)
        assert anthropic_llm.client.messages.create.call_count == 3

    async def test_retries_empty_then_succeeds(self, anthropic_llm):
        """An empty first sample is re-sampled; a populated retry is returned."""
        anthropic_llm.client.messages.create = AsyncMock(
            side_effect=[
                _tool_use_response("Answer", {"note": None}),
                _tool_use_response("Answer", {"note": "hi"}),
            ]
        )

        response = await anthropic_llm.get_json_schema_response(
            user_prompt="x", response_schema=NULLABLE_SCHEMA, schema_name="Answer"
        )

        assert response.content == '{"note":"hi"}'
        assert anthropic_llm.client.messages.create.call_count == 2


class TestAnthropicReasoningEffort:
    """Tests for the configurable output_config.effort level."""

    def _llm(self, effort, *, native=True, model="claude-opus-4-8"):
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            return Anthropic(
                model=model,
                input_cost=5.0,
                output_cost=25.0,
                supports_temperature_top_p=False,
                supports_structured_outputs=native,
                reasoning_effort=effort,
                api_key="test-key",
            )

    async def test_effort_merged_into_native_output_config(self):
        llm = self._llm("medium")
        llm.client.messages.create = AsyncMock(return_value=_text_response('{"note":"hi"}'))

        await llm.get_json_schema_response(user_prompt="x", response_schema=NULLABLE_SCHEMA)

        output_config = llm.client.messages.create.call_args.kwargs["output_config"]
        assert output_config["effort"] == "medium"
        assert output_config["format"]["type"] == "json_schema"

    async def test_effort_applied_to_plain_response(self):
        llm = self._llm("low", native=False, model="claude-sonnet-5")
        llm.client.messages.create = AsyncMock(return_value=_text_response("hi"))

        await llm.get_response("hello")

        assert llm.client.messages.create.call_args.kwargs["output_config"] == {"effort": "low"}

    async def test_no_effort_omits_output_config_on_plain_response(self):
        llm = self._llm(None, native=False, model="claude-sonnet-5")
        llm.client.messages.create = AsyncMock(return_value=_text_response("hi"))

        await llm.get_response("hello")

        assert "output_config" not in llm.client.messages.create.call_args.kwargs

    def test_invalid_effort_raises(self):
        with (
            patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"),
            pytest.raises(ValueError, match="reasoning_effort"),
        ):
            Anthropic(
                model="claude-opus-4-8",
                input_cost=5.0,
                output_cost=25.0,
                reasoning_effort="turbo",
                api_key="test-key",
            )


class TestAnthropicThinking:
    """Tests for the configurable thinking mode."""

    def _llm(self, thinking, *, effort=None, native=False, model="claude-opus-4-8"):
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            return Anthropic(
                model=model,
                input_cost=5.0,
                output_cost=25.0,
                supports_temperature_top_p=False,
                supports_structured_outputs=native,
                reasoning_effort=effort,
                thinking=thinking,
                api_key="test-key",
            )

    async def test_thinking_applied_to_plain_response(self):
        llm = self._llm("adaptive")
        llm.client.messages.create = AsyncMock(return_value=_text_response("hi"))

        await llm.get_response("hello")

        assert llm.client.messages.create.call_args.kwargs["thinking"] == {"type": "adaptive"}

    async def test_thinking_and_effort_both_applied_on_native(self):
        llm = self._llm("adaptive", effort="high", native=True)
        llm.client.messages.create = AsyncMock(return_value=_text_response('{"note":"hi"}'))

        await llm.get_json_schema_response(user_prompt="x", response_schema=NULLABLE_SCHEMA)

        kwargs = llm.client.messages.create.call_args.kwargs
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"]["effort"] == "high"
        assert kwargs["output_config"]["format"]["type"] == "json_schema"

    async def test_no_thinking_omits_field(self):
        llm = self._llm(None)
        llm.client.messages.create = AsyncMock(return_value=_text_response("hi"))

        await llm.get_response("hello")

        assert "thinking" not in llm.client.messages.create.call_args.kwargs

    def test_invalid_thinking_raises(self):
        with (
            patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"),
            pytest.raises(ValueError, match="thinking mode"),
        ):
            Anthropic(
                model="claude-opus-4-8",
                input_cost=5.0,
                output_cost=25.0,
                thinking="enabled",
                api_key="test-key",
            )


class TestAnthropicInit:
    """Tests for Anthropic initialization."""

    def test_raises_configuration_error_without_api_key(self):
        """Should raise ConfigurationError when no API key is provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                Anthropic(
                    model="claude-sonnet-5",
                    input_cost=3.0,
                    output_cost=15.0,
                )

            assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_sets_provider_name(self):
        """Should set provider to 'anthropic'."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-5",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )

            assert llm.provider == "anthropic"

    def test_stores_model_and_costs(self):
        """Should store model name and cost configuration."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-5",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )

            assert llm.model == "claude-sonnet-5"
            assert llm.input_cost == 3.0
            assert llm.output_cost == 15.0

    def test_web_search_disabled_by_default(self):
        """Should have web search disabled by default."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-5",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )

            assert llm.use_web_search is False

    def test_web_search_can_be_enabled(self):
        """Should allow enabling web search."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-4-5-20250929",
                input_cost=3.0,
                output_cost=15.0,
                use_web_search=True,
                api_key="test-key",
            )

            assert llm.use_web_search is True


class TestAnthropicGetResponseStream:
    """Tests for Anthropic.get_response_stream method."""

    @pytest.fixture
    def anthropic_llm(self):
        """Create Anthropic instance with mocked client."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-5",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )
            return llm

    async def test_yields_text_chunks(self, anthropic_llm, mock_anthropic_stream_events):
        """Should yield text chunks from stream."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_stream_events)

        stream = await anthropic_llm.get_response_stream("Hello")
        chunks = [chunk async for chunk in stream]

        assert chunks == ["Hello", " world"]

    async def test_usage_populated_after_iteration(
        self, anthropic_llm, mock_anthropic_stream_events
    ):
        """Should populate usage after stream is consumed."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_stream_events)

        stream = await anthropic_llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        assert stream.usage is not None
        assert stream.usage.input_tokens == 25
        assert stream.usage.output_tokens == 10

    async def test_calculates_costs_correctly(self, anthropic_llm, mock_anthropic_stream_events):
        """Should calculate costs from stream usage."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_stream_events)

        stream = await anthropic_llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        expected_input_cost = 25 * 3.0 / TOKENS_PER_MILLION
        expected_output_cost = 10 * 15.0 / TOKENS_PER_MILLION
        assert stream.usage.input_cost == expected_input_cost
        assert stream.usage.output_cost == expected_output_cost

    async def test_collect_returns_llm_response(self, anthropic_llm, mock_anthropic_stream_events):
        """Should return LLMResponse from collect()."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_stream_events)

        stream = await anthropic_llm.get_response_stream("Hello")
        response = await stream.collect()

        assert response.content == "Hello world"
        assert response.input_tokens == 25

    async def test_passes_stream_true_to_api(self, anthropic_llm, mock_anthropic_stream_events):
        """Should pass stream=True to the API call."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_stream_events)

        await anthropic_llm.get_response_stream("Hello")

        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        assert call_kwargs["stream"] is True


class TestAnthropicWebSearch:
    """Tests for Anthropic web search wiring."""

    @pytest.fixture
    def anthropic_llm_web(self):
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            return Anthropic(
                model="claude-sonnet-4-6",
                input_cost=3.0,
                output_cost=15.0,
                use_web_search=True,
                api_key="test-key",
            )

    async def test_web_search_tool_included_in_text_call(
        self, anthropic_llm_web, mock_anthropic_text_response
    ):
        """Should include WebSearchTool20250305Param when use_web_search=True."""
        anthropic_llm_web.client.messages.create = AsyncMock(
            return_value=mock_anthropic_text_response
        )

        await anthropic_llm_web.get_response("Latest news?")

        call_kwargs = anthropic_llm_web.client.messages.create.call_args.kwargs
        tools = call_kwargs["tools"]
        assert any(
            t.get("type") == "web_search_20250305" and t.get("name") == "web_search"
            for t in tools
        )

    async def test_web_search_cost_added_to_total(
        self, anthropic_llm_web, mock_anthropic_text_response
    ):
        """Should add server_tool_use web_search_requests * $0.01 to total_cost."""
        from unittest.mock import MagicMock

        mock_anthropic_text_response.usage.server_tool_use = MagicMock(web_search_requests=2)
        anthropic_llm_web.client.messages.create = AsyncMock(
            return_value=mock_anthropic_text_response
        )

        response = await anthropic_llm_web.get_response("Latest news?")

        assert response.tool_use_cost == pytest.approx(0.02)
        expected_input_cost = 25 * 3.0 / TOKENS_PER_MILLION
        expected_output_cost = 10 * 15.0 / TOKENS_PER_MILLION
        assert response.total_cost == pytest.approx(
            expected_input_cost + expected_output_cost + 0.02
        )


class TestAnthropicPromptCaching:
    """Tests for the configurable prompt-cache breakpoint and cache costing."""

    def _make_llm(self, **kwargs):
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            return Anthropic(
                model="claude-sonnet-5",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
                **kwargs,
            )

    async def test_cache_control_present_by_default(self, mock_anthropic_text_response):
        """The system block carries an ephemeral cache_control breakpoint by default."""
        llm = self._make_llm()
        llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        await llm.get_response("Hello", system_prompt="You are helpful.")

        system = llm.client.messages.create.call_args.kwargs["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    async def test_cache_control_omitted_when_disabled(self, mock_anthropic_text_response):
        """use_prompt_caching=False drops the cache_control breakpoint."""
        llm = self._make_llm(use_prompt_caching=False)
        llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        await llm.get_response("Hello", system_prompt="You are helpful.")

        system = llm.client.messages.create.call_args.kwargs["system"]
        assert "cache_control" not in system[0]

    async def test_cache_read_and_write_costs_applied(self, mock_anthropic_text_response):
        """Cache read/write tokens are billed on top of uncached input (additive)."""
        mock_anthropic_text_response.usage.input_tokens = 25
        mock_anthropic_text_response.usage.cache_read_input_tokens = 100
        mock_anthropic_text_response.usage.cache_creation_input_tokens = 200
        llm = self._make_llm(cached_input_cost=0.3, cache_write_cost=3.75)
        llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        response = await llm.get_response("Hello")

        assert response.cached_tokens == 100
        assert response.cache_creation_tokens == 200
        expected_input = (25 * 3.0 + 100 * 0.3 + 200 * 3.75) / TOKENS_PER_MILLION
        assert response.input_cost == pytest.approx(expected_input)


class TestAnthropicMaxTokens:
    """Tests for the configurable output cap and truncation detection."""

    @pytest.fixture
    def anthropic_llm(self):
        """Anthropic instance with no configured cap (library defaults apply)."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            return Anthropic(
                model="claude-sonnet-5",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )

    @pytest.fixture
    def capped_llm(self):
        """Anthropic instance with a cap supplied the way llm_config.yaml does."""
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            return Anthropic(
                model="claude-sonnet-5",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
                max_tokens=8192,
            )

    async def test_default_cap_on_plain_text(self, anthropic_llm, mock_anthropic_text_response):
        """Should send DEFAULT_MAX_TOKENS, not the old hardcoded 1024."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        await anthropic_llm.get_response("Test prompt")

        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == DEFAULT_MAX_TOKENS

    async def test_default_cap_on_streaming(self, anthropic_llm, mock_anthropic_stream_events):
        """Streaming should get the larger default, since no HTTP read timeout applies."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_stream_events)

        await anthropic_llm.get_response_stream("Hello")

        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == DEFAULT_STREAM_MAX_TOKENS

    async def test_config_cap_overrides_default(self, capped_llm, mock_anthropic_text_response):
        """A model's configured max_tokens should win over the library default."""
        capped_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        await capped_llm.get_response("Test prompt")

        assert capped_llm.client.messages.create.call_args.kwargs["max_tokens"] == 8192

    async def test_config_cap_applies_to_streaming(
        self, capped_llm, mock_anthropic_stream_events
    ):
        """A configured cap replaces the streaming default too."""
        capped_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_stream_events)

        await capped_llm.get_response_stream("Hello")

        assert capped_llm.client.messages.create.call_args.kwargs["max_tokens"] == 8192

    async def test_per_request_cap_overrides_config(
        self, capped_llm, mock_anthropic_text_response
    ):
        """A per-request max_tokens should win over the configured value."""
        capped_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        await capped_llm.get_response("Test prompt", max_tokens=2048)

        assert capped_llm.client.messages.create.call_args.kwargs["max_tokens"] == 2048

    async def test_structured_path_uses_resolved_cap(self, capped_llm):
        """Structured output should use the same resolved cap, not a separate literal."""
        capped_llm.client.messages.create = AsyncMock(
            return_value=_tool_use_response(
                "CountryInfo", {"name": "France", "capital": "Paris", "population": 1}
            )
        )

        await capped_llm.get_structured_json_response(
            response_model=CountryInfo, user_prompt="Tell me about France"
        )

        assert capped_llm.client.messages.create.call_args.kwargs["max_tokens"] == 8192

    async def test_records_stop_reason(self, anthropic_llm, mock_anthropic_text_response):
        """A normal response should carry the provider's stop reason."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        response = await anthropic_llm.get_response("Test prompt")

        assert response.stop_reason == "end_turn"

    async def test_raises_when_truncated_before_any_text(
        self, anthropic_llm, mock_anthropic_truncated_response
    ):
        """The reported failure: empty content must raise, not return silently."""
        anthropic_llm.client.messages.create = AsyncMock(
            return_value=mock_anthropic_truncated_response
        )

        with pytest.raises(ResponseTruncatedError) as exc_info:
            await anthropic_llm.get_response("Write six sections")

        assert exc_info.value.max_tokens == DEFAULT_MAX_TOKENS
        assert exc_info.value.output_tokens == 1024
        assert exc_info.value.partial_content == ""

    async def test_raises_when_truncated_with_partial_content(
        self, anthropic_llm, mock_anthropic_partially_truncated_response
    ):
        """Partial output is still corrupt output; the partial text is preserved."""
        anthropic_llm.client.messages.create = AsyncMock(
            return_value=mock_anthropic_partially_truncated_response
        )

        with pytest.raises(ResponseTruncatedError) as exc_info:
            await anthropic_llm.get_response("Write six sections")

        assert exc_info.value.partial_content == "Section one is about"

    async def test_error_message_names_the_remedy(
        self, anthropic_llm, mock_anthropic_truncated_response
    ):
        """The message should point at the knob, since that is the whole fix."""
        anthropic_llm.client.messages.create = AsyncMock(
            return_value=mock_anthropic_truncated_response
        )

        with pytest.raises(ResponseTruncatedError, match="max_tokens"):
            await anthropic_llm.get_response("Write six sections")

    async def test_truncation_is_not_retried(
        self, anthropic_llm, mock_anthropic_truncated_response
    ):
        """Re-sampling would spend the same budget on the same ceiling."""
        create = AsyncMock(return_value=mock_anthropic_truncated_response)
        anthropic_llm.client.messages.create = create

        with pytest.raises(ResponseTruncatedError):
            await anthropic_llm.get_response("Write six sections")

        assert create.await_count == 1

    async def test_streaming_raises_after_yielding_chunks(
        self, anthropic_llm, mock_anthropic_truncated_stream_events
    ):
        """A truncated stream should fail like a truncated non-streaming call."""
        anthropic_llm.client.messages.create = AsyncMock(
            return_value=mock_anthropic_truncated_stream_events
        )

        stream = await anthropic_llm.get_response_stream("Write six sections")
        chunks = []
        with pytest.raises(ResponseTruncatedError):
            async for chunk in stream:
                chunks.append(chunk)

        assert chunks == ["Section one is about"]

    async def test_streaming_records_stop_reason(
        self, anthropic_llm, mock_anthropic_stream_events
    ):
        """A clean stream should expose its stop reason through collect()."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_stream_events)

        stream = await anthropic_llm.get_response_stream("Hello")
        response = await stream.collect()

        assert stream.stop_reason == "end_turn"
        assert response.stop_reason == "end_turn"

    async def test_structured_truncation_raises_before_parsing(self, anthropic_llm):
        """A cut-off structured response should report the cause, not a parse error."""
        truncated = _tool_use_response("CountryInfo", {}, stop_reason="max_tokens")
        anthropic_llm.client.messages.create = AsyncMock(return_value=truncated)

        with pytest.raises(ResponseTruncatedError):
            await anthropic_llm.get_structured_json_response(
                response_model=CountryInfo, user_prompt="Tell me about France"
            )

    async def test_rejects_non_positive_cap(self, anthropic_llm, mock_anthropic_text_response):
        """A zero or negative cap is a caller bug, not something to send upstream."""
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        with pytest.raises(ValueError, match="positive integer"):
            await anthropic_llm.get_response("Test prompt", max_tokens=0)


class TestAnthropicNonStreamingLimit:
    """The SDK rejects a non-streaming max_tokens over 21333 before sending.

    Regression guard: v0.22.0 pinned each model's vendor ceiling (128000) in
    config, which became the per-request default and broke every non-streaming
    call. The mocked client cannot reproduce that — the real SDK's check lives in
    messages.create — so these assert on what we resolve and reject ourselves.
    """

    @pytest.fixture
    def anthropic_llm(self):
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            return Anthropic(
                model="claude-sonnet-5",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )

    def test_default_is_under_the_sdk_limit(self):
        assert DEFAULT_MAX_TOKENS <= MAX_NONSTREAMING_TOKENS

    def test_stream_default_is_over_it(self):
        """Which is fine — streaming has no such limit — but proves they differ."""
        assert DEFAULT_STREAM_MAX_TOKENS > MAX_NONSTREAMING_TOKENS

    async def test_rejects_an_over_limit_per_request_value(
        self, anthropic_llm, mock_anthropic_text_response
    ):
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        with pytest.raises(ValueError, match="get_response_stream"):
            await anthropic_llm.get_response("hi", max_tokens=64000)

    async def test_rejects_an_over_limit_config_value(self, mock_anthropic_text_response):
        with patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"):
            llm = Anthropic(
                model="claude-sonnet-5", input_cost=3.0, output_cost=15.0,
                api_key="test-key", max_tokens=128000,
            )
        llm.client.messages.create = AsyncMock(return_value=mock_anthropic_text_response)

        with pytest.raises(ValueError, match="128000"):
            await llm.get_response("hi")

    async def test_structured_path_is_guarded_too(self, anthropic_llm):
        anthropic_llm.client.messages.create = AsyncMock()

        with pytest.raises(ValueError, match="non-streaming"):
            await anthropic_llm.get_structured_json_response(
                response_model=CountryInfo, user_prompt="Tell me about France",
                max_tokens=64000,
            )

    async def test_streaming_allows_what_non_streaming_rejects(
        self, anthropic_llm, mock_anthropic_stream_events
    ):
        anthropic_llm.client.messages.create = AsyncMock(return_value=mock_anthropic_stream_events)

        await anthropic_llm.get_response_stream("hi", max_tokens=64000)

        assert anthropic_llm.client.messages.create.call_args.kwargs["max_tokens"] == 64000
