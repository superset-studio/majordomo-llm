"""Tests for the shared OpenAI-compatible provider base.

The request/response machinery lives in ``OpenAICompatibleLLM`` and is exercised
here once through a throwaway subclass. The per-provider test modules
(test_baseten.py, test_nebius.py, ...) only assert their own wiring rather than
re-testing this surface four more times.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from pydantic import BaseModel

from majordomo_llm.base import TOKENS_PER_MILLION
from majordomo_llm.exceptions import (
    ConfigurationError,
    ProviderError,
    StructuredOutputUnsupported,
)
from majordomo_llm.providers._openai_compatible import OpenAICompatibleLLM

MODEL_ID = "acme/Test-Model-1"
INPUT_COST = 2.00
OUTPUT_COST = 6.00
CACHED_INPUT_COST = 0.20


class Example(OpenAICompatibleLLM):
    """Throwaway provider used to exercise the shared base class."""

    PROVIDER_NAME = "example"
    DISPLAY_NAME = "Example"
    DEFAULT_BASE_URL = "https://api.example.test/v1"
    API_KEY_ENV = "EXAMPLE_API_KEY"


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


COUNTRY_SCHEMA = CountryInfo.model_json_schema()


def make_llm(**kwargs):
    """Build an Example instance with a mocked OpenAI client."""
    with patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"):
        return Example(
            model=MODEL_ID,
            input_cost=INPUT_COST,
            output_cost=OUTPUT_COST,
            api_key="test-key",
            **kwargs,
        )


def api_error(message: str = "boom") -> openai.APIError:
    """Build a non-retryable openai.APIError."""
    return openai.APIError(message, httpx.Request("POST", "https://api.example.test/v1"), body=None)


@pytest.fixture
def llm():
    return make_llm()


@pytest.fixture
def text_response():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Example says hello!"
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 8
    response.usage.prompt_tokens_details = None
    return response


@pytest.fixture
def cached_text_response():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "cached hello"
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 10
    response.usage.prompt_tokens_details.cached_tokens = 60
    return response


@pytest.fixture
def json_response():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = (
        '{"name": "France", "capital": "Paris", "population": 67000000}'
    )
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 30
    response.usage.prompt_tokens_details = None
    return response


@pytest.fixture
def stream_chunks():
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello"
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = " world"
    chunk2.usage = None

    final_chunk = MagicMock()
    final_chunk.choices = []
    final_chunk.usage.prompt_tokens = 20
    final_chunk.usage.completion_tokens = 8
    final_chunk.usage.prompt_tokens_details = None

    async def stream():
        yield chunk1
        yield chunk2
        yield final_chunk

    return stream()


class TestConstruction:
    """Tests for client configuration and key resolution."""

    def test_uses_default_base_url(self):
        with patch(
            "majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"
        ) as mock_client:
            Example(model=MODEL_ID, input_cost=1.0, output_cost=2.0, api_key="k")

        assert mock_client.call_args.kwargs["base_url"] == "https://api.example.test/v1"

    def test_base_url_overrides_default(self):
        with patch(
            "majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"
        ) as mock_client:
            Example(
                model=MODEL_ID,
                input_cost=1.0,
                output_cost=2.0,
                api_key="k",
                base_url="https://gateway.test/v1",
            )

        assert mock_client.call_args.kwargs["base_url"] == "https://gateway.test/v1"

    def test_sets_provider_name(self, llm):
        assert llm.provider == "example"

    def test_resolves_key_from_env(self, monkeypatch):
        monkeypatch.setenv("EXAMPLE_API_KEY", "from-env")
        with patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI") as mock_client:
            Example(model=MODEL_ID, input_cost=1.0, output_cost=2.0)

        assert mock_client.call_args.kwargs["api_key"] == "from-env"

    def test_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("EXAMPLE_API_KEY", raising=False)
        with pytest.raises(ConfigurationError) as exc_info:
            Example(model=MODEL_ID, input_cost=1.0, output_cost=2.0)

        assert "EXAMPLE_API_KEY" in str(exc_info.value)

    def test_rejects_invalid_reasoning_effort(self):
        with pytest.raises(ValueError, match="reasoning_effort"):
            make_llm(reasoning_effort="turbo")

    def test_rejects_invalid_thinking_mode(self):
        with pytest.raises(ValueError, match="thinking mode"):
            make_llm(thinking="maybe")


class TestGatewayHeader:
    """Tests for x-majordomo-provider injection when routed through a proxy."""

    def test_absent_without_base_url(self, llm):
        assert "x-majordomo-provider" not in (llm.default_headers or {})

    def test_injected_with_base_url(self):
        llm = make_llm(base_url="https://gateway.test/v1")
        assert llm.default_headers["x-majordomo-provider"] == "example"

    def test_caller_header_wins_on_collision(self):
        llm = make_llm(
            base_url="https://gateway.test/v1",
            default_headers={"x-majordomo-provider": "custom"},
        )
        assert llm.default_headers["x-majordomo-provider"] == "custom"

    def test_preserves_other_caller_headers(self):
        llm = make_llm(
            base_url="https://gateway.test/v1",
            default_headers={"x-majordomo-team": "platform"},
        )
        assert llm.default_headers["x-majordomo-team"] == "platform"
        assert llm.default_headers["x-majordomo-provider"] == "example"


class TestGetResponse:
    """Tests for the plain-text path."""

    async def test_returns_text_content(self, llm, text_response):
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        response = await llm.get_response("Say hello")

        assert response.content == "Example says hello!"

    async def test_returns_token_counts(self, llm, text_response):
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        response = await llm.get_response("Say hello")

        assert response.input_tokens == 20
        assert response.output_tokens == 8

    async def test_calculates_costs(self, llm, text_response):
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        response = await llm.get_response("Say hello")

        assert response.input_cost == 20 * INPUT_COST / TOKENS_PER_MILLION
        assert response.output_cost == 8 * OUTPUT_COST / TOKENS_PER_MILLION
        assert response.total_cost == response.input_cost + response.output_cost

    async def test_missing_prompt_details_yields_zero_cached(self, llm, text_response):
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        response = await llm.get_response("Say hello")

        assert response.cached_tokens == 0

    async def test_cached_tokens_repriced_as_subset(self, cached_text_response):
        llm = make_llm(cached_input_cost=CACHED_INPUT_COST)
        llm.client.chat.completions.create = AsyncMock(return_value=cached_text_response)

        response = await llm.get_response("Say hello")

        # 60 of the 100 prompt tokens bill at the cache rate, 40 at full rate.
        expected = (40 * INPUT_COST + 60 * CACHED_INPUT_COST) / TOKENS_PER_MILLION
        assert response.cached_tokens == 60
        assert response.input_cost == pytest.approx(expected)

    async def test_sends_system_prompt(self, llm, text_response):
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello", system_prompt="Be terse")

        messages = llm.client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "Be terse"}
        assert messages[1] == {"role": "user", "content": "Say hello"}

    async def test_omits_system_message_when_absent(self, llm, text_response):
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello")

        messages = llm.client.chat.completions.create.call_args.kwargs["messages"]
        assert len(messages) == 1

    async def test_sends_sampling_params(self, llm, text_response):
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello", temperature=0.7, top_p=0.9)

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9

    async def test_omits_sampling_params_when_unsupported(self, text_response):
        llm = make_llm(supports_temperature_top_p=False)
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello")

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs

    async def test_wraps_api_error(self, llm):
        llm.client.chat.completions.create = AsyncMock(side_effect=api_error())

        with pytest.raises(ProviderError) as exc_info:
            await llm.get_response("Say hello")

        assert exc_info.value.provider == "example"
        assert "Example API error" in str(exc_info.value)


class TestRequestKwargs:
    """Tests for reasoning_effort / thinking forwarding."""

    async def test_omitted_by_default(self, llm, text_response):
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello")

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs
        assert "extra_body" not in call_kwargs

    async def test_forwards_reasoning_effort(self, text_response):
        llm = make_llm(reasoning_effort="high")
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello")

        assert llm.client.chat.completions.create.call_args.kwargs["reasoning_effort"] == "high"

    async def test_forwards_thinking_via_extra_body(self, text_response):
        llm = make_llm(thinking="enabled")
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello")

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


class TestGetResponseStream:
    """Tests for the streaming path."""

    async def test_yields_text_chunks(self, llm, stream_chunks):
        llm.client.chat.completions.create = AsyncMock(return_value=stream_chunks)

        stream = await llm.get_response_stream("Hello")
        chunks = [chunk async for chunk in stream]

        assert chunks == ["Hello", " world"]

    async def test_usage_populated_after_iteration(self, llm, stream_chunks):
        llm.client.chat.completions.create = AsyncMock(return_value=stream_chunks)

        stream = await llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        assert stream.usage is not None
        assert stream.usage.input_tokens == 20
        assert stream.usage.output_tokens == 8

    async def test_calculates_costs_from_stream_usage(self, llm, stream_chunks):
        llm.client.chat.completions.create = AsyncMock(return_value=stream_chunks)

        stream = await llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        assert stream.usage.input_cost == 20 * INPUT_COST / TOKENS_PER_MILLION
        assert stream.usage.output_cost == 8 * OUTPUT_COST / TOKENS_PER_MILLION

    async def test_collect_returns_llm_response(self, llm, stream_chunks):
        llm.client.chat.completions.create = AsyncMock(return_value=stream_chunks)

        stream = await llm.get_response_stream("Hello")
        response = await stream.collect()

        assert response.content == "Hello world"
        assert response.input_tokens == 20

    async def test_requests_usage_in_stream(self, llm, stream_chunks):
        llm.client.chat.completions.create = AsyncMock(return_value=stream_chunks)

        await llm.get_response_stream("Hello")

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}

    async def test_wraps_api_error(self, llm):
        llm.client.chat.completions.create = AsyncMock(side_effect=api_error())

        with pytest.raises(ProviderError) as exc_info:
            await llm.get_response_stream("Hello")

        assert exc_info.value.provider == "example"


class TestStructuredResponse:
    """Tests for the JSON-schema structured-output path."""

    async def test_returns_parsed_model(self, llm, json_response):
        llm.client.chat.completions.create = AsyncMock(return_value=json_response)

        response = await llm.get_structured_json_response(
            response_model=CountryInfo, user_prompt="Tell me about France"
        )

        assert response.content.name == "France"
        assert response.content.capital == "Paris"

    async def test_sends_strict_json_schema_format(self, llm, json_response):
        llm.client.chat.completions.create = AsyncMock(return_value=json_response)

        await llm.get_structured_json_response(
            response_model=CountryInfo, user_prompt="Tell me about France"
        )

        response_format = llm.client.chat.completions.create.call_args.kwargs["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["name"] == "CountryInfo"

    async def test_reports_usage(self, llm, json_response):
        llm.client.chat.completions.create = AsyncMock(return_value=json_response)

        response = await llm.get_structured_json_response(
            response_model=CountryInfo, user_prompt="Tell me about France"
        )

        assert response.input_tokens == 50
        assert response.output_tokens == 30

    async def test_wraps_api_error(self, llm):
        llm.client.chat.completions.create = AsyncMock(side_effect=api_error())

        with pytest.raises(ProviderError) as exc_info:
            await llm.get_structured_json_response(
            response_model=CountryInfo, user_prompt="Tell me about France"
        )

        assert exc_info.value.provider == "example"


class TestStructuredOutputCapability:
    """Tests for the supports_structured_outputs opt-out."""

    def test_defaults_to_supported(self, llm):
        assert llm.supports_structured_outputs is True

    async def test_raises_when_unsupported(self):
        llm = make_llm(supports_structured_outputs=False)
        llm.client.chat.completions.create = AsyncMock()

        with pytest.raises(StructuredOutputUnsupported) as exc_info:
            await llm.get_structured_json_response(
                response_model=CountryInfo, user_prompt="Tell me about France"
            )

        assert exc_info.value.provider == "example"
        assert exc_info.value.model == MODEL_ID

    async def test_does_not_call_the_api_when_unsupported(self):
        llm = make_llm(supports_structured_outputs=False)
        llm.client.chat.completions.create = AsyncMock()

        with pytest.raises(StructuredOutputUnsupported):
            await llm.get_structured_json_response(
                response_model=CountryInfo, user_prompt="Tell me about France"
            )

        llm.client.chat.completions.create.assert_not_called()

    async def test_raw_schema_path_also_raises(self):
        llm = make_llm(supports_structured_outputs=False)
        llm.client.chat.completions.create = AsyncMock()

        with pytest.raises(StructuredOutputUnsupported):
            await llm.get_json_schema_response(
                user_prompt="Tell me about France", response_schema=COUNTRY_SCHEMA
            )

    async def test_text_still_works_when_unsupported(self, text_response):
        llm = make_llm(supports_structured_outputs=False)
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        response = await llm.get_response("Say hello")

        assert response.content == "Example says hello!"


class TestSamplingParams:
    """Tests for the no-default sampling policy.

    The library sends temperature/top_p only when the caller asks for them, so a
    provider applies its own documented default instead of one this library
    invented. Models whose deployment rejects the parameters never receive them.
    """

    async def test_omitted_by_default(self, llm, text_response):
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello")

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs

    async def test_sent_when_explicit(self, llm, text_response):
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello", temperature=0.8, top_p=0.95)

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.95

    async def test_each_param_is_independent(self, llm, text_response):
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello", temperature=0.8)

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8
        assert "top_p" not in call_kwargs

    async def test_zero_temperature_is_sent(self, llm, text_response):
        # 0.0 is falsy but explicit; it must not be confused with "unset".
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello", temperature=0.0)

        assert llm.client.chat.completions.create.call_args.kwargs["temperature"] == 0.0

    async def test_dropped_when_model_rejects_them(self, text_response):
        llm = make_llm(supports_temperature_top_p=False)
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await llm.get_response("Say hello", temperature=0.8, top_p=0.95)

        call_kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs

    async def test_silent_when_explicit_values_are_dropped(self, text_response, caplog):
        llm = make_llm(supports_temperature_top_p=False)
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        with caplog.at_level(logging.WARNING, logger="majordomo_llm.base"):
            await llm.get_response("Say hello", temperature=0.8)

        assert caplog.text == ""

    async def test_silent_when_nothing_was_requested(self, text_response, caplog):
        llm = make_llm(supports_temperature_top_p=False)
        llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        with caplog.at_level(logging.WARNING, logger="majordomo_llm.base"):
            await llm.get_response("Say hello")

        assert caplog.text == ""

    async def test_applies_to_streaming(self, llm, stream_chunks):
        llm.client.chat.completions.create = AsyncMock(return_value=stream_chunks)

        await llm.get_response_stream("Hello")

        assert "temperature" not in llm.client.chat.completions.create.call_args.kwargs

    async def test_applies_to_structured_output(self, llm, json_response):
        llm.client.chat.completions.create = AsyncMock(return_value=json_response)

        await llm.get_structured_json_response(
            response_model=CountryInfo, user_prompt="Tell me about France"
        )

        assert "temperature" not in llm.client.chat.completions.create.call_args.kwargs
