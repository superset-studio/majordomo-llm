"""Tests for the Amazon Bedrock provider."""

from contextlib import asynccontextmanager
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
    ResponseTruncatedError,
)
from majordomo_llm.providers import Bedrock


class CountryInfo(BaseModel):
    """Test model for structured responses."""

    name: str
    capital: str
    population: int


COUNTRY_SCHEMA = CountryInfo.model_json_schema()


def _make_bedrock(model: str = "us.anthropic.claude-sonnet-4-5-v1:0") -> Bedrock:
    return Bedrock(
        model=model,
        input_cost=3.0,
        output_cost=15.0,
        api_key="test-key",
        region="us-east-1",
    )


def _install_mock_client(bedrock: Bedrock) -> MagicMock:
    """Replace bedrock._client with an async context manager returning a mock."""
    client = MagicMock()
    client.converse = AsyncMock()
    client.converse_stream = AsyncMock()

    @asynccontextmanager
    async def fake_client():
        yield client

    bedrock._client = fake_client  # type: ignore[method-assign]
    return client


def _converse_response(text: str = "Paris is the capital of France.") -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "usage": {"inputTokens": 25, "outputTokens": 10, "cacheReadInputTokens": 0},
        "stopReason": "end_turn",
    }


def _tool_use_response(name: str, value: dict) -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "t1", "name": name, "input": value}}],
            }
        },
        "usage": {"inputTokens": 50, "outputTokens": 30, "cacheReadInputTokens": 5},
        "stopReason": "tool_use",
    }


class TestBedrockInit:
    def test_raises_configuration_error_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                Bedrock(
                    model="us.anthropic.claude-sonnet-4-5-v1:0",
                    input_cost=3.0,
                    output_cost=15.0,
                    region="us-east-1",
                )
            assert "AWS_BEARER_TOKEN_BEDROCK" in str(exc_info.value)

    def test_raises_configuration_error_without_region(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                Bedrock(
                    model="us.anthropic.claude-sonnet-4-5-v1:0",
                    input_cost=3.0,
                    output_cost=15.0,
                    api_key="test-key",
                )
            assert "region" in str(exc_info.value).lower()

    def test_region_from_env_var(self):
        with patch.dict("os.environ", {"AWS_REGION": "us-west-2"}, clear=True):
            llm = Bedrock(
                model="us.anthropic.claude-sonnet-4-5-v1:0",
                input_cost=3.0,
                output_cost=15.0,
                api_key="test-key",
            )
            assert llm.region == "us-west-2"

    def test_sets_provider_name(self):
        llm = _make_bedrock()
        assert llm.provider == "bedrock"

    def test_stores_model_and_costs(self):
        llm = _make_bedrock()
        assert llm.model == "us.anthropic.claude-sonnet-4-5-v1:0"
        assert llm.input_cost == 3.0
        assert llm.output_cost == 15.0


class TestBedrockGetResponse:
    async def test_returns_text_content(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        response = await llm.get_response("What is the capital of France?")

        assert response.content == "Paris is the capital of France."

    async def test_returns_correct_token_counts(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        response = await llm.get_response("Test prompt")

        assert response.input_tokens == 25
        assert response.output_tokens == 10
        assert response.cached_tokens == 0

    async def test_calculates_costs_correctly(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        response = await llm.get_response("Test prompt")

        expected_input_cost = 25 * 3.0 / TOKENS_PER_MILLION
        expected_output_cost = 10 * 15.0 / TOKENS_PER_MILLION
        assert response.input_cost == expected_input_cost
        assert response.output_cost == expected_output_cost
        assert response.total_cost == expected_input_cost + expected_output_cost

    async def test_passes_temperature_and_top_p(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        await llm.get_response("Test prompt", temperature=0.7, top_p=0.9)

        kwargs = client.converse.call_args.kwargs
        assert kwargs["inferenceConfig"]["temperature"] == 0.7
        assert kwargs["inferenceConfig"]["topP"] == 0.9

    async def test_omits_temperature_when_unsupported(self):
        llm = Bedrock(
            model="us.anthropic.claude-sonnet-4-5-v1:0",
            input_cost=3.0,
            output_cost=15.0,
            supports_temperature_top_p=False,
            api_key="test-key",
            region="us-east-1",
        )
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        await llm.get_response("Test prompt", temperature=0.7, top_p=0.9)

        kwargs = client.converse.call_args.kwargs
        assert "temperature" not in kwargs["inferenceConfig"]
        assert "topP" not in kwargs["inferenceConfig"]

    async def test_uses_default_system_prompt(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        await llm.get_response("Test prompt")

        kwargs = client.converse.call_args.kwargs
        assert "helpful assistant" in kwargs["system"][0]["text"]


# Substring not in _BEDROCK_STRUCTURED_OUTPUTS_SUPPORTED — exercises the
# tool-calling fallback path in the tests below.
_TOOL_CALLING_FALLBACK_MODEL = "moonshotai.kimi-k2.5"


class TestBedrockStructuredResponse:
    async def test_extracts_tool_use_content(self):
        llm = _make_bedrock(model=_TOOL_CALLING_FALLBACK_MODEL)
        client = _install_mock_client(llm)
        client.converse.return_value = _tool_use_response(
            "CountryInfo",
            {"name": "France", "capital": "Paris", "population": 67000000},
        )

        response = await llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        assert response.content.name == "France"
        assert response.content.capital == "Paris"
        assert response.content.population == 67000000

    async def test_forces_tool_choice(self):
        llm = _make_bedrock(model=_TOOL_CALLING_FALLBACK_MODEL)
        client = _install_mock_client(llm)
        client.converse.return_value = _tool_use_response(
            "CountryInfo",
            {"name": "France", "capital": "Paris", "population": 67000000},
        )

        await llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        kwargs = client.converse.call_args.kwargs
        assert kwargs["toolConfig"]["toolChoice"] == {"tool": {"name": "CountryInfo"}}
        assert kwargs["toolConfig"]["tools"][0]["toolSpec"]["name"] == "CountryInfo"

    async def test_omits_tool_choice_for_llama4(self):
        """Llama 4 on Bedrock rejects toolChoice.tool; we expose the tool but
        do not force it (model is steered via the system prompt instead)."""
        llm = _make_bedrock(model="us.meta.llama4-scout-17b-instruct-v1:0")
        client = _install_mock_client(llm)
        client.converse.return_value = _tool_use_response(
            "CountryInfo",
            {"name": "France", "capital": "Paris", "population": 67000000},
        )

        await llm.get_structured_json_response(
            response_model=CountryInfo,
            user_prompt="Tell me about France",
        )

        kwargs = client.converse.call_args.kwargs
        assert "toolChoice" not in kwargs["toolConfig"]
        # The tool itself must still be exposed so the model can call it.
        assert kwargs["toolConfig"]["tools"][0]["toolSpec"]["name"] == "CountryInfo"

    async def test_json_schema_response_uses_schema_tool(self):
        llm = _make_bedrock(model=_TOOL_CALLING_FALLBACK_MODEL)
        client = _install_mock_client(llm)
        client.converse.return_value = _tool_use_response(
            "CountryInfo",
            {"name": "France", "capital": "Paris", "population": 67000000},
        )

        await llm.get_json_schema_response(
            user_prompt="Tell me about France",
            response_schema=COUNTRY_SCHEMA,
            schema_name="CountryInfo",
        )

        kwargs = client.converse.call_args.kwargs
        spec = kwargs["toolConfig"]["tools"][0]["toolSpec"]
        assert spec["name"] == "CountryInfo"

    async def test_sends_strict_schema_to_tool_spec(self):
        """The tool inputSchema is the strict (enforced) form, not relaxed."""
        llm = _make_bedrock(model=_TOOL_CALLING_FALLBACK_MODEL)
        client = _install_mock_client(llm)
        client.converse.return_value = _tool_use_response(
            "CountryInfo",
            {"name": "France", "capital": "Paris", "population": 67000000},
        )

        await llm.get_json_schema_response(
            user_prompt="Tell me about France",
            response_schema=COUNTRY_SCHEMA,
            schema_name="CountryInfo",
        )

        sent = client.converse.call_args.kwargs["toolConfig"]["tools"][0]["toolSpec"]
        schema = sent["inputSchema"]["json"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(COUNTRY_SCHEMA["properties"].keys())

    async def test_empty_tool_result_raises(self):
        """An all-null (schema-valid) tool result is an error, not a silent success.

        Re-sampled three times, then surfaces as an EmptyStructuredResponseError
        (wrapped in RetryError on exhaustion, matching every retryable error)."""
        llm = _make_bedrock(model=_TOOL_CALLING_FALLBACK_MODEL)
        client = _install_mock_client(llm)
        client.converse.return_value = _tool_use_response("Answer", {"note": None})

        with pytest.raises(RetryError) as exc_info:
            await llm.get_json_schema_response(
                user_prompt="Answer",
                response_schema=_NULLABLE_SCHEMA,
                schema_name="Answer",
            )

        assert isinstance(exc_info.value.last_attempt.exception(), EmptyStructuredResponseError)


_NULLABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
    },
}


class _FakeRequest:
    """Stand-in for the botocore request passed to before-send handlers."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}


async def _emit_before_send(llm: Bedrock) -> _FakeRequest:
    """Trigger the session-level before-send.bedrock-runtime event and return
    the fake request after handlers have run. aiobotocore's emit is async."""
    request = _FakeRequest()
    await llm._session.emit("before-send.bedrock-runtime", request=request)
    return request


class TestBedrockProxyHeaderInjection:
    """Tests for the Steward-routing header-injection hook."""

    async def test_injects_host_and_default_headers_when_base_url_is_set(self):
        llm = Bedrock(
            model="us.anthropic.claude-sonnet-4-5-v1:0",
            input_cost=3.0,
            output_cost=15.0,
            api_key="bedrock-bearer",
            region="us-west-2",
            base_url="https://gateway.example.com",
            default_headers={"X-Majordomo-Key": "mk-1", "X-Majordomo-Feature": "shadow"},
        )

        request = await _emit_before_send(llm)

        assert request.headers["Host"] == "gateway.example.com"
        assert request.headers["X-Majordomo-Key"] == "mk-1"
        assert request.headers["X-Majordomo-Feature"] == "shadow"
        assert request.headers["X-Majordomo-Bedrock-Region"] == "us-west-2"

    async def test_no_hook_registered_when_base_url_is_unset(self):
        """Direct-AWS callers must not get proxy headers — regression guard."""
        llm = Bedrock(
            model="us.anthropic.claude-sonnet-4-5-v1:0",
            input_cost=3.0,
            output_cost=15.0,
            api_key="bedrock-bearer",
            region="us-east-1",
            default_headers={"X-Majordomo-Key": "mk-1"},
        )

        request = await _emit_before_send(llm)

        # No handler should have run — request.headers stays empty.
        assert request.headers == {}

    async def test_handles_empty_default_headers(self):
        """``default_headers=None`` is the common case for ad-hoc gateway use;
        Host and region should still land."""
        llm = Bedrock(
            model="us.anthropic.claude-sonnet-4-5-v1:0",
            input_cost=3.0,
            output_cost=15.0,
            api_key="bedrock-bearer",
            region="us-east-1",
            base_url="https://gateway.example.com",
        )

        request = await _emit_before_send(llm)

        assert request.headers["Host"] == "gateway.example.com"
        assert request.headers["X-Majordomo-Bedrock-Region"] == "us-east-1"
        assert "X-Majordomo-Key" not in request.headers


class TestBedrockStream:
    async def _stream_response(self):
        async def stream_iter():
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockDelta": {"delta": {"text": "Hello"}}}
            yield {"contentBlockDelta": {"delta": {"text": " world"}}}
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": 25,
                        "outputTokens": 10,
                        "cacheReadInputTokens": 0,
                    }
                }
            }

        return {"stream": stream_iter()}

    async def test_yields_text_chunks(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse_stream.return_value = await self._stream_response()

        stream = await llm.get_response_stream("Hello")
        chunks = [chunk async for chunk in stream]

        assert chunks == ["Hello", " world"]

    async def test_usage_populated_after_iteration(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse_stream.return_value = await self._stream_response()

        stream = await llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        assert stream.usage is not None
        assert stream.usage.input_tokens == 25
        assert stream.usage.output_tokens == 10

    async def test_collect_returns_llm_response(self):
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse_stream.return_value = await self._stream_response()

        stream = await llm.get_response_stream("Hello")
        response = await stream.collect()

        assert response.content == "Hello world"
        assert response.input_tokens == 25


def _truncated_converse_response(text: str = "") -> dict:
    content = [{"text": text}] if text else []
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "usage": {"inputTokens": 25, "outputTokens": 1024, "cacheReadInputTokens": 0},
        "stopReason": "max_tokens",
    }


class TestBedrockMaxTokens:
    """Tests for the configurable output cap and truncation detection."""

    def _sent_cap(self, client) -> int:
        return client.converse.call_args.kwargs["inferenceConfig"]["maxTokens"]

    async def test_default_cap_on_plain_text(self):
        """Should send DEFAULT_MAX_TOKENS, not the old hardcoded 1024."""
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        await llm.get_response("Test prompt")

        assert self._sent_cap(client) == DEFAULT_MAX_TOKENS

    async def test_config_cap_overrides_default(self):
        """A model's configured max_tokens should reach the inference config."""
        llm = Bedrock(
            model="deepseek.v3.2",
            input_cost=3.0,
            output_cost=15.0,
            api_key="test-key",
            region="us-east-1",
            max_tokens=163840,
        )
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        await llm.get_response("Test prompt")

        assert self._sent_cap(client) == 163840

    async def test_per_request_cap_overrides_config(self):
        """A per-request max_tokens should win over the configured value."""
        llm = Bedrock(
            model="deepseek.v3.2",
            input_cost=3.0,
            output_cost=15.0,
            api_key="test-key",
            region="us-east-1",
            max_tokens=163840,
        )
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        await llm.get_response("Test prompt", max_tokens=4096)

        assert self._sent_cap(client) == 4096

    async def test_streaming_uses_stream_default(self):
        """Streaming should get the larger default."""
        llm = _make_bedrock()
        client = _install_mock_client(llm)

        async def stream_iter():
            yield {"contentBlockDelta": {"delta": {"text": "Hi"}}}
            yield {"messageStop": {"stopReason": "end_turn"}}

        client.converse_stream.return_value = {"stream": stream_iter()}

        stream = await llm.get_response_stream("Hello")
        async for _ in stream:
            pass

        sent = client.converse_stream.call_args.kwargs["inferenceConfig"]["maxTokens"]
        assert sent == DEFAULT_STREAM_MAX_TOKENS

    async def test_records_stop_reason(self):
        """A normal response should carry the Converse stop reason."""
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _converse_response()

        response = await llm.get_response("Test prompt")

        assert response.stop_reason == "end_turn"

    async def test_raises_when_truncated(self):
        """Converse reports truncation as stopReason max_tokens; it must raise."""
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _truncated_converse_response()

        with pytest.raises(ResponseTruncatedError) as exc_info:
            await llm.get_response("Write six sections")

        assert exc_info.value.output_tokens == 1024
        assert exc_info.value.provider == "bedrock"

    async def test_truncation_is_not_retried(self):
        """Re-sampling would spend the same budget on the same ceiling."""
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        client.converse.return_value = _truncated_converse_response("Section one")

        with pytest.raises(ResponseTruncatedError):
            await llm.get_response("Write six sections")

        assert client.converse.await_count == 1

    async def test_streaming_raises_on_message_stop(self):
        """A truncated stream should fail like a truncated non-streaming call."""
        llm = _make_bedrock()
        client = _install_mock_client(llm)

        async def stream_iter():
            yield {"contentBlockDelta": {"delta": {"text": "Section one"}}}
            yield {"metadata": {"usage": {"inputTokens": 25, "outputTokens": 1024}}}
            yield {"messageStop": {"stopReason": "max_tokens"}}

        client.converse_stream.return_value = {"stream": stream_iter()}

        stream = await llm.get_response_stream("Write six sections")
        with pytest.raises(ResponseTruncatedError):
            async for _ in stream:
                pass

    async def test_structured_truncation_raises_before_extraction(self):
        """A cut-off tool call has no complete toolUse block; report the cause."""
        llm = _make_bedrock()
        client = _install_mock_client(llm)
        truncated = _tool_use_response("CountryInfo", {})
        truncated["stopReason"] = "max_tokens"
        client.converse.return_value = truncated

        with pytest.raises(ResponseTruncatedError):
            await llm.get_structured_json_response(
                response_model=CountryInfo, user_prompt="Tell me about France"
            )
