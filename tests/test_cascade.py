"""Tests for the LLMCascade class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import Future, RetryError

from majordomo_llm import LLMCascade
from majordomo_llm.exceptions import (
    ProviderError,
    ResponseParsingError,
    ResponseTruncatedError,
    StructuredOutputUnsupported,
)


def _retry_error_with_exception(exc: BaseException) -> RetryError:
    return RetryError(Future.construct(attempt_number=3, value=exc, has_exception=True))


@pytest.fixture
def mock_all_clients():
    """Mock all provider API clients and environment variables."""
    env_vars = {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENAI_API_KEY": "test-key",
        "GEMINI_API_KEY": "test-key",
        "DEEPSEEK_API_KEY": "test-key",
        "CO_API_KEY": "test-key",
    }
    with (
        patch.dict("os.environ", env_vars),
        patch("majordomo_llm.providers.anthropic.anthropic.AsyncAnthropic"),
        patch("majordomo_llm.providers.openai.openai.AsyncOpenAI"),
        patch("majordomo_llm.providers.gemini.genai.Client"),
        patch("majordomo_llm.providers.deepseek.openai.AsyncOpenAI"),
        patch("majordomo_llm.providers.cohere.cohere.AsyncClientV2"),
    ):
        yield


class TestLLMCascadeInit:
    """Tests for LLMCascade initialization."""

    def test_creates_llm_instances_for_all_providers(self, mock_all_clients):
        """Should create LLM instances for all providers in list."""
        cascade = LLMCascade(
            [
                ("anthropic", "claude-sonnet-5"),
                ("openai", "gpt-4.1"),
            ]
        )

        assert len(cascade.llms) == 2
        assert cascade.llms[0].provider == "anthropic"
        assert cascade.llms[1].provider == "openai"

    def test_sets_provider_to_cascade(self, mock_all_clients):
        """Should set provider name to 'cascade'."""
        cascade = LLMCascade(
            [
                ("anthropic", "claude-sonnet-5"),
            ]
        )

        assert cascade.provider == "cascade"

    def test_uses_primary_provider_attributes(self, mock_all_clients):
        """Should use first provider's attributes for metadata."""
        cascade = LLMCascade(
            [
                ("anthropic", "claude-sonnet-5"),
                ("openai", "gpt-4.1"),
            ]
        )

        assert cascade.model == "claude-sonnet-5"
        assert cascade.input_cost == 3.00
        assert cascade.output_cost == 15.00

    def test_raises_error_for_empty_providers(self):
        """Should raise ValueError for empty providers list."""
        with pytest.raises(ValueError) as exc_info:
            LLMCascade([])

        assert "at least one provider" in str(exc_info.value)


class TestLLMCascadeGetResponse:
    """Tests for LLMCascade.get_response method."""

    @pytest.fixture
    def cascade(self, mock_all_clients):
        """Create LLMCascade with mocked providers."""
        return LLMCascade(
            [
                ("anthropic", "claude-sonnet-5"),
                ("openai", "gpt-4.1"),
                ("gemini", "gemini-2.5-flash"),
            ]
        )

    async def test_returns_response_from_primary_provider(self, cascade):
        """Should return response from first provider when it succeeds."""
        mock_response = MagicMock()
        mock_response.content = "Response from Anthropic"

        cascade.llms[0].get_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_response("Test prompt")

        assert response.content == "Response from Anthropic"
        cascade.llms[0].get_response.assert_called_once()

    async def test_falls_back_to_second_provider_on_failure(self, cascade):
        """Should fall back to second provider when first fails."""
        mock_response = MagicMock()
        mock_response.content = "Response from OpenAI"

        cascade.llms[0].get_response = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_response("Test prompt")

        assert response.content == "Response from OpenAI"
        cascade.llms[0].get_response.assert_called_once()
        cascade.llms[1].get_response.assert_called_once()

    async def test_falls_back_when_retry_error_wraps_provider_error(self, cascade):
        """Should fall back when provider retries exhaust with ProviderError."""
        mock_response = MagicMock()
        mock_response.content = "Response from OpenAI"
        provider_error = ProviderError("Anthropic down", provider="anthropic")

        cascade.llms[0].get_response = AsyncMock(
            side_effect=_retry_error_with_exception(provider_error)
        )
        cascade.llms[1].get_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_response("Test prompt")

        assert response.content == "Response from OpenAI"
        cascade.llms[0].get_response.assert_called_once()
        cascade.llms[1].get_response.assert_called_once()

    async def test_reraises_retry_error_with_non_provider_error(self, cascade):
        """Should not swallow retry failures caused by non-provider exceptions."""
        retry_error = _retry_error_with_exception(RuntimeError("bug"))
        cascade.llms[0].get_response = AsyncMock(side_effect=retry_error)

        with pytest.raises(RetryError) as exc_info:
            await cascade.get_response("Test prompt")

        assert exc_info.value is retry_error

    async def test_falls_back_through_all_providers(self, cascade):
        """Should try all providers in order until one succeeds."""
        mock_response = MagicMock()
        mock_response.content = "Response from Gemini"

        cascade.llms[0].get_response = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_response = AsyncMock(
            side_effect=ProviderError("OpenAI down", provider="openai")
        )
        cascade.llms[2].get_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_response("Test prompt")

        assert response.content == "Response from Gemini"

    async def test_raises_error_when_all_providers_fail(self, cascade):
        """Should raise ProviderError when all providers fail."""
        cascade.llms[0].get_response = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_response = AsyncMock(
            side_effect=ProviderError("OpenAI down", provider="openai")
        )
        cascade.llms[2].get_response = AsyncMock(
            side_effect=ProviderError("Gemini down", provider="gemini")
        )

        with pytest.raises(ProviderError) as exc_info:
            await cascade.get_response("Test prompt")

        assert "All providers in cascade failed" in str(exc_info.value)
        assert exc_info.value.provider == "cascade"

    async def test_passes_arguments_to_provider(self, cascade):
        """Should pass all arguments to the provider method."""
        mock_response = MagicMock()
        cascade.llms[0].get_response = AsyncMock(return_value=mock_response)

        await cascade.get_response(
            "Test prompt",
            system_prompt="Be helpful",
            temperature=0.7,
            top_p=0.9,
        )

        cascade.llms[0].get_response.assert_called_once_with(
            user_prompt="Test prompt",
            system_prompt="Be helpful",
            temperature=0.7,
            top_p=0.9,
            extra_headers=None,
            max_tokens=None,
        )


class TestLLMCascadeGetJSONResponse:
    """Tests for LLMCascade.get_json_response method."""

    @pytest.fixture
    def cascade(self, mock_all_clients):
        """Create LLMCascade with mocked providers."""
        return LLMCascade(
            [
                ("anthropic", "claude-sonnet-5"),
                ("openai", "gpt-4.1"),
            ]
        )

    async def test_returns_json_response_from_primary(self, cascade):
        """Should return JSON response from first provider.

        Cascade dispatches at the text layer; base parses JSON at the
        cascade boundary, so mocks attach to ``get_response``.
        """
        text_response = MagicMock()
        text_response.content = '{"key": "value"}'
        text_response.input_tokens = 10
        text_response.output_tokens = 5
        text_response.cached_tokens = 0
        text_response.input_cost = 0.0
        text_response.output_cost = 0.0
        text_response.total_cost = 0.0
        text_response.response_time = 0.0

        cascade.llms[0].get_response = AsyncMock(return_value=text_response)

        response = await cascade.get_json_response("Return JSON")

        assert response.content == {"key": "value"}

    async def test_falls_back_on_json_response_failure(self, cascade):
        """Should fall back when first provider fails."""
        text_response = MagicMock()
        text_response.content = '{"fallback": "data"}'
        text_response.input_tokens = 10
        text_response.output_tokens = 5
        text_response.cached_tokens = 0
        text_response.input_cost = 0.0
        text_response.output_cost = 0.0
        text_response.total_cost = 0.0
        text_response.response_time = 0.0

        cascade.llms[0].get_response = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_response = AsyncMock(return_value=text_response)

        response = await cascade.get_json_response("Return JSON")

        assert response.content == {"fallback": "data"}

    async def test_falls_back_when_json_retry_error_wraps_provider_error(self, cascade):
        """Should fall back when provider retries exhaust with ProviderError."""
        text_response = MagicMock()
        text_response.content = '{"fallback": "data"}'
        text_response.input_tokens = 10
        text_response.output_tokens = 5
        text_response.cached_tokens = 0
        text_response.input_cost = 0.0
        text_response.output_cost = 0.0
        text_response.total_cost = 0.0
        text_response.response_time = 0.0
        provider_error = ProviderError("Anthropic down", provider="anthropic")

        cascade.llms[0].get_response = AsyncMock(
            side_effect=_retry_error_with_exception(provider_error)
        )
        cascade.llms[1].get_response = AsyncMock(return_value=text_response)

        response = await cascade.get_json_response("Return JSON")

        assert response.content == {"fallback": "data"}
        cascade.llms[0].get_response.assert_called_once()
        cascade.llms[1].get_response.assert_called_once()


class TestLLMCascadeStructuredResponse:
    """Tests for LLMCascade.get_structured_json_response method."""

    @pytest.fixture
    def cascade(self, mock_all_clients):
        """Create LLMCascade with mocked providers."""
        return LLMCascade(
            [
                ("anthropic", "claude-sonnet-5"),
                ("openai", "gpt-4.1"),
            ]
        )

    async def test_returns_structured_response_from_primary(self, cascade):
        """Should return structured response from first provider."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_response = MagicMock()
        mock_response.content = '{"name":"test"}'

        cascade.llms[0].get_json_schema_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_structured_json_response(
            response_model=TestModel,
            user_prompt="Return structured data",
        )

        assert response.content.name == "test"

    async def test_falls_back_on_structured_response_failure(self, cascade):
        """Should fall back when first provider fails for structured response."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_response = MagicMock()
        mock_response.content = '{"name":"fallback"}'

        cascade.llms[0].get_json_schema_response = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_json_schema_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_structured_json_response(
            response_model=TestModel,
            user_prompt="Return structured data",
        )

        assert response.content.name == "fallback"

    async def test_falls_back_when_structured_retry_error_wraps_provider_error(self, cascade):
        """Should fall back when structured retries exhaust with ProviderError."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str

        mock_response = MagicMock()
        mock_response.content = '{"name":"fallback"}'
        provider_error = ProviderError("Anthropic down", provider="anthropic")

        cascade.llms[0].get_json_schema_response = AsyncMock(
            side_effect=_retry_error_with_exception(provider_error)
        )
        cascade.llms[1].get_json_schema_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_structured_json_response(
            response_model=TestModel,
            user_prompt="Return structured data",
        )

        assert response.content.name == "fallback"
        cascade.llms[0].get_json_schema_response.assert_called_once()
        cascade.llms[1].get_json_schema_response.assert_called_once()


class TestLLMCascadeJSONSchemaResponse:
    """Tests for LLMCascade.get_json_schema_response method."""

    @pytest.fixture
    def cascade(self, mock_all_clients):
        """Create LLMCascade with mocked providers."""
        return LLMCascade(
            [
                ("anthropic", "claude-sonnet-5"),
                ("openai", "gpt-4.1"),
            ]
        )

    async def test_returns_schema_response_from_primary(self, cascade):
        """Should return raw schema response from first provider."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_response = MagicMock()
        mock_response.content = '{"name":"primary"}'

        cascade.llms[0].get_json_schema_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_json_schema_response(
            user_prompt="Return structured data",
            response_schema=schema,
        )

        assert response.content == '{"name":"primary"}'

    async def test_falls_back_on_unsupported_schema_response(self, cascade):
        """Should fall back when a provider/model does not support structured output."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_response = MagicMock()
        mock_response.content = '{"name":"fallback"}'

        cascade.llms[0].get_json_schema_response = AsyncMock(
            side_effect=StructuredOutputUnsupported("anthropic", "claude-test")
        )
        cascade.llms[1].get_json_schema_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_json_schema_response(
            user_prompt="Return structured data",
            response_schema=schema,
        )

        assert response.content == '{"name":"fallback"}'
        cascade.llms[0].get_json_schema_response.assert_called_once()
        cascade.llms[1].get_json_schema_response.assert_called_once()

    async def test_falls_back_on_malformed_schema_response(self, cascade):
        """Should fall back when a provider returns malformed structured output."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_response = MagicMock()
        mock_response.content = '{"name":"fallback"}'

        cascade.llms[0].get_json_schema_response = AsyncMock(
            side_effect=ResponseParsingError("Malformed output", raw_content="not json")
        )
        cascade.llms[1].get_json_schema_response = AsyncMock(return_value=mock_response)

        response = await cascade.get_json_schema_response(
            user_prompt="Return structured data",
            response_schema=schema,
        )

        assert response.content == '{"name":"fallback"}'
        cascade.llms[0].get_json_schema_response.assert_called_once()
        cascade.llms[1].get_json_schema_response.assert_called_once()


class TestLLMCascadeGetResponseStream:
    """Tests for LLMCascade.get_response_stream method."""

    @pytest.fixture
    def cascade(self, mock_all_clients):
        """Create LLMCascade with mocked providers."""
        return LLMCascade(
            [
                ("anthropic", "claude-sonnet-5"),
                ("openai", "gpt-4.1"),
                ("gemini", "gemini-2.5-flash"),
            ]
        )

    async def test_returns_stream_from_primary_provider(self, cascade):
        """Should return stream from first provider when it succeeds."""
        mock_stream = MagicMock()
        cascade.llms[0].get_response_stream = AsyncMock(return_value=mock_stream)

        result = await cascade.get_response_stream("Test prompt")

        assert result is mock_stream
        cascade.llms[0].get_response_stream.assert_called_once()

    async def test_falls_back_on_creation_error(self, cascade):
        """Should fall back to next provider when stream creation fails."""
        mock_stream = MagicMock()
        cascade.llms[0].get_response_stream = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_response_stream = AsyncMock(return_value=mock_stream)

        result = await cascade.get_response_stream("Test prompt")

        assert result is mock_stream

    async def test_raises_error_when_all_fail(self, cascade):
        """Should raise ProviderError when all providers fail."""
        cascade.llms[0].get_response_stream = AsyncMock(
            side_effect=ProviderError("Anthropic down", provider="anthropic")
        )
        cascade.llms[1].get_response_stream = AsyncMock(
            side_effect=ProviderError("OpenAI down", provider="openai")
        )
        cascade.llms[2].get_response_stream = AsyncMock(
            side_effect=ProviderError("Gemini down", provider="gemini")
        )

        with pytest.raises(ProviderError) as exc_info:
            await cascade.get_response_stream("Test prompt")

        assert "All providers in cascade failed" in str(exc_info.value)


class TestLLMCascadeHooks:
    """Hooks attached to LLMCascade fire once at the cascade boundary."""

    @pytest.fixture
    def cascade_with_hooks(self, mock_all_clients):
        from majordomo_llm import HookOutcome, HookPipeline

        self.before_calls = 0
        self.after_calls = 0
        outer = self

        class CountingHook:
            name = "counter"

            async def before_call(self_inner, prompt, ctx):
                outer.before_calls += 1
                return HookOutcome.pass_through(self_inner.name)

            async def after_call(self_inner, prompt, response, ctx):
                outer.after_calls += 1
                return HookOutcome.pass_through(self_inner.name)

        pipeline = HookPipeline([CountingHook()])
        return LLMCascade(
            [
                ("anthropic", "claude-sonnet-5"),
                ("openai", "gpt-4.1"),
            ],
            hook_pipeline=pipeline,
        )

    async def test_hooks_fire_once_across_failover(self, cascade_with_hooks):
        cascade = cascade_with_hooks
        text_response = MagicMock()
        text_response.content = "ok"
        text_response.input_tokens = 1
        text_response.output_tokens = 1
        text_response.cached_tokens = 0
        text_response.input_cost = 0.0
        text_response.output_cost = 0.0
        text_response.total_cost = 0.0
        text_response.response_time = 0.0
        text_response.deprecation_warning = None

        cascade.llms[0].get_response = AsyncMock(
            side_effect=ProviderError("primary down", provider="anthropic")
        )
        cascade.llms[1].get_response = AsyncMock(return_value=text_response)

        await cascade.get_response("prompt")

        # Hooks fire at the cascade boundary, not per failover attempt
        assert self.before_calls == 1
        assert self.after_calls == 1

    async def test_block_at_cascade_prevents_provider_calls(
        self, mock_all_clients
    ):
        from majordomo_llm import HookBlocked, HookOutcome, HookPipeline

        class Blocker:
            name = "blocker"

            async def before_call(self, prompt, ctx):
                return HookOutcome.block(self.name, "no")

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.pass_through(self.name)

        cascade = LLMCascade(
            [
                ("anthropic", "claude-sonnet-5"),
                ("openai", "gpt-4.1"),
            ],
            hook_pipeline=HookPipeline([Blocker()]),
        )
        cascade.llms[0].get_response = AsyncMock()
        cascade.llms[1].get_response = AsyncMock()

        with pytest.raises(HookBlocked):
            await cascade.get_response("prompt")

        cascade.llms[0].get_response.assert_not_called()
        cascade.llms[1].get_response.assert_not_called()


class TestLLMCascadeTruncation:
    """A truncated response is a configuration problem, not a provider outage."""

    @pytest.fixture
    def cascade(self, mock_all_clients):
        return LLMCascade(
            [
                ("anthropic", "claude-sonnet-5"),
                ("openai", "gpt-4.1"),
            ]
        )

    async def test_does_not_fail_over_on_truncation(self, cascade):
        """The next provider would truncate identically, so failing over is waste."""
        cascade.llms[0].get_response = AsyncMock(
            side_effect=ResponseTruncatedError(
                max_tokens=1024, output_tokens=1024, partial_content="", provider="anthropic"
            )
        )
        cascade.llms[1].get_response = AsyncMock()

        with pytest.raises(ResponseTruncatedError):
            await cascade.get_response("Write six sections")

        cascade.llms[1].get_response.assert_not_called()

    async def test_still_fails_over_on_provider_error(self, cascade):
        """Failover behavior for genuine provider failures is unchanged."""
        response = MagicMock()
        response.content = "from the fallback"
        cascade.llms[0].get_response = AsyncMock(
            side_effect=ProviderError("down", provider="anthropic")
        )
        cascade.llms[1].get_response = AsyncMock(return_value=response)

        result = await cascade.get_response("Test prompt")

        assert result.content == "from the fallback"

    async def test_forwards_max_tokens_to_the_member(self, cascade):
        """A per-request cap must reach whichever member handles the call."""
        cascade.llms[0].get_response = AsyncMock(return_value=MagicMock())

        await cascade.get_response("Test prompt", max_tokens=4096)

        assert cascade.llms[0].get_response.call_args.kwargs["max_tokens"] == 4096
