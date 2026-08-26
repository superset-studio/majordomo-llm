"""Tests for the base module."""

import time

import pytest
from pydantic import BaseModel

from majordomo_llm.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_STREAM_MAX_TOKENS,
    LLM,
    TOKENS_PER_MILLION,
    LLMJSONResponse,
    LLMResponse,
    LLMStreamResponse,
    LLMStructuredResponse,
    Usage,
    _StreamState,
    canonicalize_json_schema_output,
    is_empty_structured_result,
)
from majordomo_llm.exceptions import EmptyStructuredResponseError, ResponseParsingError

COUNTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "population": {"type": "integer"},
    },
    "required": ["name", "population"],
}


class TestJSONSchemaOutputHelpers:
    """Tests for JSON-schema output parsing and canonicalization."""

    def test_serializes_canonical_json(self):
        """Should sort keys and remove extra whitespace."""
        content = '{"population": 125000000, "name": "Japan"}'

        canonical = canonicalize_json_schema_output(content, COUNTRY_SCHEMA)

        assert canonical == '{"name":"Japan","population":125000000}'

    def test_repairs_markdown_fenced_json(self):
        """Should strip markdown fences before parsing."""
        content = '```json\n{"name":"Japan","population":125000000}\n```'

        canonical = canonicalize_json_schema_output(content, COUNTRY_SCHEMA)

        assert canonical == '{"name":"Japan","population":125000000}'

    def test_repairs_first_balanced_object(self):
        """Should extract the first balanced JSON object from surrounding text."""
        content = 'Here is the answer: {"name":"Japan","population":125000000} thanks.'

        canonical = canonicalize_json_schema_output(content, COUNTRY_SCHEMA)

        assert canonical == '{"name":"Japan","population":125000000}'

    def test_validation_error_includes_raw_content(self):
        """Should include raw content when schema validation fails."""
        raw_content = '{"name":"Japan","population":"many"}'

        with pytest.raises(ResponseParsingError) as exc_info:
            canonicalize_json_schema_output(raw_content, COUNTRY_SCHEMA)

        assert exc_info.value.raw_content == raw_content


NULLABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "group": {"anyOf": [{"type": "object"}, {"type": "null"}], "default": None},
        "skip_reason": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
    },
}


class TestEmptyStructuredResult:
    """Tests for is_empty_structured_result."""

    def test_empty_dict_is_empty(self):
        assert is_empty_structured_result({}) is True

    def test_all_null_values_is_empty(self):
        assert is_empty_structured_result({"group": None, "skip_reason": None}) is True

    def test_any_populated_value_is_not_empty(self):
        assert is_empty_structured_result({"group": None, "skip_reason": "n/a"}) is False

    def test_non_dict_is_not_empty(self):
        assert is_empty_structured_result([]) is False
        assert is_empty_structured_result("x") is False
        assert is_empty_structured_result(None) is False


class TestCanonicalizeRejectsEmpty:
    """canonicalize_json_schema_output surfaces empty/all-null results as errors."""

    def test_empty_dict_raises(self):
        """An empty object that is schema-valid is reported as empty, not success."""
        with pytest.raises(EmptyStructuredResponseError):
            canonicalize_json_schema_output({}, NULLABLE_SCHEMA)

    def test_all_null_object_raises(self):
        """An all-null object (the forced-tool punt signature) is reported as empty."""
        with pytest.raises(EmptyStructuredResponseError) as exc_info:
            canonicalize_json_schema_output({"group": None, "skip_reason": None}, NULLABLE_SCHEMA)

        assert exc_info.value.raw_content == '{"group":null,"skip_reason":null}'

    def test_all_null_from_string_raises(self):
        """The string-parse path is guarded too."""
        with pytest.raises(EmptyStructuredResponseError):
            canonicalize_json_schema_output('{"group": null, "skip_reason": null}', NULLABLE_SCHEMA)

    def test_populated_object_passes(self):
        """A result with at least one populated field is returned normally."""
        canonical = canonicalize_json_schema_output(
            {"group": None, "skip_reason": "busy"}, NULLABLE_SCHEMA
        )

        assert canonical == '{"group":null,"skip_reason":"busy"}'

    def test_reject_empty_false_allows_all_null(self):
        """Callers can opt out of the empty check."""
        canonical = canonicalize_json_schema_output(
            {"group": None, "skip_reason": None}, NULLABLE_SCHEMA, reject_empty=False
        )

        assert canonical == '{"group":null,"skip_reason":null}'

    def test_empty_error_is_a_response_parsing_error(self):
        """EmptyStructuredResponseError subclasses ResponseParsingError for compatibility."""
        assert issubclass(EmptyStructuredResponseError, ResponseParsingError)


class TestUsage:
    """Tests for Usage dataclass."""

    def test_usage_stores_all_fields(self):
        """Should store all usage metrics."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=10,
            input_cost=0.0003,
            output_cost=0.00075,
            total_cost=0.00105,
            response_time=1.5,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 10
        assert usage.input_cost == 0.0003
        assert usage.output_cost == 0.00075
        assert usage.total_cost == 0.00105
        assert usage.response_time == 1.5


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_includes_content_and_usage(self):
        """Should include content and inherit usage fields."""
        response = LLMResponse(
            content="Hello, world!",
            input_tokens=10,
            output_tokens=5,
            cached_tokens=0,
            input_cost=0.00003,
            output_cost=0.000075,
            total_cost=0.000105,
            response_time=0.5,
        )

        assert response.content == "Hello, world!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5


class TestLLMJSONResponse:
    """Tests for LLMJSONResponse dataclass."""

    def test_content_is_dict(self):
        """Content should be a dictionary."""
        response = LLMJSONResponse(
            content={"key": "value", "number": 42},
            input_tokens=20,
            output_tokens=10,
            cached_tokens=0,
            input_cost=0.00006,
            output_cost=0.00015,
            total_cost=0.00021,
            response_time=0.8,
        )

        assert response.content == {"key": "value", "number": 42}
        assert response.content["key"] == "value"


class TestLLMStructuredResponse:
    """Tests for LLMStructuredResponse dataclass."""

    def test_content_is_pydantic_model(self):
        """Content should be a Pydantic model instance."""

        class Person(BaseModel):
            name: str
            age: int

        person = Person(name="Alice", age=30)
        response = LLMStructuredResponse(
            content=person,
            input_tokens=30,
            output_tokens=15,
            cached_tokens=5,
            input_cost=0.00009,
            output_cost=0.000225,
            total_cost=0.000315,
            response_time=1.0,
        )

        assert response.content.name == "Alice"
        assert response.content.age == 30


class TestLLMCostCalculation:
    """Tests for LLM._calculate_costs method."""

    class ConcreteLLM(LLM):
        """Concrete implementation for testing abstract base class."""

        async def _get_response_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None, max_tokens=None,
        ):
            raise NotImplementedError()

        async def _get_response_stream_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None, max_tokens=None,
        ):
            raise NotImplementedError()

    def test_calculates_costs_correctly(self):
        """Should calculate costs based on tokens and rates."""
        llm = self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,  # $3 per million tokens
            output_cost=15.0,  # $15 per million tokens
        )

        input_cost, output_cost, total_cost = llm._calculate_costs(
            input_tokens=1000,
            output_tokens=500,
        )

        expected_input = 1000 * 3.0 / TOKENS_PER_MILLION
        expected_output = 500 * 15.0 / TOKENS_PER_MILLION

        assert input_cost == expected_input
        assert output_cost == expected_output
        assert total_cost == expected_input + expected_output

    def test_handles_zero_tokens(self):
        """Should handle zero tokens gracefully."""
        llm = self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
        )

        input_cost, output_cost, total_cost = llm._calculate_costs(
            input_tokens=0,
            output_tokens=0,
        )

        assert input_cost == 0.0
        assert output_cost == 0.0
        assert total_cost == 0.0

    def test_handles_large_token_counts(self):
        """Should handle large token counts correctly."""
        llm = self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
        )

        input_cost, output_cost, total_cost = llm._calculate_costs(
            input_tokens=1_000_000,  # 1 million tokens
            output_tokens=1_000_000,
        )

        assert input_cost == 3.0  # Exactly $3
        assert output_cost == 15.0  # Exactly $15
        assert total_cost == 18.0

    def test_subset_accounting_reprices_cached_reads(self):
        """Subset providers re-price cached tokens (a subset of input) downward."""
        llm = self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
            cached_input_cost=0.3,
        )

        input_cost, _, total_cost = llm._calculate_costs(
            input_tokens=1000,
            output_tokens=0,
            cached_tokens=400,
        )

        # 600 uncached @ $3 + 400 cached @ $0.30, all per million.
        expected_input = (600 * 3.0 + 400 * 0.3) / TOKENS_PER_MILLION
        assert input_cost == expected_input
        assert total_cost == expected_input

    def test_subset_accounting_without_rate_bills_at_input_cost(self):
        """With no cached rate, subset cached tokens stay at full input_cost (no-op)."""
        llm = self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
        )

        input_cost, _, _ = llm._calculate_costs(
            input_tokens=1000,
            output_tokens=0,
            cached_tokens=400,
        )

        assert input_cost == 1000 * 3.0 / TOKENS_PER_MILLION

    def test_additive_accounting_adds_cache_read_and_write(self):
        """Additive providers add cache read/write on top of uncached input."""

        class AdditiveLLM(self.ConcreteLLM):
            _cache_accounting = "additive"

        llm = AdditiveLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
            cached_input_cost=0.3,
            cache_write_cost=3.75,
        )

        input_cost, _, total_cost = llm._calculate_costs(
            input_tokens=1000,
            output_tokens=0,
            cached_tokens=200,
            cache_creation_tokens=300,
        )

        # input_tokens are NOT reduced by cached/creation in additive mode.
        expected_input = (1000 * 3.0 + 200 * 0.3 + 300 * 3.75) / TOKENS_PER_MILLION
        assert input_cost == expected_input
        assert total_cost == expected_input

    def test_additive_accounting_without_rates_ignores_cache_tokens(self):
        """With no cache rates, additive cache tokens contribute nothing (prior behaviour)."""

        class AdditiveLLM(self.ConcreteLLM):
            _cache_accounting = "additive"

        llm = AdditiveLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
        )

        input_cost, _, _ = llm._calculate_costs(
            input_tokens=1000,
            output_tokens=0,
            cached_tokens=200,
            cache_creation_tokens=300,
        )

        assert input_cost == 1000 * 3.0 / TOKENS_PER_MILLION


class TestLLMFullModelName:
    """Tests for LLM.get_full_model_name method."""

    class ConcreteLLM(LLM):
        """Concrete implementation for testing."""

        async def _get_response_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None, max_tokens=None,
        ):
            raise NotImplementedError()

        async def _get_response_stream_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None, max_tokens=None,
        ):
            raise NotImplementedError()

    def test_returns_provider_colon_model(self):
        """Should return 'provider:model' format."""
        llm = self.ConcreteLLM(
            provider="anthropic",
            model="claude-sonnet-5",
            input_cost=3.0,
            output_cost=15.0,
        )

        assert llm.get_full_model_name() == "anthropic:claude-sonnet-5"


class TestLLMStreamResponse:
    """Tests for LLMStreamResponse async streaming wrapper."""

    class ConcreteLLM(LLM):
        """Concrete implementation for testing abstract base class."""

        async def _get_response_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None, max_tokens=None,
        ):
            raise NotImplementedError()

        async def _get_response_stream_impl(
            self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
            extra_headers=None, max_tokens=None,
        ):
            raise NotImplementedError()

    @staticmethod
    async def _mock_stream():
        yield "Hello"
        yield " "
        yield "world"

    def _make_llm(self):
        return self.ConcreteLLM(
            provider="test",
            model="test-model",
            input_cost=3.0,
            output_cost=15.0,
        )

    def _make_stream_response(self, stream=None):
        llm = self._make_llm()
        state = _StreamState(
            input_tokens=10,
            output_tokens=5,
            cached_tokens=0,
            start_time=time.time(),
        )
        if stream is None:
            stream = self._mock_stream()
        return LLMStreamResponse(stream=stream, state=state, llm=llm)

    @pytest.mark.asyncio
    async def test_iterating_yields_chunks(self):
        """Iterating over the stream should yield each chunk in order."""
        stream = self._make_stream_response()
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert chunks == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_usage_populated_after_iteration(self):
        """Usage should be populated with correct values after consuming the stream."""
        stream = self._make_stream_response()
        async for _ in stream:
            pass

        assert stream.usage is not None
        assert stream.usage.input_tokens == 10
        assert stream.usage.output_tokens == 5
        assert stream.usage.cached_tokens == 0
        assert stream.usage.input_cost == 10 * 3.0 / TOKENS_PER_MILLION
        assert stream.usage.output_cost == 5 * 15.0 / TOKENS_PER_MILLION
        assert stream.usage.total_cost == stream.usage.input_cost + stream.usage.output_cost

    @pytest.mark.asyncio
    async def test_usage_is_none_before_consumption(self):
        """Usage should be None before the stream is consumed."""
        stream = self._make_stream_response()

        assert stream.usage is None

    @pytest.mark.asyncio
    async def test_collect_returns_llm_response(self):
        """collect() should return an LLMResponse with correct content and usage."""
        stream = self._make_stream_response()
        response = await stream.collect()

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello world"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.cached_tokens == 0
        assert response.input_cost == 10 * 3.0 / TOKENS_PER_MILLION
        assert response.output_cost == 5 * 15.0 / TOKENS_PER_MILLION
        assert response.total_cost == response.input_cost + response.output_cost

    @pytest.mark.asyncio
    async def test_on_complete_callback_fires(self):
        """_on_complete callback should be called with Usage and content after iteration."""
        stream = self._make_stream_response()
        callback_args = {}

        def on_complete(usage, content):
            callback_args["usage"] = usage
            callback_args["content"] = content

        stream._on_complete = on_complete

        async for _ in stream:
            pass

        assert "usage" in callback_args
        assert "content" in callback_args
        assert isinstance(callback_args["usage"], Usage)
        assert callback_args["usage"].input_tokens == 10
        assert callback_args["usage"].output_tokens == 5
        assert callback_args["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_on_error_callback_fires(self):
        """_on_error callback should be called when the stream raises an exception."""

        async def failing_stream():
            yield "partial"
            raise RuntimeError("stream failed")

        stream = self._make_stream_response(stream=failing_stream())
        error_args = {}

        def on_error(exc):
            error_args["exception"] = exc

        stream._on_error = on_error

        with pytest.raises(RuntimeError, match="stream failed"):
            async for _ in stream:
                pass

        assert "exception" in error_args
        assert isinstance(error_args["exception"], RuntimeError)
        assert str(error_args["exception"]) == "stream failed"


class _RecordingLLM(LLM):
    """Test double that returns a canned LLMResponse and records prompts."""

    def __init__(self, content: str = "response", **kwargs):
        super().__init__(
            provider="test", model="test-model", input_cost=1.0, output_cost=2.0,
            **kwargs,
        )
        self.canned_content = content
        self.calls: list[str] = []
        self.schema_calls: list[str] = []

    async def _get_response_impl(
        self, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0,
        extra_headers=None, max_tokens=None,
    ):
        self.calls.append(user_prompt)
        return LLMResponse(
            content=self.canned_content,
            input_tokens=10, output_tokens=20, cached_tokens=0,
            input_cost=0.01, output_cost=0.02, total_cost=0.03,
            response_time=0.1,
        )

    async def _get_response_stream_impl(self, *args, **kwargs):
        raise NotImplementedError()

    async def _get_json_schema_response(
        self, user_prompt, response_schema, system_prompt=None,
        schema_name="Response", schema_description=None,
        temperature=0.3, top_p=1.0, extra_headers=None, max_tokens=None,
    ):
        self.schema_calls.append(user_prompt)
        return LLMResponse(
            content=self.canned_content,
            input_tokens=10, output_tokens=20, cached_tokens=0,
            input_cost=0.01, output_cost=0.02, total_cost=0.03,
            response_time=0.1,
        )


class TestLLMWithoutHooks:
    """Regression guard: an LLM with no hook_pipeline behaves identically to today."""

    @pytest.mark.asyncio
    async def test_get_response_passes_through(self):
        llm = _RecordingLLM(content="hello")
        response = await llm.get_response("prompt")
        assert response.content == "hello"
        assert response.input_tokens == 10
        assert llm.calls == ["prompt"]

    @pytest.mark.asyncio
    async def test_get_json_schema_response_passes_through(self):
        llm = _RecordingLLM(content='{"name":"x","population":1}')
        response = await llm.get_json_schema_response(
            user_prompt="prompt", response_schema=COUNTRY_SCHEMA,
        )
        assert response.content == '{"name":"x","population":1}'
        assert llm.schema_calls == ["prompt"]


class TestLLMWithHooks:
    """Hooks attached to the LLM base class wrap text-producing calls."""

    @pytest.mark.asyncio
    async def test_redact_in_after_replaces_content_preserving_usage(self):
        from majordomo_llm import HookOutcome, HookPipeline

        class Hook:
            name = "redactor"

            async def before_call(self, prompt, ctx):
                return HookOutcome.pass_through(self.name)

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.redact(self.name, "REDACTED", "test")

        pipeline = HookPipeline([Hook()])
        llm = _RecordingLLM(content="secret stuff", hook_pipeline=pipeline)
        response = await llm.get_response("prompt")
        assert response.content == "REDACTED"
        # usage preserved
        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.total_cost == 0.03

    @pytest.mark.asyncio
    async def test_block_in_before_prevents_impl_call(self):
        from majordomo_llm import HookBlocked, HookOutcome, HookPipeline

        class Hook:
            name = "blocker"

            async def before_call(self, prompt, ctx):
                return HookOutcome.block(self.name, "no")

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.pass_through(self.name)

        pipeline = HookPipeline([Hook()])
        llm = _RecordingLLM(hook_pipeline=pipeline)
        with pytest.raises(HookBlocked):
            await llm.get_response("prompt")
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_caller_metadata_propagates_to_hook(self):
        from majordomo_llm import HookContext, HookOutcome, HookPipeline

        seen: list[HookContext] = []

        class Hook:
            name = "spy"

            async def before_call(self, prompt, ctx):
                seen.append(ctx)
                return HookOutcome.pass_through(self.name)

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.pass_through(self.name)

        pipeline = HookPipeline([Hook()])
        llm = _RecordingLLM(hook_pipeline=pipeline)
        await llm.get_response("prompt", caller_metadata={"feature": "drafting"})
        assert seen[0].caller_metadata == {"feature": "drafting"}

    @pytest.mark.asyncio
    async def test_get_json_response_runs_hooks_through_get_response(self):
        from majordomo_llm import HookOutcome, HookPipeline

        class Hook:
            name = "rewriter"

            async def before_call(self, prompt, ctx):
                return HookOutcome.pass_through(self.name)

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.redact(self.name, '{"final": true}', "rewrite")

        pipeline = HookPipeline([Hook()])
        llm = _RecordingLLM(content='{"initial": true}', hook_pipeline=pipeline)
        response = await llm.get_json_response("prompt")
        assert response.content == {"final": True}

    @pytest.mark.asyncio
    async def test_get_json_schema_response_runs_hooks(self):
        from majordomo_llm import HookOutcome, HookPipeline

        class Hook:
            name = "rewriter"

            async def before_call(self, prompt, ctx):
                return HookOutcome.pass_through(self.name)

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.redact(
                    self.name, '{"name":"y","population":2}', "rewrite"
                )

        pipeline = HookPipeline([Hook()])
        llm = _RecordingLLM(
            content='{"name":"x","population":1}', hook_pipeline=pipeline,
        )
        response = await llm.get_json_schema_response(
            user_prompt="prompt", response_schema=COUNTRY_SCHEMA,
        )
        assert response.content == '{"name":"y","population":2}'


class _CapRecordingLLM(LLM):
    """Test double that records the cap each call resolved to."""

    def __init__(self, **kwargs):
        super().__init__(
            provider="test", model="test-model", input_cost=1.0, output_cost=2.0,
            **kwargs,
        )
        self.resolved: list[int] = []

    async def _get_response_impl(
        self, user_prompt, system_prompt=None, temperature=None, top_p=None,
        extra_headers=None, max_tokens=None,
    ):
        self.resolved.append(self._resolve_max_tokens(max_tokens))
        return LLMResponse(
            # Valid JSON so the same double serves get_json_response.
            content='{"ok": true}',
            input_tokens=10, output_tokens=20, cached_tokens=0,
            input_cost=0.01, output_cost=0.02, total_cost=0.03,
            response_time=0.1, stop_reason="end_turn",
        )

    async def _get_response_stream_impl(
        self, user_prompt, system_prompt=None, temperature=None, top_p=None,
        extra_headers=None, max_tokens=None,
    ):
        self.resolved.append(self._resolve_max_tokens(max_tokens, streaming=True))
        raise NotImplementedError()


class TestResolveMaxTokens:
    """Tests for the single place the output cap is decided."""

    def test_falls_back_to_library_default(self):
        llm = _CapRecordingLLM()
        assert llm._resolve_max_tokens(None) == DEFAULT_MAX_TOKENS

    def test_streaming_gets_the_larger_default(self):
        llm = _CapRecordingLLM()
        assert llm._resolve_max_tokens(None, streaming=True) == DEFAULT_STREAM_MAX_TOKENS

    def test_config_value_beats_both_defaults(self):
        llm = _CapRecordingLLM(max_tokens=8192)
        assert llm._resolve_max_tokens(None) == 8192
        assert llm._resolve_max_tokens(None, streaming=True) == 8192

    def test_per_request_value_beats_config(self):
        llm = _CapRecordingLLM(max_tokens=8192)
        assert llm._resolve_max_tokens(2048) == 2048

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_non_positive(self, bad):
        llm = _CapRecordingLLM()
        with pytest.raises(ValueError, match="positive integer"):
            llm._resolve_max_tokens(bad)

    @pytest.mark.asyncio
    async def test_public_method_threads_the_override(self):
        llm = _CapRecordingLLM(max_tokens=8192)
        await llm.get_response("prompt", max_tokens=1234)
        assert llm.resolved == [1234]

    @pytest.mark.asyncio
    async def test_get_json_response_threads_the_override(self):
        llm = _CapRecordingLLM()
        await llm.get_json_response('{"a": 1}', max_tokens=4321)
        assert llm.resolved == [4321]


class TestStopReasonPropagation:
    """stop_reason must survive every path that rebuilds an LLMResponse."""

    @pytest.mark.asyncio
    async def test_survives_plain_response(self):
        llm = _CapRecordingLLM()
        response = await llm.get_response("prompt")
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_survives_hook_rewrite(self):
        """The rewrite path rebuilds the response; it must not drop fields."""
        from majordomo_llm import HookOutcome, HookPipeline

        class Hook:
            name = "redactor"

            async def before_call(self, prompt, ctx):
                return HookOutcome.pass_through(self.name)

            async def after_call(self, prompt, response, ctx):
                return HookOutcome.redact(self.name, "REDACTED", "test")

        llm = _CapRecordingLLM(hook_pipeline=HookPipeline([Hook()]))
        response = await llm.get_response("prompt")

        assert response.content == "REDACTED"
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_survives_stream_collect(self):
        state = _StreamState(input_tokens=5, output_tokens=7, stop_reason="end_turn")

        async def chunks():
            yield "hi"

        llm = _CapRecordingLLM()
        stream = LLMStreamResponse(stream=chunks(), state=state, llm=llm)
        response = await stream.collect()

        assert stream.stop_reason == "end_turn"
        assert response.stop_reason == "end_turn"

    def test_defaults_to_none(self):
        """Providers that report no stop reason leave the field empty."""
        response = LLMResponse(
            content="x", input_tokens=1, output_tokens=1, cached_tokens=0,
            input_cost=0.0, output_cost=0.0, total_cost=0.0, response_time=0.1,
        )
        assert response.stop_reason is None
