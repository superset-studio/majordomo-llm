"""Logging wrapper for LLM instances."""

import asyncio
from datetime import UTC, datetime
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from majordomo_llm.base import (
    LLM,
    LLMJSONResponse,
    LLMResponse,
    LLMStreamResponse,
    LLMStructuredResponse,
    Usage,
)
from majordomo_llm.logging.interfaces import DatabaseAdapter, StorageAdapter
from majordomo_llm.logging.models import LogEntry

T = TypeVar("T", bound=BaseModel)


class LoggingLLM(LLM):
    """Wrapper that adds logging to any LLM instance.

    Logs all requests asynchronously (fire-and-forget) without blocking
    the main request flow. Stores metrics in a database and optionally
    stores request/response bodies in S3.

    Example:
        >>> from majordomo_llm import get_llm_instance
        >>> from majordomo_llm.logging import LoggingLLM, PostgresAdapter, S3Adapter
        >>>
        >>> llm = get_llm_instance("anthropic", "claude-sonnet-5")
        >>> db = await PostgresAdapter.create(host="localhost", ...)
        >>> storage = await S3Adapter.create(bucket="my-bucket")
        >>> logged_llm = LoggingLLM(llm, db, storage)
        >>>
        >>> response = await logged_llm.get_response("Hello!")
    """

    def __init__(
        self,
        llm: LLM,
        database: DatabaseAdapter,
        storage: StorageAdapter | None = None,
    ) -> None:
        """Initialize the logging wrapper.

        Args:
            llm: The LLM instance to wrap.
            database: Database adapter for storing metrics.
            storage: Optional storage adapter for request/response bodies.
        """
        super().__init__(
            provider=llm.provider,
            model=llm.model,
            input_cost=llm.input_cost,
            output_cost=llm.output_cost,
            supports_temperature_top_p=llm.supports_temperature_top_p,
            use_web_search=llm.use_web_search,
        )
        self._llm = llm
        self._database = database
        self._storage = storage
        self._pending_tasks: set[asyncio.Task[None]] = set()

    async def _get_response_impl(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Unused — :meth:`get_response` overrides the public method directly."""
        raise NotImplementedError(
            "LoggingLLM logs at the public method layer; _get_response_impl "
            "is never called"
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
        """Unused — :meth:`get_response_stream` overrides the public method directly."""
        raise NotImplementedError(
            "LoggingLLM logs at the public method layer; _get_response_stream_impl "
            "is never called"
        )

    async def _log_request(
        self,
        request_body: dict[str, Any],
        response_content: str | dict[str, Any] | None,
        response: Usage | None,
        status: str,
        error_message: str | None,
    ) -> None:
        """Log a request (internal, runs as fire-and-forget task)."""
        request_id = uuid4()
        s3_request_key: str | None = None
        s3_response_key: str | None = None

        if self._storage:
            s3_request_key, s3_response_key = await self._storage.upload(
                request_id, request_body, response_content
            )

        entry = LogEntry(
            request_id=request_id,
            provider=self.provider,
            model=self.model,
            timestamp=datetime.now(UTC),
            response_time=response.response_time if response else None,
            input_tokens=response.input_tokens if response else None,
            output_tokens=response.output_tokens if response else None,
            cached_tokens=response.cached_tokens if response else None,
            cache_creation_tokens=response.cache_creation_tokens if response else None,
            input_cost=response.input_cost if response else None,
            output_cost=response.output_cost if response else None,
            total_cost=response.total_cost if response else None,
            s3_request_key=s3_request_key,
            s3_response_key=s3_response_key,
            status=status,
            error_message=error_message,
            api_key_hash=self._llm.api_key_hash,
            api_key_alias=self._llm.api_key_alias,
        )

        await self._database.insert(entry)

    def _fire_and_forget(
        self,
        request_body: dict[str, Any],
        response_content: str | dict[str, Any] | None,
        response: Usage | None,
        status: str,
        error_message: str | None,
    ) -> None:
        """Schedule logging as a background task."""
        task = asyncio.create_task(
            self._log_request(
                request_body=request_body,
                response_content=response_content,
                response=response,
                status=status,
                error_message=error_message,
            )
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

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
    ) -> LLMResponse:
        """Get a plain text response from the LLM with logging."""
        request_body = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        try:
            response = await self._llm.get_response(
                user_prompt, system_prompt, temperature, top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
                caller_metadata=caller_metadata,
            )
            self._fire_and_forget(
                request_body=request_body,
                response_content=response.content,
                response=response,
                status="success",
                error_message=None,
            )
            return response
        except Exception as e:
            self._fire_and_forget(
                request_body=request_body,
                response_content=None,
                response=None,
                status="error",
                error_message=str(e),
            )
            raise

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
    ) -> LLMStreamResponse:
        """Get a streaming text response from the LLM with logging."""
        del caller_metadata
        request_body = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        try:
            stream = await self._llm.get_response_stream(
                user_prompt, system_prompt, temperature, top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
            )
        except Exception as e:
            self._fire_and_forget(
                request_body=request_body,
                response_content=None,
                response=None,
                status="error",
                error_message=str(e),
            )
            raise

        def on_complete(usage: Usage, content: str) -> None:
            self._fire_and_forget(
                request_body=request_body,
                response_content=content,
                response=usage,
                status="success",
                error_message=None,
            )

        def on_error(error: Exception) -> None:
            self._fire_and_forget(
                request_body=request_body,
                response_content=None,
                response=None,
                status="error",
                error_message=str(error),
            )

        stream._on_complete = on_complete
        stream._on_error = on_error
        return stream

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
    ) -> LLMJSONResponse:
        """Get a JSON response from the LLM with logging."""
        request_body = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        try:
            response = await self._llm.get_json_response(
                user_prompt, system_prompt, temperature, top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
                caller_metadata=caller_metadata,
            )
            self._fire_and_forget(
                request_body=request_body,
                response_content=response.content,
                response=response,
                status="success",
                error_message=None,
            )
            return response
        except Exception as e:
            self._fire_and_forget(
                request_body=request_body,
                response_content=None,
                response=None,
                status="error",
                error_message=str(e),
            )
            raise

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
    ) -> LLMStructuredResponse:
        """Get a structured response validated against a Pydantic model with logging."""
        request_body = {
            "response_model": response_model.__name__,
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        try:
            response = await self._llm.get_structured_json_response(
                response_model, user_prompt, system_prompt, temperature, top_p,
                extra_headers=extra_headers,
                max_tokens=max_tokens,
                caller_metadata=caller_metadata,
            )
            self._fire_and_forget(
                request_body=request_body,
                response_content=response.content.model_dump(),
                response=response,
                status="success",
                error_message=None,
            )
            return response
        except Exception as e:
            self._fire_and_forget(
                request_body=request_body,
                response_content=None,
                response=None,
                status="error",
                error_message=str(e),
            )
            raise

    async def flush(self) -> None:
        """Wait for all pending logging tasks to complete."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)

    async def close(self) -> None:
        """Wait for pending tasks and close database and storage connections."""
        await self.flush()
        await self._database.close()
        if self._storage:
            await self._storage.close()
