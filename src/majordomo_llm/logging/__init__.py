"""Logging module for majordomo-llm.

Provides asynchronous request logging with support for:
- PostgreSQL, MySQL, and SQLite for metrics storage
- S3 or local file system for request/response body storage

Usage:
    >>> from majordomo_llm import get_llm_instance
    >>> from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter
    >>>
    >>> llm = get_llm_instance("anthropic", "claude-sonnet-5")
    >>> db = await SqliteAdapter.create("llm_logs.db")
    >>> storage = await FileStorageAdapter.create("./llm_logs")
    >>> logged_llm = LoggingLLM(llm, db, storage)
    >>>
    >>> # All requests are now logged asynchronously
    >>> response = await logged_llm.get_response("Hello!")

Note:
    Requires optional dependencies: pip install majordomo-llm[logging]
"""

from majordomo_llm.logging.adapters import (
    FileStorageAdapter,
    MySQLAdapter,
    PostgresAdapter,
    S3Adapter,
    SqliteAdapter,
)
from majordomo_llm.logging.image_wrapper import LoggingImageModel
from majordomo_llm.logging.interfaces import DatabaseAdapter, StorageAdapter
from majordomo_llm.logging.models import LogEntry
from majordomo_llm.logging.wrapper import LoggingLLM

__all__ = [
    "LoggingLLM",
    "LoggingImageModel",
    "DatabaseAdapter",
    "StorageAdapter",
    "FileStorageAdapter",
    "MySQLAdapter",
    "PostgresAdapter",
    "S3Adapter",
    "SqliteAdapter",
    "LogEntry",
]
