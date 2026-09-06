from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import aiosqlite
from sqlalchemy.engine import make_url

from app.config import settings

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:  # pragma: no cover - import path depends on installed extras.
    SqliteSaver = None  # type: ignore[assignment]
    AsyncSqliteSaver = None  # type: ignore[assignment]


_sync_checkpointer = None
_sync_connection: sqlite3.Connection | None = None
_async_checkpointer = None
_async_connection: aiosqlite.Connection | None = None
_async_lock = asyncio.Lock()


def _resolve_sqlite_path(database_url: str) -> str:
    parsed = make_url(database_url)
    if not parsed.drivername.startswith("sqlite"):
        raise RuntimeError("Only sqlite database URLs are currently supported for LangGraph checkpointing.")

    database = parsed.database or ""
    if not database.strip():
        raise RuntimeError("SQLite database URL must include a database path for checkpointing.")

    if database == ":memory:":
        return database

    path = Path(database).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


async def ensure_checkpointer_ready():
    global _async_checkpointer, _async_connection

    if _async_checkpointer is not None:
        return _async_checkpointer

    if AsyncSqliteSaver is None:
        return None

    async with _async_lock:
        if _async_checkpointer is not None:
            return _async_checkpointer

        sqlite_path = _resolve_sqlite_path(settings.database_url)
        _async_connection = await aiosqlite.connect(sqlite_path)
        _async_checkpointer = AsyncSqliteSaver(_async_connection)
        await _async_checkpointer.setup()
        return _async_checkpointer


def ensure_sync_checkpointer_ready():
    global _sync_checkpointer, _sync_connection

    if _sync_checkpointer is not None:
        return _sync_checkpointer

    if SqliteSaver is None:
        return None

    sqlite_path = _resolve_sqlite_path(settings.database_url)
    _sync_connection = sqlite3.connect(
        sqlite_path,
        check_same_thread=False,
    )
    _sync_checkpointer = SqliteSaver(_sync_connection)
    _sync_checkpointer.setup()
    return _sync_checkpointer


async def reset_checkpointer_connections() -> None:
    global _sync_checkpointer, _sync_connection, _async_checkpointer, _async_connection

    _sync_checkpointer = None
    if _sync_connection is not None:
        _sync_connection.close()
    _sync_connection = None

    _async_checkpointer = None
    if _async_connection is not None:
        await _async_connection.close()
    _async_connection = None
