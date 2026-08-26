from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings

_postgres_saver: AsyncPostgresSaver | None = None


async def get_async_postgres_saver() -> AsyncPostgresSaver:
    global _postgres_saver

    if _postgres_saver is None:
        _postgres_saver = AsyncPostgresSaver.from_conn_string(settings.database_url)
        await _postgres_saver.setup()
    return _postgres_saver


async def ensure_checkpointer_ready() -> AsyncPostgresSaver:
    return await get_async_postgres_saver()
