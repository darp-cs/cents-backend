from pathlib import Path

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_async_engine(settings.database_url, pool_pre_ping=True, future=True, connect_args=connect_args)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session


def _ensure_sqlite_database_path() -> None:
    if not settings.database_url.startswith("sqlite"):
        return

    database = make_url(settings.database_url).database
    if not database or database == ":memory:" or database.startswith("file:"):
        return

    db_path = Path(database)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        db_path.touch()


async def init_db():
    _ensure_sqlite_database_path()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
