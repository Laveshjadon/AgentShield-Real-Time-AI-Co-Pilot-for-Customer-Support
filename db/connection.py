"""Sets up async database access for FastAPI and sync access for scripts."""

from __future__ import annotations

import contextlib
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)
from config.settings import Settings
from config.logger import get_logger

logger = get_logger("db.connection")




_sync_engine = None          
_sync_session_factory = None

_async_engine: AsyncEngine | None = None

_async_session_factory: async_sessionmaker[AsyncSession] | None = None






def get_engine():
    """Return the sync SQLAlchemy engine, creating it on first use.

    FastAPI routes should use get_db() instead.
    """
    global _sync_engine
    if _sync_engine is None:
        settings = Settings()
        logger.info(
            "Connecting to PostgreSQL (sync) at %s:%s/%s",
            settings.POSTGRES_HOST,
            settings.POSTGRES_PORT,
            settings.POSTGRES_DB,
        )
        _sync_engine = create_engine(
            settings.DATABASE_URL,          
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
            echo=False,
        )
        try:
            with _sync_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Sync database connection verified.")
        except Exception as exc:
            logger.error("Sync database connection failed: %s", exc)
            raise
    return _sync_engine


def get_session() -> Session:
    """Return a new sync session for scripts."""
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _sync_session_factory()






def _build_async_engine() -> AsyncEngine:
    """builds the async engine with asyncpg"""
    settings = Settings()
    logger.info(
        "Initialising async PostgreSQL engine (asyncpg) at %s:%s/%s",
        settings.POSTGRES_HOST,
        settings.POSTGRES_PORT,
        settings.POSTGRES_DB,
    )
    return create_async_engine(
        settings.ASYNC_DATABASE_URL,   
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        echo=False,
    )


def get_async_engine() -> AsyncEngine:
    """Return the async engine, creating it on first use."""
    global _async_engine
    if _async_engine is None:
        _async_engine = _build_async_engine()
    return _async_engine


def _get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _async_session_factory






async def get_db():
    """this is what fastapi uses to get a db session for each request.

    example:

        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(MyModel))
            ...

    Handles closing and rollback for each request.
    """
    session_factory = _get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()






def test_connection() -> bool:
    """tests if the db is up and makes sure pgvector is installed.

    can run it with: ``python -m db.connection``
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        logger.info("PostgreSQL version: %s", version[:80])
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    return True


if __name__ == "__main__":
    test_connection()
