"""Runs async Alembic migrations using settings from .env and the ORM models."""

from __future__ import annotations

import asyncio
import sys
import os

from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context





PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)




from config.settings import Settings                  # noqa: E402
from db.models import Base                            # noqa: E402




config = context.config




if config.config_file_name is not None:
    fileConfig(config.config_file_name)








target_metadata = Base.metadata




_settings = Settings()
ASYNC_DB_URL: str = _settings.ASYNC_DATABASE_URL   






def _get_context_kwargs() -> dict:
    """Common keyword arguments for both online and offline context."""
    return {
        "target_metadata": target_metadata,
        
        "compare_type": True,
        
        "compare_server_default": True,
        
        "include_schemas": True,
        
        "render_as_batch": False,
    }






def run_migrations_offline() -> None:
    """Run migrations without a live DB connection.

    Generates raw SQL that can be reviewed and applied manually.
    This is useful for environments where direct DB access is restricted.
    """
    context.configure(
        url=ASYNC_DB_URL,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **_get_context_kwargs(),
    )

    with context.begin_transaction():
        context.run_migrations()






def do_run_migrations(connection: Connection) -> None:
    """Synchronous inner function executed inside run_sync().

    SQLAlchemy's ``AsyncConnection.run_sync()`` bridges the async connection
    into a sync context that Alembic can use via its standard synchronous API.
    """
    context.configure(
        connection=connection,
        **_get_context_kwargs(),
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine, open a connection, and run migrations.

    ``NullPool`` is used because Alembic migrations are short-lived CLI
    operations — connection pooling provides no benefit and can mask errors.
    """
    connectable = create_async_engine(
        ASYNC_DB_URL,
        poolclass=pool.NullPool,
        echo=False,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migration mode.

    Detects whether there is already a running event loop (pytest-asyncio,
    Jupyter, etc.) and handles both cases safely.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        
        asyncio.run(run_async_migrations())
    else:
        
        
        loop.run_until_complete(run_async_migrations())






if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
