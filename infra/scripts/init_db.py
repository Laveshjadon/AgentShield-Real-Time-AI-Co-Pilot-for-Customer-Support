"""
AgentShield - Database Initialization Script
just run this one time before starting the app to setup the db

run it like:
    python -m scripts.init_db
"""
import asyncio
from sqlalchemy import text
from src.core.db import get_engine, get_async_engine
from src.core.db import Base
from config.logger import get_logger

logger = get_logger("scripts.init_db")


def enable_pgvector(engine):
    """turns on pgvector so we can do the ai stuff, sync because it's easier"""
    logger.info("Enabling pgvector extension...")
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    logger.info("pgvector extension enabled!")


async def create_tables_async():  
    """Create all tables with the asynchronous database engine."""
    logger.info("Creating database tables (async)...")
    async_engine = get_async_engine()
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Tables created successfully!")


def verify_tables(engine):
    """just double checking everything actually got created"""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename")
        )
        tables = [row[0] for row in result.fetchall()]
    logger.info("Tables in database: %s", tables)

    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'knowledge_chunks' AND column_name = 'embedding'"
            )
        )
        row = result.fetchone()
        if row:
            logger.info("Vector column 'embedding' verified!")
        else:
            logger.warning("Vector column 'embedding' NOT found - check models.py")
    return tables


def init_database():
    engine = get_engine()
    enable_pgvector(engine)
    asyncio.run(create_tables_async())
    tables = verify_tables(engine)

    print()
    print("=" * 50)
    print("  Database initialization complete!")
    print("=" * 50)
    for table in tables:
        print(f"    - {table}")
    print()
    print("  Run the app with: python main.py")
    print("=" * 50)


if __name__ == "__main__":
    init_database()
