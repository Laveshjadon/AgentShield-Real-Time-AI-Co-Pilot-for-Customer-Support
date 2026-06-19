from __future__ import annotations
# --- FROM: backend/db/models.py ---
"""Defines the knowledge, call log, and agent wellness database tables."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    func,
)
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector


Base = declarative_base()


class KnowledgeChunk(Base):
    """
    holding all the text chunks and vectors.
    this is what the rag pipeline uses to look up answers.
    """

    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    
    content = Column(Text, nullable=False)

    
    source_file = Column(String(500), nullable=True)     
    document_type = Column(String(50), nullable=True)    
    category = Column(String(100), nullable=True)        
    chunk_index = Column(Integer, nullable=True)         
    page_number = Column(Integer, nullable=True)         

    
    embedding = Column(Vector(384), nullable=True)

    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        preview = self.content[:50] if self.content else ""
        return f"<KnowledgeChunk(id={self.id}, source='{self.source_file}', preview='{preview}...')>"


class CallLog(Base):
    """
    logging all the calls in here.
    keeps the transcript and also if the caller was being a jerk.
    """

    __tablename__ = "call_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    
    agent_id = Column(String(50), nullable=False)        
    call_id = Column(String(100), nullable=True)         

    
    transcript = Column(Text, nullable=True)             
    transcript_redacted = Column(Text, nullable=True)    

    
    toxicity_score = Column(Float, default=0.0)          
    toxicity_label = Column(String(20), default="CLEAN") 
    is_abusive = Column(Boolean, default=False)
    toxicity_explanation = Column(Text, nullable=True)    

    
    base_score = Column(Float, nullable=True)
    adjusted_score = Column(Float, nullable=True)
    aggressive_call_flag = Column(Boolean, default=False)

    
    avg_pitch = Column(Float, nullable=True)
    avg_loudness = Column(Float, nullable=True)
    speaking_rate_wpm = Column(Float, nullable=True)

    
    duration_seconds = Column(Integer, nullable=True)
    language = Column(String(10), default="en")          

    
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<CallLog(id={self.id}, agent='{self.agent_id}', toxicity={self.toxicity_label})>"


class AgentWellness(Base):
    """
    tracking how stressed the agents are.
    use this to tell them to take a break so they don't burn out.
    """

    __tablename__ = "agent_wellness"

    id = Column(Integer, primary_key=True, autoincrement=True)

    
    agent_id = Column(String(50), nullable=False)

    
    stress_score = Column(Float, default=0.0)            
    total_calls = Column(Integer, default=0)
    toxic_calls = Column(Integer, default=0)
    consecutive_toxic = Column(Integer, default=0)       

    
    break_recommended = Column(Boolean, default=False)
    break_duration_minutes = Column(Integer, default=0)  
    break_reason = Column(String(200), nullable=True)

    
    shift_date = Column(DateTime, server_default=func.now())
    last_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<AgentWellness(agent='{self.agent_id}', stress={self.stress_score}, toxic_calls={self.toxic_calls})>"


# --- FROM: backend/db/connection.py ---
"""Sets up async database access for FastAPI and sync access for scripts."""



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


