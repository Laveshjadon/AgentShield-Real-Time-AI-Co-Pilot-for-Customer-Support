"""Creates the initial AgentShield tables and enables pgvector.

Revision ID: 05be1278d5d5
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector


revision: str = "05be1278d5d5"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all AgentShield tables and enable pgvector."""

    
    
    
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    
    
    
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_file", sa.String(length=500), nullable=True),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=True),
        sa.Column(
            "embedding",
            Vector(384),          
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding_cosine
        ON knowledge_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )

    
    
    
    op.create_table(
        "call_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        
        sa.Column("agent_id", sa.String(length=50), nullable=False),
        sa.Column("call_id", sa.String(length=100), nullable=True),
        
        sa.Column("transcript", sa.Text(), nullable=True),
        sa.Column("transcript_redacted", sa.Text(), nullable=True),
        
        sa.Column("toxicity_score", sa.Float(), nullable=True),
        sa.Column("toxicity_label", sa.String(length=20), nullable=True),
        sa.Column("is_abusive", sa.Boolean(), nullable=True),
        sa.Column("toxicity_explanation", sa.Text(), nullable=True),
        
        sa.Column("base_score", sa.Float(), nullable=True),
        sa.Column("adjusted_score", sa.Float(), nullable=True),
        sa.Column("aggressive_call_flag", sa.Boolean(), nullable=True),
        
        sa.Column("avg_pitch", sa.Float(), nullable=True),
        sa.Column("avg_loudness", sa.Float(), nullable=True),
        sa.Column("speaking_rate_wpm", sa.Float(), nullable=True),
        
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("language", sa.String(length=10), nullable=True),
        
        sa.Column(
            "started_at",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    
    op.create_index(
        "ix_call_logs_agent_id",
        "call_logs",
        ["agent_id"],
        unique=False,
    )
    
    op.create_index(
        "ix_call_logs_started_at",
        "call_logs",
        ["started_at"],
        unique=False,
    )

    
    
    
    op.create_table(
        "agent_wellness",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        
        sa.Column("agent_id", sa.String(length=50), nullable=False),
        
        sa.Column("stress_score", sa.Float(), nullable=True),
        sa.Column("total_calls", sa.Integer(), nullable=True),
        sa.Column("toxic_calls", sa.Integer(), nullable=True),
        sa.Column("consecutive_toxic", sa.Integer(), nullable=True),
        
        sa.Column("break_recommended", sa.Boolean(), nullable=True),
        sa.Column("break_duration_minutes", sa.Integer(), nullable=True),
        sa.Column("break_reason", sa.String(length=200), nullable=True),
        
        sa.Column(
            "shift_date",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "last_updated",
            sa.DateTime(),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_agent_wellness_agent_id",
        "agent_wellness",
        ["agent_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop all AgentShield tables in reverse dependency order."""
    op.drop_index("ix_agent_wellness_agent_id", table_name="agent_wellness")
    op.drop_table("agent_wellness")

    op.drop_index("ix_call_logs_started_at", table_name="call_logs")
    op.drop_index("ix_call_logs_agent_id", table_name="call_logs")
    op.drop_table("call_logs")

    op.drop_index(
        "ix_knowledge_chunks_embedding_cosine",
        table_name="knowledge_chunks",
    )
    op.drop_table("knowledge_chunks")

    
    op.execute("DROP EXTENSION IF EXISTS vector")
