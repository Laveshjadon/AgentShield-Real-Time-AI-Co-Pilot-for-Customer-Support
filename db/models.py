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
