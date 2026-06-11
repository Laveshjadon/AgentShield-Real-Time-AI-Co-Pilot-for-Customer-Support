"""
just the pydantic stuff for what gets saved to redis.
mostly for tracking the session and the agent's stress level.

notes to self:
- gave everything a default so it doesn't crash if redis is missing a field
- times are just floats
- saving booleans as "1" or "0" cause redis is weird about it
"""

from __future__ import annotations

import time
from typing import List, Optional
from pydantic import BaseModel, Field






class TurnRecord(BaseModel):
    """one line of dialogue that gets saved to redis"""

    speaker: str                          
    text: str
    timestamp: float = Field(default_factory=time.time)
    ai_suggestion: Optional[str] = None
    toxicity_score: float = 0.0
    toxicity_level: str = "safe"


class SessionRecord(BaseModel):
    """
    basic info about the call. keeping the transcript separate
    so we don't have to load the whole text every time we update something small.
    """

    session_id: str
    agent_id: str
    start_time: float = Field(default_factory=time.time)
    is_active: bool = True
    turn_count: int = 0
    peak_toxicity_score: float = 0.0
    peak_toxicity_level: str = "safe"
    language: str = "en"

    
    model_config = {"coerce_numbers_to_str": False}

    @classmethod
    def from_redis_hash(cls, raw: dict) -> "SessionRecord":
        """
        takes what redis gives us and turns it back into this object.
        Handle strings and bytes returned by different Redis clients.
        """
        
        decoded = {
            (k.decode() if isinstance(k, bytes) else k): (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in raw.items()
        }
        for bool_field in ("is_active",):
            if bool_field in decoded:
                v = decoded[bool_field]
                if isinstance(v, str):
                    decoded[bool_field] = v.lower() not in ("false", "0", "")
        return cls.model_validate(decoded)

    def to_redis_hash(self) -> dict:
        """turns it into a dict of strings so redis doesn't complain"""
        d = self.model_dump()
        
        d["is_active"] = "1" if d["is_active"] else "0"
        return {k: str(v) for k, v in d.items()}






class WellnessRecord(BaseModel):
    """
    how stressed the agent is right now. updates after every call
    """

    agent_id: str
    wellness_score: float = 100.0      
    consecutive_toxic: int = 0
    total_calls: int = 0
    toxic_calls: int = 0
    shift_start: float = Field(default_factory=time.time)
    last_updated: float = Field(default_factory=time.time)

    @classmethod
    def from_redis_hash(cls, raw: dict) -> "WellnessRecord":
        """same as above, just making sure we don't pass bytes to pydantic"""
        decoded = {
            (k.decode() if isinstance(k, bytes) else k): (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in raw.items()
        }
        return cls.model_validate(decoded)

    def to_redis_hash(self) -> dict:
        return {k: str(v) for k, v in self.model_dump().items()}


class CallHistoryEntry(BaseModel):
    """
    record of a finished call. we just append this to the list
    """

    session_id: str
    timestamp: float = Field(default_factory=time.time)
    duration_seconds: float
    peak_toxicity_score: float
    peak_toxicity_level: str
    is_toxic: bool
    wellness_score_after: float
    customer_sentiment: str = "neutral"






class ActiveSessionSummary(BaseModel):
    session_id: str
    agent_id: str
    is_active: bool
    turn_count: int
    start_time: float


class ActiveAgentSummary(BaseModel):
    agent_id: str
    wellness_score: float
    stress_level: str
    needs_break: bool
