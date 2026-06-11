"""
handles all the redis database stuff for the call sessions.

basically:
- session hash: stores the main info
- turns list: stores all the text dialogue
- active set: tracks who is currently on a call

we keep things around for 2 hours (ttl) just in case we need to look at it
after the call drops. making sure everything is atomic so we don't overwrite stuff.
"""

from __future__ import annotations

import json
import time
from typing import List, Optional

from fastapi import Depends, HTTPException
from redis.asyncio import Redis

from config.logger import get_logger
from session.client import get_redis
from session.schemas import (
    ActiveSessionSummary,
    SessionRecord,
    TurnRecord,
)

logger = get_logger("session.manager")


SESSION_TTL_SECONDS = 7_200          
_PREFIX = "agentshield"


def _session_key(session_id: str) -> str:
    return f"{_PREFIX}:session:{session_id}"


def _turns_key(session_id: str) -> str:
    return f"{_PREFIX}:session:{session_id}:turns"


def _active_set_key() -> str:
    return f"{_PREFIX}:active_sessions"


class SessionManager:
    """the main class for saving and loading call data from redis"""

    def __init__(self, redis: Redis) -> None:
        self._r = redis

    
    
    

    async def create_session(self, session_id: str, agent_id: str) -> SessionRecord:
        """starts a new call session, throws an error if that id is already used"""
        hash_key = _session_key(session_id)

        
        created = await self._r.hsetnx(hash_key, "session_id", session_id)
        if not created:
            raise HTTPException(
                status_code=409,
                detail=f"Session '{session_id}' already exists.",
            )

        record = SessionRecord(session_id=session_id, agent_id=agent_id)
        mapping = record.to_redis_hash()
        
        mapping.pop("session_id", None)

        pipe = self._r.pipeline()
        pipe.hset(hash_key, mapping=mapping)
        pipe.expire(hash_key, SESSION_TTL_SECONDS)
        pipe.expire(_turns_key(session_id), SESSION_TTL_SECONDS)
        pipe.sadd(_active_set_key(), session_id)
        await pipe.execute()

        logger.info("[SessionManager] Created session %s for agent %s", session_id, agent_id)
        return record

    
    
    

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """just gets the session info, returns none if missing"""
        raw = await self._r.hgetall(_session_key(session_id))
        if not raw:
            return None
        return SessionRecord.from_redis_hash(raw)

    async def require_session(self, session_id: str) -> SessionRecord:
        """same as above but throws a 404 error instead of none"""
        session = await self.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found. Start a call first.",
            )
        return session

    async def session_exists(self, session_id: str) -> bool:
        """quick check to see if a session is in the active list"""
        return bool(await self._r.sismember(_active_set_key(), session_id))

    async def get_turns(
        self,
        session_id: str,
        max_turns: int = 50,
    ) -> List[TurnRecord]:
        """grabs the last few messages from the chat"""
        raw_list = await self._r.lrange(
            _turns_key(session_id),
            -max_turns,   
            -1,
        )
        turns = []
        for raw in raw_list:
            try:
                turns.append(TurnRecord.model_validate(json.loads(raw)))
            except Exception as exc:
                logger.warning("[SessionManager] Failed to parse turn: %s", exc)
        return turns

    async def list_active_sessions(self) -> List[ActiveSessionSummary]:
        """gets a list of all the calls happening right now for the dashboard"""
        session_ids = await self._r.smembers(_active_set_key())
        if not session_ids:
            return []

        pipe = self._r.pipeline()
        for sid in session_ids:
            pipe.hgetall(_session_key(sid))
        results = await pipe.execute()

        summaries = []
        for raw in results:
            if not raw:
                continue
            try:
                rec = SessionRecord.from_redis_hash(raw)
                summaries.append(
                    ActiveSessionSummary(
                        session_id=rec.session_id,
                        agent_id=rec.agent_id,
                        is_active=rec.is_active,
                        turn_count=rec.turn_count,
                        start_time=rec.start_time,
                    )
                )
            except Exception as exc:
                logger.warning("[SessionManager] Failed to parse session summary: %s", exc)
        return summaries

    
    
    

    async def add_turn(
        self,
        session_id: str,
        turn: TurnRecord,
    ) -> int:
        """adds a new message to the chat list and updates the count"""
        hash_key = _session_key(session_id)
        turns_key = _turns_key(session_id)
        serialised = json.dumps(turn.model_dump())

        pipe = self._r.pipeline()
        pipe.rpush(turns_key, serialised)
        pipe.hincrby(hash_key, "turn_count", 1)

        
        if turn.toxicity_score > 0:
            pipe.hset(
                hash_key,
                mapping={
                    "peak_toxicity_score": str(turn.toxicity_score),
                    "peak_toxicity_level": turn.toxicity_level,
                },
            )

        pipe.expire(hash_key, SESSION_TTL_SECONDS)
        pipe.expire(turns_key, SESSION_TTL_SECONDS)
        results = await pipe.execute()

        
        new_count = results[0]
        logger.debug(
            "[SessionManager] Turn added to %s (%s turns total)", session_id, new_count
        )
        return new_count

    async def end_session(self, session_id: str) -> SessionRecord:
        """closes the call but leaves the data in redis for a while so we can read it"""
        hash_key = _session_key(session_id)

        pipe = self._r.pipeline()
        pipe.hset(hash_key, "is_active", "0")
        pipe.srem(_active_set_key(), session_id)
        await pipe.execute()

        
        raw = await self._r.hgetall(hash_key)
        record = SessionRecord.from_redis_hash(raw)
        logger.info("[SessionManager] Session %s ended.", session_id)
        return record

    async def delete_session(self, session_id: str) -> None:
        """completely wipes the session from redis, mostly for testing"""
        pipe = self._r.pipeline()
        pipe.delete(_session_key(session_id))
        pipe.delete(_turns_key(session_id))
        pipe.srem(_active_set_key(), session_id)
        await pipe.execute()
        logger.info("[SessionManager] Session %s deleted.", session_id)

    async def get_transcript(
        self,
        session_id: str,
        max_turns: int = 50,
    ) -> str:
        """puts all the chat messages together into one big string for the ai"""
        turns = await self.get_turns(session_id, max_turns=max_turns)
        return "\n".join(
            f"{t.speaker.capitalize()}: {t.text}" for t in turns
        )






def get_session_manager(redis: Redis = Depends(get_redis)) -> SessionManager:
    """just a helper for fastapi routes to get the session manager"""
    return SessionManager(redis=redis)
