"""
so this is the wellness store. basically keeps all the agent wellness stuff in redis. 
this way if the server crashes we don't lose the scores and all the copies of the app can see them.

keys i'm using:
- agentshield:wellness:{agent_id} -> hash for the basic record
- agentshield:wellness:{agent_id}:history -> list for call history
- agentshield:active_agents -> set of all the agents we are tracking

i put a 12 hour timer on these since shifts are like 8 hours max, gives us some buffer.
The history list is bounded with LTRIM to prevent unbounded memory growth.

all the math for scores is in the wellness tracker, this is just for saving the data.
"""

from __future__ import annotations

import json
import time
from typing import List, Optional

from fastapi import Depends
from redis.asyncio import Redis

from config.logger import get_logger
from session.client import get_redis
from session.schemas import CallHistoryEntry, WellnessRecord

logger = get_logger("session.wellness_store")

WELLNESS_TTL_SECONDS = 43_200       
HISTORY_MAX_ENTRIES = 500           
_PREFIX = "agentshield"


def _wellness_key(agent_id: str) -> str:
    return f"{_PREFIX}:wellness:{agent_id}"


def _history_key(agent_id: str) -> str:
    return f"{_PREFIX}:wellness:{agent_id}:history"


def _active_agents_key() -> str:
    return f"{_PREFIX}:active_agents"


class WellnessStore:
    """
    class to handle the redis db stuff for agent wellness.
    just throw it in fastapi routes with Depends. 
    kinda easy to mock with FakeRedis for tests too.
    """

    def __init__(self, redis: Redis) -> None:
        self._r = redis

    
    
    

    async def get(self, agent_id: str) -> Optional[WellnessRecord]:
        """grabs the record from redis, returns None if it doesn't exist yet"""
        raw = await self._r.hgetall(_wellness_key(agent_id))
        if not raw:
            return None
        return WellnessRecord.from_redis_hash(raw)

    async def get_or_init(self, agent_id: str) -> WellnessRecord:
        """tries to get the record, if not there just makes a new one"""
        record = await self.get(agent_id)
        if record is None:
            record = await self._init(agent_id)
        return record

    async def get_history(
        self,
        agent_id: str,
        max_entries: int = 100,
    ) -> List[CallHistoryEntry]:
        """gets the latest calls from history. newest ones are at the end"""
        raw_list = await self._r.lrange(
            _history_key(agent_id),
            -max_entries,
            -1,
        )
        entries = []
        for raw in raw_list:
            try:
                entries.append(CallHistoryEntry.model_validate(json.loads(raw)))
            except Exception as exc:
                logger.warning("[WellnessStore] Failed to parse history entry: %s", exc)
        return entries

    async def list_all_agents(self) -> List[WellnessRecord]:
        """fetches all the agents we are tracking right now. using a pipeline to make it fast"""
        agent_ids = await self._r.smembers(_active_agents_key())
        if not agent_ids:
            return []

        pipe = self._r.pipeline()
        for aid in agent_ids:
            pipe.hgetall(_wellness_key(aid))
        results = await pipe.execute()

        records = []
        for raw in results:
            if not raw:
                continue
            try:
                records.append(WellnessRecord.from_redis_hash(raw))
            except Exception as exc:
                logger.warning("[WellnessStore] Failed to parse agent record: %s", exc)
        return records

    
    
    

    async def _init(self, agent_id: str) -> WellnessRecord:
        """sets up a totally new record in redis. mainly used by get_or_init"""
        record = WellnessRecord(agent_id=agent_id)
        await self._write_record(agent_id, record)
        await self._r.sadd(_active_agents_key(), agent_id)
        logger.info("[WellnessStore] Initialised wellness record for agent %s", agent_id)
        return record

    async def _write_record(self, agent_id: str, record: WellnessRecord) -> None:
        """saves everything in one go and resets the expiration timer"""
        mapping = record.to_redis_hash()
        mapping["last_updated"] = str(time.time())

        pipe = self._r.pipeline()
        pipe.hset(_wellness_key(agent_id), mapping=mapping)
        pipe.expire(_wellness_key(agent_id), WELLNESS_TTL_SECONDS)
        await pipe.execute()

    async def save_state(
        self,
        agent_id: str,
        *,
        wellness_score: float,
        consecutive_toxic: int,
        total_calls: int,
        toxic_calls: int,
    ) -> WellnessRecord:
        """
        saves the current tracker numbers back to redis.
        routes call this after doing their math.
        i'm only updating the changing numbers and keeping the original shift start time.
        """
        
        existing_raw = await self._r.hget(_wellness_key(agent_id), "shift_start")
        shift_start = float(existing_raw) if existing_raw else time.time()

        record = WellnessRecord(
            agent_id=agent_id,
            wellness_score=wellness_score,
            consecutive_toxic=consecutive_toxic,
            total_calls=total_calls,
            toxic_calls=toxic_calls,
            shift_start=shift_start,
            last_updated=time.time(),
        )
        await self._write_record(agent_id, record)
        await self._r.sadd(_active_agents_key(), agent_id)
        return record

    async def append_call_history(
        self,
        agent_id: str,
        entry: CallHistoryEntry,
    ) -> None:
        """
        adds a call to the history list.
        i trim it so it doesn't get super big and crash everything, and reset the timer too.
        """
        serialised = json.dumps(entry.model_dump())
        history_key = _history_key(agent_id)

        pipe = self._r.pipeline()
        pipe.rpush(history_key, serialised)
        pipe.ltrim(history_key, -HISTORY_MAX_ENTRIES, -1)
        pipe.expire(history_key, WELLNESS_TTL_SECONDS)
        await pipe.execute()

    async def apply_break(
        self,
        agent_id: str,
        break_minutes: int,
    ) -> WellnessRecord:
        """
        updates redis directly when an agent takes a break.
        gives them up to 40 points back (2 points per min), maxes out at 100.
        also resets the consecutive toxic counter.
        """
        record = await self.get_or_init(agent_id)

        recovery = min(break_minutes * 2.0, 40.0)
        new_score = min(100.0, record.wellness_score + recovery)

        updated = WellnessRecord(
            agent_id=agent_id,
            wellness_score=new_score,
            consecutive_toxic=0,
            total_calls=record.total_calls,
            toxic_calls=record.toxic_calls,
            shift_start=record.shift_start,
            last_updated=time.time(),
        )
        await self._write_record(agent_id, updated)
        logger.info(
            "[WellnessStore] Break applied for %s: score %.1f → %.1f",
            agent_id,
            record.wellness_score,
            new_score,
        )
        return updated






def get_wellness_store(redis: Redis = Depends(get_redis)) -> WellnessStore:
    """
    just a helper for fastapi to inject the store with our redis connection.
    pretty standard dependency injection thing.
    """
    return WellnessStore(redis=redis)
