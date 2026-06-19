from __future__ import annotations
# --- FROM: backend/session/schemas.py ---
"""
just the pydantic stuff for what gets saved to redis.
mostly for tracking the session and the agent's stress level.

notes to self:
- gave everything a default so it doesn't crash if redis is missing a field
- times are just floats
- saving booleans as "1" or "0" cause redis is weird about it
"""



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



# --- FROM: backend/session/client.py ---
"""
redis connection setup stuff. basically keeping one pool open so we don't spam connections.
init_redis() sets it up when the app starts.
close_redis() cleans it up when we quit.
get_redis() is how you actually get a connection to use it somewhere else.
"""



from redis.asyncio import ConnectionPool, Redis
from redis.exceptions import RedisError

from config.settings import Settings
from config.logger import get_logger

logger = get_logger("session.client")
settings = Settings()


_pool: ConnectionPool | None = None
_fake_client = None


class RedisUnavailableError(RuntimeError):
    """throws an error if redis goes down or isn't started yet"""






async def init_redis() -> None:
    """starts the pool and checks if it actually works, otherwise crashes right away"""
    global _pool, _fake_client
    _pool = ConnectionPool.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=50,
    )

    
    client = Redis(connection_pool=_pool)
    try:
        await client.ping()
        logger.info("[Redis] Connected — pool initialised (max_connections=50)")
    except RedisError as exc:
        _pool = None
        if getattr(settings, "REDIS_ALLOW_FAKE", False):
            import fakeredis.aioredis

            _fake_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
            logger.warning(
                "[Redis] Unavailable; using in-memory fakeredis for local development."
            )
            return
        raise RedisUnavailableError(f"Redis ping failed: {exc}") from exc
    finally:
        await client.aclose()


async def close_redis() -> None:
    """closes everything down nicely so we don't leave connections hanging"""
    global _pool, _fake_client
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        logger.info("[Redis] Connection pool closed.")
    if _fake_client is not None:
        await _fake_client.aclose()
        _fake_client = None






def get_redis() -> Redis:
    """just grabs a connection from the pool. pretty fast and doesn't open new sockets."""
    if _fake_client is not None:
        return _fake_client
    if _pool is None:
        raise RedisUnavailableError(
            "Redis pool is not initialised. "
            "Ensure init_redis() was called during application startup."
        )
    return Redis(connection_pool=_pool)






async def ping_redis() -> bool:
    """just pings redis, returns false if it's dead instead of crashing"""
    try:
        client = get_redis()
        result = await client.ping()
        await client.aclose()
        return bool(result)
    except Exception as exc:
        logger.warning("[Redis] Health ping failed: %s", exc)
        return False



# --- FROM: backend/session/wellness_store.py ---
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



import json
import time
from typing import List, Optional

from fastapi import Depends
from redis.asyncio import Redis

from config.logger import get_logger

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



# --- FROM: backend/session/session_manager.py ---
"""
handles all the redis database stuff for the call sessions.

basically:
- session hash: stores the main info
- turns list: stores all the text dialogue
- active set: tracks who is currently on a call

we keep things around for 2 hours (ttl) just in case we need to look at it
after the call drops. making sure everything is atomic so we don't overwrite stuff.
"""



import json
import time
from typing import List, Optional

from fastapi import Depends, HTTPException
from redis.asyncio import Redis

from config.logger import get_logger


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

        # Only update peak toxicity when the new score is strictly higher.
        if turn.toxicity_score > 0:
            current_peak_raw = await self._r.hget(hash_key, "peak_toxicity_score")
            current_peak = float(current_peak_raw) if current_peak_raw else 0.0
            if turn.toxicity_score > current_peak:
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


# --- FROM: backend/core/conversation_manager.py ---
"""
main class that ties everything together for a live call.
handles the vad → stt pipeline, feeds text into the rag system,
and keeps track of the conversation so the ai has context.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import numpy as np

from config.settings import Settings
from config.logger import get_logger
from src.audio.stt import VADEngine
from src.audio.stt import Transcriber
from src.retrieval.generation import generate_answer
from src.audio.tts import TTSEngine

logger = get_logger("core.conversation_manager")
settings = Settings()


@dataclass
class Turn:
    speaker: str
    text: str
    timestamp: float = field(default_factory=time.time)
    ai_suggestion: Optional[str] = None


@dataclass
class CallSession:
    session_id: str
    agent_id: str
    start_time: float = field(default_factory=time.time)
    turns: List[Turn] = field(default_factory=list)
    is_active: bool = True

    def get_transcript(self, max_turns: int = 10) -> str:
        recent = self.turns[-max_turns:]
        return "\n".join(f"{t.speaker.capitalize()}: {t.text}" for t in recent)

    def add_turn(self, speaker: str, text: str, ai_suggestion: str = None):
        turn = Turn(speaker=speaker, text=text, ai_suggestion=ai_suggestion)
        self.turns.append(turn)
        return turn


_GLOBAL_VAD = None
_GLOBAL_TRANSCRIBER = None
_GLOBAL_TTS = None

def get_vad():
    global _GLOBAL_VAD
    if _GLOBAL_VAD is None:
        _GLOBAL_VAD = VADEngine(threshold=0.25)
    return _GLOBAL_VAD

def get_transcriber():
    global _GLOBAL_TRANSCRIBER
    if _GLOBAL_TRANSCRIBER is None:
        _GLOBAL_TRANSCRIBER = Transcriber()
    return _GLOBAL_TRANSCRIBER

def get_tts():
    global _GLOBAL_TTS
    if _GLOBAL_TTS is None:
        _GLOBAL_TTS = TTSEngine()
    return _GLOBAL_TTS


class ConversationManager:
    def __init__(self, session_id: str = "default", agent_id: str = "agent_001"):
        logger.info("Initializing ConversationManager...")
        # Laptop and headset microphones often produce quieter 250 ms chunks.
        # A lower threshold keeps live speech from being discarded before STT.
        self.vad = get_vad()
        self.transcriber = get_transcriber()
        self.tts = get_tts()
        self.session = CallSession(session_id=session_id, agent_id=agent_id)
        self.audio_buffer = []
        self.SAMPLE_RATE = 16000
        
        
        self.silence_chunks = 0
        self.is_speaking = False
        self.SILENCE_THRESHOLD_CHUNKS = 6  # 6 × 250ms = 1.5s of silence to end utterance
        self.MAX_BUFFER_SAMPLES = self.SAMPLE_RATE * 5  
        
        self.on_suggestion_ready: Optional[Callable] = None
        self.on_transcription_ready: Optional[Callable] = None
        logger.info(f"ConversationManager ready. Session: {session_id}")

    async def process_audio_chunk(self, audio_chunk: np.ndarray, speaker: str = "customer") -> Optional[str]:
        if not self.session.is_active:
            logger.warning("Session is not active. Ignoring audio.")
            return None

        has_speech = self.vad.contains_speech(audio_chunk, self.SAMPLE_RATE)
        logger.info(f"[VAD] has_speech={has_speech}  buffer={len(self.audio_buffer)}  silence_chunks={self.silence_chunks}")

        if has_speech:
            self.is_speaking = True
            self.silence_chunks = 0
            self.audio_buffer.extend(audio_chunk.tolist())
            logger.debug(f"[VAD] Speech detected. Buffer size: {len(self.audio_buffer)}")
        else:
            if self.is_speaking:
                self.silence_chunks += 1
                self.audio_buffer.extend(audio_chunk.tolist())
                logger.debug(f"[VAD] Silence detected (chunk {self.silence_chunks}/{self.SILENCE_THRESHOLD_CHUNKS})")
            else:
                
                return None

        
        trigger_transcription = False
        if self.is_speaking and self.silence_chunks >= self.SILENCE_THRESHOLD_CHUNKS:
            logger.debug("[VAD] Silence threshold reached. Triggering transcription.")
            trigger_transcription = True
        elif len(self.audio_buffer) >= self.MAX_BUFFER_SAMPLES:
            logger.debug("[VAD] Max buffer limit reached (5s). Force triggering transcription.")
            trigger_transcription = True
            
        if not trigger_transcription:
            return None

        
        audio_to_transcribe = np.array(self.audio_buffer, dtype=np.float32)
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_chunks = 0
        
        logger.info(f"Transcribing audio buffer of {len(audio_to_transcribe)} samples ({len(audio_to_transcribe)/self.SAMPLE_RATE:.1f}s)")
        result = self.transcriber.transcribe(audio_to_transcribe)
        text = result.get("transcript", "") if isinstance(result, dict) else result
        
        if not text or len(text.strip()) < 3:
            return None
            
        logger.info(f"[{speaker.upper()}] said: '{text}'")
        if self.on_transcription_ready:
            await self.on_transcription_ready(speaker, text)

        self.session.add_turn(speaker=speaker, text=text)

        suggestion = None
        if speaker == "customer":
            suggestion = await self._generate_suggestion()

        return suggestion

    async def _generate_suggestion(self) -> Optional[str]:
        """Run the full hybrid RAG pipeline and return an agent suggestion."""
        try:
            transcript = self.session.get_transcript(max_turns=6)
            last_customer_line = [t.text for t in self.session.turns if t.speaker == "customer"]
            latest_utterance = last_customer_line[-1] if last_customer_line else ""
            if not latest_utterance:
                return None
            # Pass no pre-fetched context so generate_answer runs hybrid retrieval itself.
            suggestion = await asyncio.to_thread(
                generate_answer,
                transcript=transcript,
                latest_utterance=latest_utterance,
            )
            if suggestion:
                logger.info(f"[AI SUGGESTION] {suggestion[:80]}...")
                if self.session.turns:
                    self.session.turns[-1].ai_suggestion = suggestion
                if self.on_suggestion_ready:
                    await self.on_suggestion_ready(suggestion)
            return suggestion
        except Exception as e:
            logger.error(f"Failed to generate suggestion: {e}")
            return None

    async def speak(self, text: str, language: str = None) -> Optional[str]:
        audio_path = await self.tts.synthesize(text, force_language=language)
        return audio_path

    def end_session(self) -> dict:
        self.session.is_active = False
        duration = time.time() - self.session.start_time
        summary = {
            "session_id": self.session.session_id,
            "agent_id": self.session.agent_id,
            "duration_seconds": round(duration, 1),
            "total_turns": len(self.session.turns),
            "full_transcript": self.session.get_transcript(max_turns=999),
        }
        logger.info(f"Session {self.session.session_id} ended. Duration: {duration:.0f}s, Turns: {len(self.session.turns)}")
        return summary




if __name__ == "__main__":
    import uuid

    async def simulate():
        print("\n" + "="*55)
        print("  AgentShield - Conversation Simulation Test")
        print("="*55)
        manager = ConversationManager(session_id=uuid.uuid4().hex[:8], agent_id="test_agent")

        async def on_transcription(speaker, text):
            print(f"\n  [{speaker.upper()}]: {text}")

        async def on_suggestion(suggestion):
            print(f"\n  [AI SUGGESTION]:\n{suggestion}")

        manager.on_transcription_ready = on_transcription
        manager.on_suggestion_ready = on_suggestion
        print("\n  Simulating customer complaint about refund...")
        print("-"*55)
        manager.session.add_turn("agent", "Thank you for calling TechNova. How can I help?")
        manager.session.add_turn("customer", "I bought a router 10 days ago and it stopped working. I want a refund.")
        suggestion = await manager._generate_suggestion()
        print("\n" + "="*55)
        audio_path = await manager.speak("Thank you for calling. I'm looking into this for you right now.")
        print(f"  Audio file: {audio_path}")
        summary = manager.end_session()
        print(f"\n  Session: {summary['session_id']}, Turns: {summary['total_turns']}, Duration: {summary['duration_seconds']}s")
        print("="*55 + "\n")

    asyncio.run(simulate())


