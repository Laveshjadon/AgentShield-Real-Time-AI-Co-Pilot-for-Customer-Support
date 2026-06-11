"""Manages transcription, AI responses, and conversation sessions."""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Callable
import numpy as np

from config.settings import Settings
from config.logger import get_logger
from stt.vad import VADEngine
from stt.whisper_engine import Transcriber
from rag.retriever import retrieve_context
from rag.generator import generate_answer
from tts.edge_tts_engine import TTSEngine

logger = get_logger("core.conversation_manager")
settings = Settings()


_redis_client = None


async def get_redis():
    """Return a Redis client if Redis is available."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis.asyncio as aioredis
            _redis_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            await _redis_client.ping()
            logger.info("[Redis] Connected to %s", settings.REDIS_URL)
        except Exception as e:
            logger.warning("[Redis] Could not connect: %s; sessions will be in-memory only", e)
            _redis_client = None
    return _redis_client


async def save_session_to_redis(session_id: str, session_data: dict):
    """Persist session data in Redis for two hours."""
    r = await get_redis()
    if r:
        try:
            await r.setex(f"session:{session_id}", 7200, json.dumps(session_data))
        except Exception as e:
            logger.warning("[Redis] Failed to save session %s: %s", session_id, e)


async def load_session_from_redis(session_id: str) -> Optional[dict]:
    """Load session data from Redis."""
    r = await get_redis()
    if r:
        try:
            data = await r.get(f"session:{session_id}")
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning("[Redis] Failed to load session %s: %s", session_id, e)
    return None


async def delete_session_from_redis(session_id: str):
    """Delete session data from Redis."""
    r = await get_redis()
    if r:
        try:
            await r.delete(f"session:{session_id}")
        except Exception as e:
            logger.warning("[Redis] Failed to delete session %s: %s", session_id, e)


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


class ConversationManager:
    def __init__(self, session_id: str = "default", agent_id: str = "agent_001"):
        logger.info("Initializing ConversationManager...")
        self.vad = VADEngine(threshold=0.5)
        self.transcriber = Transcriber()
        self.tts = TTSEngine()
        self.session = CallSession(session_id=session_id, agent_id=agent_id)
        self.audio_buffer = []
        self.SAMPLE_RATE = 16000
        
        
        self.silence_chunks = 0
        self.is_speaking = False
        self.SILENCE_THRESHOLD_CHUNKS = 3  
        self.MAX_BUFFER_SAMPLES = self.SAMPLE_RATE * 5  
        
        self.on_suggestion_ready: Optional[Callable] = None
        self.on_transcription_ready: Optional[Callable] = None
        logger.info(f"ConversationManager ready. Session: {session_id}")

    async def process_audio_chunk(self, audio_chunk: np.ndarray, speaker: str = "customer") -> Optional[str]:
        if not self.session.is_active:
            logger.warning("Session is not active. Ignoring audio.")
            return None

        has_speech = self.vad.contains_speech(audio_chunk, self.SAMPLE_RATE)
        
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
        
        
        await save_session_to_redis(self.session.session_id, {
            "session_id": self.session.session_id,
            "agent_id": self.session.agent_id,
            "turns": len(self.session.turns),
            "is_active": self.session.is_active,
        })
        
        suggestion = None
        if speaker == "customer":
            suggestion = await self._generate_suggestion()
            
        return suggestion

    async def _generate_suggestion(self) -> Optional[str]:
        try:
            transcript = self.session.get_transcript(max_turns=6)
            last_customer_line = [t.text for t in self.session.turns if t.speaker == "customer"]
            query = last_customer_line[-1] if last_customer_line else ""
            context = retrieve_context(query)
            if not context:
                logger.info("No relevant knowledge found for this query.")
                return None
            suggestion = generate_answer(
                transcript=transcript,
                context=context,
                latest_utterance=query,
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



async def compute_adjusted_score(agent_id: str, db) -> dict:
    """
    Calculate an adjusted score that reduces the weight of aggressive calls.
    """
    from sqlalchemy import select
    from db.models import CallLog

    result = await db.execute(
        select(CallLog).where(
            CallLog.agent_id == agent_id,
            CallLog.started_at >= datetime.utcnow() - timedelta(hours=8)
        )
    )
    calls = result.scalars().all()

    if not calls:
        return {"agent_id": agent_id, "adjusted_score": None, "reason": "no_calls"}

    clean = [c.base_score for c in calls if not c.aggressive_call_flag and c.base_score is not None]
    aggressive = [c.base_score for c in calls if c.aggressive_call_flag and c.base_score is not None]

    if not clean:
        return {"agent_id": agent_id, "adjusted_score": None, "reason": "all_aggressive"}

    clean_avg = sum(clean) / len(clean)
    agg_avg = sum(aggressive) / len(aggressive) if aggressive else 0
    total_weight = len(clean) + 0.5 * len(aggressive)
    adjusted = (clean_avg * len(clean) + agg_avg * 0.5 * len(aggressive)) / total_weight
    all_scores = clean + aggressive

    return {
        "agent_id": agent_id,
        "clean_calls": len(clean),
        "aggressive_calls": len(aggressive),
        "raw_avg": round(sum(all_scores) / len(all_scores), 3),
        "adjusted_score": round(adjusted, 3),
    }


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
