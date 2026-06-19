# --- FROM: backend/api/routes/calls.py ---
"""Handles calls, text analysis, audio, and Redis-backed sessions."""

import os
import uuid
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request, WebSocket, WebSocketDisconnect, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field

from src.api.limiter import limiter
from config.logger import get_logger
from src.core.session import ConversationManager
from src.analysis.toxicity import ToxicityAnalyzer
from src.analysis.wellness import WellnessTracker
from src.core.session import SessionManager, get_session_manager
from src.core.session import WellnessStore, get_wellness_store
from src.core.session import TurnRecord, CallHistoryEntry
from src.audio.stt import AudioConverter, AudioConversionError, AudioValidationError
from src.audio.stt import StreamingAudioNormalizer
from src.core.db import get_db
from src.core.db import CallLog
from src.analysis.wellness import compute_adjusted_score

logger = get_logger("api.calls")
master_router = APIRouter()

calls_router = APIRouter(prefix="/api/calls", tags=["Calls"])
router = calls_router


_live_managers: dict[str, ConversationManager] = {}


_toxicity_analyzer: ToxicityAnalyzer | None = None


def _get_toxicity_analyzer() -> ToxicityAnalyzer:
    global _toxicity_analyzer
    if _toxicity_analyzer is None:
        _toxicity_analyzer = ToxicityAnalyzer()
    return _toxicity_analyzer


_ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac", ".aac"}






class StartCallRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=50)
    session_id: Optional[str] = Field(None, min_length=1, max_length=100)


class StartCallResponse(BaseModel):
    session_id: str
    agent_id: str
    message: str


class EndCallRequest(BaseModel):
    session_id: str
    agent_id: str


class TextAnalyseRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=1, max_length=2000)
    speaker: str = Field(default="customer", pattern="^(customer|agent)$")


class TextAnalyseResponse(BaseModel):
    session_id: str
    speaker: str
    transcribed_text: str
    ai_suggestion: Optional[str]
    toxicity_score: float
    toxicity_level: str
    alert_message: Optional[str]






def _get_or_create_manager(session_id: str, agent_id: str) -> ConversationManager:
    """
    grabs the local ML manager or makes a new one if it's missing.
    this happens if the request hits a different server pod.
    """
    if session_id not in _live_managers:
        _live_managers[session_id] = ConversationManager(
            session_id=session_id,
            agent_id=agent_id,
        )
    return _live_managers[session_id]


async def _get_or_create_tracker(
    agent_id: str,
    ws: WellnessStore,
) -> WellnessTracker:
    """pulls the wellness state from redis or makes a new one"""
    record = await ws.get_or_init(agent_id)
    return WellnessTracker(
        agent_id=agent_id,
        initial_state=record.model_dump(),
    )






@router.post("/start", response_model=StartCallResponse)
async def start_call(
    req: StartCallRequest,
    sm: SessionManager = Depends(get_session_manager),
    ws: WellnessStore = Depends(get_wellness_store),
):
    """starts up a new call"""
    session_id = req.session_id or uuid.uuid4().hex[:10]

    
    await sm.create_session(session_id=session_id, agent_id=req.agent_id)

    
    await ws.get_or_init(req.agent_id)

    
    _get_or_create_manager(session_id, req.agent_id)

    logger.info("Call session started: %s (Agent: %s)", session_id, req.agent_id)
    return StartCallResponse(
        session_id=session_id,
        agent_id=req.agent_id,
        message=f"Session {session_id} started successfully.",
    )


@router.post("/end/{session_id}")
async def end_call(
    session_id: str,
    agent_id: str,
    agent_rating: float = Body(75.0, embed=True),
    sm: SessionManager = Depends(get_session_manager),
    ws: WellnessStore = Depends(get_wellness_store),
    db: AsyncSession = Depends(get_db),
):
    """ends the call and updates the stress score"""
    session = await sm.require_session(session_id)

    
    manager = _get_or_create_manager(session_id, session.agent_id)
    summary = manager.end_session()

    
    await sm.end_session(session_id)

    
    tracker = await _get_or_create_tracker(agent_id, ws)
    tracker.log_call(
        session_id=session_id,
        duration_seconds=summary["duration_seconds"],
        peak_toxicity_score=session.peak_toxicity_score,
        peak_toxicity_level=session.peak_toxicity_level,
        customer_sentiment="neutral",
    )
    await ws.save_state(agent_id, **tracker.to_state_dict())
    await ws.append_call_history(
        agent_id,
        CallHistoryEntry(
            session_id=session_id,
            duration_seconds=summary["duration_seconds"],
            peak_toxicity_score=session.peak_toxicity_score,
            peak_toxicity_level=session.peak_toxicity_level,
            is_toxic=session.peak_toxicity_level not in ("safe", "clean"),
            wellness_score_after=tracker._wellness_score,
        ),
    )

    
    _live_managers.pop(session_id, None)
    logger.info("Call session ended: %s", session_id)
    
    
    is_aggressive = True if session.peak_toxicity_score >= 0.5 else False
    call_log = CallLog(
        agent_id=agent_id,
        call_id=session_id,
        transcript=summary.get("full_transcript", ""),
        toxicity_score=session.peak_toxicity_score,
        toxicity_label=session.peak_toxicity_level,
        is_abusive=session.peak_toxicity_level not in ("safe", "clean"),
        duration_seconds=summary["duration_seconds"],
        base_score=agent_rating,
        aggressive_call_flag=is_aggressive
    )
    db.add(call_log)
    
    
    return {"status": "ended", "summary": summary}


@router.get("/agent/{agent_id}/performance")
async def get_performance(
    agent_id: str,
    db: AsyncSession = Depends(get_db)
):
    """gets the agent's performance scores from the db"""
    result = await db.execute(select(CallLog).where(CallLog.agent_id == agent_id))
    calls = result.scalars().all()
    
    clean_scores = [c.base_score for c in calls if not c.aggressive_call_flag and c.base_score is not None]
    aggressive_scores = [c.base_score for c in calls if c.aggressive_call_flag and c.base_score is not None]
    
    raw_avg, adj_avg, clean_cnt, agg_cnt = compute_adjusted_score(clean_scores, aggressive_scores)
    
    return {
        "agent_id": agent_id,
        "raw_average": raw_avg,
        "adjusted_average": adj_avg,
        "clean_calls": clean_cnt,
        "aggressive_calls": agg_cnt
    }


@router.post("/analyse-text", response_model=TextAnalyseResponse)
@limiter.limit("30/minute")
async def analyse_text(
    request: Request,
    req: TextAnalyseRequest,
    sm: SessionManager = Depends(get_session_manager),
):
    """
    takes text, checks if it's toxic, and generates a reply.
    """
    session = await sm.require_session(req.session_id)
    manager = _get_or_create_manager(req.session_id, session.agent_id)

    
    tox_result = _get_toxicity_analyzer().analyse(req.text)

    
    turn = TurnRecord(
        speaker=req.speaker,
        text=req.text,
        toxicity_score=tox_result.score,
        toxicity_level=tox_result.level,
    )

    
    ai_suggestion: Optional[str] = None
    if req.speaker == "customer":
        
        manager.session.add_turn(speaker=req.speaker, text=req.text)
        ai_suggestion = await manager._generate_suggestion()
        if ai_suggestion:
            turn.ai_suggestion = ai_suggestion
    else:
        manager.session.add_turn(speaker=req.speaker, text=req.text)

    
    try:
        await sm.add_turn(req.session_id, turn)
    except Exception as exc:
        logger.warning("[analyse_text] Failed to persist turn to Redis: %s", exc)

    logger.info(
        "[API] turn_analysed | session=%s | speaker=%s | text_len=%d | "
        "toxicity=%.2f | suggestion=%s",
        req.session_id, req.speaker, len(req.text),
        tox_result.score, bool(ai_suggestion),
    )

    return TextAnalyseResponse(
        session_id=req.session_id,
        speaker=req.speaker,
        transcribed_text=req.text,
        ai_suggestion=ai_suggestion,
        toxicity_score=tox_result.score,
        toxicity_level=tox_result.level,
        alert_message=tox_result.alert_message or None,
    )


@router.post("/transcribe-audio/{session_id}")
@limiter.limit("20/minute")
async def transcribe_audio(
    request: Request,
    session_id: str,
    audio: UploadFile = File(...),
    sm: SessionManager = Depends(get_session_manager),
):
    """
    takes an uploaded audio file, converts it, and runs whisper on it.
    """
    ext = os.path.splitext(audio.filename or "")[1].lower()
    if ext not in _ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Accepted: {', '.join(_ALLOWED_AUDIO_EXTENSIONS)}",
        )

    session = await sm.require_session(session_id)
    manager = _get_or_create_manager(session_id, session.agent_id)

        
    tmp_input_path = None
    try:
        audio_bytes = await audio.read()
        
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_input_path = tmp.name

        
        with AudioConverter(max_size_bytes=25*1024*1024, max_duration_seconds=600.0) as converter:
            converted_path = converter.convert(tmp_input_path)
            
            
            audio_array, _ = sf.read(converted_path, dtype="float32")
            
            
            suggestion = await manager.process_audio_chunk(audio_array, speaker="customer")

            
            if manager.session.turns:
                latest = manager.session.turns[-1]
                turn = TurnRecord(
                    speaker="customer",
                    text=latest.text,
                    timestamp=latest.timestamp,
                    ai_suggestion=suggestion,
                )
                try:
                    await sm.add_turn(session_id, turn)
                except Exception as exc:
                    logger.warning("[transcribe_audio] Failed to persist turn: %s", exc)

            return {"session_id": session_id, "ai_suggestion": suggestion, "status": "processed"}

    except AudioValidationError as e:
        logger.warning(f"Audio validation failed: {e}")
        raise HTTPException(status_code=413, detail=str(e))
    except AudioConversionError as e:
        logger.error(f"Audio conversion failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Audio transcription error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during audio processing.")
    finally:
        if tmp_input_path and os.path.exists(tmp_input_path):
            try:
                os.remove(tmp_input_path)
            except Exception:
                pass


import base64
import json
import asyncio

@router.websocket("/ws/audio/{session_id}")
async def ws_audio_endpoint(
    websocket: WebSocket,
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
):
    """
    websocket for streaming audio live.
    we get base64 encoded audio chunks and send back transcriptions.
    """
    await websocket.accept()

    
    session_exists = await sm.session_exists(session_id)
    if not session_exists:
        await websocket.send_json({"type": "error", "message": "Session not found", "status": 404})
        await websocket.close()
        return

    session_record = await sm.get_session(session_id)
    agent_id = session_record.agent_id
    manager = _get_or_create_manager(session_id, agent_id)

    
    
    normalizer: StreamingAudioNormalizer | None = None

    
    expected_chunk_index = 0
    chunk_buffer = {}

    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                payload = json.loads(data)
            except Exception:
                await websocket.send_json({"type": "error", "message": "Invalid JSON payload"})
                continue

            msg_type = payload.get("type", "audio")

            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            
            if msg_type == "audio":
                chunk_index = payload.get("chunk_index")
                audio_b64 = payload.get("audio_data")
                audio_format = payload.get("audio_format", "encoded")
                sample_rate = payload.get("sample_rate", 16000)

                if chunk_index is None or not audio_b64:
                    await websocket.send_json({"type": "error", "message": "Missing chunk_index or audio_data"})
                    continue

                speaker = payload.get("speaker", "customer")
                if speaker not in ("customer", "agent"):
                    speaker = "customer"

                
                if chunk_index < expected_chunk_index:
                    continue

                try:
                    raw_bytes = base64.b64decode(audio_b64)
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Failed to decode base64: {e}"})
                    continue

                
                chunk_buffer[chunk_index] = raw_bytes

                
                while expected_chunk_index in chunk_buffer:
                    chunk_to_process = chunk_buffer.pop(expected_chunk_index)
                    expected_chunk_index += 1

                    if audio_format == "pcm_s16le":
                        if sample_rate != 16000:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Raw PCM must use a 16000 Hz sample rate",
                            })
                            continue
                        if len(chunk_to_process) % 2:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Invalid PCM16 audio chunk",
                            })
                            continue
                        normalized_chunks = [
                            np.frombuffer(chunk_to_process, dtype="<i2").astype(np.float32)
                            / 32768.0
                        ]
                    else:
                        
                        if normalizer is None:
                            normalizer = StreamingAudioNormalizer(sample_rate=16000)
                            await normalizer.start()
                        await normalizer.feed_chunk(chunk_to_process)
                        normalized_chunks = []
                        while True:
                            normalized_np = await normalizer.read_normalized(max_bytes=8192)
                            if normalized_np.size == 0:
                                break
                            normalized_chunks.append(normalized_np)

                    for normalized_np in normalized_chunks:
                        
                        num_turns_before = len(manager.session.turns)
                        
                        try:
                            suggestion = await manager.process_audio_chunk(normalized_np, speaker=speaker)
                            num_turns_after = len(manager.session.turns)
                            
                            
                            if num_turns_after > num_turns_before:
                                latest_turn = manager.session.turns[-1]
                                
                                if speaker == "customer":
                                    
                                    tox_result = _get_toxicity_analyzer().analyse(latest_turn.text)
                                    tox_score = tox_result.score
                                    tox_level = tox_result.level
                                    tox_alert = tox_result.alert_message
                                else:
                                    
                                    tox_score = 0.0
                                    tox_level = "safe"
                                    tox_alert = None
                                    
                                turn_record = TurnRecord(
                                    speaker=speaker,
                                    text=latest_turn.text,
                                    timestamp=latest_turn.timestamp,
                                    ai_suggestion=suggestion,
                                    toxicity_score=tox_score,
                                    toxicity_level=tox_level
                                )
                                
                                try:
                                    await sm.add_turn(session_id, turn_record)
                                except Exception as exc:
                                    logger.warning("[ws_audio] Failed to persist turn: %s", exc)

                                await websocket.send_json({
                                    "type": "transcription",
                                    "speaker": speaker,
                                    "text": latest_turn.text,
                                    "suggestion": suggestion,
                                    "toxicity_score": tox_score,
                                    "toxicity_level": tox_level,
                                    "alert_message": tox_alert
                                })
                                
                        except Exception as e:
                            logger.error(f"Error processing chunk {expected_chunk_index-1}: {e}")
                            await websocket.send_json({"type": "error", "message": "Internal processing error"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        if normalizer is not None:
            await normalizer.close()



@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
):
    """gets the whole chat history from redis"""
    session = await sm.require_session(session_id)
    transcript = await sm.get_transcript(session_id, max_turns=50)
    return {
        "session_id": session_id,
        "agent_id": session.agent_id,
        "is_active": session.is_active,
        "total_turns": session.turn_count,
        "transcript": transcript,
    }


@router.get("/active")
async def list_active_sessions(
    sm: SessionManager = Depends(get_session_manager),
):
    """gets a list of all active calls"""
    sessions = await sm.list_active_sessions()
    return {
        "active_sessions": [
            {
                "session_id": s.session_id,
                "agent_id": s.agent_id,
                "turns": s.turn_count,
            }
            for s in sessions
        ],
        "total": len(sessions),
    }


# --- FROM: backend/api/routes/knowledge.py ---
"""
Knowledge Base API Routes
stuff for uploading docs and doing RAG search on them.
"""

import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from pydantic import BaseModel, Field
from typing import Optional

from src.api.limiter import limiter  
from src.retrieval.hybrid import retrieve_context
from src.ingestion.indexer import process_and_store_documents
from config.logger import get_logger

logger = get_logger("api.knowledge")
knowledge_router = APIRouter(prefix="/api/knowledge", tags=["Knowledge Base"])
router = knowledge_router

UPLOAD_DIR = "data/knowledge_base"
os.makedirs(UPLOAD_DIR, exist_ok=True)


ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


class SearchRequest(BaseModel):  
    query: str = Field(..., min_length=1, max_length=500)
    top_k: Optional[int] = Field(default=3, ge=1, le=10)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    upload a doc to the KB.
    it automatically chunks it up and throws it into pgvector.
    """
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Accepted: .txt, .md, .pdf, .docx"
        )

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"File uploaded: {file.filename}")

        
        process_and_store_documents(UPLOAD_DIR)

        return {
            "status": "success",
            "filename": file.filename,
            "message": f"Document '{file.filename}' uploaded and indexed successfully."
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
@limiter.limit("60/minute")  
async def search_knowledge(request: Request, req: SearchRequest):
    """does a search in pgvector using the query string."""
    
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        context = retrieve_context(req.query, top_k=req.top_k)
        return {
            "query": req.query,
            "results": context if context else "No relevant documents found.",
            "found": bool(context)
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    """lists all the docs we have so far"""
    
    files = [
        f for f in os.listdir(UPLOAD_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
    ]
    return {
        "documents": files,
        "total": len(files),
        "directory": UPLOAD_DIR
    }


# --- FROM: backend/api/routes/wellness.py ---
"""Provides wellness routes using agent state stored in Redis."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.analysis.wellness import WellnessTracker, _score_to_level, RECOMMENDATIONS
from src.core.db import get_db
from src.core.session import WellnessStore, get_wellness_store
from config.logger import get_logger

logger = get_logger("api.wellness")
wellness_router = APIRouter(prefix="/api/wellness", tags=["Agent Wellness"])
router = wellness_router


class BreakRequest(BaseModel):
    duration_minutes: int = Field(default=10, ge=1, le=60)






async def _load_tracker(agent_id: str, ws: WellnessStore) -> WellnessTracker:
    """loads up the tracker with whatever is in redis right now"""
    record = await ws.get_or_init(agent_id)
    return WellnessTracker(agent_id=agent_id, initial_state=record.model_dump())






@router.get("/{agent_id}/status")
async def get_wellness_status(
    agent_id: str,
    ws: WellnessStore = Depends(get_wellness_store),
):
    """gets how stressed the agent is right now"""
    record = await ws.get_or_init(agent_id)
    score = record.wellness_score
    level = _score_to_level(score)
    needs_break = (
        score < 30
        or record.consecutive_toxic >= 3
        or level == "critical"
    )

    alert = ""
    if level == "critical":
        alert = f"BURNOUT ALERT: Wellness at {score:.1f}/100. Break required NOW."
    elif level == "high":
        alert = f"Stress Alert: Wellness at {score:.1f}/100. Break recommended."
    elif record.consecutive_toxic >= 3:
        alert = "3+ consecutive toxic calls. Agent needs immediate support."
    elif level == "moderate":
        alert = f"Mild stress detected. Wellness at {score:.1f}/100."

    return {
        "agent_id": agent_id,
        "wellness_score": round(score, 1),
        "stress_level": level,
        "total_calls_today": record.total_calls,
        "toxic_calls_today": record.toxic_calls,
        "consecutive_toxic_calls": record.consecutive_toxic,
        "needs_break": needs_break,
        "alert_message": alert,
        "recommendation": RECOMMENDATIONS[level],
    }


@router.post("/{agent_id}/break")
async def log_break(
    agent_id: str,
    req: BreakRequest,
    ws: WellnessStore = Depends(get_wellness_store),
):
    """logs a break and boosts their wellness score back up in redis"""
    updated = await ws.apply_break(agent_id, req.duration_minutes)
    level = _score_to_level(updated.wellness_score)
    return {
        "message": f"Break of {req.duration_minutes} minutes logged for {agent_id}.",
        "wellness_score_after_break": round(updated.wellness_score, 1),
        "stress_level": level,
    }


@router.get("/{agent_id}/report")
async def get_shift_report(
    agent_id: str,
    ws: WellnessStore = Depends(get_wellness_store),
):
    """Return the full report for the agent's shift."""
    import time
    record = await ws.get_or_init(agent_id)
    shift_duration = (time.time() - record.shift_start) / 60
    status_level = _score_to_level(record.wellness_score)

    return {
        "agent_id": agent_id,
        "shift_duration_minutes": round(shift_duration, 1),
        "final_wellness_score": round(record.wellness_score, 1),
        "final_stress_level": status_level,
        "total_calls": record.total_calls,
        "toxic_calls": record.toxic_calls,
        "toxic_percentage": round(
            (record.toxic_calls / record.total_calls * 100)
            if record.total_calls > 0 else 0.0,
            1,
        ),
        "recommendation": RECOMMENDATIONS[status_level],
    }


@router.get("/{agent_id}/score")
async def get_adjusted_score(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """grabs the adjusted score from postgres so we don't punish agents for mean callers"""
    try:
        result = await db.execute(
            text("""
                SELECT agent_id, base_score, aggressive_call_flag
                FROM call_logs
                WHERE agent_id = :agent_id
                AND started_at >= NOW() - INTERVAL '8 hours'
            """),
            {"agent_id": agent_id},
        )
        rows = result.fetchall()

        if not rows:
            return {"agent_id": agent_id, "adjusted_score": None, "reason": "no_calls"}

        clean = [r[1] for r in rows if not r[2] and r[1] is not None]
        aggressive = [r[1] for r in rows if r[2] and r[1] is not None]

        if not clean:
            return {"agent_id": agent_id, "adjusted_score": None, "reason": "all_aggressive"}

        clean_avg = sum(clean) / len(clean)
        agg_avg = sum(aggressive) / len(aggressive) if aggressive else 0.0
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
    except Exception as exc:
        logger.error("Failed to compute adjusted score for %s: %s", agent_id, exc)
        raise HTTPException(status_code=500, detail="Score computation failed")


@router.get("/")
async def list_all_agents(
    ws: WellnessStore = Depends(get_wellness_store),
):
    """lists all agents. works across replicas now so that's cool."""
    records = await ws.list_all_agents()
    return {
        "agents": [
            {
                "agent_id": r.agent_id,
                "wellness_score": round(r.wellness_score, 1),
                "stress_level": _score_to_level(r.wellness_score),
                "needs_break": (
                    r.wellness_score < 30
                    or r.consecutive_toxic >= 3
                    or _score_to_level(r.wellness_score) == "critical"
                ),
            }
            for r in records
        ],
        "total_agents": len(records),
    }

# All routes are now defined on their sub-routers. Only now do we include them
# into master_router — FastAPI snapshots routes at include_router() call time,
# so this must happen AFTER all @router decorators have been applied above.
master_router.include_router(calls_router)
master_router.include_router(knowledge_router)
master_router.include_router(wellness_router)

router = master_router

