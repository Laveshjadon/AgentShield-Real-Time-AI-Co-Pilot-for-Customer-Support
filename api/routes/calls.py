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

from api.limiter import limiter
from config.logger import get_logger
from core.conversation_manager import ConversationManager
from analysis.toxicity_analyzer import ToxicityAnalyzer
from analysis.wellness_tracker import WellnessTracker
from session.session_manager import SessionManager, get_session_manager
from session.wellness_store import WellnessStore, get_wellness_store
from session.schemas import TurnRecord, CallHistoryEntry
from stt.audio_converter import AudioConverter, AudioConversionError, AudioValidationError
from stt.streaming_normalizer import StreamingAudioNormalizer
from db.connection import get_db
from db.models import CallLog
from analysis.scoring import compute_adjusted_score

logger = get_logger("api.calls")
router = APIRouter(prefix="/api/calls", tags=["Calls"])


_live_managers: dict[str, ConversationManager] = {}


_toxicity_analyzer = ToxicityAnalyzer()


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

    
    tox_result = _toxicity_analyzer.analyse(req.text)

    
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
import heapq
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
                                    
                                    tox_result = _toxicity_analyzer.analyse(latest_turn.text)
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
