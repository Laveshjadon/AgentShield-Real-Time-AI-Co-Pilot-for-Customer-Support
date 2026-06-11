"""Provides wellness routes using agent state stored in Redis."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from analysis.wellness_tracker import WellnessTracker, _score_to_level, RECOMMENDATIONS
from db.connection import get_db
from session.wellness_store import WellnessStore, get_wellness_store
from config.logger import get_logger

logger = get_logger("api.wellness")
router = APIRouter(prefix="/api/wellness", tags=["Agent Wellness"])


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
