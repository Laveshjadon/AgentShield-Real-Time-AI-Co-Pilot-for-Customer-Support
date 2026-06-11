"""Tracks agent stress and recommends breaks when wellness is low."""

import time
from dataclasses import dataclass, field
from typing import Optional, List

from config.logger import get_logger

logger = get_logger("analysis.wellness")





@dataclass
class WellnessStatus:
    agent_id: str
    wellness_score: float       
    stress_level: str           
    total_calls_today: int
    toxic_calls_today: int
    consecutive_toxic_calls: int
    needs_break: bool
    alert_message: str
    recommendation: str
    timestamp: float = field(default_factory=time.time)





SCORE_THRESHOLDS = {
    "low":      (75, 100),  
    "moderate": (50, 74),   
    "high":     (25, 49),   
    "critical": (0,  24),   
}

RECOMMENDATIONS = {
    "low":      "Agent is doing well. No action needed.",
    "moderate": "Agent showing mild stress. Consider offering a 5-min break soon.",
    "high":     "Agent under significant stress. Recommend a break after this call.",
    "critical": "URGENT: Agent is burned out. Pause call queue and send to break NOW.",
}


def _score_to_level(score: float) -> str:
    for level, (low, high) in SCORE_THRESHOLDS.items():
        if low <= score <= high:
            return level
    return "critical"





class WellnessTracker:

    def __init__(self, agent_id: str, initial_state: Optional[dict] = None):
        """
        Args:
            agent_id: The agent this tracker belongs to.
            initial_state: Optional dict hydrated from a ``WellnessRecord``
                (from Redis) so the tracker resumes mid-shift state instead
                of starting fresh.  Keys: wellness_score, consecutive_toxic,
                total_calls, toxic_calls.
        """
        self.agent_id = agent_id
        self.shift_start = time.time()

        
        self._call_history: List[dict] = []   

        
        if initial_state:
            self._wellness_score = float(initial_state.get("wellness_score", 100.0))
            self._consecutive_toxic = int(initial_state.get("consecutive_toxic", 0))
            
            _total = int(initial_state.get("total_calls", 0))
            _toxic = int(initial_state.get("toxic_calls", 0))
            
            for _ in range(_total):
                self._call_history.append({"is_toxic": False})
            for i in range(_total - _toxic, _total):
                self._call_history[i]["is_toxic"] = True
        else:
            self._wellness_score = 100.0
            self._consecutive_toxic = 0

        
        self.RECOVERY_PER_SAFE_CALL = 5.0

        
        self.TOXICITY_PENALTY = {
            "safe":     0.0,
            "warning":  3.0,
            "danger":   10.0,
            "critical": 20.0,
        }

        logger.info(f"WellnessTracker initialized for agent: {agent_id}")

    def log_call(
        self,
        session_id: str,
        duration_seconds: float,
        peak_toxicity_score: float,
        peak_toxicity_level: str,
        customer_sentiment: str = "neutral"
    ):
        """
        logs a call and adjusts the score.
        the manager calls this when a session wraps up.
        """
        is_toxic = peak_toxicity_level in ("warning", "danger", "critical")

        
        if is_toxic:
            self._consecutive_toxic += 1
        else:
            self._consecutive_toxic = 0
            
            self._wellness_score = min(100.0, self._wellness_score + self.RECOVERY_PER_SAFE_CALL)

        
        penalty = self.TOXICITY_PENALTY.get(peak_toxicity_level, 0.0)

        
        if is_toxic and duration_seconds > 300:  
            penalty *= 1.5

        
        if self._consecutive_toxic >= 3:
            penalty *= 1.25

        self._wellness_score = max(0.0, self._wellness_score - penalty)

        
        call_record = {
            "session_id": session_id,
            "timestamp": time.time(),
            "duration_seconds": duration_seconds,
            "peak_toxicity_score": peak_toxicity_score,
            "peak_toxicity_level": peak_toxicity_level,
            "is_toxic": is_toxic,
            "wellness_score_after": round(self._wellness_score, 1),
        }
        self._call_history.append(call_record)

        logger.info(
            f"Call logged for {self.agent_id}: "
            f"toxicity={peak_toxicity_level}, "
            f"wellness={self._wellness_score:.1f}, "
            f"consecutive_toxic={self._consecutive_toxic}"
        )
        
        

        return self.get_wellness_status()

    def get_wellness_status(self) -> WellnessStatus:
        """Return the current wellness status."""
        score = round(self._wellness_score, 1)
        level = _score_to_level(score)

        total_calls = len(self._call_history)
        toxic_calls = sum(1 for c in self._call_history if c["is_toxic"])

        
        needs_break = (
            score < 30 or
            self._consecutive_toxic >= 3 or
            level == "critical"
        )

        
        alert = ""
        if level == "critical":
            alert = f"BURNOUT ALERT: Wellness at {score}/100. Break required NOW."
        elif level == "high":
            alert = f"Stress Alert: Wellness at {score}/100. Break recommended."
        elif self._consecutive_toxic >= 3:
            alert = f"3+ consecutive toxic calls. Agent needs immediate support."
        elif level == "moderate":
            alert = f"Mild stress detected. Wellness at {score}/100."

        return WellnessStatus(
            agent_id=self.agent_id,
            wellness_score=score,
            stress_level=level,
            total_calls_today=total_calls,
            toxic_calls_today=toxic_calls,
            consecutive_toxic_calls=self._consecutive_toxic,
            needs_break=needs_break,
            alert_message=alert,
            recommendation=RECOMMENDATIONS[level],
        )

    def apply_break(self, break_duration_minutes: int = 10):
        """
        Apply recovery for a completed break and return the updated status.
        """
        recovery = min(break_duration_minutes * 2.0, 40.0)  
        self._wellness_score = min(100.0, self._wellness_score + recovery)
        self._consecutive_toxic = 0  
        logger.info(f"Agent {self.agent_id} took a {break_duration_minutes}-min break. Score: {self._wellness_score:.1f}")
        return self.get_wellness_status()

    def to_state_dict(self) -> dict:
        """
        Return the mutable fields persisted in Redis.
        """
        return {
            "wellness_score": round(self._wellness_score, 2),
            "consecutive_toxic": self._consecutive_toxic,
            "total_calls": len(self._call_history),
            "toxic_calls": sum(1 for c in self._call_history if c.get("is_toxic")),
        }

    def get_shift_report(self) -> dict:
        """generates a summary for the whole shift"""
        status = self.get_wellness_status()
        shift_duration = (time.time() - self.shift_start) / 60  

        return {
            "agent_id": self.agent_id,
            "shift_duration_minutes": round(shift_duration, 1),
            "final_wellness_score": status.wellness_score,
            "final_stress_level": status.stress_level,
            "total_calls": status.total_calls_today,
            "toxic_calls": status.toxic_calls_today,
            "toxic_percentage": round(
                (status.toxic_calls_today / status.total_calls_today * 100)
                if status.total_calls_today > 0 else 0, 1
            ),
            "recommendation": status.recommendation,
        }


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AgentShield - Agent Wellness Tracker Test")
    print("="*55)

    tracker = WellnessTracker(agent_id="test_agent_001")

    
    calls = [
        ("sess_01", 120, 0.0,  "safe",     "positive"),
        ("sess_02", 90,  0.0,  "safe",     "neutral"),
        ("sess_03", 240, 0.45, "warning",  "negative"),
        ("sess_04", 300, 0.74, "danger",   "negative"),
        ("sess_05", 420, 0.82, "critical", "negative"),  
        ("sess_06", 360, 0.75, "danger",   "negative"),  
        ("sess_07", 60,  0.0,  "safe",     "positive"),  
    ]

    print(f"\n  Simulating shift for agent: test_agent_001")
    print("-"*55)

    for sess_id, duration, score, level, sentiment in calls:
        status = tracker.log_call(
            session_id=sess_id,
            duration_seconds=duration,
            peak_toxicity_score=score,
            peak_toxicity_level=level,
            customer_sentiment=sentiment,
        )

        print(f"\n  Call {sess_id} [{level.upper()}]")
        print(f"    Wellness Score : {status.wellness_score}/100")
        print(f"    Stress Level   : {status.stress_level.upper()}")
        if status.alert_message:
            print(f"    *** {status.alert_message} ***")
        if status.needs_break:
            print(f"    --> Recommending a BREAK!")

    
    print("\n" + "-"*55)
    print("  Agent takes a 10-minute break...")
    after_break = tracker.apply_break(10)
    print(f"  Score after break: {after_break.wellness_score}/100")

    
    print("\n" + "="*55)
    print("  SHIFT REPORT")
    print("="*55)
    report = tracker.get_shift_report()
    for k, v in report.items():
        print(f"  {k:<30} : {v}")
    print("="*55 + "\n")
