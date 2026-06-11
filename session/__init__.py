"""
session package for agentshield
================================
basically all the redis session stuff lives here - manages call sessions
and agent wellness state

you can import stuff like:
    from session.client import init_redis, close_redis, get_redis, ping_redis
    from session.session_manager import SessionManager, get_session_manager
    from session.wellness_store import WellnessStore, get_wellness_store
    from session.schemas import SessionRecord, TurnRecord, WellnessRecord, CallHistoryEntry
"""

from session.client import (
    init_redis,
    close_redis,
    get_redis,
    ping_redis,
    RedisUnavailableError,
)
from session.session_manager import SessionManager, get_session_manager
from session.wellness_store import WellnessStore, get_wellness_store
from session.schemas import (
    SessionRecord,
    TurnRecord,
    WellnessRecord,
    CallHistoryEntry,
    ActiveSessionSummary,
    ActiveAgentSummary,
)

__all__ = [
    "init_redis",
    "close_redis",
    "get_redis",
    "ping_redis",
    "RedisUnavailableError",
    "SessionManager",
    "get_session_manager",
    "WellnessStore",
    "get_wellness_store",
    "SessionRecord",
    "TurnRecord",
    "WellnessRecord",
    "CallHistoryEntry",
    "ActiveSessionSummary",
    "ActiveAgentSummary",
]
