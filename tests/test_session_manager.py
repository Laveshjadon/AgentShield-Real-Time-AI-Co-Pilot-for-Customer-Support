"""
tests for the session manager
using fakeredis so i don't have to spin up a real database every time
just run pytest to test it
"""

from __future__ import annotations

import json
import time
import pytest
import pytest_asyncio
import fakeredis.aioredis

from fastapi import HTTPException

from session.session_manager import SessionManager, SESSION_TTL_SECONDS
from session.wellness_store import WellnessStore, WELLNESS_TTL_SECONDS
from session.schemas import (
    TurnRecord,
    CallHistoryEntry,
    WellnessRecord,
)




@pytest_asyncio.fixture
async def fake_redis():
    """gives me a clean fake redis for each test run"""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def sm(fake_redis):
    """session manager but with fake redis"""
    return SessionManager(redis=fake_redis)


@pytest_asyncio.fixture
async def ws(fake_redis):
    """wellness store using fake redis so it doesn't break"""
    return WellnessStore(redis=fake_redis)




class TestCreateSession:
    async def test_create_session_success(self, sm: SessionManager):
        record = await sm.create_session("sess_001", "agent_a")
        assert record.session_id == "sess_001"
        assert record.agent_id == "agent_a"
        assert record.is_active is True
        assert record.turn_count == 0

    async def test_create_session_duplicate_raises_409(self, sm: SessionManager):
        await sm.create_session("sess_dup", "agent_a")
        with pytest.raises(HTTPException) as exc_info:
            await sm.create_session("sess_dup", "agent_b")
        assert exc_info.value.status_code == 409

    async def test_create_adds_to_active_set(self, sm: SessionManager, fake_redis):
        await sm.create_session("sess_set", "agent_a")
        members = await fake_redis.smembers("agentshield:active_sessions")
        assert "sess_set" in members

    async def test_create_sets_ttl_on_hash(self, sm: SessionManager, fake_redis):
        await sm.create_session("sess_ttl", "agent_a")
        ttl = await fake_redis.ttl("agentshield:session:sess_ttl")
        assert ttl > 0
        assert ttl <= SESSION_TTL_SECONDS


class TestGetSession:
    async def test_get_session_returns_record(self, sm: SessionManager):
        await sm.create_session("sess_get", "agent_a")
        record = await sm.get_session("sess_get")
        assert record is not None
        assert record.session_id == "sess_get"

    async def test_get_session_missing_returns_none(self, sm: SessionManager):
        result = await sm.get_session("nonexistent")
        assert result is None

    async def test_require_session_raises_404_when_missing(self, sm: SessionManager):
        with pytest.raises(HTTPException) as exc_info:
            await sm.require_session("nonexistent")
        assert exc_info.value.status_code == 404

    async def test_session_exists_true(self, sm: SessionManager):
        await sm.create_session("sess_ex", "agent_a")
        assert await sm.session_exists("sess_ex") is True

    async def test_session_exists_false_for_unknown(self, sm: SessionManager):
        assert await sm.session_exists("unknown_sess") is False


class TestAddTurn:
    async def test_add_turn_increments_count(self, sm: SessionManager):
        await sm.create_session("sess_turn", "agent_a")
        turn = TurnRecord(speaker="customer", text="Hello!")
        count = await sm.add_turn("sess_turn", turn)
        assert count == 1
        count2 = await sm.add_turn("sess_turn", TurnRecord(speaker="agent", text="Hi!"))
        assert count2 == 2

    async def test_add_turn_persisted_to_redis(self, sm: SessionManager, fake_redis):
        await sm.create_session("sess_persist", "agent_a")
        turn = TurnRecord(speaker="customer", text="Test text", toxicity_score=0.5, toxicity_level="warning")
        await sm.add_turn("sess_persist", turn)

        raw = await fake_redis.lrange("agentshield:session:sess_persist:turns", 0, -1)
        assert len(raw) == 1
        parsed = json.loads(raw[0])
        assert parsed["text"] == "Test text"
        assert parsed["speaker"] == "customer"

    async def test_get_turns_returns_correct_count(self, sm: SessionManager):
        await sm.create_session("sess_gturns", "agent_a")
        for i in range(5):
            await sm.add_turn("sess_gturns", TurnRecord(speaker="customer", text=f"msg {i}"))
        turns = await sm.get_turns("sess_gturns", max_turns=3)
        assert len(turns) == 3
        assert turns[-1].text == "msg 4"

    async def test_get_turns_refreshes_ttl(self, sm: SessionManager, fake_redis):
        await sm.create_session("sess_ttl_turn", "agent_a")
        await sm.add_turn("sess_ttl_turn", TurnRecord(speaker="customer", text="ping"))
        ttl = await fake_redis.ttl("agentshield:session:sess_ttl_turn:turns")
        assert ttl > 0

    async def test_get_transcript_formats_correctly(self, sm: SessionManager):
        await sm.create_session("sess_tscr", "agent_a")
        await sm.add_turn("sess_tscr", TurnRecord(speaker="customer", text="I need help"))
        await sm.add_turn("sess_tscr", TurnRecord(speaker="agent", text="Sure, let me check"))
        transcript = await sm.get_transcript("sess_tscr")
        assert "Customer: I need help" in transcript
        assert "Agent: Sure, let me check" in transcript


class TestEndSession:
    async def test_end_session_sets_inactive(self, sm: SessionManager):
        await sm.create_session("sess_end", "agent_a")
        record = await sm.end_session("sess_end")
        assert record.is_active is False

    async def test_end_session_removes_from_active_set(self, sm: SessionManager, fake_redis):
        await sm.create_session("sess_remove", "agent_a")
        await sm.end_session("sess_remove")
        members = await fake_redis.smembers("agentshield:active_sessions")
        assert "sess_remove" not in members

    async def test_session_data_still_readable_after_end(self, sm: SessionManager):
        """making sure the data doesn't get wiped immediately when the session ends"""
        await sm.create_session("sess_post_end", "agent_a")
        await sm.add_turn("sess_post_end", TurnRecord(speaker="customer", text="Last words"))
        await sm.end_session("sess_post_end")
        record = await sm.get_session("sess_post_end")
        assert record is not None
        assert record.turn_count == 1


class TestListActiveSessions:
    async def test_list_empty_when_no_sessions(self, sm: SessionManager):
        result = await sm.list_active_sessions()
        assert result == []

    async def test_list_returns_all_active(self, sm: SessionManager):
        await sm.create_session("sess_la1", "agent_a")
        await sm.create_session("sess_la2", "agent_b")
        sessions = await sm.list_active_sessions()
        ids = {s.session_id for s in sessions}
        assert "sess_la1" in ids
        assert "sess_la2" in ids

    async def test_list_excludes_ended_sessions(self, sm: SessionManager):
        await sm.create_session("sess_ended", "agent_a")
        await sm.create_session("sess_live", "agent_b")
        await sm.end_session("sess_ended")
        sessions = await sm.list_active_sessions()
        ids = {s.session_id for s in sessions}
        assert "sess_ended" not in ids
        assert "sess_live" in ids




class TestWellnessStoreInit:
    async def test_get_or_init_creates_fresh_record(self, ws: WellnessStore):
        record = await ws.get_or_init("agent_fresh")
        assert record.agent_id == "agent_fresh"
        assert record.wellness_score == 100.0
        assert record.consecutive_toxic == 0
        assert record.total_calls == 0

    async def test_get_or_init_idempotent(self, ws: WellnessStore):
        r1 = await ws.get_or_init("agent_idem")
        r2 = await ws.get_or_init("agent_idem")
        assert r1.wellness_score == r2.wellness_score

    async def test_get_returns_none_for_unknown(self, ws: WellnessStore):
        result = await ws.get("totally_unknown_agent")
        assert result is None

    async def test_adds_to_active_agents_set(self, ws: WellnessStore, fake_redis):
        await ws.get_or_init("agent_set")
        members = await fake_redis.smembers("agentshield:active_agents")
        assert "agent_set" in members

    async def test_ttl_set_on_wellness_key(self, ws: WellnessStore, fake_redis):
        await ws.get_or_init("agent_ttl")
        ttl = await fake_redis.ttl("agentshield:wellness:agent_ttl")
        assert ttl > 0
        assert ttl <= WELLNESS_TTL_SECONDS


class TestWellnessSaveState:
    async def test_save_state_persists_score(self, ws: WellnessStore):
        await ws.get_or_init("agent_save")
        await ws.save_state(
            "agent_save",
            wellness_score=72.5,
            consecutive_toxic=1,
            total_calls=5,
            toxic_calls=2,
        )
        record = await ws.get("agent_save")
        assert record is not None
        assert abs(record.wellness_score - 72.5) < 0.01
        assert record.consecutive_toxic == 1
        assert record.total_calls == 5
        assert record.toxic_calls == 2

    async def test_save_state_preserves_shift_start(self, ws: WellnessStore, fake_redis):
        await ws.get_or_init("agent_ss")
        original_start = float(await fake_redis.hget("agentshield:wellness:agent_ss", "shift_start"))
        await ws.save_state(
            "agent_ss",
            wellness_score=80.0,
            consecutive_toxic=0,
            total_calls=3,
            toxic_calls=0,
        )
        updated_start = float(await fake_redis.hget("agentshield:wellness:agent_ss", "shift_start"))
        assert abs(original_start - updated_start) < 1.0  


class TestWellnessHistory:
    async def test_append_call_history(self, ws: WellnessStore, fake_redis):
        await ws.get_or_init("agent_hist")
        entry = CallHistoryEntry(
            session_id="s1",
            duration_seconds=120.0,
            peak_toxicity_score=0.3,
            peak_toxicity_level="warning",
            is_toxic=True,
            wellness_score_after=87.0,
        )
        await ws.append_call_history("agent_hist", entry)
        raw = await fake_redis.lrange("agentshield:wellness:agent_hist:history", 0, -1)
        assert len(raw) == 1
        parsed = json.loads(raw[0])
        assert parsed["session_id"] == "s1"

    async def test_get_history_returns_entries(self, ws: WellnessStore):
        await ws.get_or_init("agent_ghist")
        for i in range(5):
            await ws.append_call_history(
                "agent_ghist",
                CallHistoryEntry(
                    session_id=f"s{i}",
                    duration_seconds=60.0,
                    peak_toxicity_score=0.0,
                    peak_toxicity_level="safe",
                    is_toxic=False,
                    wellness_score_after=100.0,
                ),
            )
        history = await ws.get_history("agent_ghist", max_entries=3)
        assert len(history) == 3

    async def test_history_capped_at_max_entries(self, ws: WellnessStore, fake_redis):
        """checking if it actually trims the list so we don't use up all the memory"""
        from session.wellness_store import HISTORY_MAX_ENTRIES
        await ws.get_or_init("agent_cap")
        
        for i in range(HISTORY_MAX_ENTRIES + 10):
            await ws.append_call_history(
                "agent_cap",
                CallHistoryEntry(
                    session_id=f"s{i}",
                    duration_seconds=30.0,
                    peak_toxicity_score=0.0,
                    peak_toxicity_level="safe",
                    is_toxic=False,
                    wellness_score_after=100.0,
                ),
            )
        length = await fake_redis.llen("agentshield:wellness:agent_cap:history")
        assert length <= HISTORY_MAX_ENTRIES


class TestWellnessApplyBreak:
    async def test_break_increases_score(self, ws: WellnessStore):
        await ws.get_or_init("agent_brk")
        await ws.save_state(
            "agent_brk",
            wellness_score=50.0,
            consecutive_toxic=2,
            total_calls=10,
            toxic_calls=4,
        )
        updated = await ws.apply_break("agent_brk", break_minutes=10)
        
        assert updated.wellness_score > 50.0
        assert abs(updated.wellness_score - 70.0) < 0.01

    async def test_break_caps_score_at_100(self, ws: WellnessStore):
        await ws.get_or_init("agent_cap_brk")
        await ws.save_state(
            "agent_cap_brk",
            wellness_score=95.0,
            consecutive_toxic=0,
            total_calls=2,
            toxic_calls=0,
        )
        updated = await ws.apply_break("agent_cap_brk", break_minutes=30)
        assert updated.wellness_score <= 100.0

    async def test_break_resets_consecutive_toxic(self, ws: WellnessStore):
        await ws.get_or_init("agent_cons")
        await ws.save_state(
            "agent_cons",
            wellness_score=40.0,
            consecutive_toxic=5,
            total_calls=8,
            toxic_calls=5,
        )
        updated = await ws.apply_break("agent_cons", break_minutes=10)
        assert updated.consecutive_toxic == 0


class TestListAllAgents:
    async def test_list_all_agents_returns_all(self, ws: WellnessStore):
        await ws.get_or_init("agent_all_1")
        await ws.get_or_init("agent_all_2")
        records = await ws.list_all_agents()
        ids = {r.agent_id for r in records}
        assert "agent_all_1" in ids
        assert "agent_all_2" in ids

    async def test_list_all_agents_empty_when_none(self, ws: WellnessStore):
        result = await ws.list_all_agents()
        assert result == []
