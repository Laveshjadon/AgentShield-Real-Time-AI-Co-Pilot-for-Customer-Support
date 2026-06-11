"""Tests for WebSocket audio streaming."""
import pytest
import asyncio
import base64
import json
import numpy as np
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from api.main import app
from api.routes import calls
from session.session_manager import get_session_manager
from session.wellness_store import get_wellness_store
import fakeredis.aioredis

@pytest.fixture
def fake_redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    asyncio.run(client.aclose())

@pytest.fixture
def test_client(fake_redis):
    
    from session.session_manager import SessionManager
    from session.wellness_store import WellnessStore
    import session.client
    
    
    import api.main
    async def mock_init_redis(): pass
    async def mock_close_redis(): pass
    api.main.init_redis = mock_init_redis
    api.main.close_redis = mock_close_redis
    
    sm = SessionManager(redis=fake_redis)
    ws = WellnessStore(redis=fake_redis)
    
    app.dependency_overrides[get_session_manager] = lambda: sm
    app.dependency_overrides[get_wellness_store] = lambda: ws

    class FakeSession:
        def __init__(self):
            self.turns = []

    class FakeManager:
        def __init__(self):
            self.session = FakeSession()

        async def process_audio_chunk(self, _audio, speaker="customer"):
            return None

    original_manager_factory = calls._get_or_create_manager
    calls._get_or_create_manager = lambda _session_id, _agent_id: FakeManager()
    
    with TestClient(app) as client:
        yield client, sm

    calls._get_or_create_manager = original_manager_factory
    app.dependency_overrides.clear()

def test_ws_rejects_invalid_session(test_client):
    client, sm = test_client
    
    with client.websocket_connect("/api/calls/ws/audio/invalid_session_id") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert data["status"] == 404

def test_ws_accepts_valid_session_and_heartbeat(test_client):
    client, sm = test_client
    
    
    asyncio.run(sm.create_session("sess_ws_1", "agent_1"))
    
    with client.websocket_connect("/api/calls/ws/audio/sess_ws_1") as websocket:
        
        websocket.send_json({"type": "ping"})
        data = websocket.receive_json()
        assert data["type"] == "pong"

def test_ws_processes_audio_chunks(test_client):
    client, sm = test_client
    
    asyncio.run(sm.create_session("sess_ws_2", "agent_1"))
    
    with client.websocket_connect("/api/calls/ws/audio/sess_ws_2") as websocket:
        
        
        pcm_data = np.zeros(4000, dtype=np.int16)
        b64_data = base64.b64encode(pcm_data.tobytes()).decode("utf-8")
        
        
        websocket.send_json({
            "type": "audio",
            "chunk_index": 0,
            "audio_data": b64_data,
            "audio_format": "pcm_s16le",
            "sample_rate": 16000,
        })
        
        websocket.send_json({
            "type": "audio",
            "chunk_index": 1,
            "audio_data": b64_data,
            "audio_format": "pcm_s16le",
            "sample_rate": 16000,
        })
        
        
        
        
        websocket.send_json({"type": "ping"})
        data = websocket.receive_json()
        assert data["type"] == "pong"

def test_ws_out_of_order_chunks(test_client):
    client, sm = test_client
    
    asyncio.run(sm.create_session("sess_ws_3", "agent_1"))
    
    with client.websocket_connect("/api/calls/ws/audio/sess_ws_3") as websocket:
        pcm_data = np.zeros(4000, dtype=np.int16)
        b64_data = base64.b64encode(pcm_data.tobytes()).decode("utf-8")
        
        
        websocket.send_json({
            "type": "audio",
            "chunk_index": 1,
            "audio_data": b64_data,
            "audio_format": "pcm_s16le",
            "sample_rate": 16000,
        })
        
        websocket.send_json({
            "type": "audio",
            "chunk_index": 0,
            "audio_data": b64_data,
            "audio_format": "pcm_s16le",
            "sample_rate": 16000,
        })
        
        
        websocket.send_json({"type": "ping"})
        data = websocket.receive_json()
        assert data["type"] == "pong"
