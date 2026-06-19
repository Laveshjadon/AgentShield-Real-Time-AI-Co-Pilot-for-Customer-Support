"""
AgentShield - WebSocket Audio Client
====================================
this just connects to the websocket and sends your mic audio in pieces.

you need to install these first:
    pip install websockets sounddevice numpy

how to run it:
    python scripts/ws_audio_client.py --session-id <your_session_id>
"""

import argparse
import asyncio
import base64
import json
import logging
import sys

try:
    import websockets
    import sounddevice as sd
    import numpy as np
except ImportError:
    print("Missing dependencies. Run: pip install websockets sounddevice numpy")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ws_client")


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 250  
CHUNK_SAMPLES = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000.0))

class WSAudioClient:
    def __init__(self, session_id: str, server_url: str = "ws://localhost:8080"):
        self.session_id = session_id
        self.ws_url = f"{server_url}/api/calls/ws/audio/{session_id}"
        self.chunk_index = 0
        self.ws = None
        self.is_running = False

    async def connect(self):
        """try to connect, and keep trying if it fails"""
        while self.is_running:
            try:
                logger.info(f"Connecting to {self.ws_url}...")
                async with websockets.connect(self.ws_url) as ws:
                    self.ws = ws
                    logger.info("Connected successfully.")
                    
                    
                    heartbeat_task = asyncio.create_task(self.heartbeat_loop())
                    receive_task = asyncio.create_task(self.receive_loop())
                    
                    
                    done, pending = await asyncio.wait(
                        [heartbeat_task, receive_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    
                    for task in pending:
                        task.cancel()
                        
                    logger.warning("Disconnected. Reconnecting in 3 seconds...")
            except Exception as e:
                logger.error(f"Connection failed: {e}")
            
            self.ws = None
            if self.is_running:
                await asyncio.sleep(3.0)

    async def heartbeat_loop(self):
        """just pings every 10 secs so the connection doesn't drop"""
        try:
            while True:
                await asyncio.sleep(10)
                if self.ws and self.ws.open:
                    await self.ws.send(json.dumps({"type": "ping"}))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")

    async def receive_loop(self):
        """waits for the server to send back text and ai suggestions"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "pong":
                    pass 
                elif msg_type == "transcription":
                    print("\n" + "="*50)
                    print(f"👤 Customer: {data.get('text')}")
                    if data.get("toxicity_level") != "safe":
                        print(f"⚠️  Toxicity: {data.get('toxicity_level').upper()}")
                    if data.get('suggestion'):
                        print(f"🤖 Suggestion:\n{data.get('suggestion')}")
                    print("="*50 + "\n")
                elif msg_type == "error":
                    logger.error(f"Server error: {data.get('message')}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed by server.")
        except Exception as e:
            logger.error(f"Receive loop error: {e}")

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags):
        """gets called every time we have new audio from the mic"""
        if status:
            logger.warning(f"Audio status: {status}")
            
        if self.ws and self.ws.open:
            
            
            pcm16_data = (indata[:, 0] * 32767).astype(np.int16)
            b64_data = base64.b64encode(pcm16_data.tobytes()).decode("utf-8")
            
            payload = {
                "type": "audio",
                "chunk_index": self.chunk_index,
                "audio_data": b64_data
            }
            self.chunk_index += 1
            
            
            asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps(payload)),
                asyncio.get_running_loop()
            )

    async def start(self):
        self.is_running = True
        logger.info("Starting audio stream...")
        
        
        connect_task = asyncio.create_task(self.connect())
        
        
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=CHUNK_SAMPLES,
            callback=self.audio_callback
        )
        
        with stream:
            logger.info("Microphone active. Speak to test streaming. Press Ctrl+C to stop.")
            try:
                
                while self.is_running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping...")
                self.is_running = False
                connect_task.cancel()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True, help="Active session ID to stream audio to")
    parser.add_argument("--url", default="ws://localhost:8080", help="WebSocket base URL")
    args = parser.parse_args()
    
    client = WSAudioClient(args.session_id, args.url)
    try:
        asyncio.run(client.start())
    except KeyboardInterrupt:
        pass
