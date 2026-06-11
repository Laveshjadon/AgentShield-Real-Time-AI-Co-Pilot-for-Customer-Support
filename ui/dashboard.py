"""NiceGUI dashboard for live agent support.

Run with:
    python -m ui.dashboard
"""

import asyncio
import json
import httpx
from nicegui import ui, app
from datetime import datetime

from config.logger import get_logger

logger = get_logger("ui.dashboard")




API_BASE = "http://localhost:8080"




class AppState:
    def __init__(self):
        self.session_id: str | None = None
        self.agent_id: str = "agent_001"
        self.call_active: bool = False
        self.wellness_score: float = 100.0
        self.stress_level: str = "low"
        self.transcript_entries: list[dict] = []
        self.toxicity_score: float = 0.0
        self.toxicity_level: str = "safe"
        self.last_suggestion: str = ""
        self.alert_message: str = ""





async def api_post(endpoint: str, body: dict) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(f"{API_BASE}{endpoint}", json=body)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.error(f"API POST {endpoint} failed: {e}")
        return None

async def api_get(endpoint: str) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{API_BASE}{endpoint}")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.error(f"API GET {endpoint} failed: {e}")
        return None





LEVEL_COLORS = {
    "safe":     "#22c55e",  
    "warning":  "#eab308",  
    "danger":   "#f97316",  
    "critical": "#ef4444",  
}

LEVEL_BG = {
    "safe":     "bg-green-900/30",
    "warning":  "bg-yellow-900/40",
    "danger":   "bg-orange-900/50",
    "critical": "bg-red-900/60",
}

WELLNESS_COLORS = {
    "low":      "#22c55e",
    "moderate": "#eab308",
    "high":     "#f97316",
    "critical": "#ef4444",
}





def create_dashboard():
    state = AppState()

    
    ui.add_head_html("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
      * { font-family: 'Inter', sans-serif; box-sizing: border-box; }
      body { background: #0f1117; color: #e2e8f0; margin: 0; }
      .card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
      }
      .card-title {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 12px;
      }
      .transcript-entry {
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 13px;
        line-height: 1.5;
        animation: fadeIn 0.3s ease;
      }
      .transcript-customer {
        background: #1e3a5f;
        border-left: 3px solid #3b82f6;
      }
      .transcript-agent {
        background: #1a2e1a;
        border-left: 3px solid #22c55e;
      }
      .suggestion-box {
        background: linear-gradient(135deg, #1a2744 0%, #1e3a5f 100%);
        border: 1px solid #3b82f6;
        border-radius: 10px;
        padding: 16px;
        font-size: 13px;
        line-height: 1.7;
        white-space: pre-wrap;
        min-height: 120px;
      }
      .toxicity-bar-outer {
        background: #1e2535;
        border-radius: 99px;
        height: 12px;
        width: 100%;
        overflow: hidden;
      }
      .toxicity-bar-inner {
        height: 12px;
        border-radius: 99px;
        transition: width 0.5s ease, background 0.5s ease;
      }
      .wellness-circle {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        margin: 0 auto 10px;
        font-size: 22px;
        font-weight: 700;
        border: 4px solid;
        transition: border-color 0.5s ease, color 0.5s ease;
      }
      .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
      }
      .alert-banner {
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        animation: pulse 1s ease infinite;
      }
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.75; }
      }
      .nicegui-input input {
        background: #1a1f2e !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 8px !important;
      }
      .header-bar {
        background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 100%);
        border-bottom: 1px solid #2d3748;
        padding: 12px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .logo-text {
        font-size: 20px;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
    </style>
    <style>
      .mic-btn {
        border: none; cursor: pointer; border-radius: 8px;
        padding: 8px 16px; font-weight: 700; font-size: 13px;
        transition: all 0.3s ease;
      }
      .mic-btn-idle {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
      }
      .mic-btn-recording {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        animation: micPulse 1.2s ease infinite;
      }
      @keyframes micPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
        50%      { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
      }
      .speaker-btn {
        border: none; cursor: pointer; border-radius: 6px;
        padding: 6px 14px; font-weight: 600; font-size: 11px;
        letter-spacing: 0.5px; transition: all 0.2s ease;
      }
      .speaker-btn-active {
        color: white;
      }
      .speaker-btn-inactive {
        background: #1e2535; color: #64748b;
      }
      .recording-indicator {
        display: flex; align-items: center; gap: 6px;
        font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
      }
      .rec-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #ef4444; animation: micPulse 1.2s ease infinite;
      }
    </style>
    <script>
    window.AgentShieldAudio = (function() {
        let audioContext = null;
        let mediaStream = null;
        let workletNode = null;
        let ws = null;
        let chunkIndex = 0;
        let speaker = 'customer';
        let isRecording = false;
        let queue = [];
        let pingInterval = null;
        let sampleBuffer = [];
        let recognition = null;
        const TARGET_RATE = 16000;
        const CHUNK_SAMPLES = 4000;

        function resample(input, fromRate, toRate) {
            if (fromRate === toRate) return input;
            const ratio = fromRate / toRate;
            const len = Math.floor(input.length / ratio);
            const out = new Float32Array(len);
            for (let i = 0; i < len; i++) {
                const srcIdx = i * ratio;
                const idx = Math.floor(srcIdx);
                const frac = srcIdx - idx;
                out[i] = idx + 1 < input.length
                    ? input[idx] * (1 - frac) + input[idx + 1] * frac
                    : input[idx];
            }
            return out;
        }

        function float32ToInt16Base64(float32Arr) {
            const pcm16 = new Int16Array(float32Arr.length);
            for (let i = 0; i < float32Arr.length; i++) {
                const s = Math.max(-1, Math.min(1, float32Arr[i]));
                pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            const bytes = new Uint8Array(pcm16.buffer);
            let binary = '';
            for (let i = 0; i < bytes.length; i += 8192) {
                binary += String.fromCharCode.apply(null, bytes.subarray(i, Math.min(i + 8192, bytes.length)));
            }
            return btoa(binary);
        }

        function sendChunk(samples) {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            const b64 = float32ToInt16Base64(samples);
            ws.send(JSON.stringify({
                type: 'audio',
                chunk_index: chunkIndex++,
                audio_data: b64,
                audio_format: 'pcm_s16le',
                sample_rate: TARGET_RATE,
                speaker: speaker
            }));
        }

        function startLiveSpeechRecognition() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                queue.push({
                    type: 'info',
                    message: 'Live browser captions are not supported in this browser.'
                });
                return;
            }

            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-IN';

            recognition.onresult = function(event) {
                let interim = '';
                let finalText = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript.trim();
                    if (event.results[i].isFinal) {
                        finalText += transcript + ' ';
                    } else {
                        interim += transcript + ' ';
                    }
                }

                if (interim.trim()) {
                    queue.push({
                        type: 'live_transcription',
                        speaker: speaker,
                        text: interim.trim()
                    });
                }
                if (finalText.trim()) {
                    queue.push({
                        type: 'browser_transcription',
                        speaker: speaker,
                        text: finalText.trim()
                    });
                }
            };

            recognition.onerror = function(event) {
                if (event.error === 'no-speech' || event.error === 'aborted') {
                    return;
                }
                queue.push({
                    type: 'error',
                    message: 'Browser speech recognition error: ' + event.error
                });
            };

            recognition.onend = function() {
                if (isRecording && recognition) {
                    try { recognition.start(); } catch(e) {}
                }
            };

            try {
                recognition.start();
            } catch(e) {
                queue.push({
                    type: 'error',
                    message: 'Could not start browser live captions'
                });
            }
        }

        return {
            async startCapture(sessionId, apiBase, initialSpeaker) {
                if (isRecording) {
                    speaker = initialSpeaker;
                    return true;
                }
                speaker = initialSpeaker;
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({
                        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
                    });

                    audioContext = new AudioContext();
                    const source = audioContext.createMediaStreamSource(mediaStream);
                    const nativeRate = audioContext.sampleRate;

                    // Use ScriptProcessorNode (widely supported)
                    const processor = audioContext.createScriptProcessor(4096, 1, 1);
                    source.connect(processor);
                    processor.connect(audioContext.destination);

                    // WebSocket
                    const wsBase = apiBase.replace(/^http/, 'ws');
                    ws = new WebSocket(wsBase + '/api/calls/ws/audio/' + sessionId);

                    ws.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            if (data.type === 'transcription') {
                                queue.push(data);
                            } else if (data.type === 'error') {
                                queue.push(data);
                            }
                        } catch(e) {}
                    };

                    ws.onerror = function() {
                        queue.push({type: 'error', message: 'Audio WebSocket connection failed'});
                    };
                    ws.onclose = function() {
                        if (isRecording) {
                            queue.push({type: 'error', message: 'Audio connection closed'});
                        }
                    };

                    chunkIndex = 0;
                    sampleBuffer = [];
                    isRecording = true;
                    startLiveSpeechRecognition();

                    processor.onaudioprocess = function(e) {
                        if (!isRecording) return;
                        const input = e.inputBuffer.getChannelData(0);
                        const resampled = resample(input, nativeRate, TARGET_RATE);

                        for (let i = 0; i < resampled.length; i++) {
                            sampleBuffer.push(resampled[i]);
                        }

                        while (sampleBuffer.length >= CHUNK_SAMPLES) {
                            const chunk = new Float32Array(sampleBuffer.splice(0, CHUNK_SAMPLES));
                            sendChunk(chunk);
                        }
                    };

                    workletNode = processor;

                    pingInterval = setInterval(function() {
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({ type: 'ping' }));
                        }
                    }, 10000);

                    return true;
                } catch(err) {
                    console.error('[AgentShield] Mic error:', err);
                    return false;
                }
            },

            stopCapture() {
                isRecording = false;
                if (recognition) {
                    recognition.onend = null;
                    try { recognition.stop(); } catch(e) {}
                    recognition = null;
                }
                if (pingInterval) { clearInterval(pingInterval); pingInterval = null; }
                if (workletNode) { workletNode.disconnect(); workletNode = null; }
                if (audioContext) { audioContext.close().catch(function(){}); audioContext = null; }
                if (mediaStream) { mediaStream.getTracks().forEach(function(t){ t.stop(); }); mediaStream = null; }

                if (ws && ws.readyState === WebSocket.OPEN) {
                    if (sampleBuffer.length > 0) {
                        const finalChunk = new Float32Array(CHUNK_SAMPLES);
                        finalChunk.set(sampleBuffer.slice(0, CHUNK_SAMPLES));
                        sendChunk(finalChunk);
                    }

                    // Supply enough trailing silence for VAD to close the utterance.
                    const silence = new Float32Array(CHUNK_SAMPLES);
                    sendChunk(silence);
                    sendChunk(silence);
                    sendChunk(silence);

                    const closingSocket = ws;
                    setTimeout(function() {
                        closingSocket.close();
                    }, 1000);
                    ws = null;
                } else if (ws) {
                    ws.close();
                    ws = null;
                }
                sampleBuffer = [];
            },

            setSpeaker(s) { speaker = s; },
            getSpeaker() { return speaker; },
            isActive() { return isRecording; },

            dequeue() { return queue.shift() || null; },
            queueLength() { return queue.length; }
        };
    })();
    </script>
    """)

    
    transcript_col = None
    suggestion_label = None
    tox_bar = None
    tox_level_badge = None
    tox_score_label = None
    wellness_circle = None
    wellness_label = None
    alert_banner = None
    header_wellness = None
    start_btn = None
    end_btn = None

    
    def refresh_toxicity_ui():
        pct = int(state.toxicity_score * 100)
        color = LEVEL_COLORS.get(state.toxicity_level, "#22c55e")
        tox_bar.style(f"width:{pct}%; background:{color};")
        tox_score_label.set_text(f"{pct}%")
        tox_level_badge.set_text(state.toxicity_level.upper())
        tox_level_badge.style(f"background:{color}33; color:{color};")

        if state.alert_message:
            alert_banner.set_text(f"⚠ {state.alert_message}")
            alert_banner.style(f"background:{color}22; border:1px solid {color}; color:{color};")
            alert_banner.set_visibility(True)
        else:
            alert_banner.set_visibility(False)

    def refresh_wellness_ui():
        score = int(state.wellness_score)
        color = WELLNESS_COLORS.get(state.stress_level, "#22c55e")
        wellness_circle.style(f"border-color:{color}; color:{color};")
        wellness_label.set_text(f"{score}/100")
        header_wellness.set_text(f"Wellness: {score}/100")

    def add_transcript_entry(speaker: str, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        css_class = "transcript-customer" if speaker == "customer" else "transcript-agent"
        speaker_label = "Customer" if speaker == "customer" else "Agent"
        with transcript_col:
            ui.html(f"""
                <div class="transcript-entry {css_class}">
                    <span style="font-size:10px;opacity:0.6;">[{ts}] {speaker_label}</span><br/>
                    {text}
                </div>
            """)

    
    
    with ui.element("div").classes("header-bar"):
        ui.html('<span class="logo-text">🛡 AgentShield</span>')
        with ui.row().classes("items-center gap-4"):
            ui.label(f"Agent: {state.agent_id}").style("color:#94a3b8; font-size:13px;")
            header_wellness = ui.label("Wellness: 100/100").style("color:#22c55e; font-size:13px; font-weight:600;")
            status_dot = ui.html('<span style="width:8px;height:8px;background:#ef4444;border-radius:50%;display:inline-block;"></span>')

    
    with ui.row().classes("w-full gap-4 p-4").style("min-height: calc(100vh - 60px); background: #0f1117;"):

        
        with ui.column().classes("flex-1").style("min-width:300px; max-width:380px;"):
            with ui.element("div").classes("card").style("height:100%;"):
                ui.html('<div class="card-title">Live Transcript</div>')
                transcript_col = ui.column().classes("w-full").style(
                    "max-height: calc(100vh - 220px); overflow-y:auto; gap:0;"
                )
                with transcript_col:
                    ui.html("""
                        <div style="text-align:center; color:#475569; font-size:12px; padding:20px 0;">
                            Start a call to see the transcript
                        </div>
                    """)

        
        with ui.column().classes("flex-1").style("min-width:320px;"):
            with ui.element("div").classes("card").style("height:100%;"):
                ui.html('<div class="card-title">AI Suggestion</div>')
                suggestion_label = ui.html(
                    '<div class="suggestion-box" style="color:#475569;">Waiting for customer to speak...</div>'
                ).style("width:100%;")

                ui.html('<div class="card-title" style="margin-top:16px;">Alert</div>')
                alert_banner = ui.label("No active alerts").classes("alert-banner").style(
                    "background:#1e2535; color:#475569; border:1px solid #2d3748;"
                )
                alert_banner.set_visibility(True)

        
        with ui.column().style("width:260px; gap:16px;"):

            
            with ui.element("div").classes("card"):
                ui.html('<div class="card-title">Toxicity Meter</div>')
                with ui.row().classes("items-center justify-between mb-2"):
                    tox_score_label = ui.label("0%").style("font-size:13px; font-weight:600; color:#22c55e;")
                    tox_level_badge = ui.label("SAFE").classes("badge").style(
                        "background:#22c55e33; color:#22c55e;"
                    )
                with ui.element("div").classes("toxicity-bar-outer"):
                    tox_bar = ui.element("div").classes("toxicity-bar-inner").style(
                        "width:0%; background:#22c55e;"
                    )

            
            with ui.element("div").classes("card"):
                ui.html('<div class="card-title">Agent Wellness</div>')
                wellness_circle = ui.element("div").classes("wellness-circle").style(
                    "border-color:#22c55e; color:#22c55e;"
                )
                with wellness_circle:
                    wellness_label = ui.label("100").style("font-size:22px; font-weight:700;")
                    ui.label("/ 100").style("font-size:10px; opacity:0.6;")

                async def take_break():
                    data = await api_get(f"/api/wellness/{state.agent_id}/status")
                    if data:
                        state.wellness_score = data["wellness_score"]
                        state.stress_level = data["stress_level"]
                        refresh_wellness_ui()
                    await api_post(f"/api/wellness/{state.agent_id}/break", {"duration_minutes": 10})
                    ui.notify("Break logged! Wellness score recovering.", color="positive")

                ui.button("Log Break (10 min)", on_click=take_break).props("flat").style(
                    "width:100%; font-size:12px; color:#60a5fa; margin-top:8px;"
                )

            
            with ui.element("div").classes("card"):
                ui.html('<div class="card-title">Quick Knowledge Search</div>')
                search_input = ui.input(placeholder="Search knowledge base...").style(
                    "width:100%; font-size:13px;"
                )
                search_result = ui.label("").style(
                    "font-size:12px; color:#94a3b8; margin-top:8px; white-space:pre-wrap; line-height:1.5;"
                )

                async def do_search():
                    q = search_input.value.strip()
                    if not q:
                        return
                    data = await api_post("/api/knowledge/search", {"query": q})
                    if data and data.get("found"):
                        search_result.set_text(data["results"][:300] + "...")
                    else:
                        search_result.set_text("No results found.")

                ui.button("Search", on_click=do_search).props("flat").style(
                    "width:100%; font-size:12px; background:#1e3a5f; color:#60a5fa; margin-top:6px; border-radius:8px;"
                )

    
    with ui.footer().style("background:#13182a; border-top:1px solid #2d3748; padding:14px 24px;"):
        with ui.row().classes("w-full items-center gap-3"):

            
            mic_recording = False
            current_speaker = "customer"
            ws_poll_timer = None

            
            mic_cust_btn_ref = None
            mic_agent_btn_ref = None
            rec_indicator_ref = None
            rec_indicator_label_ref = None
            live_caption_ref = None

            
            async def process_ws_transcription(data: dict):
                """deals with the transcribed text coming from the js side"""
                if data.get("type") == "error":
                    message = data.get("message", "Microphone audio processing failed")
                    logger.warning("Dashboard audio error: %s", message)
                    ui.notify(message, color="negative", close_button=True)
                    return
                if data.get("type") == "info":
                    ui.notify(data.get("message", ""), color="warning", close_button=True)
                    return
                if data.get("type") == "live_transcription":
                    text = data.get("text", "")
                    if text and live_caption_ref:
                        live_caption_ref.set_text(text)
                    return
                if data.get("type") == "browser_transcription":
                    speaker = data.get("speaker", "customer")
                    text = data.get("text", "").strip()
                    if live_caption_ref:
                        live_caption_ref.set_text("")
                    if not text:
                        return

                    if speaker == "customer":
                        api_data = await api_post(
                            "/api/calls/analyse-text",
                            {
                                "session_id": state.session_id,
                                "speaker": speaker,
                                "text": text,
                            },
                        )
                        if api_data:
                            data = {
                                "speaker": api_data.get("speaker", "customer"),
                                "text": api_data.get("transcribed_text", ""),
                                "suggestion": api_data.get("ai_suggestion") or "",
                                "toxicity_score": api_data.get("toxicity_score", 0.0),
                                "toxicity_level": api_data.get("toxicity_level", "safe"),
                                "alert_message": api_data.get("alert_message") or "",
                            }
                        else:
                            add_transcript_entry(speaker, text)
                            return
                    else:
                        add_transcript_entry(speaker, text)
                        return

                speaker = data.get("speaker", "customer")
                text = data.get("text", "")
                suggestion = data.get("suggestion") or ""
                tox_score = data.get("toxicity_score", 0.0)
                tox_level = data.get("toxicity_level", "safe")
                alert_msg = data.get("alert_message") or ""

                if not text:
                    return

                
                add_transcript_entry(speaker, text)

                
                state.toxicity_score = tox_score
                state.toxicity_level = tox_level
                state.alert_message = alert_msg
                refresh_toxicity_ui()

                
                if speaker == "customer" and suggestion and suggestion.strip():
                    formatted = (
                        suggestion
                        .replace("\n", "<br/>")
                        .replace("* ", "<br/>&bull; ")
                    )
                    suggestion_label.set_content(
                        f'<div class="suggestion-box">{formatted}</div>'
                    )
                elif speaker == "customer":
                    suggestion_label.set_content(
                        '<div class="suggestion-box">'
                        '<span style="color:#eab308;font-weight:600;">No KB match found</span>'
                        '<br/><br/>'
                        '<span style="color:#94a3b8;">'
                        'No policy found for this query.<br/>'
                        'Use your judgement or search the Knowledge Base panel.'
                        '</span>'
                        '</div>'
                    )

            
            async def poll_ws_queue():
                """pulls items out of the queue from javascript"""
                try:
                    result = await ui.run_javascript(
                        'return window.AgentShieldAudio.dequeue();',
                        timeout=2.0,
                    )
                    if result:
                        if isinstance(result, str):
                            result = json.loads(result)
                        await process_ws_transcription(result)
                except Exception:
                    pass  

            
            async def start_call():
                data = await api_post("/api/calls/start", {"agent_id": state.agent_id})
                if not data:
                    ui.notify(
                        "Failed to start call — backend unreachable. "
                        "Make sure the FastAPI server (port 8080) and Redis are running.",
                        color="negative",
                        close_button=True,
                    )
                    return
                state.session_id = data["session_id"]
                state.call_active = True
                start_btn.set_visibility(False)
                end_btn.set_visibility(True)
                status_dot.set_content('<span style="width:8px;height:8px;background:#22c55e;border-radius:50%;display:inline-block;animation:pulse 1.5s infinite;"></span>')
                with transcript_col:
                    ui.html("""
                        <div class="transcript-entry transcript-agent">
                            <span style="font-size:10px;opacity:0.6;">System</span><br/>
                            Call session started. Listening...
                        </div>
                    """)
                ui.notify(f"Call started: {state.session_id}", color="positive")

            
            async def toggle_customer_mic():
                if not state.call_active:
                    ui.notify("Start a call first!", color="negative")
                    return
                    
                nonlocal current_speaker, mic_recording, ws_poll_timer
                if mic_recording and current_speaker == "customer":
                    await ui.run_javascript('window.AgentShieldAudio.stopCapture();')
                    mic_recording = False
                    mic_cust_btn_ref.style("background: linear-gradient(135deg, #3b82f6, #2563eb); animation: none;")
                    rec_indicator_ref.set_visibility(False)
                else:
                    current_speaker = "customer"
                    api_base = "http://localhost:8080"
                    success = await ui.run_javascript(
                        f'return window.AgentShieldAudio.startCapture("{state.session_id}", "{api_base}", "customer");',
                        timeout=15.0,
                    )
                    if success:
                        mic_recording = True
                        mic_cust_btn_ref.style("background: linear-gradient(135deg, #ef4444, #dc2626); animation: pulse 2s infinite;")
                        mic_agent_btn_ref.style("background: linear-gradient(135deg, #10b981, #059669); animation: none;")
                        rec_indicator_label_ref.set_text("Listening to Customer...")
                        rec_indicator_ref.set_visibility(True)
                        if not ws_poll_timer:
                            ws_poll_timer = ui.timer(0.3, poll_ws_queue)
                        ws_poll_timer.activate()
                    else:
                        ui.notify("Mic access denied or error.", color="negative")

            async def toggle_agent_mic():
                if not state.call_active:
                    ui.notify("Start a call first!", color="negative")
                    return
                    
                nonlocal current_speaker, mic_recording, ws_poll_timer
                if mic_recording and current_speaker == "agent":
                    await ui.run_javascript('window.AgentShieldAudio.stopCapture();')
                    mic_recording = False
                    mic_agent_btn_ref.style("background: linear-gradient(135deg, #10b981, #059669); animation: none;")
                    rec_indicator_ref.set_visibility(False)
                else:
                    current_speaker = "agent"
                    api_base = "http://localhost:8080"
                    success = await ui.run_javascript(
                        f'return window.AgentShieldAudio.startCapture("{state.session_id}", "{api_base}", "agent");',
                        timeout=15.0,
                    )
                    if success:
                        mic_recording = True
                        mic_agent_btn_ref.style("background: linear-gradient(135deg, #ef4444, #dc2626); animation: pulse 2s infinite;")
                        mic_cust_btn_ref.style("background: linear-gradient(135deg, #3b82f6, #2563eb); animation: none;")
                        rec_indicator_label_ref.set_text("Listening to Agent...")
                        rec_indicator_ref.set_visibility(True)
                        if not ws_poll_timer:
                            ws_poll_timer = ui.timer(0.3, poll_ws_queue)
                        ws_poll_timer.activate()
                    else:
                        ui.notify("Mic access denied or error.", color="negative")

            
            async def end_call():
                nonlocal mic_recording, ws_poll_timer
                if not state.session_id:
                    return
                
                if mic_recording:
                    await ui.run_javascript('window.AgentShieldAudio.stopCapture();')
                    mic_recording = False
                    mic_cust_btn_ref.style("background: linear-gradient(135deg, #3b82f6, #2563eb); animation: none;")
                    mic_agent_btn_ref.style("background: linear-gradient(135deg, #10b981, #059669); animation: none;")
                    rec_indicator_ref.set_visibility(False)
                if ws_poll_timer:
                    ws_poll_timer.cancel()
                    ws_poll_timer = None

                await api_post(
                    f"/api/calls/end/{state.session_id}?agent_id={state.agent_id}",
                    {"agent_rating": 75.0},
                )
                state.call_active = False
                state.session_id = None
                start_btn.set_visibility(True)
                end_btn.set_visibility(False)
                status_dot.set_content('<span style="width:8px;height:8px;background:#ef4444;border-radius:50%;display:inline-block;"></span>')
                ui.notify("Call ended.", color="warning")

            
            async def analyse_input():
                text = text_input.value.strip()
                if not text or not state.call_active:
                    if not state.call_active:
                        ui.notify("Start a call first!", color="negative")
                    return

                text_input.value = ""
                add_transcript_entry("customer", text)

                suggestion_label.set_content(
                    '<div class="suggestion-box" style="color:#64748b;font-style:italic;">'
                    'Analysing query...'
                    '</div>'
                )

                data = await api_post("/api/calls/analyse-text", {
                    "session_id": state.session_id,
                    "text": text,
                    "speaker": "customer"
                })

                if not data:
                    suggestion_label.set_content(
                        '<div class="suggestion-box" style="color:#ef4444;">'
                        '<strong>API Error</strong><br/>'
                        'Could not reach the backend on port 8080.'
                        '</div>'
                    )
                    return

                state.toxicity_score = data.get("toxicity_score", 0.0)
                state.toxicity_level = data.get("toxicity_level", "safe")
                state.alert_message  = data.get("alert_message") or ""
                refresh_toxicity_ui()

                suggestion = data.get("ai_suggestion") or ""
                if suggestion and suggestion.strip():
                    formatted = (
                        suggestion
                        .replace("\n", "<br/>")
                        .replace("* ", "<br/>&bull; ")
                    )
                    suggestion_label.set_content(
                        f'<div class="suggestion-box">{formatted}</div>'
                    )
                else:
                    suggestion_label.set_content(
                        '<div class="suggestion-box">'
                        '<span style="color:#eab308;font-weight:600;">No KB match found</span>'
                        '<br/><br/>'
                        '<span style="color:#94a3b8;">'
                        'No policy found for this query.<br/>'
                        'Use your judgement or search the Knowledge Base panel.'
                        '</span>'
                        '</div>'
                    )

                wellness_data = await api_get(f"/api/wellness/{state.agent_id}/status")
                if wellness_data:
                    state.wellness_score = wellness_data["wellness_score"]
                    state.stress_level   = wellness_data["stress_level"]
                    refresh_wellness_ui()

            
            start_btn = ui.button("START CALL", on_click=start_call).style(
                "background: linear-gradient(135deg, #16a34a, #15803d); color:white; "
                "font-weight:700; font-size:13px; border-radius:8px; padding:8px 20px;"
            )

            end_btn = ui.button("END CALL", on_click=end_call).style(
                "background: linear-gradient(135deg, #dc2626, #b91c1c); color:white; "
                "font-weight:700; font-size:13px; border-radius:8px; padding:8px 20px;"
            )
            end_btn.set_visibility(False)

            
            ui.element("div").style("width:1px; height:28px; background:#2d3748;")

            
            mic_cust_btn_ref = ui.button("🎤 CUSTOMER MIC", on_click=toggle_customer_mic).style(
                "background: linear-gradient(135deg, #3b82f6, #2563eb); color:white; "
                "font-weight:700; font-size:13px; border-radius:8px; padding:8px 16px;"
            )

            
            mic_agent_btn_ref = ui.button("🎤 AGENT MIC", on_click=toggle_agent_mic).style(
                "background: linear-gradient(135deg, #10b981, #059669); color:white; "
                "font-weight:700; font-size:13px; border-radius:8px; padding:8px 16px;"
            )

            
            with ui.row().classes("items-center gap-2").style("margin-left:12px;") as rec_indicator_ref:
                ui.element("div").style(
                    "width:10px; height:10px; border-radius:50%; "
                    "background:#ef4444; animation: pulse 2s infinite;"
                )
                rec_indicator_label_ref = ui.label("Listening...").style("color:#ef4444; font-weight:600; font-size:13px;")
            rec_indicator_ref.set_visibility(False)

            live_caption_ref = ui.label("").style(
                "min-width:220px; max-width:420px; color:#facc15; font-size:13px; "
                "font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"
            )

            
            ui.element("div").style("width:1px; height:28px; background:#2d3748;")

            
            text_input = ui.input(placeholder="Type speech and press Enter...").style(
                "flex:1; font-size:13px;"
            )
            text_input.on("keydown.enter", analyse_input)

            ui.button("Analyse", on_click=analyse_input).style(
                "background: linear-gradient(135deg, #2563eb, #1d4ed8); color:white; "
                "font-weight:600; font-size:13px; border-radius:8px; padding:8px 20px;"
            )


@ui.page("/")
def index():
    create_dashboard()


if __name__ == "__main__":
    ui.run(
        host="0.0.0.0",
        port=8081,
        title="AgentShield",
        dark=True,
        reload=False
    )
