"""
Start the FastAPI backend and NiceGUI dashboard for local development.

FastAPI runs on port 8080 and the dashboard runs on port 8081.
"""

import os
import sys
import subprocess
import time
import urllib.request
import urllib.error
import socket
from config.settings import Settings
from config.logger import get_logger

settings = Settings()
logger = get_logger("main")

sys.stdout.reconfigure(encoding="utf-8")

# How long to wait for FastAPI to come up before giving up and opening the UI anyway.
_API_READY_TIMEOUT_SECONDS = 60
_API_HEALTH_URL = "http://127.0.0.1:8080/api/info"


def start_api():
    """Start the FastAPI server in a separate process."""
    logger.info("Starting FastAPI backend on port 8080...")
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "src.api.main:app",
            "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning",
        ],
        env={"PYTHONPATH": ".", **os.environ},
    )


def wait_for_api(timeout: int = _API_READY_TIMEOUT_SECONDS) -> bool:
    """
    Poll the API health endpoint until it responds or we time out.
    FastAPI loads Whisper + VAD + embeddings on startup so this can take 20-30s
    on first run. Without this the UI opens before the backend is ready and
    every button click returns 'backend unreachable'.
    """
    print(f"  Waiting for FastAPI to be ready (up to {timeout}s)...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(_API_HEALTH_URL, timeout=2) as resp:
                if resp.status == 200:
                    print("  FastAPI is ready.\n", flush=True)
                    return True
        except urllib.error.HTTPError:
            # Got an HTTP response (even 4xx/5xx) — the server port is open and
            # responding, which is all we need before opening the dashboard.
            print("  FastAPI is ready.\n", flush=True)
            return True
        except Exception:
            # Connection refused or timeout — server not up yet, keep polling.
            pass
        time.sleep(1)

    print(
        f"  WARNING: FastAPI did not respond within {timeout}s. "
        "Opening dashboard anyway — some features may not work yet.\n",
        flush=True,
    )
    return False


def free_port(port: int) -> None:
    """
    Kill every process holding the given port on Windows.
    The previous version used isdigit() on the raw output — that silently
    failed when Get-NetTCPConnection returned multiple lines (one PID per
    connection), so we never killed anything. Now we split and kill all of them.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) != 0:
            return  # port is free, nothing to do

    print(f"  Port {port} is in use — attempting to free it...", flush=True)
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             f"(Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue).OwningProcess"],
            capture_output=True, text=True, timeout=5,
        )
        pids = list({p for p in result.stdout.split() if p.isdigit()})
        for pid in pids:
            try:
                subprocess.run(["taskkill", "/PID", pid, "/F"],
                               timeout=5, capture_output=True)
            except Exception:
                pass
        if pids:
            print(f"  Killed PID(s) {', '.join(pids)} holding port {port}.", flush=True)
        time.sleep(2)  # give OS time to fully release the socket
    except Exception as exc:
        print(f"  Could not auto-free port {port}: {exc}. Kill it manually and retry.", flush=True)


def start_ui():
    """Run the NiceGUI dashboard."""
    from nicegui import ui as nicegui_ui
    import ui.app  # noqa: F401

    logger.info("Starting NiceGUI dashboard on port 8081...")
    nicegui_ui.run(
        host="0.0.0.0",
        port=8081,
        title="AgentShield",
        dark=True,
        reload=False,
        show=False,
    )


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  AgentShield - Starting All Services")
    print("=" * 55)
    print("  FastAPI Backend  -> http://localhost:8080")
    print("  API Docs         -> http://localhost:8080/docs")
    print("  Agent Dashboard  -> http://localhost:8081")
    print("=" * 55 + "\n")

    free_port(8080)
    free_port(8081)
    api_process = start_api()

    try:
        wait_for_api()
        start_ui()
    finally:
        api_process.terminate()
        api_process.wait()
