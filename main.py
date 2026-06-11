"""
Start the FastAPI backend and NiceGUI dashboard for local development.

FastAPI runs on port 8080 and the dashboard runs on port 8081.
"""

import sys
import subprocess
import time
from config.settings import Settings
from config.logger import get_logger

settings = Settings()
logger = get_logger("main")

sys.stdout.reconfigure(encoding="utf-8")


def start_api():
    """Start the FastAPI server in a separate process."""
    logger.info("Starting FastAPI backend on port 8080...")
    
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning"],
        env={"PYTHONPATH": ".", **sys.modules['os'].environ}
    )


def start_ui():
    """Run the NiceGUI dashboard."""
    from nicegui import ui as nicegui_ui
    import ui.dashboard  # noqa: F401

    logger.info("Starting NiceGUI dashboard on port 8081...")
    nicegui_ui.run(
        host="0.0.0.0",
        port=8081,
        title="AgentShield",
        dark=True,
        reload=False,
        show=True
    )


if __name__ == "__main__":
    import os
    print("\n" + "="*55)
    print("  AgentShield - Starting All Services")
    print("="*55)
    print("  FastAPI Backend  -> http://localhost:8080")
    print("  API Docs         -> http://localhost:8080/docs")
    print("  Agent Dashboard  -> http://localhost:8081")
    print("="*55 + "\n")

    
    api_process = start_api()

    try:
        
        start_ui()
    finally:
        api_process.terminate()
        api_process.wait()
