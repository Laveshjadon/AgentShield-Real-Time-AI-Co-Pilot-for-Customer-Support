"""Sets up the AgentShield API and its routes."""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler  
from slowapi.errors import RateLimitExceeded

from src.api.limiter import limiter
from src.api import routes
from config.settings import Settings
from config.logger import get_logger
from src.core.session import init_redis, close_redis, ping_redis

logger = get_logger("api.main")
settings = Settings()



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown hooks."""
    logger.info("AgentShield API starting up...")
    logger.info(f"Server: {settings.APP_HOST}:{settings.APP_PORT}")

    
    await init_redis()

    from src.core.session import get_vad, get_transcriber, get_tts
    logger.info("Pre-loading ML models globally...")
    get_vad()
    get_transcriber()
    get_tts()

    # print the registered route table so we can debug 404s
    _call_routes = [
        f"  {list(r.methods)} {r.path}"
        for r in app.routes
        if hasattr(r, "methods") and hasattr(r, "path")
    ]
    logger.info(f"Registered routes ({len(_call_routes)}):\n" + "\n".join(_call_routes))

    yield


    await close_redis()
    logger.info("AgentShield API shutting down.")



app = FastAPI(
    title="AgentShield API",
    description=(
        "Real-time AI assistant for call center agents. "
        "Provides live transcription, RAG-based suggestions, "
        "toxicity detection, and agent wellness monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please slow down requests."}
    )
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(routes.calls_router)
app.include_router(routes.knowledge_router)
app.include_router(routes.wellness_router)




@app.get("/api/debug/routes", tags=["Debug"])
async def debug_routes():
    """Returns every route FastAPI has registered. Use this to diagnose 404s."""
    return {
        "routes": [
            {"methods": sorted(r.methods), "path": r.path, "name": r.name}
            for r in app.routes
            if hasattr(r, "methods")
        ]
    }


@app.get("/", tags=["Health"])
@app.get("/api/info", tags=["Health"])
async def root():
    import os
    return {
        "service": "AgentShield API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "pid": os.getpid(),
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """checks if db and redis are alive. helpful for debugging"""
    from src.core.db import get_async_engine
    from sqlalchemy import text

    checks = {
        "api": "ok",
        "database": "unknown",
        "redis": "unknown",
    }

    try:
        engine = get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    checks["redis"] = "ok" if await ping_redis() else "unreachable (using in-memory)"

    # Only require api and database to be ok for 200 status
    all_ok = checks["api"] == "ok" and checks["database"] == "ok"
    status_code = 200 if all_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_ok else "degraded",
            "checks": checks,
            "timestamp": time.time()
        }
    )


@app.post("/api/analysis/toxicity", tags=["Analysis"])
@limiter.limit("60/minute")
async def analyse_toxicity(request: Request, body: dict):
    """Return toxicity scoring for a text sample."""
    from src.analysis.toxicity import ToxicityAnalyzer
    text = body.get("text", "")
    if not text:
        return JSONResponse(status_code=400, content={"error": "text field required"})

    analyser = ToxicityAnalyzer()
    result = analyser.analyse(text)
    return {
        "text": result.text,
        "score": result.score,
        "level": result.level,
        "is_toxic": result.is_toxic,
        "alert_message": result.alert_message,
        "flags": result.flags,
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )
