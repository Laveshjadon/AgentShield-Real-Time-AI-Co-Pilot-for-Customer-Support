"""Sets up the AgentShield API and its routes."""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler  
from slowapi.errors import RateLimitExceeded

from api.limiter import limiter
from api.routes import calls, knowledge, wellness
from config.settings import Settings
from config.logger import get_logger
from session.client import init_redis, close_redis, ping_redis

logger = get_logger("api.main")
settings = Settings()



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown hooks."""
    logger.info("AgentShield API starting up...")
    logger.info(f"Server: {settings.APP_HOST}:{settings.APP_PORT}")

    
    await init_redis()

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


app.include_router(calls.router)
app.include_router(knowledge.router)
app.include_router(wellness.router)




@app.get("/api/info", tags=["Health"])
async def root():
    return {
        "service": "AgentShield API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """checks if db and redis are alive. helpful for debugging"""
    from db.connection import get_async_engine
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

    checks["redis"] = "ok" if await ping_redis() else "error: unreachable"

    all_ok = all(v == "ok" for v in checks.values())
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
    from analysis.toxicity_analyzer import ToxicityAnalyzer
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
        "api.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )
