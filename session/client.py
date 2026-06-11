"""
redis connection setup stuff. basically keeping one pool open so we don't spam connections.
init_redis() sets it up when the app starts.
close_redis() cleans it up when we quit.
get_redis() is how you actually get a connection to use it somewhere else.
"""

from __future__ import annotations

from redis.asyncio import ConnectionPool, Redis
from redis.exceptions import RedisError

from config.settings import Settings
from config.logger import get_logger

logger = get_logger("session.client")
settings = Settings()


_pool: ConnectionPool | None = None
_fake_client = None


class RedisUnavailableError(RuntimeError):
    """throws an error if redis goes down or isn't started yet"""






async def init_redis() -> None:
    """starts the pool and checks if it actually works, otherwise crashes right away"""
    global _pool, _fake_client
    _pool = ConnectionPool.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=50,
    )

    
    client = Redis(connection_pool=_pool)
    try:
        await client.ping()
        logger.info("[Redis] Connected — pool initialised (max_connections=50)")
    except RedisError as exc:
        _pool = None
        if getattr(settings, "REDIS_ALLOW_FAKE", False):
            import fakeredis.aioredis

            _fake_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
            logger.warning(
                "[Redis] Unavailable; using in-memory fakeredis for local development."
            )
            return
        raise RedisUnavailableError(f"Redis ping failed: {exc}") from exc
    finally:
        await client.aclose()


async def close_redis() -> None:
    """closes everything down nicely so we don't leave connections hanging"""
    global _pool, _fake_client
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        logger.info("[Redis] Connection pool closed.")
    if _fake_client is not None:
        await _fake_client.aclose()
        _fake_client = None






def get_redis() -> Redis:
    """just grabs a connection from the pool. pretty fast and doesn't open new sockets."""
    if _fake_client is not None:
        return _fake_client
    if _pool is None:
        raise RedisUnavailableError(
            "Redis pool is not initialised. "
            "Ensure init_redis() was called during application startup."
        )
    return Redis(connection_pool=_pool)






async def ping_redis() -> bool:
    """just pings redis, returns false if it's dead instead of crashing"""
    try:
        client = get_redis()
        result = await client.ping()
        await client.aclose()
        return bool(result)
    except Exception as exc:
        logger.warning("[Redis] Health ping failed: %s", exc)
        return False
