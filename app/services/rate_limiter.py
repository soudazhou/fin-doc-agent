# =============================================================================
# Rate Limiter — Redis-Based Per-Key Sliding Window
# =============================================================================
#
# Implements a sliding window counter using Redis sorted sets (ZSET).
# Each request adds an entry with its timestamp as the score. On each
# check, entries older than the window are pruned and the remaining
# count is compared against the limit.
#
# DESIGN DECISION: Sliding window over fixed window. Fixed windows
# allow burst traffic at window boundaries (e.g., 100 requests at
# 0:59 + 100 at 1:00 = 200 in 2 seconds). Sliding windows distribute
# the limit evenly.
#
# DESIGN DECISION: Graceful degradation. If Redis is unavailable,
# rate limiting is silently bypassed (log a warning, allow the request).
# This prevents Redis outages from blocking the entire API.
#
# Uses Redis db 2 (db 0/1 reserved for Celery).
# =============================================================================

from __future__ import annotations

import logging
import time

from fastapi import HTTPException

from app.config import settings
from app.db.models import ApiKey

logger = logging.getLogger(__name__)

# Lazy Redis connection
_redis_client = None


def _get_rate_limit_redis():
    """Lazily create and cache the async Redis client for rate limiting."""
    global _redis_client
    if _redis_client is None:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(
            settings.rate_limit_redis_url,
            decode_responses=True,
        )
    return _redis_client


async def check_rate_limit(api_key: ApiKey | None) -> None:
    """
    Check whether the request should be rate-limited.

    Uses a sliding window counter per API key.
    Window = 60 seconds, limit = api_key.rate_limit_rpm or settings default.

    Raises:
        HTTPException 429: Rate limit exceeded (includes Retry-After header).

    No-op when:
    - Auth is disabled (api_key is None)
    - Redis is unavailable (graceful degradation)
    """
    if api_key is None:
        return

    limit = api_key.rate_limit_rpm or settings.rate_limit_rpm
    redis_key = f"ratelimit:apikey:{api_key.id}"
    window_seconds = 60

    try:
        r = _get_rate_limit_redis()
        now = time.time()
        window_start = now - window_seconds

        pipe = r.pipeline()
        # Remove entries outside the window
        pipe.zremrangebyscore(redis_key, 0, window_start)
        # Count entries in the window
        pipe.zcard(redis_key)
        # Add current request
        pipe.zadd(redis_key, {str(now): now})
        # Set TTL to auto-cleanup
        pipe.expire(redis_key, window_seconds + 10)
        results = await pipe.execute()

        current_count = results[1]  # zcard result

        if current_count >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. "
                f"Limit: {limit} requests/minute.",
                headers={"Retry-After": str(window_seconds)},
            )

    except HTTPException:
        raise  # Re-raise 429
    except Exception as e:
        # Redis unavailable — graceful degradation
        logger.warning(
            "Rate limiter unavailable (Redis error): %s. "
            "Allowing request through.",
            e,
        )
