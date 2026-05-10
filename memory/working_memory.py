"""
memory/working_memory.py — WorkingMemory

Redis-backed working memory for live conversation state.
Stores the PipelineContext per session with TTL auto-expiry.

Keys:
  wm:{session_id}          → serialised PipelineContext (JSON)
  wm:{session_id}:messages → list of Message objects (JSON)
  wm:{user_id}:patterns    → current confirmed patterns (JSON, longer TTL)

All methods are async (redis.asyncio).
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis
import structlog

from config import get_settings
from schemas.session import Message, PipelineContext

logger = structlog.get_logger(__name__)
settings = get_settings()


class WorkingMemory:
    """
    Thin async wrapper around Redis for session-scoped working memory.

    Lifecycle:
      1. Orchestrator calls save_context() after each state transition.
      2. Orchestrator calls load_context() to resume a session.
      3. On session close, flush_to_episodic() is called, then delete_session().
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    # ── Context ────────────────────────────────────────────────────────────

    async def save_context(self, ctx: PipelineContext) -> None:
        """Persist full pipeline context to Redis."""
        r = self._get_redis()
        key = f"wm:{ctx.session_id}"
        value = ctx.model_dump_json()
        await r.set(key, value, ex=settings.redis_working_memory_ttl)
        logger.debug("working_memory.saved", session_id=ctx.session_id, state=ctx.state.value)

    async def load_context(self, session_id: str) -> PipelineContext | None:
        """Load pipeline context for a session, or None if expired/missing."""
        r = self._get_redis()
        key = f"wm:{session_id}"
        raw = await r.get(key)
        if raw is None:
            logger.debug("working_memory.miss", session_id=session_id)
            return None
        ctx = PipelineContext.model_validate_json(raw)
        logger.debug("working_memory.loaded", session_id=session_id, state=ctx.state.value)
        return ctx

    async def delete_session(self, session_id: str) -> None:
        """Remove all keys for a session (called after persist to episodic)."""
        r = self._get_redis()
        keys = [
            f"wm:{session_id}",
            f"wm:{session_id}:messages",
            f"wm:{session_id}:clarification",
        ]
        if keys:
            await r.delete(*keys)
        logger.info("working_memory.deleted", session_id=session_id)

    # ── Messages ───────────────────────────────────────────────────────────

    async def append_message(self, session_id: str, message: Message) -> None:
        """Append a message to the session's conversation list."""
        r = self._get_redis()
        key = f"wm:{session_id}:messages"
        await r.rpush(key, message.model_dump_json())
        await r.expire(key, settings.redis_working_memory_ttl)

    async def get_messages(self, session_id: str) -> list[Message]:
        """Retrieve all messages for a session."""
        r = self._get_redis()
        key = f"wm:{session_id}:messages"
        raw_list = await r.lrange(key, 0, -1)
        return [Message.model_validate_json(raw) for raw in raw_list]

    # ── Clarification state ────────────────────────────────────────────────

    async def set_clarification_pending(
        self,
        session_id: str,
        questions: list[str],
    ) -> None:
        r = self._get_redis()
        key = f"wm:{session_id}:clarification"
        await r.set(
            key,
            json.dumps(questions),
            ex=settings.redis_working_memory_ttl,
        )

    async def get_clarification_pending(self, session_id: str) -> list[str] | None:
        r = self._get_redis()
        key = f"wm:{session_id}:clarification"
        raw = await r.get(key)
        return json.loads(raw) if raw else None

    async def clear_clarification_pending(self, session_id: str) -> None:
        r = self._get_redis()
        await r.delete(f"wm:{session_id}:clarification")

    # ── Pattern cache ──────────────────────────────────────────────────────

    async def cache_patterns(
        self,
        user_id: str,
        patterns: list[dict[str, Any]],
    ) -> None:
        """Cache the user's current confirmed patterns (short TTL)."""
        r = self._get_redis()
        key = f"wm:{user_id}:patterns"
        await r.set(
            key,
            json.dumps(patterns),
            ex=settings.redis_pattern_cache_ttl,
        )

    async def get_cached_patterns(self, user_id: str) -> list[dict[str, Any]] | None:
        r = self._get_redis()
        key = f"wm:{user_id}:patterns"
        raw = await r.get(key)
        return json.loads(raw) if raw else None

    async def invalidate_pattern_cache(self, user_id: str) -> None:
        r = self._get_redis()
        await r.delete(f"wm:{user_id}:patterns")

    # ── Rate limiting ──────────────────────────────────────────────────────

    async def check_rate_limit(
        self,
        user_id: str,
        limit: int = 10,
        window_seconds: int = 3600,
    ) -> tuple[bool, int]:
        """
        Token-bucket rate limiter. Returns (allowed, remaining_tokens).
        """
        r = self._get_redis()
        key = f"ratelimit:{user_id}"
        count = await r.incr(key)
        if count == 1:
            await r.expire(key, window_seconds)
        remaining = max(0, limit - count)
        allowed = count <= limit
        if not allowed:
            logger.warning("working_memory.rate_limited", user_id=user_id, count=count)
        return allowed, remaining

    # ── Health ─────────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        try:
            r = self._get_redis()
            return await r.ping()
        except Exception as exc:
            logger.error("working_memory.ping_failed", error=str(exc))
            return False

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()