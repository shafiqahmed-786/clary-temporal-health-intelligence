"""
memory/summary_compressor.py — SummaryCompressor

Compresses sessions older than COMPRESS_AFTER_DAYS into a rolling
summary to keep the context window bounded regardless of conversation length.

The summary preserves:
  - All confirmed patterns (with dates)
  - All symptom occurrences with dates
  - Key behavior/lifestyle signals
  - User-reported improvements

Raw sessions are never deleted — they remain in PostgreSQL and ChromaDB
for audit and deep analysis — but excluded from the live context window.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import openai
import structlog

from config import get_settings
from memory.episodic_store import EpisodicStore

logger = structlog.get_logger(__name__)
settings = get_settings()

_COMPRESS_SYSTEM_PROMPT = """
You are a medical data summariser for a health AI assistant called Clary.
Your job is to compress a list of older health session records into a
concise rolling summary that preserves ALL medically relevant information.

The summary MUST preserve:
1. Every symptom occurrence with its date (e.g., "acidity on Jan 5, Jan 28")
2. Every identified trigger with its date
3. Any confirmed patterns already noted
4. Any improvements or positive changes reported
5. Any behavioral trends (sleep, diet, exercise)

The summary must NOT:
- Diagnose conditions
- Include opinions or analysis (just facts from the sessions)
- Exceed 300 words

Format: concise prose with dates. Start with: "Health history summary:"
""".strip()


class SummaryCompressor:
    """
    Builds and caches rolling summaries of a user's older sessions.

    Cache strategy:
      - Summary stored in Redis with 24-hour TTL.
      - Rebuilt when: TTL expired, new old session added, or explicit invalidate().
    """

    def __init__(
        self,
        episodic: EpisodicStore,
        working_memory: Any,  # WorkingMemory — avoid circular import with string hint
    ) -> None:
        self._episodic = episodic
        self._wm = working_memory
        self._openai: openai.AsyncOpenAI | None = None

    def _get_openai(self) -> openai.AsyncOpenAI:
        if self._openai is None:
            self._openai = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        return self._openai

    async def get_or_build_summary(self, user_id: str) -> str:
        """
        Return cached summary or build a new one from old sessions.
        Returns empty string if no old sessions exist.
        """
        # Check cache
        r = self._wm._get_redis()
        cache_key = f"summary:{user_id}"
        cached = await r.get(cache_key)
        if cached:
            logger.debug("summary_compressor.cache_hit", user_id=user_id)
            return cached

        return await self._build_and_cache(user_id)

    async def invalidate(self, user_id: str) -> None:
        """Invalidate the cached summary so it rebuilds on next access."""
        r = self._wm._get_redis()
        await r.delete(f"summary:{user_id}")
        logger.info("summary_compressor.invalidated", user_id=user_id)

    async def _build_and_cache(self, user_id: str) -> str:
        """Build summary from old sessions and cache it."""
        cutoff = datetime.utcnow() - timedelta(days=settings.summary_compress_after_days)

        all_events = await self._episodic.get_all_events(user_id)
        old_events = [
            e for e in all_events
            if e.get("metadata", {}).get("timestamp_epoch", 0) < cutoff.timestamp()
        ]

        if not old_events:
            return ""

        sessions_text = self._format_sessions_for_compression(old_events)

        oai = self._get_openai()
        try:
            response = await oai.chat.completions.create(
                model=settings.openai_model,
                temperature=0.1,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": _COMPRESS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Compress these {len(old_events)} health sessions:\n\n"
                            + sessions_text
                        ),
                    },
                ],
            )
            summary = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("summary_compressor.llm_error", error=str(exc), user_id=user_id)
            summary = self._fallback_summary(old_events)

        # Cache for 24 hours
        r = self._wm._get_redis()
        await r.set(f"summary:{user_id}", summary, ex=86400)
        logger.info(
            "summary_compressor.built",
            user_id=user_id,
            source_events=len(old_events),
            summary_chars=len(summary),
        )
        return summary

    @staticmethod
    def _format_sessions_for_compression(events: list[dict]) -> str:
        lines = []
        for e in events:
            meta = e.get("metadata", {})
            epoch = meta.get("timestamp_epoch", 0)
            date = datetime.fromtimestamp(epoch).strftime("%Y-%m-%d") if epoch else "?"
            symptoms = meta.get("symptoms", "—")
            triggers = meta.get("triggers", "—")
            doc = e.get("document", "")[:150]
            lines.append(f"[{date}] symptoms=[{symptoms}] triggers=[{triggers}] | {doc}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_summary(events: list[dict]) -> str:
        """Simple non-LLM fallback if OpenAI call fails."""
        symptom_dates: dict[str, list[str]] = {}
        for e in events:
            meta = e.get("metadata", {})
            epoch = meta.get("timestamp_epoch", 0)
            date = datetime.fromtimestamp(epoch).strftime("%Y-%m-%d") if epoch else "?"
            for sym in (meta.get("symptoms", "") or "").split(","):
                sym = sym.strip()
                if sym:
                    symptom_dates.setdefault(sym, []).append(date)

        parts = ["Health history summary:"]
        for sym, dates in symptom_dates.items():
            parts.append(f"  - {sym}: reported on {', '.join(dates)}")
        return "\n".join(parts)