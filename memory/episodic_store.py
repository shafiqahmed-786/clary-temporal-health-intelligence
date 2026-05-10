"""
memory/episodic_store.py — EpisodicStore

Per-user episodic memory backed by ChromaDB.
Stores HealthEvents as embeddings with rich temporal metadata,
enabling both semantic similarity search and time-range filtering.

Two query strategies:
  1. semantic_search(query, user_id) → find past sessions by meaning
  2. temporal_range_query(user_id, start, end) → find sessions by date
  3. Combined: semantic_search with temporal pre-filter (the key pattern)

All embeddings are computed via OpenAI's async API (text-embedding-3-small)
and passed directly to ChromaDB — bypassing ChromaDB's sync embedding function.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

import chromadb
import openai
import structlog

from config import get_settings
from schemas.event import HealthEvent

logger = structlog.get_logger(__name__)
settings = get_settings()

# ChromaDB collection name pattern: "episodic_{user_id}"
_COLLECTION_PREFIX = settings.episodic_collection_prefix


class EpisodicStore:
    """
    Per-user episodic memory store.

    One ChromaDB collection per user: "episodic_{user_id}".
    Collections are created lazily on first access.

    Thread/async safety: all public methods are async.
    ChromaDB's HTTP client handles concurrent requests internally.
    """

    def __init__(self) -> None:
        self._client: chromadb.AsyncHttpClient | None = None
        self._collections: dict[str, Any] = {}  # user_id → collection
        self._openai: openai.AsyncOpenAI | None = None

    async def _get_client(self) -> chromadb.AsyncHttpClient:
        if self._client is None:
            self._client = await chromadb.AsyncHttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
        return self._client

    def _get_openai(self) -> openai.AsyncOpenAI:
        if self._openai is None:
            self._openai = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        return self._openai

    async def _get_collection(self, user_id: str) -> Any:
        """Get or create the ChromaDB collection for a user."""
        if user_id not in self._collections:
            client = await self._get_client()
            name = f"{_COLLECTION_PREFIX}_{user_id}"
            collection = await client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[user_id] = collection
            logger.info("episodic_store.collection_ready", user_id=user_id, name=name)
        return self._collections[user_id]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via OpenAI async API."""
        oai = self._get_openai()
        response = await oai.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    # ── Write ──────────────────────────────────────────────────────────────

    async def store_event(self, event: HealthEvent) -> None:
        """Embed and store a single HealthEvent."""
        collection = await self._get_collection(event.user_id)
        text = event.to_embedding_text()
        embeddings = await self._embed([text])
        metadata = event.to_chroma_metadata()

        await collection.upsert(
            ids=[str(event.event_id)],
            embeddings=embeddings,
            documents=[text],
            metadatas=[metadata],
        )
        logger.debug(
            "episodic_store.stored",
            user_id=event.user_id,
            event_id=str(event.event_id),
            session_id=event.session_id,
        )

    async def store_events(self, events: list[HealthEvent]) -> None:
        """Batch store multiple HealthEvents efficiently."""
        if not events:
            return

        # Group by user_id for batch embedding
        by_user: dict[str, list[HealthEvent]] = {}
        for ev in events:
            by_user.setdefault(ev.user_id, []).append(ev)

        for user_id, user_events in by_user.items():
            collection = await self._get_collection(user_id)
            texts = [e.to_embedding_text() for e in user_events]
            embeddings = await self._embed(texts)
            ids = [str(e.event_id) for e in user_events]
            metadatas = [e.to_chroma_metadata() for e in user_events]

            await collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(
                "episodic_store.batch_stored",
                user_id=user_id,
                count=len(user_events),
            )

    # ── Read: semantic ─────────────────────────────────────────────────────

    async def semantic_search(
        self,
        user_id: str,
        query: str,
        top_k: int | None = None,
        symptom_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find past sessions most semantically similar to query.

        Returns list of {document, metadata, distance} dicts.
        """
        k = top_k or settings.episodic_semantic_top_k
        collection = await self._get_collection(user_id)
        query_embedding = await self._embed([query])

        where: dict = {"user_id": {"$eq": user_id}}
        if symptom_filter:
            where["symptoms"] = {"$contains": symptom_filter.lower()}

        results = await collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return self._unpack_results(results)

    # ── Read: temporal range ───────────────────────────────────────────────

    async def temporal_range_query(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
        symptom_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve all events in a date range — used for lag window lookback.

        This is the fundamental temporal query: "what happened in the
        6 weeks before this symptom event?"
        """
        collection = await self._get_collection(user_id)

        start_epoch = start.timestamp()
        end_epoch = end.timestamp()

        where: dict[str, Any] = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"timestamp_epoch": {"$gte": start_epoch}},
                {"timestamp_epoch": {"$lte": end_epoch}},
            ]
        }

        if symptom_filter:
            norm = symptom_filter.lower()
            where["$and"].append({"symptoms": {"$contains": norm}})

        # Fetch without semantic ranking (get all matching)
        results = await collection.get(
            where=where,
            include=["documents", "metadatas"],
        )

        logger.debug(
            "episodic_store.temporal_range",
            user_id=user_id,
            start=start.isoformat(),
            end=end.isoformat(),
            found=len(results.get("ids", [])),
        )

        return self._unpack_get_results(results)

    async def get_all_events(self, user_id: str) -> list[dict[str, Any]]:
        """Retrieve the complete event history for a user (for full timeline build)."""
        collection = await self._get_collection(user_id)
        results = await collection.get(
            where={"user_id": {"$eq": user_id}},
            include=["documents", "metadatas"],
        )
        events = self._unpack_get_results(results)
        # Sort by timestamp ascending
        events.sort(key=lambda e: e["metadata"].get("timestamp_epoch", 0))
        logger.info(
            "episodic_store.get_all",
            user_id=user_id,
            total_events=len(events),
        )
        return events

    async def get_recent_events(
        self, user_id: str, n: int | None = None
    ) -> list[dict[str, Any]]:
        """Return the N most recent events for context window assembly."""
        n = n or settings.episodic_recent_n
        all_events = await self.get_all_events(user_id)
        return all_events[-n:]

    async def get_events_by_symptom(
        self, user_id: str, symptom: str
    ) -> list[dict[str, Any]]:
        """Return all events containing a specific symptom."""
        collection = await self._get_collection(user_id)
        results = await collection.get(
            where={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"symptoms": {"$contains": symptom.lower()}},
                ]
            },
            include=["documents", "metadatas"],
        )
        return self._unpack_get_results(results)

    # ── Lag window query (composite) ───────────────────────────────────────

    async def lag_window_query(
        self,
        user_id: str,
        symptom_event_date: datetime,
        max_lag_days: int,
    ) -> list[dict[str, Any]]:
        """
        Retrieve all events in the lag window BEFORE a symptom event.
        This is called by PatternAgent for every symptom occurrence.
        """
        end = symptom_event_date
        start = end - timedelta(days=max_lag_days)
        return await self.temporal_range_query(user_id=user_id, start=start, end=end)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack_results(results: dict) -> list[dict[str, Any]]:
        """Unpack ChromaDB query() results (returns lists of lists)."""
        out = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            out.append({
                "id": doc_id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else 1.0,
            })
        return out

    @staticmethod
    def _unpack_get_results(results: dict) -> list[dict[str, Any]]:
        """Unpack ChromaDB get() results (returns flat lists)."""
        out = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for i, doc_id in enumerate(ids):
            out.append({
                "id": doc_id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": None,
            })
        return out

    async def event_count(self, user_id: str) -> int:
        """Return total number of stored events for a user."""
        collection = await self._get_collection(user_id)
        results = await collection.get(
            where={"user_id": {"$eq": user_id}},
            include=[],
        )
        return len(results.get("ids", []))