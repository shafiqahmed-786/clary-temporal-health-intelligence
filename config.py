"""
config.py — Centralised settings via pydantic-settings.
All env vars loaded from .env; never hardcode secrets.
"""

from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o", description="Chat completion model")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model"
    )
    openai_temperature: float = Field(0.1, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(4096, ge=256)

    # ── ChromaDB ──────────────────────────────────────────────────────────
    chroma_host: str = Field("localhost")
    chroma_port: int = Field(8000)
    chroma_persist_dir: str = Field("./data/chroma")
    # Collections
    episodic_collection_prefix: str = Field("episodic")
    semantic_collection_name: str = Field("medical_knowledge")

    # ── Neo4j ─────────────────────────────────────────────────────────────
    neo4j_uri: str = Field("bolt://localhost:7687")
    neo4j_user: str = Field("neo4j")
    neo4j_password: str = Field(...)
    neo4j_database: str = Field("neo4j")

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379/0")
    redis_working_memory_ttl: int = Field(14400, description="Seconds — 4 hours")
    redis_pattern_cache_ttl: int = Field(300, description="5 min pattern cache")

    # ── PostgreSQL ────────────────────────────────────────────────────────
    postgres_url: str = Field(
        "postgresql+asyncpg://clary:clary@localhost:5432/clary"
    )

    # ── Agent behaviour ───────────────────────────────────────────────────
    max_agent_retries: int = Field(3, ge=1, le=10)
    retry_base_delay: float = Field(1.0, ge=0.1)
    retry_max_delay: float = Field(30.0)

    # ── Temporal reasoning ────────────────────────────────────────────────
    min_evidence_count: int = Field(2, description="Candidate pattern threshold")
    confirmed_evidence_count: int = Field(3, description="Confirmed pattern threshold")
    max_lag_days: int = Field(84, description="12 weeks — max temporal look-back")
    min_trigger_consistency: float = Field(
        1.0, description="Trigger must appear in ALL occurrences"
    )

    # ── Context window budgets (tokens) ───────────────────────────────────
    ctx_profile_tokens: int = Field(100)
    ctx_summary_tokens: int = Field(350)
    ctx_recent_sessions_tokens: int = Field(900)
    ctx_semantic_tokens: int = Field(400)
    ctx_patterns_tokens: int = Field(250)
    ctx_current_tokens: int = Field(250)

    # ── Retrieval ─────────────────────────────────────────────────────────
    episodic_recent_n: int = Field(10, description="Last N verbatim sessions")
    episodic_semantic_top_k: int = Field(3, description="Top-K similarity hits")
    summary_compress_after_days: int = Field(30, description="Compress older sessions")

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_min_f1: float = Field(0.85)
    eval_temporal_tolerance_days: int = Field(7)

    @field_validator("openai_api_key")
    @classmethod
    def key_not_empty(cls, v: str) -> str:
        if not v or v == "sk-...":
            raise ValueError("openai_api_key must be set in .env")
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]