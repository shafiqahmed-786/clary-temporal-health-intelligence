"""
graph/graph_schema.py — Neo4j property graph schema definitions.

Defines:
  - NodeLabel enum
  - RelationshipType enum
  - Node/edge property specs as dataclasses
  - Cypher DDL for constraints and indexes (run once at startup)

Design principles:
  - Node identity is always a unique string ID (user_id, session_id, etc.)
  - Timestamps stored as both ISO string (human) and epoch int (range query)
  - Causal edges carry lag_days and confidence
  - All relationship properties are nullable (schema-lenient)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeLabel(str, Enum):
    """Neo4j node labels."""
    USER = "User"
    SESSION = "Session"
    SYMPTOM = "Symptom"
    TRIGGER = "Trigger"
    PATTERN = "Pattern"
    MECHANISM = "Mechanism"   # Biological mechanism from lag registry


class RelationshipType(str, Enum):
    """Neo4j relationship types."""
    HAS_SESSION = "HAS_SESSION"
    REPORTED_SYMPTOM = "REPORTED_SYMPTOM"
    CONTAINS_TRIGGER = "CONTAINS_TRIGGER"
    PRECEDES = "PRECEDES"              # Session → Session (temporal chain)
    SIMILAR_TO = "SIMILAR_TO"          # Session → Session (semantic similarity)
    CAUSED_BY = "CAUSED_BY"            # Symptom → Trigger (causal)
    EXPLAINED_BY = "EXPLAINED_BY"      # Pattern → Mechanism
    EVIDENCE_FOR = "EVIDENCE_FOR"      # Session → Pattern
    DOWNSTREAM_OF = "DOWNSTREAM_OF"    # Symptom → Symptom (cascade)


# ── Property schemas ───────────────────────────────────────────────────────────

@dataclass
class UserNode:
    user_id: str
    name: str = ""
    age: int | None = None
    location: str = ""
    created_at: str = ""

    def to_props(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}


@dataclass
class SessionNode:
    session_id: str
    user_id: str
    timestamp_iso: str
    timestamp_epoch: float
    severity: str = "unknown"
    summary: str = ""
    symptoms: str = ""   # comma-separated
    triggers: str = ""   # comma-separated

    def to_props(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SymptomNode:
    name: str                    # normalised lowercase
    display_name: str = ""
    body_location: str = ""

    def to_props(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}


@dataclass
class TriggerNode:
    name: str                    # normalised lowercase
    display_name: str = ""
    trigger_type: str = "unknown"  # food|behavior|stress|environmental

    def to_props(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}


@dataclass
class PatternNode:
    pattern_id: str
    user_id: str
    symptom: str
    trigger: str
    status: str = "watching"
    confidence: str = "low"
    occurrence_count: int = 0
    lag_days_min: int = 0
    lag_days_max: int = 0
    first_detected_at: str = ""
    last_confirmed_at: str = ""

    def to_props(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class MechanismNode:
    name: str
    description: str = ""
    lag_min_days: int = 0
    lag_max_days: int = 0

    def to_props(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v or isinstance(v, int)}


# ── Relationship property schemas ──────────────────────────────────────────────

@dataclass
class PrecedesRel:
    lag_days: float
    same_symptom: bool = False
    shared_symptoms: str = ""

    def to_props(self) -> dict[str, Any]:
        return {"lag_days": self.lag_days, "same_symptom": self.same_symptom,
                "shared_symptoms": self.shared_symptoms}


@dataclass
class CausedByRel:
    confidence: str = "moderate"
    evidence_count: int = 0
    mean_lag_days: float = 0.0

    def to_props(self) -> dict[str, Any]:
        return {"confidence": self.confidence, "evidence_count": self.evidence_count,
                "mean_lag_days": self.mean_lag_days}


@dataclass
class EvidenceForRel:
    occurrence_number: int = 0
    lag_days_actual: float = 0.0

    def to_props(self) -> dict[str, Any]:
        return {"occurrence_number": self.occurrence_number, "lag_days_actual": self.lag_days_actual}


# ── Cypher DDL (run once at startup) ──────────────────────────────────────────

CONSTRAINT_STATEMENTS = [
    # Unique node constraints
    "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
    "CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE",
    "CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT trigger_name_unique IF NOT EXISTS FOR (t:Trigger) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT pattern_id_unique IF NOT EXISTS FOR (p:Pattern) REQUIRE p.pattern_id IS UNIQUE",
    "CREATE CONSTRAINT mechanism_name_unique IF NOT EXISTS FOR (m:Mechanism) REQUIRE m.name IS UNIQUE",
]

INDEX_STATEMENTS = [
    # Temporal indexes for range queries
    "CREATE INDEX session_epoch_idx IF NOT EXISTS FOR (s:Session) ON (s.timestamp_epoch)",
    "CREATE INDEX session_user_idx IF NOT EXISTS FOR (s:Session) ON (s.user_id)",
    "CREATE INDEX pattern_user_idx IF NOT EXISTS FOR (p:Pattern) ON (p.user_id)",
    "CREATE INDEX pattern_status_idx IF NOT EXISTS FOR (p:Pattern) ON (p.status)",
]

ALL_DDL = CONSTRAINT_STATEMENTS + INDEX_STATEMENTS