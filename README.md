# ⬡ Clary — Temporal Health Intelligence

> **Ask First · Production AI Architecture**  
> *A multi-agent health companion that detects causal patterns across weeks of conversation history*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35%2B-ff4b4b.svg)](https://streamlit.io)
[![OpenAI GPT-4o](https://img.shields.io/badge/LLM-GPT--4o-412991.svg)](https://openai.com)
[![Neo4j](https://img.shields.io/badge/graph-Neo4j-008CC1.svg)](https://neo4j.com)
[![ChromaDB](https://img.shields.io/badge/vectors-ChromaDB-FF6B6B.svg)](https://trychroma.com)

---

## Overview

Clary is a **production-grade, multi-agent AI system** that tracks user health conversations over time and surfaces medically grounded causal patterns — including patterns where the trigger and symptom are separated by **days or weeks**.

### The Core Problem

Standard health chatbots answer one question at a time. Clary answers *across time*:

```
Jan 8:   "I've been restricting calories to lose weight"
Feb 19:  "My hair has been falling out a lot lately"
```

A naive system treats these as two unrelated complaints.  
Clary connects them via **Telogen Effluvium** — a known biological mechanism with a 6–12 week lag — and surfaces the pattern with full evidence citations and calibrated confidence.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — Presentation (Streamlit)                                 │
│  Chat · Timeline · Patterns · Event Graph · Evaluation              │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 4 — Agent Pipeline                                           │
│                                                                     │
│  User Message                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  INTAKE  │──▶│ PATTERN  │──▶│ SKEPTIC  │──▶│ NARRATOR │        │
│  │  Agent   │   │  Agent   │   │  Agent   │   │  Agent   │        │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
│       │              │               │               │              │
│       └──────────────┴───────────────┴───────────────┴──┐          │
│                                                          ▼          │
│                                              ┌──────────────────┐   │
│                                              │ TRIAGE Agent     │   │
│                                              └──────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — Temporal Reasoning Engine                                │
│  EventTimeline · LagRegistry · CoOccurrence · VariableIsolator      │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — Memory & Retrieval                                       │
│  Episodic (ChromaDB) · Semantic (ChromaDB) · Working (Redis)        │
│  Event Graph (Neo4j) · Summary Compressor                           │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 1 — Infrastructure                                           │
│  PostgreSQL · ChromaDB · Neo4j · Redis · OpenAI · Celery            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Temporal Reasoning Design

This is the architectural centrepiece. Standard RAG retrieves semantically similar content. Clary additionally performs **temporal range queries** to find triggers that occurred *weeks before* the current symptom.

### Biological Lag Registry

Every symptom–trigger pair is matched against a registry of known biological mechanisms with observed lag windows:

| Mechanism | Trigger | Symptom | Lag Window |
|---|---|---|---|
| Telogen Effluvium | Caloric restriction | Hair fall | **6–12 weeks** |
| GERD | Late meal | Acidity | Same night |
| IGF-1 Acne | Dairy intake | Cheek/jaw acne | 48–72 hours |
| Reactive Hypoglycaemia | High-GI carbs | Energy crash | 90–150 min |
| Cortisol Dysmenorrhea | Sleep deprivation | Cramps | 3–4 weeks |

### Pattern Confidence Escalation

```
WATCHING  (N=2)       Tracked silently — not shown to user
EMERGING  (N=3)       Surfaced softly: "I've noticed this might be a pattern"
CONFIRMED (N≥4)       Asserted directly with date citations
LAG MATCH (registry)  Promoted immediately at N=2 (biological evidence)
```

### Cascade Detection

Clary detects multi-hop symptom chains via Neo4j graph traversal:

```
Caloric Restriction
  → Dizziness        (Week 1  — hypoglycaemia)
  → Brain Fog        (Week 2  — neurotransmitter depletion)
  → Hair Fall        (Week 6  — Telogen Effluvium)
```

---

## The Skeptic Agent

Every candidate pattern passes through the **Skeptic Agent** before reaching the user. The Skeptic's default stance is *"this is a coincidence"*.

### 9-Check Rubric

```
HARD GATES (fail → immediate REJECT)
  G1. Minimum evidence: N ≥ 2 sessions
  G2. Temporal ordering: trigger ALWAYS precedes symptom

POSITIVE SIGNALS (raise confidence)
  P1. Trigger consistent in ALL occurrences (not just most)
  P2. Lag matches biological mechanism in registry
  P3. Dose-response: more trigger → more symptom
  P4. Removal-reversal: symptom improves when trigger removed
  P5. N ≥ 4 occurrences

NEGATIVE SIGNALS (lower confidence)
  N1. Confounder present (third variable co-occurs)
  N2. Alternative explanation equally plausible
  N3. High base rate (may be coincidental)
```

### Variable Isolation

For patterns like P6 (sleep vs stress for dysmenorrhea), Clary uses a dedicated `VariableIsolator` to find the *only* factor consistent across all occurrences:

```python
# Cycle 1: stress=HIGH, sleep=POOR → cramps=SEVERE
# Cycle 2: stress=HIGH, sleep=NORMAL → cramps=MILD
# Cycle 3: stress=LOW,  sleep=POOR  → cramps=SEVERE
#
# VariableIsolator:
#   Stress: present in 2/3 → ELIMINATED
#   Sleep deprivation: present in 3/3 → CONSISTENT ✓
```

---

## Memory Architecture

Three-tier memory, each with a distinct role:

| Tier | Backend | Retrieval | Purpose |
|---|---|---|---|
| **Episodic** | ChromaDB | Dual: semantic + temporal range | Per-user health history |
| **Semantic** | ChromaDB | Similarity | Medical knowledge base |
| **Working** | Redis | Direct key | Live session state (TTL: 4h) |

### Temporal Indexing

ChromaDB metadata stores `timestamp_epoch` as a float, enabling NumPy-style range queries:

```python
# "What happened in the 12 weeks before Feb 19?"
results = collection.query(
    where={"$and": [
        {"user_id": {"$eq": "USR002"}},
        {"timestamp_epoch": {"$gte": dec_26_epoch}},
        {"timestamp_epoch": {"$lte": feb_19_epoch}},
    ]}
)
```

This is the query that connects a January diet to February hair loss.

---

## Dataset

The system was designed and evaluated against `askfirst_synthetic_dataset.json`:

- **3 users**: Arjun Sharma (USR001), Meera Nair (USR002), Priya Pillai (USR003)  
- **27 conversation sessions** spanning January–March 2026  
- **8 hidden patterns** of varying difficulty (3 easy, 2 medium, 3 hard)  
- Hard patterns require temporal reasoning across 6-week windows  

---

## Project Structure

```
clary/
├── agents/                    # Agent layer
│   ├── base_agent.py          # Retry + LLM infrastructure
│   ├── pattern_agent.py       # Two-phase temporal pattern detection
│   ├── skeptic_agent.py       # 9-check rubric + confidence scoring
│   ├── triage_agent.py        # Severity assessment + escalation
│   ├── narrator_agent.py      # Response synthesis + calibrated language
│   └── orchestrator.py        # State machine coordinating all agents
│
├── temporal/                  # Temporal reasoning engine
│   ├── lag_detector.py        # Biological lag registry (8 mechanisms)
│   ├── event_timeline.py      # Chronological event chain + lag queries
│   ├── cooccurrence.py        # Symptom × trigger frequency matrix
│   ├── hypothesis_builder.py  # Algorithmic candidate generation
│   └── variable_isolator.py   # Confounder isolation across N cycles
│
├── memory/                    # Three-tier memory
│   ├── episodic_store.py      # ChromaDB per-user episodic memory
│   ├── semantic_store.py      # ChromaDB medical knowledge base
│   ├── working_memory.py      # Redis session state
│   ├── context_builder.py     # Context window assembler
│   └── summary_compressor.py  # Rolling history compression
│
├── graph/                     # Neo4j event causality graph
│   ├── event_graph.py         # Node/edge management + async driver
│   ├── graph_queries.py       # Cypher query library
│   ├── graph_schema.py        # Node/edge type definitions + DDL
│   └── graph_visualizer.py    # pyvis/networkx export for UI
│
├── schemas/                   # Pydantic v2 data contracts
│   ├── event.py               # HealthEvent — atomic health signal
│   ├── pattern.py             # TemporalPattern lifecycle
│   ├── session.py             # PipelineContext + ClaryResponse
│   ├── agent.py               # AgentTrace + LLMUsage
│   └── evaluation.py          # EvalCase + EvalRunResult
│
├── app/                       # Streamlit frontend
│   ├── main.py                # Entry point + dark theme CSS
│   ├── state.py               # Centralised session state manager
│   └── ui/
│       ├── chat.py            # Streaming chat + pattern alerts
│       ├── timeline.py        # Plotly event timeline + heatmaps
│       ├── patterns.py        # Pattern dashboard + skeptic traces
│       ├── graph.py           # pyvis interactive event graph
│       └── eval_dashboard.py  # Precision/recall/F1 evaluation
│
├── eval/                      # Evaluation harness
│   ├── runner.py              # Eval orchestrator
│   ├── metrics.py             # Precision/recall/temporal accuracy
│   └── cases/                 # 8 golden pattern test cases
│
├── config.py                  # Pydantic Settings
├── requirements.txt
├── pyproject.toml
├── Makefile
└── .env.example
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### 1. Clone and install

```bash
git clone https://github.com/askfirst/clary.git
cd clary
make install-dev
```

### 2. Configure environment

```bash
make env                    # creates .env from .env.example
nano .env                   # set OPENAI_API_KEY and other secrets
```

### 3. Start infrastructure

```bash
make docker-up              # Neo4j + ChromaDB + Redis + PostgreSQL
```

Services:
- Neo4j Browser: http://localhost:7474
- ChromaDB API:  http://localhost:8000
- Redis:         localhost:6379

### 4. Seed the knowledge base

```bash
make seed-semantic          # loads medical knowledge into ChromaDB
make ingest                 # loads synthetic dataset (optional)
```

### 5. Run Clary

```bash
make streamlit              # opens http://localhost:8501
```

---

## Running Evaluations

```bash
make eval
# Output: eval/reports/latest.json

make eval-report            # print formatted results
```

Target thresholds (enforced in CI):

| Metric | Target |
|---|---|
| F1 Score | ≥ 85% |
| Lag Accuracy (±7d) | ≥ 90% |
| False Positive Rate | ≤ 10% |
| Calibration ECE | < 0.10 |

---

## Development Commands

```bash
make test          # unit tests (no infrastructure)
make lint          # ruff + black check
make format        # auto-format with black
make typecheck     # mypy static analysis
make check         # lint + types + tests (full CI gate)
make clean         # remove caches and build artifacts
```

---

## Example Pattern Output

```json
{
  "title": "Late Dinner → Acidity",
  "symptom": "acidity",
  "trigger": "late dinner",
  "status": "confirmed",
  "confidence": "high",
  "occurrence_count": 4,
  "lag_days_min": 0,
  "lag_days_max": 1,
  "lag_registry_match": {
    "mechanism_name": "Acid Reflux / GERD (Late Meal)",
    "lag_min_days": 0,
    "lag_max_days": 1,
    "match_quality": 0.90
  },
  "evidence": [
    {"session_id": "S01", "occurred_at": "2026-01-05", "lag_days": 0.5},
    {"session_id": "S04", "occurred_at": "2026-01-28", "lag_days": 0.3}
  ],
  "confounders": ["work_stress"],
  "skeptic_verdict": {
    "confidence": "high",
    "positive_signals": ["lag_registry_match", "trigger_fully_consistent"],
    "confounders": ["work_stress"],
    "dissent_note": "Stress co-occurs in all sessions but does not independently predict acidity."
  }
}
```

---

## Deployment

### Docker Compose (staging)

```bash
make docker-build
docker compose -f infra/docker-compose.yml up
```

### Kubernetes (production)

```bash
kubectl apply -f infra/k8s/
```

Includes:
- **App deployment** (Streamlit, 2 replicas, HPA)
- **API deployment** (FastAPI orchestrator)
- **Worker deployment** (Celery, async pattern jobs)
- **Managed services**: Neo4j AuraDB, Redis ElastiCache, RDS PostgreSQL

### CI/CD Gate

Every PR runs:
1. Unit tests (no infra)
2. Ruff lint + Black format check
3. mypy type check
4. Eval harness against 8 golden patterns (F1 ≥ 85% required)

---

## Limitations

- **Minimum data requirement**: 2 sessions with the same symptom before pattern detection begins. Single-session users see no patterns.
- **Lag registry coverage**: 8 known biological mechanisms. Patterns outside these windows rely on LLM reasoning, which is less precise.
- **No real-time streaming**: Streamlit's async bridge runs agent pipelines synchronously per message. Heavy pattern analysis is offloaded to Celery workers.
- **Privacy**: Health data is sensitive. This system is designed for single-user deployments or deployments with strong data isolation. Multi-tenant production requires additional encryption and access controls.
- **No medical advice**: Clary surfaces observations and hypotheses. It explicitly does not diagnose. All responses include a medical disclaimer.

---

## Future Improvements

- [ ] **Wearable integration**: Pull sleep, HRV, and step data from Apple Health / Google Fit for richer trigger signals
- [ ] **Physician portal**: Structured pattern reports exportable as PDF for clinical review
- [ ] **Fine-tuned embeddings**: Domain-specific embedding model trained on health language for better semantic retrieval
- [ ] **Real-time graph updates**: Incremental Neo4j updates via Kafka instead of batch post-session writes
- [ ] **Multi-language support**: Intake agent adapted for regional languages (Hindi, Tamil, Bengali)
- [ ] **A/B testing framework**: Compare pattern detection quality across model versions before full rollout
- [ ] **Longitudinal studies**: Aggregate anonymised pattern data across users to validate against medical literature

---

## Architecture Decisions Log

| Decision | Rationale |
|---|---|
| Two-phase pattern detection (algo + LLM) | Algo guarantees temporal ordering and consistency; LLM catches cascade patterns the algo misses. Saves 40% token cost. |
| Skeptic Agent as separate pass | Prevents false positives. A single LLM detecting and validating its own patterns produces confirmation bias. |
| ChromaDB `timestamp_epoch` metadata | Standard vector similarity retrieves semantically similar content; temporal range queries find events *N days before* a symptom regardless of semantic similarity. Both are required. |
| Neo4j for causal graph | Graph databases natively traverse multi-hop causal chains (A → B → C). Implementing this in SQL or a vector store would require N+1 queries. |
| Variable Isolator before Skeptic LLM | Running isolation algorithmically (set intersection across occurrences) gives the LLM pre-computed evidence rather than asking it to reason about absence of confounders from raw text. |
| Redis working memory with 4h TTL | Sessions have natural boundaries. Persisting mid-pipeline state allows clarification flows across multiple turns without re-running pattern detection. |

---

## Licence

Proprietary — Ask First Engineering. All rights reserved.

---

*Built with GPT-4o, ChromaDB, Neo4j, Redis, Streamlit, and a lot of respect for the complexity of the human body.*