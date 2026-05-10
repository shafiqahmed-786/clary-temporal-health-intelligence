"""
graph/graph_queries.py — Cypher query library.

All Cypher queries used in the application are defined here as constants
or builder functions. Never inline Cypher in business logic — always
use this module.

Query categories:
  - Writes: upsert nodes, create edges
  - Pattern support: find consistent prior triggers, variable isolation
  - Cascade detection: symptom chain traversal
  - Retrieval: user timeline, pattern evidence
  - Analytics: co-occurrence, lag distribution
"""

from __future__ import annotations

# ══ UPSERT QUERIES ═════════════════════════════════════════════════════════════

UPSERT_USER = """
MERGE (u:User {user_id: $user_id})
ON CREATE SET u.name = $name, u.age = $age,
              u.location = $location, u.created_at = $created_at
ON MATCH SET  u.name = $name, u.age = $age
RETURN u
"""

UPSERT_SESSION = """
MERGE (s:Session {session_id: $session_id})
ON CREATE SET s.user_id      = $user_id,
              s.timestamp_iso  = $timestamp_iso,
              s.timestamp_epoch = $timestamp_epoch,
              s.severity       = $severity,
              s.summary        = $summary,
              s.symptoms       = $symptoms,
              s.triggers       = $triggers
ON MATCH SET  s.severity = $severity,
              s.summary  = $summary
RETURN s
"""

UPSERT_SYMPTOM = """
MERGE (sym:Symptom {name: $name})
ON CREATE SET sym.display_name = $display_name,
              sym.body_location = $body_location
RETURN sym
"""

UPSERT_TRIGGER = """
MERGE (t:Trigger {name: $name})
ON CREATE SET t.display_name = $display_name,
              t.trigger_type  = $trigger_type
RETURN t
"""

UPSERT_PATTERN = """
MERGE (p:Pattern {pattern_id: $pattern_id})
ON CREATE SET p.user_id          = $user_id,
              p.symptom           = $symptom,
              p.trigger           = $trigger,
              p.status            = $status,
              p.confidence        = $confidence,
              p.occurrence_count  = $occurrence_count,
              p.lag_days_min      = $lag_days_min,
              p.lag_days_max      = $lag_days_max,
              p.first_detected_at = $first_detected_at,
              p.last_confirmed_at = $last_confirmed_at
ON MATCH SET  p.status            = $status,
              p.confidence        = $confidence,
              p.occurrence_count  = $occurrence_count,
              p.last_confirmed_at = $last_confirmed_at
RETURN p
"""

UPSERT_MECHANISM = """
MERGE (m:Mechanism {name: $name})
ON CREATE SET m.description  = $description,
              m.lag_min_days = $lag_min_days,
              m.lag_max_days = $lag_max_days
RETURN m
"""

# ══ EDGE CREATION ══════════════════════════════════════════════════════════════

LINK_USER_SESSION = """
MATCH (u:User {user_id: $user_id})
MATCH (s:Session {session_id: $session_id})
MERGE (u)-[:HAS_SESSION]->(s)
"""

LINK_SESSION_SYMPTOM = """
MATCH (s:Session {session_id: $session_id})
MATCH (sym:Symptom {name: $symptom_name})
MERGE (s)-[r:REPORTED_SYMPTOM]->(sym)
ON CREATE SET r.severity = $severity, r.timestamp_epoch = $timestamp_epoch
"""

LINK_SESSION_TRIGGER = """
MATCH (s:Session {session_id: $session_id})
MATCH (t:Trigger {name: $trigger_name})
MERGE (s)-[r:CONTAINS_TRIGGER]->(t)
ON CREATE SET r.certainty = $certainty
"""

LINK_SESSION_PRECEDES = """
MATCH (s1:Session {session_id: $session_id_from})
MATCH (s2:Session {session_id: $session_id_to})
MERGE (s1)-[r:PRECEDES]->(s2)
ON CREATE SET r.lag_days = $lag_days,
              r.same_symptom = $same_symptom,
              r.shared_symptoms = $shared_symptoms
"""

LINK_SYMPTOM_CAUSED_BY = """
MATCH (sym:Symptom {name: $symptom_name})
MATCH (t:Trigger {name: $trigger_name})
MERGE (sym)-[r:CAUSED_BY]->(t)
ON CREATE SET r.confidence    = $confidence,
              r.evidence_count = $evidence_count,
              r.mean_lag_days  = $mean_lag_days
ON MATCH SET  r.confidence    = $confidence,
              r.evidence_count = $evidence_count,
              r.mean_lag_days  = $mean_lag_days
"""

LINK_PATTERN_MECHANISM = """
MATCH (p:Pattern {pattern_id: $pattern_id})
MATCH (m:Mechanism {name: $mechanism_name})
MERGE (p)-[r:EXPLAINED_BY]->(m)
ON CREATE SET r.match_quality = $match_quality
"""

LINK_SESSION_EVIDENCE_FOR_PATTERN = """
MATCH (s:Session {session_id: $session_id})
MATCH (p:Pattern {pattern_id: $pattern_id})
MERGE (s)-[r:EVIDENCE_FOR]->(p)
ON CREATE SET r.occurrence_number  = $occurrence_number,
              r.lag_days_actual     = $lag_days_actual
"""

LINK_SYMPTOM_DOWNSTREAM = """
MATCH (s1:Symptom {name: $symptom_upstream})
MATCH (s2:Symptom {name: $symptom_downstream})
MERGE (s2)-[r:DOWNSTREAM_OF]->(s1)
ON CREATE SET r.weeks_delay = $weeks_delay,
              r.user_id     = $user_id
"""

# ══ ANALYTICAL READ QUERIES ════════════════════════════════════════════════════

GET_USER_TIMELINE = """
MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session)
RETURN s.session_id   AS session_id,
       s.timestamp_iso AS timestamp,
       s.symptoms      AS symptoms,
       s.triggers      AS triggers,
       s.severity      AS severity,
       s.summary       AS summary
ORDER BY s.timestamp_epoch ASC
"""

GET_SESSIONS_IN_LAG_WINDOW = """
MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session)
WHERE s.timestamp_epoch >= $start_epoch
  AND s.timestamp_epoch <= $end_epoch
RETURN s.session_id, s.timestamp_iso, s.symptoms, s.triggers, s.severity
ORDER BY s.timestamp_epoch ASC
"""

GET_CONSISTENT_PRIOR_TRIGGERS = """
// Find triggers present before EVERY occurrence of a given symptom for a user.
// Uses PRECEDES chain limited to lag window.
MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(sym_session:Session)
      -[:REPORTED_SYMPTOM]->(sym:Symptom {name: $symptom})
WITH collect(sym_session) AS symptom_sessions, count(sym_session) AS total

MATCH (trigger_session:Session {user_id: $user_id})
      -[:CONTAINS_TRIGGER]->(t:Trigger)
MATCH (trigger_session)-[:PRECEDES]->(sym_session2:Session)
      -[:REPORTED_SYMPTOM]->(:Symptom {name: $symptom})
WHERE trigger_session.timestamp_epoch < sym_session2.timestamp_epoch
  AND (sym_session2.timestamp_epoch - trigger_session.timestamp_epoch) / 86400.0 <= $max_lag_days

WITH t.name AS trigger_name,
     count(DISTINCT sym_session2) AS hit_count,
     total,
     avg((sym_session2.timestamp_epoch - trigger_session.timestamp_epoch) / 86400.0) AS mean_lag

WHERE hit_count = total
RETURN trigger_name, hit_count, total, mean_lag
ORDER BY hit_count DESC
"""

GET_PATTERNS_FOR_USER = """
MATCH (p:Pattern {user_id: $user_id})
OPTIONAL MATCH (p)-[:EXPLAINED_BY]->(m:Mechanism)
RETURN p.pattern_id    AS pattern_id,
       p.symptom        AS symptom,
       p.trigger        AS trigger,
       p.status         AS status,
       p.confidence     AS confidence,
       p.occurrence_count AS n,
       p.lag_days_min   AS lag_min,
       p.lag_days_max   AS lag_max,
       m.name           AS mechanism
ORDER BY p.occurrence_count DESC
"""

GET_PATTERN_EVIDENCE_SESSIONS = """
MATCH (s:Session)-[e:EVIDENCE_FOR]->(p:Pattern {pattern_id: $pattern_id})
RETURN s.session_id      AS session_id,
       s.timestamp_iso   AS timestamp,
       e.lag_days_actual  AS lag_days,
       e.occurrence_number AS occ_num
ORDER BY s.timestamp_epoch ASC
"""

DETECT_CASCADE_CHAINS = """
// Find symptom downstream chains of depth 1-4 for a user.
MATCH path = (root:Symptom)-[:DOWNSTREAM_OF*1..4 {user_id: $user_id}]->(leaf:Symptom)
WHERE (root)<-[:REPORTED_SYMPTOM]-(:Session {user_id: $user_id})
  AND (leaf)<-[:REPORTED_SYMPTOM]-(:Session {user_id: $user_id})
RETURN [node IN nodes(path) | node.name] AS chain,
       length(path)                        AS depth
ORDER BY depth DESC
LIMIT 10
"""

VARIABLE_ISOLATION_QUERY = """
// For each occurrence of a pattern, get ALL triggers present in the lag window.
// Used to find triggers CONSISTENT across all occurrences.
MATCH (p:Pattern {pattern_id: $pattern_id})<-[:EVIDENCE_FOR]-(s:Session)
MATCH (prior:Session {user_id: $user_id})
      -[:CONTAINS_TRIGGER]->(t:Trigger)
WHERE prior.timestamp_epoch < s.timestamp_epoch
  AND (s.timestamp_epoch - prior.timestamp_epoch) / 86400.0 <= $max_lag_days

WITH t.name AS trigger_name,
     count(DISTINCT s) AS session_hit_count,
     count(DISTINCT s) AS total_occurrences

RETURN trigger_name,
       session_hit_count,
       CASE WHEN session_hit_count = total_occurrences THEN true ELSE false END AS is_consistent
ORDER BY is_consistent DESC, session_hit_count DESC
"""

GET_SIMILAR_SESSIONS = """
MATCH (s1:Session {session_id: $session_id})-[:SIMILAR_TO]-(s2:Session)
RETURN s2.session_id, s2.timestamp_iso, s2.symptoms, s2.triggers
LIMIT $limit
"""

GET_USER_GRAPH_SUBGRAPH = """
// Fetch the entire subgraph for a user (for visualisation).
MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session)
OPTIONAL MATCH (s)-[:REPORTED_SYMPTOM]->(sym:Symptom)
OPTIONAL MATCH (s)-[:CONTAINS_TRIGGER]->(t:Trigger)
OPTIONAL MATCH (s)-[:EVIDENCE_FOR]->(p:Pattern)
OPTIONAL MATCH (p)-[:EXPLAINED_BY]->(m:Mechanism)
OPTIONAL MATCH (sym)-[:CAUSED_BY]->(t2:Trigger)
RETURN u, s, sym, t, p, m, t2
LIMIT 200
"""