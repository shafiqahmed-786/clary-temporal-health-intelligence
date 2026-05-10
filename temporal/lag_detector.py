"""
temporal/lag_detector.py — Biological lag registry.

Encodes known medical mechanisms that produce delayed symptoms.
This is the domain knowledge that allows Clary to connect a trigger
on Jan 8 to a symptom on Feb 19 (Telogen Effluvium, 6-week lag).

Each entry is a LagEntry. The registry exposes:
  - lookup(symptom, trigger, observed_lag_days) → LagMatch | None
  - get_max_lookback(symptom) → int (max days to look back)
  - get_all_entries() → list[LagEntry]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from schemas.pattern import LagRegistryMatch

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class LagEntry:
    """A single biological mechanism with its trigger/symptom profile."""

    mechanism_name: str
    description: str

    # Symptom keywords (any match triggers lookup)
    symptom_keywords: tuple[str, ...]
    # Trigger keywords (any match triggers lookup)
    trigger_keywords: tuple[str, ...]

    lag_min_days: int
    lag_max_days: int

    # How strongly we trust this mechanism
    base_confidence: float  # 0.0 – 1.0
    source: str = "internal_registry"
    dose_response_expected: bool = False
    removal_reversal_expected: bool = False


# ─── Registry entries ─────────────────────────────────────────────────────────

_REGISTRY: list[LagEntry] = [
    LagEntry(
        mechanism_name="Telogen Effluvium",
        description=(
            "Sudden metabolic stress (caloric restriction, nutritional deficiency, "
            "severe illness, extreme stress) shifts hair follicles into telogen "
            "phase; shedding begins 6–12 weeks later when the resting phase ends."
        ),
        symptom_keywords=(
            "hair loss", "hair fall", "hair shedding", "hair thinning",
            "losing hair", "hairfall",
        ),
        trigger_keywords=(
            "calorie restriction", "caloric deficit", "crash diet", "low calorie",
            "nutritional deficiency", "nutrient deficiency", "iron deficiency",
            "severe stress", "crash", "fasting", "undereating",
        ),
        lag_min_days=42,   # 6 weeks
        lag_max_days=84,   # 12 weeks
        base_confidence=0.92,
        dose_response_expected=True,
        removal_reversal_expected=True,
    ),
    LagEntry(
        mechanism_name="Dairy-Induced Acne (IGF-1 / Hormonal Pathway)",
        description=(
            "Dairy consumption raises IGF-1 and stimulates sebum production, "
            "leading to comedone formation and acne breakouts 48–72 hours later."
        ),
        symptom_keywords=(
            "acne", "pimple", "breakout", "skin breakout", "cheek acne",
            "jawline acne", "cystic acne", "spots", "blemishes",
        ),
        trigger_keywords=(
            "dairy", "milk", "cheese", "yogurt", "curd", "paneer",
            "ice cream", "butter", "cream", "whey",
        ),
        lag_min_days=2,
        lag_max_days=4,
        base_confidence=0.78,
        dose_response_expected=True,
        removal_reversal_expected=True,
    ),
    LagEntry(
        mechanism_name="Acid Reflux / GERD (Late Meal)",
        description=(
            "Eating within 2–3 hours of lying down allows gastric acid to reflux "
            "into the oesophagus; symptoms typically manifest within 1–3 hours."
        ),
        symptom_keywords=(
            "acidity", "acid reflux", "heartburn", "burning", "stomach pain",
            "chest burning", "regurgitation", "gerd", "bloating", "nausea",
        ),
        trigger_keywords=(
            "late dinner", "late eating", "late night meal", "eating late",
            "dinner after 10", "dinner after 9", "night eating",
            "late food", "midnight snack",
        ),
        lag_min_days=0,
        lag_max_days=1,
        base_confidence=0.90,
        dose_response_expected=False,
        removal_reversal_expected=True,
    ),
    LagEntry(
        mechanism_name="Dehydration Headache",
        description=(
            "Dehydration and caffeine (a diuretic) compound fluid loss; "
            "the brain slightly contracts, causing tension-type headaches "
            "within 3–6 hours of insufficient fluid intake."
        ),
        symptom_keywords=(
            "headache", "head pain", "migraine", "head throbbing",
            "pressure headache", "tension headache",
        ),
        trigger_keywords=(
            "low water", "not drinking water", "dehydrated", "dehydration",
            "coffee", "caffeine", "skipped water", "forgot water",
            "work pressure", "sitting all day",
        ),
        lag_min_days=0,
        lag_max_days=1,
        base_confidence=0.88,
        dose_response_expected=True,
        removal_reversal_expected=True,
    ),
    LagEntry(
        mechanism_name="Post-Prandial Hypoglycaemia (Energy Crash)",
        description=(
            "High-glycaemic-index carbohydrates without protein cause a rapid "
            "blood glucose spike followed by reactive hypoglycaemia 90–150 min "
            "after eating, producing fatigue and brain fog."
        ),
        symptom_keywords=(
            "energy crash", "afternoon slump", "fatigue", "brain fog",
            "low energy", "sluggish", "tired", "exhausted", "crash",
            "can't focus", "concentration",
        ),
        trigger_keywords=(
            "high carb", "white rice", "pasta", "bread", "sugary", "sweet",
            "no protein", "skipped protein", "carb heavy", "dessert lunch",
            "roti only", "just carbs",
        ),
        lag_min_days=0,
        lag_max_days=1,
        base_confidence=0.85,
        dose_response_expected=True,
        removal_reversal_expected=True,
    ),
    LagEntry(
        mechanism_name="Sleep Deprivation → Cortisol Dysmenorrhea",
        description=(
            "Chronic sleep deprivation elevates cortisol and disrupts "
            "prostaglandin regulation, worsening menstrual cramp severity "
            "in the cycle following the deprivation period."
        ),
        symptom_keywords=(
            "cramps", "menstrual cramps", "period pain", "dysmenorrhea",
            "painful period", "period cramps", "bad period",
        ),
        trigger_keywords=(
            "sleep deprivation", "poor sleep", "bad sleep", "late nights",
            "not sleeping", "insomnia", "sleep debt", "screen time",
            "2am", "3am", "midnight",
        ),
        lag_min_days=14,
        lag_max_days=35,
        base_confidence=0.70,
        dose_response_expected=True,
        removal_reversal_expected=False,
    ),
    LagEntry(
        mechanism_name="Sleep Deprivation → Anxiety Cascade",
        description=(
            "Cumulative sleep debt raises amygdala reactivity and suppresses "
            "prefrontal regulation, producing anxiety symptoms that worsen "
            "progressively over weeks of insufficient sleep."
        ),
        symptom_keywords=(
            "anxiety", "anxious", "worried", "panic", "restless",
            "on edge", "overwhelmed", "heart racing", "nervous",
        ),
        trigger_keywords=(
            "sleep deprivation", "poor sleep", "bad sleep", "late nights",
            "not sleeping", "insomnia", "screen time before bed",
            "5 hours sleep", "6 hours sleep",
        ),
        lag_min_days=14,
        lag_max_days=56,
        base_confidence=0.72,
        dose_response_expected=True,
        removal_reversal_expected=True,
    ),
    LagEntry(
        mechanism_name="Nutritional Cascade (Caloric Restriction)",
        description=(
            "Severe caloric restriction produces sequential downstream effects: "
            "dizziness within days (hypoglycaemia), cognitive fog within 1–2 weeks "
            "(neurotransmitter depletion), then hair loss at 6–12 weeks (telogen shift)."
        ),
        symptom_keywords=(
            "dizziness", "lightheaded", "brain fog", "foggy", "concentration",
            "hair loss", "hair fall",
        ),
        trigger_keywords=(
            "calorie restriction", "eating less", "diet", "low calorie",
            "skipping meals", "crash diet", "800 calories", "1000 calories",
        ),
        lag_min_days=1,
        lag_max_days=84,
        base_confidence=0.82,
        dose_response_expected=True,
        removal_reversal_expected=True,
    ),
]


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation for keyword matching."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


class LagRegistry:
    """
    Singleton registry. Matches (symptom, trigger, observed_lag_days) tuples
    against known biological mechanisms.
    """

    def __init__(self, entries: list[LagEntry] | None = None) -> None:
        self._entries: list[LagEntry] = entries or _REGISTRY

    # ── Public API ────────────────────────────────────────────────────────

    def lookup(
        self,
        symptom: str,
        trigger: str,
        observed_lag_days: float | None = None,
    ) -> LagRegistryMatch | None:
        """
        Return the best matching LagRegistryMatch or None.

        Matching logic:
          1. At least one symptom keyword must match the symptom string.
          2. At least one trigger keyword must match the trigger string.
          3. If observed_lag_days provided, it must fall within [min, max].
          4. Among multiple matches, return highest base_confidence.
        """
        norm_symptom = _normalise(symptom)
        norm_trigger = _normalise(trigger)

        candidates: list[tuple[float, LagEntry]] = []

        for entry in self._entries:
            symptom_hit = any(kw in norm_symptom for kw in entry.symptom_keywords)
            trigger_hit = any(kw in norm_trigger for kw in entry.trigger_keywords)

            if not (symptom_hit and trigger_hit):
                continue

            # If lag observed, check it falls in window (with 20% tolerance)
            match_quality = entry.base_confidence
            if observed_lag_days is not None:
                tol_min = entry.lag_min_days * 0.8
                tol_max = entry.lag_max_days * 1.2
                if not (tol_min <= observed_lag_days <= tol_max):
                    continue  # lag outside window — skip
                # Scale quality by how central the lag is
                mid = (entry.lag_min_days + entry.lag_max_days) / 2
                distance = abs(observed_lag_days - mid) / max(mid, 1)
                match_quality *= max(0.6, 1.0 - distance * 0.4)

            candidates.append((match_quality, entry))

        if not candidates:
            logger.debug(
                "lag_registry.no_match",
                symptom=symptom,
                trigger=trigger,
                observed_lag_days=observed_lag_days,
            )
            return None

        # Pick highest quality match
        best_quality, best_entry = max(candidates, key=lambda t: t[0])

        logger.info(
            "lag_registry.match",
            mechanism=best_entry.mechanism_name,
            quality=round(best_quality, 3),
            symptom=symptom,
            trigger=trigger,
        )

        return LagRegistryMatch(
            mechanism_name=best_entry.mechanism_name,
            description=best_entry.description,
            lag_min_days=best_entry.lag_min_days,
            lag_max_days=best_entry.lag_max_days,
            match_quality=round(best_quality, 3),
            source=best_entry.source,
        )

    def get_max_lookback(self, symptom: str) -> int:
        """
        Return the maximum lag_max_days across all entries matching
        the given symptom. Used to determine how far back to query
        the episodic store when a symptom is reported.
        """
        norm = _normalise(symptom)
        max_days = 7  # floor: always look back at least a week
        for entry in self._entries:
            if any(kw in norm for kw in entry.symptom_keywords):
                max_days = max(max_days, entry.lag_max_days)
        return max_days

    def get_all_entries(self) -> list[LagEntry]:
        return list(self._entries)

    def describe_all(self) -> str:
        """Human-readable summary for injecting into LLM prompts."""
        lines = ["Known biological lag windows (for reference):"]
        for e in self._entries:
            lines.append(
                f"  • {e.mechanism_name}: "
                f"trigger=[{', '.join(e.trigger_keywords[:3])}...] → "
                f"symptom=[{', '.join(e.symptom_keywords[:3])}...] "
                f"lag={e.lag_min_days}–{e.lag_max_days}d"
            )
        return "\n".join(lines)


# Module-level singleton
lag_registry = LagRegistry()