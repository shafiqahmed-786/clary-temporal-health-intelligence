"""
memory/semantic_store.py — SemanticStore

Shared medical knowledge base, not per-user.
Stores medical mechanism descriptions, symptom profiles, and treatment
context from the biological lag registry and curated sources.

Used by PatternAgent and NarratorAgent to ground responses in
evidence-based medical knowledge.
"""

from __future__ import annotations

from typing import Any

import chromadb
import openai
import structlog

from config import get_settings
from temporal.lag_detector import lag_registry

logger = structlog.get_logger(__name__)
settings = get_settings()

_SEED_DOCUMENTS = [
    {
        "id": "med_001",
        "text": (
            "Telogen Effluvium: Hair follicles have a growth cycle of anagen (growing), "
            "catagen (transitioning), and telogen (resting/shedding). Severe metabolic "
            "stress — including caloric restriction below 1200 kcal/day, extreme dieting, "
            "nutritional deficiencies (iron, biotin, zinc), major illness, or emotional "
            "shock — can synchronise follicles into telogen phase. Shedding begins 6–12 "
            "weeks after the stressor. The condition is typically self-limiting if the "
            "stressor is removed."
        ),
        "metadata": {"category": "dermatology", "mechanism": "telogen_effluvium"},
    },
    {
        "id": "med_002",
        "text": (
            "Dairy and Acne: Milk and dairy products contain bioactive molecules including "
            "insulin-like growth factor 1 (IGF-1), hormones, and leucine. IGF-1 stimulates "
            "sebum production and keratinocyte proliferation, promoting comedone formation. "
            "Hormones in dairy can also dysregulate androgen signalling. Breakouts typically "
            "appear 48–72 hours after dairy consumption. The face — particularly cheeks and "
            "jawline — is most affected. Dose-response relationship is commonly observed."
        ),
        "metadata": {"category": "dermatology", "mechanism": "dairy_acne"},
    },
    {
        "id": "med_003",
        "text": (
            "GERD and Late Eating: Gastro-oesophageal reflux disease (GERD) is worsened by "
            "eating within 2–3 hours of lying down. Horizontal position reduces the mechanical "
            "advantage of the lower oesophageal sphincter, allowing gastric acid to reflux. "
            "Large meals, fatty foods, caffeine, and stress further reduce sphincter tone. "
            "Symptoms — burning chest or epigastric pain, regurgitation — begin within 1–3 "
            "hours of the triggering meal. Long-term management includes eating at least "
            "3 hours before bedtime and elevating the head of the bed."
        ),
        "metadata": {"category": "gastroenterology", "mechanism": "gerd_late_meal"},
    },
    {
        "id": "med_004",
        "text": (
            "Reactive Hypoglycaemia: High-glycaemic-index meals (white rice, bread, sugary "
            "drinks) cause rapid blood glucose spikes followed by reactive hypoglycaemia "
            "90–150 minutes post-meal. Symptoms include fatigue, poor concentration, "
            "irritability, and the characteristic 'afternoon slump'. Protein co-ingestion "
            "significantly blunts the glycaemic response. The 2–4pm energy crash experienced "
            "by many individuals is often explained by this mechanism combined with circadian "
            "cortisol decline."
        ),
        "metadata": {"category": "endocrinology", "mechanism": "reactive_hypoglycaemia"},
    },
    {
        "id": "med_005",
        "text": (
            "Sleep Deprivation and Cortisol: Chronic sleep restriction below 7 hours raises "
            "evening cortisol levels. Cortisol is a catabolic hormone that, when chronically "
            "elevated, impairs immune function, increases gut permeability, disrupts menstrual "
            "prostaglandin regulation (worsening dysmenorrhea), and sensitises the amygdala "
            "(increasing anxiety). Effects accumulate over weeks. A 5-night restriction "
            "study showed cortisol elevation comparable to major psychological stressors."
        ),
        "metadata": {"category": "sleep_medicine", "mechanism": "sleep_cortisol"},
    },
    {
        "id": "med_006",
        "text": (
            "Dehydration Headache: Mild dehydration (1–2% body weight fluid loss) causes "
            "brain volume reduction, triggering meningeal pain receptors. Caffeine is a "
            "diuretic that worsens dehydration. The combination of low water intake and "
            "caffeine consumption (coffee, tea) is a common headache trigger, particularly "
            "in office environments where drinking is easily forgotten during focused work. "
            "Headache typically develops 3–6 hours into dehydration. Rehydration provides "
            "relief within 30–60 minutes."
        ),
        "metadata": {"category": "neurology", "mechanism": "dehydration_headache"},
    },
]


class SemanticStore:
    """
    Shared medical knowledge base in ChromaDB.
    Seeded with lag registry descriptions and curated medical documents.
    """

    def __init__(self) -> None:
        self._client: chromadb.AsyncHttpClient | None = None
        self._collection: Any = None
        self._openai: openai.AsyncOpenAI | None = None
        self._seeded = False

    async def _get_client(self) -> chromadb.AsyncHttpClient:
        if self._client is None:
            self._client = await chromadb.AsyncHttpClient(
                host=settings.chroma_host, port=settings.chroma_port
            )
        return self._client

    def _get_openai(self) -> openai.AsyncOpenAI:
        if self._openai is None:
            self._openai = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        return self._openai

    async def _get_collection(self) -> Any:
        if self._collection is None:
            client = await self._get_client()
            self._collection = await client.get_or_create_collection(
                name=settings.semantic_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        oai = self._get_openai()
        response = await oai.embeddings.create(
            model=settings.openai_embedding_model, input=texts
        )
        return [item.embedding for item in response.data]

    async def seed(self, force: bool = False) -> None:
        """Seed the collection with core medical knowledge on first run."""
        if self._seeded and not force:
            return

        collection = await self._get_collection()
        count_result = await collection.count()
        if count_result > 0 and not force:
            self._seeded = True
            return

        # Combine seed docs with lag registry descriptions
        all_docs = list(_SEED_DOCUMENTS)
        for entry in lag_registry.get_all_entries():
            all_docs.append({
                "id": f"lag_{entry.mechanism_name.replace(' ', '_').lower()}",
                "text": f"{entry.mechanism_name}: {entry.description}",
                "metadata": {
                    "category": "lag_registry",
                    "mechanism": entry.mechanism_name,
                    "lag_min_days": entry.lag_min_days,
                    "lag_max_days": entry.lag_max_days,
                },
            })

        texts = [d["text"] for d in all_docs]
        embeddings = await self._embed(texts)

        await collection.upsert(
            ids=[d["id"] for d in all_docs],
            embeddings=embeddings,
            documents=texts,
            metadatas=[d["metadata"] for d in all_docs],
        )
        self._seeded = True
        logger.info("semantic_store.seeded", document_count=len(all_docs))

    async def search(
        self,
        query: str,
        top_k: int = 3,
        category_filter: str | None = None,
    ) -> list[dict]:
        """Search medical knowledge base by semantic similarity."""
        collection = await self._get_collection()
        query_emb = await self._embed([query])

        where = {}
        if category_filter:
            where["category"] = {"$eq": category_filter}

        results = await collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        out = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i in range(len(ids)):
            out.append({
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "distance": dists[i],
            })
        return out

    async def get_mechanism_context(self, symptom: str, trigger: str) -> str:
        """
        Return relevant medical context for a (symptom, trigger) pair.
        Used to ground NarratorAgent responses.
        """
        query = f"{symptom} caused by {trigger}"
        results = await self.search(query, top_k=2)
        if not results:
            return ""
        return "\n\n".join(r["document"] for r in results)