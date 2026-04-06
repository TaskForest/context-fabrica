from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

from ..adapters import GraphStore, Reranker, RecordStore
from ..config import HybridStoreSettings, NamespacePolicy, ScoringWeights
from ..embedding import Embedder, build_default_embedder, chunk_text
from ..entity import extract_entities, extract_relations
from ..models import KnowledgeRecord, MemoryKind, MemoryStage, QueryResult, Relation
from ..policy import decide_memory_tier, promote_record as _promote_record_fn
from ..projection import GraphProjection, build_graph_projection
from ..scoring import ScoringMode, ScoringPipeline
from ..synthesis import build_observation_record
from ..temporal import extract_time_range
from .kuzu import KuzuGraphProjectionAdapter
from .postgres import PostgresPgvectorAdapter


@dataclass(frozen=True)
class HybridWritePlan:
    record_id: str
    graph_projection: GraphProjection


class HybridMemoryStore:
    """Unified memory store: persistent storage + full scoring pipeline.

    Accepts any RecordStore and optional GraphStore implementation.
    Ships with Postgres + Kuzu as defaults, but works with SQLite
    or any custom adapter implementing the protocols.

    Construction options:

        # Protocol-based (recommended)
        store = HybridMemoryStore(store=SQLiteRecordStore("./memory.db"))
        store = HybridMemoryStore(store=my_postgres, graph=my_kuzu)

        # Settings-based (backward-compatible, builds Postgres + Kuzu)
        store = HybridMemoryStore(settings=HybridStoreSettings(...))
    """

    def __init__(
        self,
        settings: HybridStoreSettings | None = None,
        *,
        store: RecordStore | None = None,
        graph: GraphStore | None = None,
        embedder: Embedder | None = None,
        scoring: ScoringMode = "hybrid",
        weights: ScoringWeights | None = None,
        reranker: Reranker | None = None,
        rerank_weight: float = 0.15,
        namespace_policies: dict[str, NamespacePolicy] | None = None,
    ) -> None:
        self.settings = settings
        if store is not None:
            self.store: RecordStore = store
            self.graph: GraphStore | None = graph
            dims = 1536
        elif settings is not None:
            self.store = PostgresPgvectorAdapter(settings.postgres)
            self.graph = KuzuGraphProjectionAdapter(settings.kuzu)
            dims = settings.postgres.embedding_dimensions
        else:
            raise TypeError("Provide either 'store' or 'settings'")

        # Backward-compatible aliases used by older call sites and tests.
        self.postgres = self.store if isinstance(self.store, PostgresPgvectorAdapter) else None
        self.kuzu = self.graph if isinstance(self.graph, KuzuGraphProjectionAdapter) else None
        self.embedder = embedder or build_default_embedder(dimensions=dims)

        self._scoring = ScoringPipeline(
            scoring=scoring,
            weights=weights,
            reranker=reranker,
            rerank_weight=rerank_weight,
            namespace_policies=namespace_policies,
        )

    # ── Bootstrap ──

    def bootstrap(self) -> None:
        self.store.bootstrap()
        if self.graph is not None:
            self.graph.bootstrap()

    # Keep backward-compatible alias
    def bootstrap_postgres(self) -> None:
        self.store.bootstrap()

    def _ensure_scoring_bootstrapped(self) -> None:
        """Lazily bootstrap BM25 index and graph from store on first query."""
        if self._scoring.bootstrapped:
            return
        texts = self.store.list_all_texts() if hasattr(self.store, "list_all_texts") else []
        relations = self.store.list_all_relations() if hasattr(self.store, "list_all_relations") else []
        self._scoring.bootstrap_from_store(texts, relations)

    # ── Ingest ──

    def ingest(
        self,
        text: str,
        *,
        source: str = "unknown",
        domain: str = "global",
        namespace: str = "default",
        confidence: float = 0.6,
        tags: Iterable[str] | None = None,
        metadata: dict[str, object] | None = None,
        record_id: str | None = None,
        auto_stage: bool = True,
        entities: list[str] | None = None,
        relations: list[Relation] | None = None,
        stage: MemoryStage | None = None,
        kind: MemoryKind | None = None,
        occurred_from: datetime | None = None,
        occurred_to: datetime | None = None,
        infer_occurrence: bool = True,
    ) -> KnowledgeRecord:
        """Ingest text into the memory store with full persistence and index updates.

        Builds a KnowledgeRecord, persists it with embeddings and relations to the
        backing store, and incrementally updates the in-memory BM25 and graph indexes.

        Args:
            text: The knowledge text to store.
            source: Provenance label (e.g. ``"design-doc"``, ``"agent"``).
            domain: Knowledge domain for filtering (e.g. ``"auth"``, ``"payments"``).
            namespace: Tenant or team namespace.
            confidence: Trust score between 0.0 and 1.0.
            tags: Optional categorization tags.
            metadata: Arbitrary key-value metadata.
            record_id: Explicit ID; auto-generated UUID if omitted.
            auto_stage: Whether to auto-classify stage/kind via policy.
            entities: Caller-provided entities; extracted from text if omitted.
            relations: Caller-provided relations; extracted from text if omitted.
            stage: Explicit stage override (applied after auto_stage).
            kind: Explicit kind override (applied after auto_stage).
            occurred_from: Start of the event-time window.
            occurred_to: End of the event-time window.
            infer_occurrence: Attempt to infer occurrence window from text.

        Returns:
            The persisted KnowledgeRecord.

        Raises:
            TypeError: If *text* is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        rid = record_id or str(uuid4())
        created_at = datetime.now(tz=timezone.utc)
        inferred_occurrence = None
        if infer_occurrence and occurred_from is None and occurred_to is None:
            inferred_occurrence = extract_time_range(text, now=created_at)
        record = KnowledgeRecord(
            record_id=rid,
            text=text,
            source=source,
            domain=domain,
            namespace=namespace,
            created_at=created_at,
            confidence=max(0.0, min(1.0, confidence)),
            tags=list(tags or []),
            metadata=dict(metadata or {}),
            occurred_from=occurred_from or (inferred_occurrence[0] if inferred_occurrence else None),
            occurred_to=occurred_to or (inferred_occurrence[1] if inferred_occurrence else None),
        )
        if auto_stage:
            decision = decide_memory_tier(record)
            record.stage = decision.stage
            record.kind = decision.kind
            record.metadata.setdefault("promotion_rationale", decision.rationale)
        if stage is not None:
            record.stage = stage
        if kind is not None:
            record.kind = kind

        # Resolve entities and relations
        resolved_entities = entities if entities is not None else extract_entities(text)
        if relations is not None:
            resolved_relations = list(relations)
        else:
            resolved_relations = [
                Relation(left, rel_type, right, weight=1.0)
                for left, rel_type, right in extract_relations(text, resolved_entities)
            ]

        # Persist via write_text (record + chunks + relations)
        self.write_text(record, _entities=resolved_entities, _relations=resolved_relations)

        # Update in-memory scoring indexes incrementally
        self._scoring.index_record(rid, text, resolved_entities, resolved_relations)

        return record

    # ── Query ──

    def query(
        self,
        prompt: str,
        *,
        top_k: int = 5,
        hops: int | None = None,
        domain: str | None = None,
        namespace: str | None = None,
        now: datetime | None = None,
        as_of: datetime | None = None,
        include_staged: bool | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        rerank_top_n: int | None = None,
    ) -> list[QueryResult]:
        """Search memory using the full multi-signal scoring pipeline.

        Combines embedding similarity (from the persistent store), BM25 lexical
        scoring, knowledge-graph traversal, temporal overlap, recency, and
        confidence into a single fused score per candidate.  An optional
        second-stage reranker can further refine the top results.

        Args:
            prompt: Natural-language query.
            top_k: Maximum results to return.
            hops: Graph traversal depth (default from namespace policy or 2).
            domain: Filter candidates to a domain.
            namespace: Filter candidates to a namespace (and apply its policy).
            now: Reference time for recency scoring.
            as_of: Point-in-time for validity filtering.
            include_staged: Include draft/staged records in results.
            time_range: Explicit query time window; inferred from *prompt* if omitted.
            rerank_top_n: How many top candidates to pass through the reranker.

        Returns:
            Ranked list of :class:`QueryResult` with score breakdowns.

        Raises:
            TypeError: If *prompt* is not a string.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        self._ensure_scoring_bootstrapped()

        # Resolve namespace policy for hops
        policy = self._scoring._namespace_policies.get(namespace) if namespace is not None else None
        effective_hops = hops if hops is not None else (policy.default_hops if policy and policy.default_hops is not None else 2)

        # Step 1: embedding candidates from store
        query_embedding = self.embedder.embed(prompt)
        candidate_limit = max(top_k * 4, 50)
        embedding_results = self.store.semantic_search(
            query_embedding, domain=domain, namespace=namespace, top_k=candidate_limit,
        )
        embedding_scores = {r.record.record_id: r.semantic_score for r in embedding_results}

        # Step 2: BM25 candidates from in-memory index
        bm25_scores = self._scoring.index.score(prompt)

        # Step 3: graph candidates from in-memory graph
        query_entities = extract_entities(prompt)
        graph_scores = self._scoring.graph.records_for_entities(query_entities, hops=effective_hops)

        # Step 4: collect all candidate IDs, build candidates dict
        all_candidate_ids = set(embedding_scores) | set(bm25_scores) | set(graph_scores)
        candidates: dict[str, KnowledgeRecord] = {}
        for r in embedding_results:
            candidates[r.record.record_id] = r.record
        # Fetch remaining candidates not already in embedding results
        for rid in all_candidate_ids - set(candidates):
            record = self.store.fetch_record(rid)
            if record is not None:
                candidates[rid] = record

        # Step 5: full scoring pipeline
        return self._scoring.score_candidates(
            prompt, candidates, embedding_scores, bm25_scores, graph_scores,
            top_k=top_k, domain=domain, namespace=namespace,
            now=now, as_of=as_of, include_staged=include_staged,
            time_range=time_range, rerank_top_n=rerank_top_n,
        )

    # ── Record operations ──

    @property
    def records(self) -> dict[str, KnowledgeRecord]:
        """Backward-compat property. Loads records from store."""
        recs = self.store.list_records(limit=100_000)
        return {r.record_id: r for r in recs}

    def related_records(self, record_id: str, hops: int = 1, top_k: int = 8) -> list[KnowledgeRecord]:
        self._ensure_scoring_bootstrapped()
        entities = list(self._scoring.graph.record_entities(record_id))
        if not entities:
            return []
        scores = self._scoring.graph.records_for_entities(entities, hops=hops)
        ranked_ids = [rid for rid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True) if rid != record_id]
        now = datetime.now(tz=timezone.utc)
        results: list[KnowledgeRecord] = []
        for rid in ranked_ids[:top_k]:
            record = self.store.fetch_record(rid)
            if record is not None and ScoringPipeline._is_valid(record, now):
                results.append(record)
        return results

    def invalidate_record(
        self,
        record_id: str,
        *,
        invalidated_at: datetime | None = None,
        reason: str = "superseded",
    ) -> None:
        """Soft-delete a record by setting its validity end time.

        The record remains in the store for audit purposes but is excluded
        from future queries.

        Raises:
            KeyError: If *record_id* does not exist.
        """
        record = self.store.fetch_record(record_id)
        if record is None:
            raise KeyError(record_id)
        record.valid_to = invalidated_at or datetime.now(tz=timezone.utc)
        record.metadata["invalid_reason"] = reason
        self.store.upsert_record(record)

    def supersede_record_by_text(
        self,
        old_record_id: str,
        new_text: str,
        *,
        source: str = "unknown",
        domain: str | None = None,
        confidence: float | None = None,
        record_id: str | None = None,
        reason: str = "updated",
        entities: list[str] | None = None,
        relations: list[Relation] | None = None,
    ) -> KnowledgeRecord:
        """Supersede by providing new text instead of a pre-built KnowledgeRecord."""
        old = self.store.fetch_record(old_record_id)
        if old is None:
            raise KeyError(old_record_id)
        self.invalidate_record(old_record_id, reason=reason)
        new = self.ingest(
            new_text,
            source=source or old.source,
            domain=domain or old.domain,
            namespace=old.namespace,
            confidence=confidence if confidence is not None else old.confidence,
            tags=list(old.tags),
            metadata={**old.metadata, "supersession_reason": reason},
            record_id=record_id,
            entities=entities,
            relations=relations,
            occurred_from=old.occurred_from,
            occurred_to=old.occurred_to,
        )
        new.supersedes = old_record_id
        self.store.upsert_record(new)
        return new

    def supersession_chain(self, record_id: str) -> list[KnowledgeRecord]:
        return self.store.supersession_chain(record_id)

    def promote_record(
        self,
        source_record_id: str,
        *,
        reviewed_at: datetime | None = None,
        reason: str = "manual_review",
    ) -> KnowledgeRecord:
        """Promote a staged record to canonical status.

        Sets ``stage`` to ``"canonical"``, records the review timestamp,
        and enqueues graph projection if a graph store is configured.

        Raises:
            KeyError: If *source_record_id* does not exist.
        """
        record = self.store.fetch_record(source_record_id)
        if record is None:
            raise KeyError(source_record_id)
        record = _promote_record_fn(record, reviewed_at=reviewed_at)
        self.store.upsert_record(record)
        self.store.record_promotion(source_record_id, record.record_id, reason, record.reviewed_at or datetime.now(tz=timezone.utc))
        if self.graph is not None:
            self.store.enqueue_projection(record.record_id)
        return record

    def synthesize_observation(
        self,
        record_ids: list[str],
        *,
        record_id: str | None = None,
        source: str = "observation-synthesizer",
        max_sentences: int = 3,
    ) -> KnowledgeRecord:
        records = [self.store.fetch_record(rid) for rid in record_ids]
        records = [r for r in records if r is not None]
        observation = build_observation_record(
            records, record_id=record_id, source=source, max_sentences=max_sentences,
        )
        return self.ingest(
            observation.text,
            source=observation.source,
            domain=observation.domain,
            namespace=observation.namespace,
            confidence=observation.confidence,
            tags=observation.tags,
            metadata=observation.metadata,
            record_id=observation.record_id,
            auto_stage=False,
            stage=observation.stage,
            kind=observation.kind,
            occurred_from=observation.occurred_from,
            occurred_to=observation.occurred_to,
        )

    # ── Low-level persistence (existing API) ──

    def write_plan(self, record: KnowledgeRecord) -> HybridWritePlan:
        projection = build_graph_projection(record)
        return HybridWritePlan(
            record_id=record.record_id,
            graph_projection=projection,
        )

    def write_record(
        self,
        record: KnowledgeRecord,
        *,
        chunks: list[tuple[str, list[float], int]] | None = None,
    ) -> HybridWritePlan:
        plan = self.write_plan(record)
        self.store.upsert_record(record)
        if chunks is not None:
            self.store.replace_chunks(record.record_id, chunks)

        relation_rows = [
            (record.record_id, rel.source_entity, rel.relation, rel.target_entity, rel.weight)
            for rel in plan.graph_projection.relations
        ]
        if relation_rows:
            self.store.replace_relations(record.record_id, relation_rows)

        if record.stage in {"canonical", "pattern"} and self.graph is not None:
            self.store.enqueue_projection(record.record_id)

        return plan

    def write_text(
        self,
        record: KnowledgeRecord,
        *,
        max_chars: int = 800,
        overlap: int = 120,
        _entities: list[str] | None = None,
        _relations: list[Relation] | None = None,
    ) -> HybridWritePlan:
        chunks = [
            (chunk.text, self.embedder.embed(chunk.text), chunk.chunk_index)
            for chunk in chunk_text(record.text, max_chars=max_chars, overlap=overlap)
        ]
        if not chunks:
            chunks = [(record.text, self.embedder.embed(record.text), 0)]

        # Use caller-provided entities/relations if available (from ingest)
        if _entities is not None or _relations is not None:
            projection = GraphProjection(
                record_id=record.record_id,
                entities=_entities or [],
                relations=_relations or [],
            )
            plan = HybridWritePlan(record_id=record.record_id, graph_projection=projection)
            self.store.upsert_record(record)
            self.store.replace_chunks(record.record_id, chunks)
            relation_rows = [
                (record.record_id, rel.source_entity, rel.relation, rel.target_entity, rel.weight)
                for rel in projection.relations
            ]
            if relation_rows:
                self.store.replace_relations(record.record_id, relation_rows)
            if record.stage in {"canonical", "pattern"} and self.graph is not None:
                self.store.enqueue_projection(record.record_id)
            return plan

        return self.write_record(record, chunks=chunks)

    def list_records(
        self,
        *,
        domain: str | None = None,
        namespace: str | None = None,
        stage: str | None = None,
        limit: int = 100,
    ) -> list[KnowledgeRecord]:
        return self.store.list_records(domain=domain, namespace=namespace, stage=stage, limit=limit)

    def delete_record(self, record_id: str) -> bool:
        return self.store.delete_record(record_id)

    def semantic_search(
        self,
        query_embedding: list[float],
        *,
        domain: str | None = None,
        namespace: str | None = None,
        top_k: int = 5,
    ) -> list[QueryResult]:
        """Embedding-only search delegated to the backing store.

        For full multi-signal scoring use :meth:`query` instead.
        """
        return self.store.semantic_search(query_embedding, domain=domain, namespace=namespace, top_k=top_k)

    def supersede_record(
        self,
        old_record_id: str,
        new_record: KnowledgeRecord,
        *,
        reason: str = "updated",
    ) -> HybridWritePlan:
        """Replace an old record with a new one, invalidating the old."""
        old = self.store.fetch_record(old_record_id)
        if old is None:
            raise KeyError(old_record_id)
        old.valid_to = datetime.now(tz=timezone.utc)
        old.metadata["invalid_reason"] = reason
        self.store.upsert_record(old)
        new_record.supersedes = old_record_id
        return self.write_text(new_record)
