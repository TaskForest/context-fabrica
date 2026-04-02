from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..adapters import GraphStore, RecordStore
from ..config import HybridStoreSettings
from ..embedding import Embedder, build_default_embedder, chunk_text
from ..models import KnowledgeRecord
from ..projection import GraphProjection, build_graph_projection
from .kuzu import KuzuGraphProjectionAdapter
from .postgres import PostgresPgvectorAdapter


@dataclass(frozen=True)
class HybridWritePlan:
    record_id: str
    graph_projection: GraphProjection


class HybridMemoryStore:
    """Orchestrates record storage with optional graph projection.

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
    ) -> None:
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

        self.embedder = embedder or build_default_embedder(dimensions=dims)

    def bootstrap(self) -> None:
        self.store.bootstrap()
        if self.graph is not None:
            self.graph.bootstrap()

    # Keep backward-compatible alias
    def bootstrap_postgres(self) -> None:
        self.store.bootstrap()

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
    ) -> HybridWritePlan:
        chunks = [
            (chunk.text, self.embedder.embed(chunk.text), chunk.chunk_index)
            for chunk in chunk_text(record.text, max_chars=max_chars, overlap=overlap)
        ]
        if not chunks:
            chunks = [(record.text, self.embedder.embed(record.text), 0)]
        return self.write_record(record, chunks=chunks)

    def promote_record(
        self,
        source_record_id: str,
        *,
        reviewed_at: datetime | None = None,
        reason: str = "manual_review",
    ) -> KnowledgeRecord:
        record = self.store.fetch_record(source_record_id)
        if record is None:
            raise KeyError(source_record_id)
        record.stage = "canonical"
        record.reviewed_at = reviewed_at or datetime.now(tz=timezone.utc)
        if record.kind == "note":
            record.kind = "fact"
        self.store.upsert_record(record)
        self.store.record_promotion(source_record_id, record.record_id, reason, record.reviewed_at)
        if self.graph is not None:
            self.store.enqueue_projection(record.record_id)
        return record

    def semantic_search(
        self,
        query_embedding: list[float],
        *,
        domain: str | None = None,
        top_k: int = 5,
    ) -> list[Any]:
        return self.store.semantic_search(query_embedding, domain=domain, top_k=top_k)
