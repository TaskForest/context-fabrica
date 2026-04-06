from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .models import KnowledgeRecord, QueryResult


@dataclass
class RetrievedChunk:
    record_id: str
    score: float
    source: str


class RecordStore(Protocol):
    """Protocol for persistent record storage backends.

    Implement this to plug in any storage backend (SQLite, Postgres,
    DuckDB, etc.) into HybridMemoryStore.
    """

    def bootstrap(self) -> None: ...

    def upsert_record(self, record: KnowledgeRecord) -> None: ...

    def upsert_records(self, records: list[KnowledgeRecord]) -> None: ...

    def fetch_record(self, record_id: str) -> KnowledgeRecord | None: ...

    def fetch_record_with_chunks(self, record_id: str) -> tuple[KnowledgeRecord, list[tuple[str, list[float], int]]] | None: ...

    def replace_chunks(self, record_id: str, chunks: list[tuple[str, list[float], int]]) -> None: ...

    def replace_relations(self, record_id: str, relations: list[tuple[str, str, str, str, float]]) -> None: ...

    def record_promotion(self, source_record_id: str, target_record_id: str, reason: str, promoted_at: datetime) -> None: ...

    def semantic_search(
        self,
        query_embedding: list[float],
        *,
        domain: str | None = None,
        namespace: str | None = None,
        top_k: int = 5,
    ) -> list[QueryResult]: ...

    def delete_record(self, record_id: str) -> bool: ...

    def list_records(
        self,
        *,
        domain: str | None = None,
        namespace: str | None = None,
        stage: str | None = None,
        limit: int = 100,
    ) -> list[KnowledgeRecord]: ...

    def expire_records(self, *, before: datetime) -> int: ...

    def decay_confidence(self, *, older_than_days: int, decay_factor: float = 0.95) -> int: ...

    def purge_expired(self) -> int: ...

    def supersession_chain(self, record_id: str) -> list[KnowledgeRecord]: ...

    def record_outcome(self, record_id: str, query_text: str, outcome: str, *, delta: float = 0.0) -> None: ...

    def outcome_summary(self, record_id: str) -> dict[str, int]: ...

    def enqueue_projection(self, record_id: str) -> None: ...

    def list_all_texts(self, *, namespace: str | None = None) -> list[tuple[str, str]]:
        """Return (record_id, text) pairs for all valid records.

        Used by the scoring pipeline to bootstrap the BM25 index.
        """
        ...

    def list_all_relations(self, *, namespace: str | None = None) -> list[tuple[str, str, str, str, float]]:
        """Return (record_id, source_entity, relation_type, target_entity, weight) tuples.

        Used by the scoring pipeline to bootstrap the in-memory knowledge graph.
        """
        ...


class GraphStore(Protocol):
    """Protocol for optional graph projection backends.

    Implement this to plug in any graph store (Kuzu, Neo4j, Memgraph,
    etc.) for relation-heavy multi-hop traversal.
    """

    def bootstrap(self) -> None: ...

    def project(self, projection: Any, *, domain: str, source: str) -> None: ...


class Reranker(Protocol):
    """Protocol for optional second-stage rerankers."""

    def score(self, query: str, record: KnowledgeRecord) -> float: ...


class TrustPolicyAdapter(Protocol):
    def is_source_allowed(self, source: str) -> bool: ...

    def score(self, *, confidence: float, source: str, created_at: datetime) -> float: ...
