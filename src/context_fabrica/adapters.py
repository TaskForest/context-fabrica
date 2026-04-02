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

    def fetch_record(self, record_id: str) -> KnowledgeRecord | None: ...

    def replace_chunks(self, record_id: str, chunks: list[tuple[str, list[float], int]]) -> None: ...

    def replace_relations(self, record_id: str, relations: list[tuple[str, str, str, str, float]]) -> None: ...

    def record_promotion(self, source_record_id: str, target_record_id: str, reason: str, promoted_at: datetime) -> None: ...

    def semantic_search(
        self,
        query_embedding: list[float],
        *,
        domain: str | None = None,
        top_k: int = 5,
    ) -> list[QueryResult]: ...

    def enqueue_projection(self, record_id: str) -> None: ...


class GraphStore(Protocol):
    """Protocol for optional graph projection backends.

    Implement this to plug in any graph store (Kuzu, Neo4j, Memgraph,
    etc.) for relation-heavy multi-hop traversal.
    """

    def bootstrap(self) -> None: ...

    def project(self, projection: Any, *, domain: str, source: str) -> None: ...


class TrustPolicyAdapter(Protocol):
    def is_source_allowed(self, source: str) -> bool: ...

    def score(self, *, confidence: float, source: str, created_at: datetime) -> float: ...
