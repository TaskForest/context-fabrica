from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from math import sqrt
from typing import Any, cast

from ..models import KnowledgeRecord, QueryResult


class SQLiteRecordStore:
    """Zero-dependency persistent record store using stdlib sqlite3.

    Stores records, chunks, relations, and promotions in a single
    SQLite database file. Semantic search uses brute-force cosine
    similarity over stored embeddings — suitable for local development
    and single-agent workloads up to ~50k records.

    Usage:
        from context_fabrica.storage.sqlite import SQLiteRecordStore
        from context_fabrica import HybridMemoryStore

        store = HybridMemoryStore(store=SQLiteRecordStore("./memory.db"))
        store.bootstrap()
    """

    def __init__(self, path: str = "context_fabrica.db") -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def bootstrap(self) -> None:
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS memory_records (
                record_id TEXT PRIMARY KEY,
                text_content TEXT NOT NULL,
                source TEXT NOT NULL,
                domain TEXT NOT NULL,
                confidence REAL NOT NULL,
                memory_stage TEXT NOT NULL DEFAULT 'canonical',
                memory_kind TEXT NOT NULL DEFAULT 'fact',
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                supersedes TEXT,
                reviewed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS memory_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL REFERENCES memory_records(record_id) ON DELETE CASCADE,
                chunk_text TEXT NOT NULL,
                embedding TEXT,
                chunk_index INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS memory_relations (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL REFERENCES memory_records(record_id) ON DELETE CASCADE,
                source_entity TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0
            );

            CREATE TABLE IF NOT EXISTS memory_promotions (
                promotion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_record_id TEXT NOT NULL REFERENCES memory_records(record_id) ON DELETE CASCADE,
                target_record_id TEXT NOT NULL REFERENCES memory_records(record_id) ON DELETE CASCADE,
                promoted_at TEXT NOT NULL,
                reason TEXT NOT NULL,
                UNIQUE (source_record_id, target_record_id)
            );

            CREATE INDEX IF NOT EXISTS idx_records_domain ON memory_records(domain);
            CREATE INDEX IF NOT EXISTS idx_records_stage ON memory_records(memory_stage);
            CREATE INDEX IF NOT EXISTS idx_chunks_record ON memory_chunks(record_id);
            CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_entity);
            CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_entity);
        """)
        self.conn.commit()

    def upsert_record(self, record: KnowledgeRecord) -> None:
        self.conn.execute(
            """INSERT INTO memory_records (
                record_id, text_content, source, domain, confidence,
                memory_stage, memory_kind, tags, metadata,
                created_at, valid_from, valid_to, supersedes, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                text_content=excluded.text_content,
                source=excluded.source,
                domain=excluded.domain,
                confidence=excluded.confidence,
                memory_stage=excluded.memory_stage,
                memory_kind=excluded.memory_kind,
                tags=excluded.tags,
                metadata=excluded.metadata,
                created_at=excluded.created_at,
                valid_from=excluded.valid_from,
                valid_to=excluded.valid_to,
                supersedes=excluded.supersedes,
                reviewed_at=excluded.reviewed_at
            """,
            (
                record.record_id,
                record.text,
                record.source,
                record.domain,
                record.confidence,
                record.stage,
                record.kind,
                json.dumps(record.tags),
                json.dumps(record.metadata),
                record.created_at.isoformat(),
                record.valid_from.isoformat(),
                record.valid_to.isoformat() if record.valid_to else None,
                record.supersedes,
                record.reviewed_at.isoformat() if record.reviewed_at else None,
            ),
        )
        self.conn.commit()

    def fetch_record(self, record_id: str) -> KnowledgeRecord | None:
        row = self.conn.execute(
            "SELECT record_id, text_content, source, domain, confidence, "
            "memory_stage, memory_kind, tags, metadata, "
            "created_at, valid_from, valid_to, supersedes, reviewed_at "
            "FROM memory_records WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def replace_chunks(self, record_id: str, chunks: list[tuple[str, list[float], int]]) -> None:
        self.conn.execute("DELETE FROM memory_chunks WHERE record_id = ?", (record_id,))
        for chunk_text, embedding, chunk_index in chunks:
            self.conn.execute(
                "INSERT INTO memory_chunks (record_id, chunk_text, embedding, chunk_index) "
                "VALUES (?, ?, ?, ?)",
                (record_id, chunk_text, json.dumps(embedding), chunk_index),
            )
        self.conn.commit()

    def replace_relations(self, record_id: str, relations: list[tuple[str, str, str, str, float]]) -> None:
        self.conn.execute("DELETE FROM memory_relations WHERE record_id = ?", (record_id,))
        for row in relations:
            self.conn.execute(
                "INSERT INTO memory_relations (record_id, source_entity, relation_type, target_entity, weight) "
                "VALUES (?, ?, ?, ?, ?)",
                row,
            )
        self.conn.commit()

    def record_promotion(self, source_record_id: str, target_record_id: str, reason: str, promoted_at: datetime) -> None:
        self.conn.execute(
            "INSERT INTO memory_promotions (source_record_id, target_record_id, promoted_at, reason) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(source_record_id, target_record_id) DO UPDATE SET "
            "promoted_at=excluded.promoted_at, reason=excluded.reason",
            (source_record_id, target_record_id, promoted_at.isoformat(), reason),
        )
        self.conn.commit()

    def enqueue_projection(self, record_id: str) -> None:
        # No-op for SQLite — graph projection is optional
        pass

    def semantic_search(
        self,
        query_embedding: list[float],
        *,
        domain: str | None = None,
        top_k: int = 5,
    ) -> list[QueryResult]:
        now = datetime.now(tz=timezone.utc).isoformat()

        query = (
            "SELECT r.record_id, r.text_content, r.source, r.domain, r.confidence, "
            "r.memory_stage, r.memory_kind, r.tags, r.metadata, "
            "r.created_at, r.valid_from, r.valid_to, r.supersedes, r.reviewed_at, "
            "c.embedding "
            "FROM memory_chunks c "
            "JOIN memory_records r ON r.record_id = c.record_id "
            "WHERE r.memory_stage <> 'staged' "
            "AND r.valid_from <= ? "
            "AND (r.valid_to IS NULL OR r.valid_to >= ?) "
        )
        params: list[Any] = [now, now]

        if domain is not None:
            query += "AND r.domain = ? "
            params.append(domain)

        rows = self.conn.execute(query, params).fetchall()

        scored: list[tuple[float, Any]] = []
        for row in rows:
            embedding_json = row[14]
            if embedding_json is None:
                continue
            stored_embedding = json.loads(embedding_json)
            sim = self._cosine_similarity(query_embedding, stored_embedding)
            if sim > 0.0:
                scored.append((sim, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[QueryResult] = []
        for score, row in scored[:top_k]:
            record = self._row_to_record(row[:14])
            results.append(
                QueryResult(
                    record=record,
                    score=score,
                    semantic_score=score,
                    graph_score=0.0,
                    recency_score=0.0,
                    confidence_score=record.confidence,
                    rationale=["semantic_match", f"stage:{record.stage}", f"kind:{record.kind}"],
                )
            )
        return results

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sqrt(sum(x * x for x in a)) or 1.0
        norm_b = sqrt(sum(x * x for x in b)) or 1.0
        return max(dot / (norm_a * norm_b), 0.0)

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> KnowledgeRecord:
        return KnowledgeRecord(
            record_id=str(row[0]),
            text=str(row[1]),
            source=str(row[2]),
            domain=str(row[3]),
            confidence=float(row[4]),
            stage=cast(Any, str(row[5])),
            kind=cast(Any, str(row[6])),
            tags=json.loads(row[7]) if isinstance(row[7], str) else list(row[7]),
            metadata=json.loads(row[8]) if isinstance(row[8], str) else dict(row[8]),
            created_at=datetime.fromisoformat(str(row[9])),
            valid_from=datetime.fromisoformat(str(row[10])),
            valid_to=datetime.fromisoformat(str(row[11])) if row[11] else None,
            supersedes=str(row[12]) if row[12] else None,
            reviewed_at=datetime.fromisoformat(str(row[13])) if row[13] else None,
        )
