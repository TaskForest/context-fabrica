from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
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
                namespace TEXT NOT NULL DEFAULT 'default',
                confidence REAL NOT NULL,
                memory_stage TEXT NOT NULL DEFAULT 'canonical',
                memory_kind TEXT NOT NULL DEFAULT 'fact',
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                supersedes TEXT,
                reviewed_at TEXT,
                occurred_from TEXT,
                occurred_to TEXT
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

            CREATE TABLE IF NOT EXISTS memory_outcomes (
                outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL REFERENCES memory_records(record_id) ON DELETE CASCADE,
                query_text TEXT NOT NULL,
                outcome TEXT NOT NULL,
                delta REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_outcomes_record ON memory_outcomes(record_id);
            CREATE INDEX IF NOT EXISTS idx_records_namespace ON memory_records(namespace);
            CREATE INDEX IF NOT EXISTS idx_records_domain ON memory_records(domain);
            CREATE INDEX IF NOT EXISTS idx_records_stage ON memory_records(memory_stage);
            CREATE INDEX IF NOT EXISTS idx_chunks_record ON memory_chunks(record_id);
            CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_entity);
            CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_entity);
        """)
        self._ensure_record_column("occurred_from", "TEXT")
        self._ensure_record_column("occurred_to", "TEXT")
        self.conn.commit()

    _SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    def _ensure_record_column(self, name: str, column_type: str) -> None:
        if not self._SAFE_IDENTIFIER.match(name) or not self._SAFE_IDENTIFIER.match(column_type):
            raise ValueError(f"Invalid column name or type: {name!r} {column_type!r}")
        columns = {
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(memory_records)").fetchall()
        }
        if name not in columns:
            self.conn.execute(f"ALTER TABLE memory_records ADD COLUMN {name} {column_type}")

    def _upsert_record_params(self, record: KnowledgeRecord) -> tuple[Any, ...]:
        return (
            record.record_id,
            record.text,
            record.source,
            record.domain,
            record.namespace,
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
            record.occurred_from.isoformat() if record.occurred_from else None,
            record.occurred_to.isoformat() if record.occurred_to else None,
        )

    _UPSERT_SQL = """INSERT INTO memory_records (
            record_id, text_content, source, domain, namespace, confidence,
            memory_stage, memory_kind, tags, metadata,
            created_at, valid_from, valid_to, supersedes, reviewed_at, occurred_from, occurred_to
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_id) DO UPDATE SET
            text_content=excluded.text_content,
            source=excluded.source,
            domain=excluded.domain,
            namespace=excluded.namespace,
            confidence=excluded.confidence,
            memory_stage=excluded.memory_stage,
            memory_kind=excluded.memory_kind,
            tags=excluded.tags,
            metadata=excluded.metadata,
            created_at=excluded.created_at,
            valid_from=excluded.valid_from,
            valid_to=excluded.valid_to,
            supersedes=excluded.supersedes,
            reviewed_at=excluded.reviewed_at,
            occurred_from=excluded.occurred_from,
            occurred_to=excluded.occurred_to
        """

    def upsert_record(self, record: KnowledgeRecord) -> None:
        self.conn.execute(self._UPSERT_SQL, self._upsert_record_params(record))
        self.conn.commit()

    def fetch_record_with_chunks(self, record_id: str) -> tuple[KnowledgeRecord, list[tuple[str, list[float], int]]] | None:
        record = self.fetch_record(record_id)
        if record is None:
            return None
        rows = self.conn.execute(
            "SELECT chunk_text, embedding, chunk_index FROM memory_chunks "
            "WHERE record_id = ? ORDER BY chunk_index",
            (record_id,),
        ).fetchall()
        chunks = [
            (str(row[0]), json.loads(row[1]) if row[1] else [], int(row[2]))
            for row in rows
        ]
        return (record, chunks)

    def upsert_records(self, records: list[KnowledgeRecord]) -> None:
        for record in records:
            self.conn.execute(self._UPSERT_SQL, self._upsert_record_params(record))
        self.conn.commit()

    def fetch_record(self, record_id: str) -> KnowledgeRecord | None:
        row = self.conn.execute(
            "SELECT record_id, text_content, source, domain, namespace, confidence, "
            "memory_stage, memory_kind, tags, metadata, "
            "created_at, valid_from, valid_to, supersedes, reviewed_at, occurred_from, occurred_to "
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

    def list_records(
        self,
        *,
        domain: str | None = None,
        namespace: str | None = None,
        stage: str | None = None,
        limit: int = 100,
    ) -> list[KnowledgeRecord]:
        query = (
            "SELECT record_id, text_content, source, domain, namespace, confidence, "
            "memory_stage, memory_kind, tags, metadata, "
            "created_at, valid_from, valid_to, supersedes, reviewed_at, occurred_from, occurred_to "
            "FROM memory_records WHERE 1=1 "
        )
        params: list[Any] = []
        if domain is not None:
            query += "AND domain = ? "
            params.append(domain)
        if namespace is not None:
            query += "AND namespace = ? "
            params.append(namespace)
        if stage is not None:
            query += "AND memory_stage = ? "
            params.append(stage)
        query += "ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def delete_record(self, record_id: str) -> bool:
        """Delete a record and cascade to chunks, relations, and promotions."""
        cur = self.conn.execute("DELETE FROM memory_records WHERE record_id = ?", (record_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def expire_records(self, *, before: datetime) -> int:
        """Soft-expire records created before the given timestamp."""
        now = datetime.now(tz=timezone.utc).isoformat()
        cur = self.conn.execute(
            "UPDATE memory_records SET valid_to = ? "
            "WHERE valid_to IS NULL AND created_at < ?",
            (now, before.isoformat()),
        )
        self.conn.commit()
        return cur.rowcount

    def decay_confidence(self, *, older_than_days: int, decay_factor: float = 0.95) -> int:
        """Multiply confidence by decay_factor for records older than N days."""
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=older_than_days)).isoformat()
        cur = self.conn.execute(
            "UPDATE memory_records SET confidence = confidence * ? "
            "WHERE created_at < ? AND confidence > 0.01 AND valid_to IS NULL",
            (decay_factor, cutoff),
        )
        self.conn.commit()
        return cur.rowcount

    def purge_expired(self) -> int:
        """Hard-delete records whose valid_to is in the past."""
        now = datetime.now(tz=timezone.utc).isoformat()
        cur = self.conn.execute(
            "DELETE FROM memory_records WHERE valid_to IS NOT NULL AND valid_to < ?",
            (now,),
        )
        self.conn.commit()
        return cur.rowcount

    def record_outcome(
        self,
        record_id: str,
        query_text: str,
        outcome: str,
        *,
        delta: float = 0.0,
    ) -> None:
        """Record whether a retrieved record was useful.

        outcome: "useful", "not_useful", "partially_useful", "misleading"
        delta: confidence adjustment to apply (e.g. +0.05 for useful, -0.1 for misleading)
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO memory_outcomes (record_id, query_text, outcome, delta, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (record_id, query_text, outcome, delta, now),
        )
        if delta != 0.0:
            self.conn.execute(
                "UPDATE memory_records SET confidence = min(1.0, max(0.0, confidence + ?)) WHERE record_id = ?",
                (delta, record_id),
            )
        self.conn.commit()

    def outcome_summary(self, record_id: str) -> dict[str, int]:
        """Count outcomes by type for a record."""
        rows = self.conn.execute(
            "SELECT outcome, count(*) FROM memory_outcomes WHERE record_id = ? GROUP BY outcome",
            (record_id,),
        ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def supersession_chain(self, record_id: str) -> list[KnowledgeRecord]:
        """Walk supersession chain backward from record to origin."""
        chain: list[KnowledgeRecord] = []
        seen: set[str] = set()
        current_id: str | None = record_id
        while current_id and current_id not in seen:
            seen.add(current_id)
            record = self.fetch_record(current_id)
            if record is None:
                break
            chain.append(record)
            current_id = record.supersedes
        return chain

    def enqueue_projection(self, record_id: str) -> None:
        # No-op for SQLite — graph projection is optional
        pass

    def semantic_search(
        self,
        query_embedding: list[float],
        *,
        domain: str | None = None,
        namespace: str | None = None,
        top_k: int = 5,
    ) -> list[QueryResult]:
        now = datetime.now(tz=timezone.utc).isoformat()

        query = (
            "SELECT r.record_id, r.text_content, r.source, r.domain, r.namespace, r.confidence, "
            "r.memory_stage, r.memory_kind, r.tags, r.metadata, "
            "r.created_at, r.valid_from, r.valid_to, r.supersedes, r.reviewed_at, r.occurred_from, r.occurred_to, "
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
        if namespace is not None:
            query += "AND r.namespace = ? "
            params.append(namespace)

        rows = self.conn.execute(query, params).fetchall()

        scored: list[tuple[float, Any]] = []
        for row in rows:
            embedding_json = row[17]
            if embedding_json is None:
                continue
            stored_embedding = json.loads(embedding_json)
            sim = self._cosine_similarity(query_embedding, stored_embedding)
            if sim > 0.0:
                scored.append((sim, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[QueryResult] = []
        for score, row in scored[:top_k]:
            record = self._row_to_record(row[:17])
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
            namespace=str(row[4]),
            confidence=float(row[5]),
            stage=cast(Any, str(row[6])),
            kind=cast(Any, str(row[7])),
            tags=json.loads(row[8]) if isinstance(row[8], str) else list(row[8]),
            metadata=json.loads(row[9]) if isinstance(row[9], str) else dict(row[9]),
            created_at=datetime.fromisoformat(str(row[10])),
            valid_from=datetime.fromisoformat(str(row[11])),
            valid_to=datetime.fromisoformat(str(row[12])) if row[12] else None,
            supersedes=str(row[13]) if row[13] else None,
            reviewed_at=datetime.fromisoformat(str(row[14])) if row[14] else None,
            occurred_from=datetime.fromisoformat(str(row[15])) if row[15] else None,
            occurred_to=datetime.fromisoformat(str(row[16])) if row[16] else None,
        )

    def list_all_texts(self, *, namespace: str | None = None) -> list[tuple[str, str]]:
        query = "SELECT record_id, text_content FROM memory_records WHERE valid_to IS NULL"
        params: list[Any] = []
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        return [(str(r[0]), str(r[1])) for r in self.conn.execute(query, params).fetchall()]

    def list_all_relations(self, *, namespace: str | None = None) -> list[tuple[str, str, str, str, float]]:
        if namespace is not None:
            query = (
                "SELECT mr.record_id, mr.source_entity, mr.relation_type, mr.target_entity, mr.weight "
                "FROM memory_relations mr "
                "JOIN memory_records r ON r.record_id = mr.record_id "
                "WHERE r.namespace = ?"
            )
            rows = self.conn.execute(query, (namespace,)).fetchall()
        else:
            query = "SELECT record_id, source_entity, relation_type, target_entity, weight FROM memory_relations"
            rows = self.conn.execute(query).fetchall()
        return [(str(r[0]), str(r[1]), str(r[2]), str(r[3]), float(r[4])) for r in rows]
