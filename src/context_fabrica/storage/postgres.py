from __future__ import annotations

from contextlib import suppress
import json
from importlib import import_module
from typing import Any, cast

from ..config import PostgresSettings
from ..models import KnowledgeRecord, QueryResult


class PostgresPgvectorAdapter:
    def __init__(self, settings: PostgresSettings) -> None:
        self.settings = settings

    def bootstrap_statements(self) -> list[str]:
        schema = self.settings.schema
        dims = self.settings.embedding_dimensions
        return [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            f"CREATE SCHEMA IF NOT EXISTS {schema};",
            f"CREATE TABLE IF NOT EXISTS {schema}.memory_records ("
            "record_id TEXT PRIMARY KEY, "
            "text_content TEXT NOT NULL, "
            "source TEXT NOT NULL, "
            "domain TEXT NOT NULL, "
            "confidence DOUBLE PRECISION NOT NULL, "
            "memory_stage TEXT NOT NULL DEFAULT 'canonical', "
            "memory_kind TEXT NOT NULL DEFAULT 'fact', "
            "tags JSONB NOT NULL DEFAULT '[]'::jsonb, "
            "metadata JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "created_at TIMESTAMPTZ NOT NULL, "
            "valid_from TIMESTAMPTZ NOT NULL, "
            "valid_to TIMESTAMPTZ NULL, "
            "supersedes TEXT NULL, "
            "reviewed_at TIMESTAMPTZ NULL"
            ");",
            f"ALTER TABLE {schema}.memory_records ADD COLUMN IF NOT EXISTS memory_stage TEXT NOT NULL DEFAULT 'canonical';",
            f"ALTER TABLE {schema}.memory_records ADD COLUMN IF NOT EXISTS memory_kind TEXT NOT NULL DEFAULT 'fact';",
            f"ALTER TABLE {schema}.memory_records ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ NULL;",
            f"CREATE TABLE IF NOT EXISTS {schema}.memory_chunks ("
            "chunk_id BIGSERIAL PRIMARY KEY, "
            f"record_id TEXT NOT NULL REFERENCES {schema}.memory_records(record_id) ON DELETE CASCADE, "
            "chunk_text TEXT NOT NULL, "
            f"embedding vector({dims}) NULL, "
            "chunk_index INTEGER NOT NULL DEFAULT 0"
            ");",
            f"CREATE TABLE IF NOT EXISTS {schema}.memory_relations ("
            "relation_id BIGSERIAL PRIMARY KEY, "
            f"record_id TEXT NOT NULL REFERENCES {schema}.memory_records(record_id) ON DELETE CASCADE, "
            "source_entity TEXT NOT NULL, "
            "relation_type TEXT NOT NULL, "
            "target_entity TEXT NOT NULL, "
            "weight DOUBLE PRECISION NOT NULL DEFAULT 1.0"
            ");",
            f"CREATE TABLE IF NOT EXISTS {schema}.memory_promotions ("
            "promotion_id BIGSERIAL PRIMARY KEY, "
            "source_record_id TEXT NOT NULL REFERENCES {schema}.memory_records(record_id) ON DELETE CASCADE, "
            "target_record_id TEXT NOT NULL REFERENCES {schema}.memory_records(record_id) ON DELETE CASCADE, "
            "promoted_at TIMESTAMPTZ NOT NULL, "
            "reason TEXT NOT NULL, "
            "UNIQUE (source_record_id, target_record_id)"
            ");".replace("{schema}", schema),
            f"CREATE TABLE IF NOT EXISTS {schema}.projection_jobs ("
            "job_id BIGSERIAL PRIMARY KEY, "
            "record_id TEXT NOT NULL REFERENCES {schema}.memory_records(record_id) ON DELETE CASCADE, "
            "job_type TEXT NOT NULL DEFAULT 'project_record', "
            "status TEXT NOT NULL DEFAULT 'pending', "
            "attempt_count INTEGER NOT NULL DEFAULT 0, "
            "last_error TEXT NULL, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "UNIQUE (record_id, job_type)"
            ");".replace("{schema}", schema),
            f"CREATE INDEX IF NOT EXISTS idx_{schema}_records_domain ON {schema}.memory_records(domain);",
            f"CREATE INDEX IF NOT EXISTS idx_{schema}_records_stage ON {schema}.memory_records(memory_stage);",
            f"CREATE INDEX IF NOT EXISTS idx_{schema}_records_validity ON {schema}.memory_records(valid_from, valid_to);",
            f"CREATE INDEX IF NOT EXISTS idx_{schema}_relations_source ON {schema}.memory_relations(source_entity);",
            f"CREATE INDEX IF NOT EXISTS idx_{schema}_relations_target ON {schema}.memory_relations(target_entity);",
            f"CREATE INDEX IF NOT EXISTS idx_{schema}_chunks_embedding ON {schema}.memory_chunks USING hnsw (embedding vector_cosine_ops);",
            f"CREATE INDEX IF NOT EXISTS idx_{schema}_projection_jobs_status ON {schema}.projection_jobs(status, updated_at);",
            f"CREATE OR REPLACE FUNCTION {schema}.notify_projection_job() RETURNS trigger AS $$ "
            f"BEGIN PERFORM pg_notify('{schema}_projection_jobs', NEW.record_id); RETURN NEW; END; "
            "$$ LANGUAGE plpgsql;",
            f"DROP TRIGGER IF EXISTS trg_projection_notify ON {schema}.projection_jobs;",
            f"CREATE TRIGGER trg_projection_notify AFTER INSERT OR UPDATE OF status ON {schema}.projection_jobs "
            f"FOR EACH ROW WHEN (NEW.status = 'pending') "
            f"EXECUTE FUNCTION {schema}.notify_projection_job();",
        ]

    def upsert_record_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"INSERT INTO {schema}.memory_records ("
            "record_id, text_content, source, domain, confidence, memory_stage, memory_kind, tags, metadata, created_at, valid_from, valid_to, supersedes, reviewed_at"
            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s) "
            "ON CONFLICT (record_id) DO UPDATE SET "
            "text_content = EXCLUDED.text_content, "
            "source = EXCLUDED.source, "
            "domain = EXCLUDED.domain, "
            "confidence = EXCLUDED.confidence, "
            "memory_stage = EXCLUDED.memory_stage, "
            "memory_kind = EXCLUDED.memory_kind, "
            "tags = EXCLUDED.tags, "
            "metadata = EXCLUDED.metadata, "
            "created_at = EXCLUDED.created_at, "
            "valid_from = EXCLUDED.valid_from, "
            "valid_to = EXCLUDED.valid_to, "
            "supersedes = EXCLUDED.supersedes, "
            "reviewed_at = EXCLUDED.reviewed_at;"
        )

    def replace_chunks_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"INSERT INTO {schema}.memory_chunks (record_id, chunk_text, embedding, chunk_index) "
            "VALUES (%s, %s, %s, %s);"
        )

    def delete_chunks_statement(self) -> str:
        schema = self.settings.schema
        return f"DELETE FROM {schema}.memory_chunks WHERE record_id = %s;"

    def replace_relations_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"INSERT INTO {schema}.memory_relations (record_id, source_entity, relation_type, target_entity, weight) "
            "VALUES (%s, %s, %s, %s, %s);"
        )

    def delete_relations_statement(self) -> str:
        schema = self.settings.schema
        return f"DELETE FROM {schema}.memory_relations WHERE record_id = %s;"

    def fetch_record_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"SELECT record_id, text_content, source, domain, confidence, memory_stage, memory_kind, tags, metadata, "
            "created_at, valid_from, valid_to, supersedes, reviewed_at "
            f"FROM {schema}.memory_records WHERE record_id = %s;"
        )

    def insert_promotion_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"INSERT INTO {schema}.memory_promotions (source_record_id, target_record_id, promoted_at, reason) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (source_record_id, target_record_id) DO UPDATE SET "
            "promoted_at = EXCLUDED.promoted_at, reason = EXCLUDED.reason;"
        )

    def enqueue_projection_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"INSERT INTO {schema}.projection_jobs (record_id, job_type, status, attempt_count, last_error, created_at, updated_at) "
            "VALUES (%s, 'project_record', 'pending', 0, NULL, now(), now()) "
            "ON CONFLICT (record_id, job_type) DO UPDATE SET "
            "status = 'pending', updated_at = now(), last_error = NULL;"
        )

    def claim_projection_jobs_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"UPDATE {schema}.projection_jobs j SET status = 'processing', attempt_count = attempt_count + 1, updated_at = now() "
            "WHERE j.job_id IN ("
            f"SELECT job_id FROM {schema}.projection_jobs WHERE status = 'pending' ORDER BY created_at LIMIT %s"
            ") RETURNING job_id, record_id;"
        )

    def complete_projection_job_statement(self) -> str:
        schema = self.settings.schema
        return f"UPDATE {schema}.projection_jobs SET status = 'done', updated_at = now(), last_error = NULL WHERE job_id = %s;"

    def fail_projection_job_statement(self) -> str:
        schema = self.settings.schema
        return f"UPDATE {schema}.projection_jobs SET status = 'failed', updated_at = now(), last_error = %s WHERE job_id = %s;"

    def list_projection_jobs_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"SELECT job_id, record_id, status, attempt_count, coalesce(last_error, ''), created_at, updated_at "
            f"FROM {schema}.projection_jobs ORDER BY updated_at DESC LIMIT %s;"
        )

    def retry_failed_jobs_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"UPDATE {schema}.projection_jobs SET status = 'pending', updated_at = now(), last_error = NULL "
            "WHERE status = 'failed' RETURNING job_id, record_id;"
        )

    def requeue_record_projection_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"INSERT INTO {schema}.projection_jobs (record_id, job_type, status, attempt_count, last_error, created_at, updated_at) "
            "VALUES (%s, 'project_record', 'pending', 0, NULL, now(), now()) "
            "ON CONFLICT (record_id, job_type) DO UPDATE SET status = 'pending', updated_at = now(), last_error = NULL "
            "RETURNING job_id, record_id;"
        )

    def requeue_canonical_projection_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"INSERT INTO {schema}.projection_jobs (record_id, job_type, status, attempt_count, last_error, created_at, updated_at) "
            f"SELECT record_id, 'project_record', 'pending', 0, NULL, now(), now() FROM {schema}.memory_records "
            "WHERE memory_stage IN ('canonical', 'pattern') "
            "AND (%s::text IS NULL OR domain = %s) "
            "ON CONFLICT (record_id, job_type) DO UPDATE SET status = 'pending', updated_at = now(), last_error = NULL "
            "RETURNING job_id, record_id;"
        )

    def projection_queue_summary_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"SELECT status, count(*) FROM {schema}.projection_jobs GROUP BY status ORDER BY status;"
        )

    def health_probe_statement(self) -> str:
        schema = self.settings.schema
        return (
            "SELECT current_database(), current_user, exists (select 1 from pg_extension where extname = 'vector');"
        )

    def search_statement(self) -> str:
        schema = self.settings.schema
        return (
            f"SELECT r.record_id, r.text_content, r.source, r.domain, r.confidence, r.memory_stage, r.memory_kind, c.chunk_text, "
            "1 - (c.embedding <=> %s) AS semantic_score "
            f"FROM {schema}.memory_chunks c "
            f"JOIN {schema}.memory_records r ON r.record_id = c.record_id "
            "WHERE (%s::text IS NULL OR r.domain = %s) "
            "AND r.valid_from <= %s "
            "AND (r.valid_to IS NULL OR r.valid_to >= %s) "
            "AND r.memory_stage <> 'staged' "
            "ORDER BY c.embedding <=> %s LIMIT %s;"
        )

    def upsert_record_payload(self, record: KnowledgeRecord) -> tuple[object, ...]:
        return (
            record.record_id,
            record.text,
            record.source,
            record.domain,
            record.confidence,
            record.stage,
            record.kind,
            json.dumps(record.tags),
            json.dumps(record.metadata),
            record.created_at,
            record.valid_from,
            record.valid_to,
            record.supersedes,
            record.reviewed_at,
        )

    def bootstrap(self) -> None:
        conn = self.connect()
        with conn:
            with conn.cursor() as cur:
                for statement in self.bootstrap_statements():
                    cur.execute(statement)
            conn.commit()

    def upsert_record(self, record: KnowledgeRecord) -> None:
        conn = self.connect()
        with conn:
            self._ensure_vector_registered(conn)
            with conn.cursor() as cur:
                cur.execute(self.upsert_record_statement(), self.upsert_record_payload(record))
            conn.commit()

    def record_promotion(self, source_record_id: str, target_record_id: str, reason: str, promoted_at) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(self.insert_promotion_statement(), (source_record_id, target_record_id, promoted_at, reason))
            conn.commit()

    def replace_chunks(self, record_id: str, chunks: list[tuple[str, list[float], int]]) -> None:
        conn = self.connect()
        with conn:
            self._ensure_vector_registered(conn)
            with conn.cursor() as cur:
                cur.execute(self.delete_chunks_statement(), (record_id,))
                for chunk_text, embedding, chunk_index in chunks:
                    cur.execute(
                        self.replace_chunks_statement(),
                        (record_id, chunk_text, embedding, chunk_index),
                    )
            conn.commit()

    def replace_relations(self, record_id: str, relations: list[tuple[str, str, str, str, float]]) -> None:
        conn = self.connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(self.delete_relations_statement(), (record_id,))
                for row in relations:
                    cur.execute(self.replace_relations_statement(), row)
            conn.commit()

    def enqueue_projection(self, record_id: str) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(self.enqueue_projection_statement(), (record_id,))
            conn.commit()

    def claim_projection_jobs(self, limit: int = 10) -> list[tuple[int, str]]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(self.claim_projection_jobs_statement(), (limit,))
                rows = cur.fetchall()
            conn.commit()
        return [(int(row[0]), str(row[1])) for row in rows]

    def complete_projection_job(self, job_id: int) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(self.complete_projection_job_statement(), (job_id,))
            conn.commit()

    def fail_projection_job(self, job_id: int, error: str) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(self.fail_projection_job_statement(), (error, job_id))
            conn.commit()

    def list_projection_jobs(self, limit: int = 25) -> list[tuple[int, str, str, int, str, object, object]]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(self.list_projection_jobs_statement(), (limit,))
                rows = cur.fetchall()
        return [
            (int(row[0]), str(row[1]), str(row[2]), int(row[3]), str(row[4]), row[5], row[6])
            for row in rows
        ]

    def retry_failed_jobs(self) -> list[tuple[int, str]]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(self.retry_failed_jobs_statement())
                rows = cur.fetchall()
            conn.commit()
        return [(int(row[0]), str(row[1])) for row in rows]

    def requeue_record_projection(self, record_id: str) -> tuple[int, str] | None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(self.requeue_record_projection_statement(), (record_id,))
                row = cur.fetchone()
            conn.commit()
        if row is None:
            return None
        return (int(row[0]), str(row[1]))

    def requeue_canonical_projection(self, domain: str | None = None) -> list[tuple[int, str]]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(self.requeue_canonical_projection_statement(), (domain, domain))
                rows = cur.fetchall()
            conn.commit()
        return [(int(row[0]), str(row[1])) for row in rows]

    def projection_queue_summary(self) -> dict[str, int]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(self.projection_queue_summary_statement())
                rows = cur.fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def health_probe(self) -> dict[str, object]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(self.health_probe_statement())
                row = cur.fetchone()
        if row is None:
            return {"ok": False}
        return {
            "ok": True,
            "database": str(row[0]),
            "user": str(row[1]),
            "vector_extension": bool(row[2]),
            "queue": self.projection_queue_summary(),
        }

    def fetch_record(self, record_id: str) -> KnowledgeRecord | None:
        conn = self.connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(self.fetch_record_statement(), (record_id,))
                row = cur.fetchone()
        if row is None:
            return None
        typed_row = cast(tuple[Any, ...], row)
        return KnowledgeRecord(
            record_id=str(typed_row[0]),
            text=str(typed_row[1]),
            source=str(typed_row[2]),
            domain=str(typed_row[3]),
            confidence=float(typed_row[4]),
            stage=cast(Any, str(typed_row[5])),
            kind=cast(Any, str(typed_row[6])),
            tags=list(cast(list[str], typed_row[7])),
            metadata=dict(cast(dict[str, Any], typed_row[8])),
            created_at=typed_row[9],
            valid_from=typed_row[10],
            valid_to=typed_row[11],
            supersedes=typed_row[12],
            reviewed_at=typed_row[13],
        )

    def semantic_search(
        self,
        query_embedding: list[float],
        *,
        domain: str | None = None,
        top_k: int = 5,
    ) -> list[QueryResult]:
        conn = self.connect()
        with conn:
            self._ensure_vector_registered(conn)
            with conn.cursor() as cur:
                now = self._now_utc()
                vector_query = self._vector_value(query_embedding)
                cur.execute(self.search_statement(), (vector_query, domain, domain, now, now, vector_query, top_k))
                rows = cur.fetchall()
        return [self._row_to_query_result(row) for row in rows]

    @property
    def notification_channel(self) -> str:
        return f"{self.settings.schema}_projection_jobs"

    def connect(self) -> Any:
        with suppress(ModuleNotFoundError):
            psycopg = import_module("psycopg")
            return psycopg.connect(self.settings.dsn)
        raise ModuleNotFoundError("Install context-fabrica[postgres] to use the Postgres adapter")

    def listen_connection(self) -> Any:
        """Open a connection and subscribe to projection job notifications."""
        conn = self.connect()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"LISTEN {self.notification_channel};")
        return conn

    def _ensure_vector_registered(self, conn: Any) -> None:
        register_vector = import_module("pgvector.psycopg").register_vector
        register_vector(conn)

    def _vector_value(self, values: list[float]) -> Any:
        vector_cls = import_module("pgvector").Vector
        return vector_cls(values)

    def _row_to_query_result(self, row: tuple[Any, ...]) -> QueryResult:
        record = KnowledgeRecord(
            record_id=str(row[0]),
            text=str(row[1]),
            source=str(row[2]),
            domain=str(row[3]),
            confidence=float(row[4]),
            stage=cast(Any, str(row[5])),
            kind=cast(Any, str(row[6])),
        )
        return QueryResult(
            record=record,
            score=float(row[8]),
            semantic_score=float(row[8]),
            graph_score=0.0,
            recency_score=0.0,
            confidence_score=float(row[4]),
            rationale=["semantic_match", f"stage:{row[5]}", f"kind:{row[6]}"],
        )

    def _now_utc(self):
        from datetime import datetime, timezone

        return datetime.now(tz=timezone.utc)
