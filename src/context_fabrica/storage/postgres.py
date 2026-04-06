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

    @classmethod
    def from_dsn(
        cls,
        dsn: str,
        *,
        schema: str = "context_fabrica",
        embedding_dimensions: int = 1536,
    ) -> "PostgresPgvectorAdapter":
        return cls(PostgresSettings(
            dsn=dsn,
            schema=schema,
            embedding_dimensions=embedding_dimensions,
        ))

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
            "namespace TEXT NOT NULL DEFAULT 'default', "
            "confidence DOUBLE PRECISION NOT NULL, "
            "memory_stage TEXT NOT NULL DEFAULT 'canonical', "
            "memory_kind TEXT NOT NULL DEFAULT 'fact', "
            "tags JSONB NOT NULL DEFAULT '[]'::jsonb, "
            "metadata JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "created_at TIMESTAMPTZ NOT NULL, "
            "valid_from TIMESTAMPTZ NOT NULL, "
            "valid_to TIMESTAMPTZ NULL, "
            "supersedes TEXT NULL, "
            "reviewed_at TIMESTAMPTZ NULL, "
            "occurred_from TIMESTAMPTZ NULL, "
            "occurred_to TIMESTAMPTZ NULL"
            ");",
            f"ALTER TABLE {schema}.memory_records ADD COLUMN IF NOT EXISTS namespace TEXT NOT NULL DEFAULT 'default';",
            f"ALTER TABLE {schema}.memory_records ADD COLUMN IF NOT EXISTS memory_stage TEXT NOT NULL DEFAULT 'canonical';",
            f"ALTER TABLE {schema}.memory_records ADD COLUMN IF NOT EXISTS memory_kind TEXT NOT NULL DEFAULT 'fact';",
            f"ALTER TABLE {schema}.memory_records ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ NULL;",
            f"ALTER TABLE {schema}.memory_records ADD COLUMN IF NOT EXISTS occurred_from TIMESTAMPTZ NULL;",
            f"ALTER TABLE {schema}.memory_records ADD COLUMN IF NOT EXISTS occurred_to TIMESTAMPTZ NULL;",
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
            # Ensure embedding column matches configured dimensions (handles dimension changes)
            f"DO $$ BEGIN "
            f"IF EXISTS (SELECT 1 FROM information_schema.columns "
            f"WHERE table_schema = '{schema}' AND table_name = 'memory_chunks' AND column_name = 'embedding') "
            f"AND NOT EXISTS (SELECT 1 FROM information_schema.columns "
            f"WHERE table_schema = '{schema}' AND table_name = 'memory_chunks' AND column_name = 'embedding' "
            f"AND udt_name = 'vector' AND character_maximum_length IS NOT DISTINCT FROM {dims}) "
            f"THEN "
            f"DROP INDEX IF EXISTS {schema}.idx_{schema}_chunks_embedding; "
            f"ALTER TABLE {schema}.memory_chunks ALTER COLUMN embedding TYPE vector({dims}); "
            f"END IF; "
            f"END $$;",
            f"CREATE TABLE IF NOT EXISTS {schema}.memory_outcomes ("
            "outcome_id BIGSERIAL PRIMARY KEY, "
            f"record_id TEXT NOT NULL REFERENCES {schema}.memory_records(record_id) ON DELETE CASCADE, "
            "query_text TEXT NOT NULL, "
            "outcome TEXT NOT NULL, "
            "delta DOUBLE PRECISION NOT NULL DEFAULT 0.0, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ");",
            f"CREATE INDEX IF NOT EXISTS idx_{schema}_outcomes_record ON {schema}.memory_outcomes(record_id);",
            f"CREATE INDEX IF NOT EXISTS idx_{schema}_records_namespace ON {schema}.memory_records(namespace);",
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
            "record_id, text_content, source, domain, namespace, confidence, memory_stage, memory_kind, tags, metadata, created_at, valid_from, valid_to, supersedes, reviewed_at, occurred_from, occurred_to"
            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (record_id) DO UPDATE SET "
            "text_content = EXCLUDED.text_content, "
            "source = EXCLUDED.source, "
            "domain = EXCLUDED.domain, "
            "namespace = EXCLUDED.namespace, "
            "confidence = EXCLUDED.confidence, "
            "memory_stage = EXCLUDED.memory_stage, "
            "memory_kind = EXCLUDED.memory_kind, "
            "tags = EXCLUDED.tags, "
            "metadata = EXCLUDED.metadata, "
            "created_at = EXCLUDED.created_at, "
            "valid_from = EXCLUDED.valid_from, "
            "valid_to = EXCLUDED.valid_to, "
            "supersedes = EXCLUDED.supersedes, "
            "reviewed_at = EXCLUDED.reviewed_at, "
            "occurred_from = EXCLUDED.occurred_from, "
            "occurred_to = EXCLUDED.occurred_to;"
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
            f"SELECT record_id, text_content, source, domain, namespace, confidence, memory_stage, memory_kind, tags, metadata, "
            "created_at, valid_from, valid_to, supersedes, reviewed_at, occurred_from, occurred_to "
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
            f"SELECT r.record_id, r.text_content, r.source, r.domain, r.namespace, r.confidence, "
            "r.memory_stage, r.memory_kind, r.tags, r.metadata, "
            "r.created_at, r.valid_from, r.valid_to, r.supersedes, r.reviewed_at, r.occurred_from, r.occurred_to, "
            "1 - (c.embedding <=> %s) AS semantic_score "
            f"FROM {schema}.memory_chunks c "
            f"JOIN {schema}.memory_records r ON r.record_id = c.record_id "
            "WHERE (%s::text IS NULL OR r.domain = %s) "
            "AND (%s::text IS NULL OR r.namespace = %s) "
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
            record.namespace,
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
            record.occurred_from,
            record.occurred_to,
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

    def fetch_record_with_chunks(self, record_id: str) -> tuple[KnowledgeRecord, list[tuple[str, list[float], int]]] | None:
        record = self.fetch_record(record_id)
        if record is None:
            return None
        schema = self.settings.schema
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT chunk_text, embedding, chunk_index FROM {schema}.memory_chunks "
                    "WHERE record_id = %s ORDER BY chunk_index",
                    (record_id,),
                )
                rows = cur.fetchall()
        chunks = [(str(row[0]), list(row[1]) if row[1] else [], int(row[2])) for row in rows]
        return (record, chunks)

    def upsert_records(self, records: list[KnowledgeRecord]) -> None:
        conn = self.connect()
        with conn:
            self._ensure_vector_registered(conn)
            with conn.cursor() as cur:
                stmt = self.upsert_record_statement()
                for record in records:
                    cur.execute(stmt, self.upsert_record_payload(record))
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

    def delete_record(self, record_id: str) -> bool:
        """Delete a record and cascade to chunks, relations, promotions, and projection jobs."""
        schema = self.settings.schema
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {schema}.memory_records WHERE record_id = %s", (record_id,))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def list_records(
        self,
        *,
        domain: str | None = None,
        namespace: str | None = None,
        stage: str | None = None,
        limit: int = 100,
    ) -> list[KnowledgeRecord]:
        schema = self.settings.schema
        query = (
            f"SELECT record_id, text_content, source, domain, namespace, confidence, "
            "memory_stage, memory_kind, tags, metadata, "
            "created_at, valid_from, valid_to, supersedes, reviewed_at, occurred_from, occurred_to "
            f"FROM {schema}.memory_records WHERE 1=1 "
        )
        params: list[object] = []
        if domain is not None:
            query += "AND domain = %s "
            params.append(domain)
        if namespace is not None:
            query += "AND namespace = %s "
            params.append(namespace)
        if stage is not None:
            query += "AND memory_stage = %s "
            params.append(stage)
        query += "ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [self._row_to_record(cast(tuple[Any, ...], row)) for row in rows]

    def _row_to_record(self, row: tuple[Any, ...]) -> KnowledgeRecord:
        return KnowledgeRecord(
            record_id=str(row[0]),
            text=str(row[1]),
            source=str(row[2]),
            domain=str(row[3]),
            namespace=str(row[4]),
            confidence=float(row[5]),
            stage=cast(Any, str(row[6])),
            kind=cast(Any, str(row[7])),
            tags=list(cast(list[str], row[8])),
            metadata=dict(cast(dict[str, Any], row[9])),
            created_at=row[10],
            valid_from=row[11],
            valid_to=row[12],
            supersedes=row[13],
            reviewed_at=row[14],
            occurred_from=row[15],
            occurred_to=row[16],
        )

    def expire_records(self, *, before: datetime) -> int:
        """Soft-expire records created before the given timestamp."""
        schema = self.settings.schema
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {schema}.memory_records SET valid_to = now() "
                    "WHERE valid_to IS NULL AND created_at < %s",
                    (before,),
                )
                count = cur.rowcount
            conn.commit()
        return count

    def decay_confidence(self, *, older_than_days: int, decay_factor: float = 0.95) -> int:
        """Multiply confidence by decay_factor for records older than N days."""
        schema = self.settings.schema
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {schema}.memory_records "
                    "SET confidence = confidence * %s "
                    "WHERE created_at < now() - interval '1 day' * %s "
                    "AND confidence > 0.01 AND valid_to IS NULL",
                    (decay_factor, older_than_days),
                )
                count = cur.rowcount
            conn.commit()
        return count

    def purge_expired(self) -> int:
        """Hard-delete records whose valid_to is in the past."""
        schema = self.settings.schema
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {schema}.memory_records WHERE valid_to IS NOT NULL AND valid_to < now()"
                )
                count = cur.rowcount
            conn.commit()
        return count

    def record_outcome(
        self,
        record_id: str,
        query_text: str,
        outcome: str,
        *,
        delta: float = 0.0,
    ) -> None:
        """Record whether a retrieved record was useful."""
        schema = self.settings.schema
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {schema}.memory_outcomes (record_id, query_text, outcome, delta) "
                    "VALUES (%s, %s, %s, %s)",
                    (record_id, query_text, outcome, delta),
                )
                if delta != 0.0:
                    cur.execute(
                        f"UPDATE {schema}.memory_records "
                        "SET confidence = least(1.0, greatest(0.0, confidence + %s)) "
                        "WHERE record_id = %s",
                        (delta, record_id),
                    )
            conn.commit()

    def outcome_summary(self, record_id: str) -> dict[str, int]:
        """Count outcomes by type for a record."""
        schema = self.settings.schema
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT outcome, count(*) FROM {schema}.memory_outcomes "
                    "WHERE record_id = %s GROUP BY outcome",
                    (record_id,),
                )
                rows = cur.fetchall()
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
        return self._row_to_record(cast(tuple[Any, ...], row))

    def semantic_search(
        self,
        query_embedding: list[float],
        *,
        domain: str | None = None,
        namespace: str | None = None,
        top_k: int = 5,
    ) -> list[QueryResult]:
        conn = self.connect()
        with conn:
            self._ensure_vector_registered(conn)
            with conn.cursor() as cur:
                now = self._now_utc()
                vector_query = self._vector_value(query_embedding)
                cur.execute(self.search_statement(), (vector_query, domain, domain, namespace, namespace, now, now, vector_query, top_k))
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
        record = self._row_to_record(row[:17])
        semantic_score = float(row[17])
        return QueryResult(
            record=record,
            score=semantic_score,
            semantic_score=semantic_score,
            graph_score=0.0,
            recency_score=0.0,
            confidence_score=record.confidence,
            rationale=["semantic_match", f"stage:{record.stage}", f"kind:{record.kind}"],
        )

    def list_all_texts(self, *, namespace: str | None = None) -> list[tuple[str, str]]:
        schema = self.settings.schema
        query = f"SELECT record_id, text_content FROM {schema}.memory_records WHERE valid_to IS NULL"
        params: list[object] = []
        if namespace is not None:
            query += " AND namespace = %s"
            params.append(namespace)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return [(str(row[0]), str(row[1])) for row in cur.fetchall()]

    def list_all_relations(self, *, namespace: str | None = None) -> list[tuple[str, str, str, str, float]]:
        schema = self.settings.schema
        if namespace is not None:
            query = (
                f"SELECT mr.record_id, mr.source_entity, mr.relation_type, mr.target_entity, mr.weight "
                f"FROM {schema}.memory_relations mr "
                f"JOIN {schema}.memory_records r ON r.record_id = mr.record_id "
                f"WHERE r.namespace = %s"
            )
            params: list[object] = [namespace]
        else:
            query = f"SELECT record_id, source_entity, relation_type, target_entity, weight FROM {schema}.memory_relations"
            params = []
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return [(str(r[0]), str(r[1]), str(r[2]), str(r[3]), float(r[4])) for r in cur.fetchall()]

    def _now_utc(self):
        from datetime import datetime, timezone

        return datetime.now(tz=timezone.utc)
