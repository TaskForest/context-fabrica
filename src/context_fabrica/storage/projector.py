from __future__ import annotations

import select
from contextlib import suppress
from dataclasses import dataclass
from threading import Event
from time import sleep
from typing import Any, Protocol, cast

from ..models import KnowledgeRecord

from ..projection import build_graph_projection


class ProjectionPostgres(Protocol):
    def claim_projection_jobs(self, limit: int = 10) -> list[tuple[int, str]]: ...

    def fetch_record(self, record_id: str) -> object | None: ...

    def complete_projection_job(self, job_id: int) -> None: ...

    def fail_projection_job(self, job_id: int, error: str) -> None: ...

    def list_projection_jobs(self, limit: int = 25) -> list[tuple[int, str, str, int, str, object, object]]: ...

    def retry_failed_jobs(self) -> list[tuple[int, str]]: ...

    def requeue_record_projection(self, record_id: str) -> tuple[int, str] | None: ...

    def requeue_canonical_projection(self, domain: str | None = None) -> list[tuple[int, str]]: ...

    def projection_queue_summary(self) -> dict[str, int]: ...

    @property
    def notification_channel(self) -> str: ...

    def listen_connection(self) -> Any: ...


class ProjectionGraph(Protocol):
    def bootstrap(self) -> None: ...

    def project(self, projection, *, domain: str, source: str) -> None: ...


@dataclass(frozen=True)
class ProjectionJobResult:
    job_id: int
    record_id: str
    status: str


class GraphProjectionWorker:
    def __init__(self, postgres: ProjectionPostgres, kuzu: ProjectionGraph) -> None:
        self.postgres = postgres
        self.kuzu = kuzu

    def bootstrap(self) -> None:
        self.kuzu.bootstrap()

    def process_pending(self, limit: int = 10) -> list[ProjectionJobResult]:
        results: list[ProjectionJobResult] = []
        jobs = self.postgres.claim_projection_jobs(limit=limit)
        if not jobs:
            return results

        self.kuzu.bootstrap()
        for job_id, record_id in jobs:
            record = self.postgres.fetch_record(record_id)
            if record is None:
                self.postgres.fail_projection_job(job_id, "record_missing")
                results.append(ProjectionJobResult(job_id=job_id, record_id=record_id, status="failed"))
                continue
            try:
                typed_record = cast(KnowledgeRecord, record)
                projection = build_graph_projection(typed_record)
                self.kuzu.project(projection, domain=typed_record.domain, source=typed_record.source)
                self.postgres.complete_projection_job(job_id)
                results.append(ProjectionJobResult(job_id=job_id, record_id=record_id, status="done"))
            except Exception as exc:  # noqa: BLE001
                self.postgres.fail_projection_job(job_id, str(exc))
                results.append(ProjectionJobResult(job_id=job_id, record_id=record_id, status="failed"))
        return results

    def run_forever(self, *, poll_interval: float = 2.0, batch_size: int = 10, stop_event: Event | None = None) -> None:
        event = stop_event or Event()
        listen_conn: Any | None = None
        try:
            listen_conn = self.postgres.listen_connection()
        except Exception:  # noqa: BLE001
            listen_conn = None

        while not event.is_set():
            results = self.process_pending(limit=batch_size)
            if results:
                continue
            # Wait for NOTIFY or fall back to poll_interval timeout
            if listen_conn is not None:
                try:
                    ready, _, _ = select.select([listen_conn.fileno()], [], [], poll_interval)
                    if ready:
                        # Drain all pending notifications
                        with suppress(Exception):
                            for _notify in listen_conn.notifies():
                                break  # wake up is enough, process_pending handles the work
                except Exception:  # noqa: BLE001
                    # Connection lost — fall back to polling
                    listen_conn = None
                    sleep(poll_interval)
            else:
                sleep(poll_interval)

        if listen_conn is not None:
            with suppress(Exception):
                listen_conn.close()
