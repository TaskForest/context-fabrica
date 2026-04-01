from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from time import sleep
from typing import Protocol, cast

from ..models import KnowledgeRecord

from ..projection import build_graph_projection


class ProjectionPostgres(Protocol):
    def claim_projection_jobs(self, limit: int = 10) -> list[tuple[int, str]]: ...

    def fetch_record(self, record_id: str) -> object | None: ...

    def complete_projection_job(self, job_id: int) -> None: ...

    def fail_projection_job(self, job_id: int, error: str) -> None: ...


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
        while not event.is_set():
            results = self.process_pending(limit=batch_size)
            if results:
                continue
            sleep(poll_interval)
