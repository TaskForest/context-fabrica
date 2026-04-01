from src.context_fabrica.storage.projector import GraphProjectionWorker


class _FakePostgres:
    def __init__(self) -> None:
        self.completed = []
        self.failed = []

    def claim_projection_jobs(self, limit: int = 10):
        return [(1, "r1")]

    def fetch_record(self, record_id: str):
        from src.context_fabrica.models import KnowledgeRecord

        return KnowledgeRecord(
            record_id=record_id,
            text="AuthService depends on TokenSigner.",
            source="design-doc",
            domain="platform",
            confidence=0.9,
        )

    def complete_projection_job(self, job_id: int) -> None:
        self.completed.append(job_id)

    def fail_projection_job(self, job_id: int, error: str) -> None:
        self.failed.append((job_id, error))


class _FakeKuzu:
    def __init__(self) -> None:
        self.projected = []
        self.bootstrapped = False

    def bootstrap(self) -> None:
        self.bootstrapped = True

    def project(self, projection, *, domain: str, source: str) -> None:
        self.projected.append((projection.record_id, domain, source))


def test_projector_processes_pending_jobs() -> None:
    worker = GraphProjectionWorker(_FakePostgres(), _FakeKuzu())
    results = worker.process_pending(limit=5)
    assert results[0].status == "done"


def test_projector_run_forever_returns_after_stop_event() -> None:
    from threading import Event

    worker = GraphProjectionWorker(_FakePostgres(), _FakeKuzu())
    stop = Event()
    stop.set()
    worker.run_forever(stop_event=stop)
