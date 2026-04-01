from datetime import datetime, timedelta, timezone
from src.context_fabrica.engine import DomainMemoryEngine


def test_query_prefers_graph_connected_record() -> None:
    engine = DomainMemoryEngine()
    engine.ingest(
        "PaymentsService depends on LedgerAdapter and calls RiskGateway.",
        source="design-doc",
        domain="fintech",
        confidence=0.8,
        record_id="r1",
    )
    engine.ingest(
        "LedgerAdapter writes transactions to event store.",
        source="runbook",
        domain="fintech",
        confidence=0.7,
        record_id="r2",
    )
    engine.ingest(
        "Frontend styling notes for dashboard color tokens.",
        source="wiki",
        domain="web",
        confidence=0.9,
        record_id="r3",
    )

    results = engine.query("How does PaymentsService interact with LedgerAdapter?", top_k=2, domain="fintech")
    assert results
    assert results[0].record.record_id in {"r1", "r2"}
    assert any("graph_relation" in hit.rationale for hit in results)


def test_recency_affects_ranking_when_semantic_is_similar() -> None:
    engine = DomainMemoryEngine()
    old_record = engine.ingest(
        "API Gateway timeout is 60 seconds for partner integrations.",
        source="old-doc",
        confidence=0.7,
        record_id="old",
    )
    new_record = engine.ingest(
        "API Gateway timeout is 45 seconds for partner integrations.",
        source="new-doc",
        confidence=0.7,
        record_id="new",
    )

    old_record.created_at = datetime.now(tz=timezone.utc) - timedelta(days=15)
    new_record.created_at = datetime.now(tz=timezone.utc) - timedelta(hours=3)

    results = engine.query("What is API Gateway timeout for partner integrations?", top_k=2)
    assert results[0].record.record_id == "new"


def test_related_records_returns_graph_neighbors() -> None:
    engine = DomainMemoryEngine()
    engine.ingest(
        "AuthService uses TokenSigner and depends on KeyStore.",
        record_id="a",
    )
    engine.ingest(
        "TokenSigner rotates keys from KeyStore daily.",
        record_id="b",
    )
    engine.ingest(
        "Billing service queue settings.",
        record_id="c",
    )

    related = engine.related_records("a", hops=2, top_k=3)
    assert any(item.record_id == "b" for item in related)
    assert all(item.record_id != "a" for item in related)


def test_invalidated_record_is_filtered_from_current_queries() -> None:
    engine = DomainMemoryEngine()
    engine.ingest("Build command is make release", record_id="old", confidence=0.8)
    engine.invalidate_record("old", reason="obsolete")
    engine.ingest("Build command is uv run task release", record_id="new", confidence=0.9)

    results = engine.query("What is the build command?", top_k=2)
    assert results
    assert all(hit.record.record_id != "old" for hit in results)
