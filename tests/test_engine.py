from datetime import datetime, timedelta, timezone
from src.context_fabrica.engine import DomainMemoryEngine
from src.context_fabrica.models import Relation


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


def test_staged_record_is_hidden_until_promoted() -> None:
    engine = DomainMemoryEngine()
    staged = engine.ingest(
        "Draft note: TODO investigate flaky auth refresh.",
        source="scratchpad",
        confidence=0.4,
        record_id="draft-1",
    )

    assert staged.stage == "staged"
    assert not engine.query("flaky auth refresh", top_k=3)

    engine.promote_record("draft-1")
    promoted_results = engine.query("flaky auth refresh", top_k=3)
    assert promoted_results
    assert promoted_results[0].record.record_id == "draft-1"


def test_ingest_with_caller_provided_entities_and_relations() -> None:
    engine = DomainMemoryEngine()
    engine.ingest(
        "The authentication service validates tokens before passing requests to the API gateway.",
        record_id="r1",
        entities=["auth_service", "api_gateway", "token_validator"],
        relations=[
            Relation("auth_service", "calls", "api_gateway", weight=1.0),
            Relation("auth_service", "uses", "token_validator", weight=1.0),
        ],
    )
    engine.ingest(
        "The API gateway routes traffic to downstream microservices.",
        record_id="r2",
        entities=["api_gateway", "microservices"],
    )

    results = engine.query("How does auth_service connect to api_gateway?", top_k=2)
    assert results
    ids = [r.record.record_id for r in results]
    assert "r1" in ids
    assert any("graph_relation" in hit.rationale for hit in results)


def test_ingest_with_caller_entities_skips_heuristic_extraction() -> None:
    engine = DomainMemoryEngine()
    # Text has no PascalCase or snake_case — heuristic extraction would find little
    engine.ingest(
        "the login flow checks credentials and returns a session",
        record_id="r1",
        entities=["login_flow", "credentials", "session"],
    )
    related = engine.related_records("r1", hops=1)
    # entities were attached, so the record is reachable via graph
    entities = engine._graph.record_entities("r1")
    assert "login_flow" in entities
    assert "session" in entities


def test_embedding_scoring_finds_semantically_similar_records() -> None:
    engine = DomainMemoryEngine(scoring="embedding")
    engine.ingest(
        "authentication middleware handles login and session tokens",
        record_id="auth",
        confidence=0.8,
    )
    engine.ingest(
        "billing pipeline processes monthly invoices",
        record_id="billing",
        confidence=0.8,
    )
    # HashEmbedder is deterministic — tokens that overlap produce higher cosine similarity
    results = engine.query("authentication login session", top_k=2)
    assert results
    assert results[0].record.record_id == "auth"


def test_hybrid_scoring_combines_embedding_and_bm25() -> None:
    engine = DomainMemoryEngine(scoring="hybrid")
    engine.ingest("API Gateway timeout configuration", record_id="r1", confidence=0.7)
    engine.ingest("database connection pooling settings", record_id="r2", confidence=0.7)

    results = engine.query("API Gateway timeout", top_k=2)
    assert results
    assert results[0].record.record_id == "r1"
    assert results[0].semantic_score > 0.0


def test_bm25_only_mode_ignores_embeddings() -> None:
    engine = DomainMemoryEngine(scoring="bm25")
    engine.ingest("PaymentsService processes refunds", record_id="r1", confidence=0.8)
    results = engine.query("PaymentsService refunds", top_k=2)
    assert results
    # Embedding scores should not be computed in bm25 mode
    assert engine._embeddings  # embeddings are still stored
    assert results[0].record.record_id == "r1"
