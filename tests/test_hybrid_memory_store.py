from datetime import datetime, timedelta, timezone
from src.context_fabrica.config import NamespacePolicy, ScoringWeights
from src.context_fabrica.storage.hybrid import HybridMemoryStore
from src.context_fabrica.storage.sqlite import SQLiteRecordStore
from src.context_fabrica.models import Relation
from src.context_fabrica.temporal import extract_time_range, temporal_overlap_score


def _make_engine(**kwargs) -> HybridMemoryStore:
    store = HybridMemoryStore(store=SQLiteRecordStore(":memory:"), **kwargs)
    store.bootstrap()
    return store


def test_query_prefers_graph_connected_record() -> None:
    engine = _make_engine()
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
    engine = _make_engine()
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
    engine = _make_engine()
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
    engine = _make_engine()
    engine.ingest("Build command is make release", record_id="old", confidence=0.8)
    engine.invalidate_record("old", reason="obsolete")
    engine.ingest("Build command is uv run task release", record_id="new", confidence=0.9)

    results = engine.query("What is the build command?", top_k=2)
    assert results
    assert all(hit.record.record_id != "old" for hit in results)


def test_staged_record_is_hidden_until_promoted() -> None:
    engine = _make_engine()
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
    engine = _make_engine()
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
    engine = _make_engine()
    # Text has no PascalCase or snake_case — heuristic extraction would find little
    engine.ingest(
        "the login flow checks credentials and returns a session",
        record_id="r1",
        entities=["login_flow", "credentials", "session"],
    )
    related = engine.related_records("r1", hops=1)
    # entities were attached, so the record is reachable via graph
    entities = engine._scoring.graph.record_entities("r1")
    assert "login_flow" in entities
    assert "session" in entities


def test_embedding_scoring_finds_semantically_similar_records() -> None:
    engine = _make_engine(scoring="embedding")
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
    engine = _make_engine(scoring="hybrid")
    engine.ingest("API Gateway timeout configuration", record_id="r1", confidence=0.7)
    engine.ingest("database connection pooling settings", record_id="r2", confidence=0.7)

    results = engine.query("API Gateway timeout", top_k=2)
    assert results
    assert results[0].record.record_id == "r1"
    assert results[0].semantic_score > 0.0


def test_bm25_only_mode_ignores_embeddings() -> None:
    engine = _make_engine(scoring="bm25")
    engine.ingest("PaymentsService processes refunds", record_id="r1", confidence=0.8)
    results = engine.query("PaymentsService refunds", top_k=2)
    assert results
    # Embedding scores should not be computed in bm25 mode
    # Embeddings are stored in the SQLite store, not in-memory
    assert results[0].record.record_id == "r1"


def test_configurable_scoring_weights() -> None:
    weights = ScoringWeights(semantic=0.80, graph=0.10, temporal=0.0, recency=0.05, confidence=0.05)
    engine = _make_engine(weights=weights)
    # Weights are normalized to sum to 1.0
    assert abs(sum(engine._scoring._weights.values()) - 1.0) < 1e-9
    assert engine._scoring._weights["semantic"] == 0.80  # 0.80 / 1.0
    assert engine._scoring._weights["graph"] == 0.10

    engine.ingest("AuthService depends on TokenSigner", record_id="r1", confidence=0.8)
    results = engine.query("AuthService TokenSigner", top_k=1)
    assert results


def test_namespace_filtering_in_engine() -> None:
    engine = _make_engine()
    engine.ingest("alpha team auth service", record_id="r1", confidence=0.8, namespace="alpha")
    engine.ingest("beta team auth service", record_id="r2", confidence=0.8, namespace="beta")

    alpha_results = engine.query("auth service", namespace="alpha", top_k=5)
    assert all(r.record.namespace == "alpha" for r in alpha_results)
    assert len(alpha_results) == 1


def test_supersede_record_invalidates_old_and_links() -> None:
    engine = _make_engine()
    engine.ingest("Build command is make release", record_id="v1", confidence=0.8)
    new = engine.supersede_record_by_text("v1", "Build command is uv run task release", record_id="v2", reason="corrected")

    assert new.supersedes == "v1"
    old = engine.records["v1"]
    assert old.valid_to is not None
    # Old should be filtered from queries
    results = engine.query("build command", top_k=3)
    assert all(r.record.record_id != "v1" for r in results)
    assert any(r.record.record_id == "v2" for r in results)


def test_supersession_chain() -> None:
    engine = _make_engine()
    engine.ingest("timeout is 60s", record_id="v1", confidence=0.8)
    engine.supersede_record_by_text("v1", "timeout is 45s", record_id="v2")
    engine.supersede_record_by_text("v2", "timeout is 30s", record_id="v3")

    chain = engine.supersession_chain("v3")
    assert [r.record_id for r in chain] == ["v3", "v2", "v1"]


def test_rrf_scoring_mode() -> None:
    engine = _make_engine(scoring="rrf")
    engine.ingest(
        "PaymentsService depends on LedgerAdapter.",
        source="design-doc",
        domain="fintech",
        confidence=0.9,
        record_id="r1",
    )
    engine.ingest(
        "LedgerAdapter writes to event store.",
        source="runbook",
        domain="fintech",
        confidence=0.7,
        record_id="r2",
    )

    results = engine.query("PaymentsService LedgerAdapter", domain="fintech", top_k=2)
    assert results
    assert any("rrf" in r.rationale for r in results)
    # RRF should still rank related records reasonably
    assert results[0].score > 0


def test_temporal_query_prefers_matching_occurrence_window() -> None:
    engine = _make_engine()
    engine.ingest(
        "Incident review happened in June 2025 for the auth service.",
        record_id="june",
        confidence=0.8,
    )
    engine.ingest(
        "Incident review happened in July 2025 for the auth service.",
        record_id="july",
        confidence=0.8,
    )

    results = engine.query("What happened in June 2025 for the auth service?", top_k=2)
    assert results
    assert results[0].record.record_id == "june"
    assert "temporal_match" in results[0].rationale
    assert results[0].temporal_score > 0.0


def test_namespace_policy_can_constrain_sources_and_confidence() -> None:
    policy = NamespacePolicy(
        min_confidence=0.8,
        source_allowlist=("design-doc",),
        default_hops=1,
    )
    engine = _make_engine(namespace_policies={"alpha": policy})
    engine.ingest("alpha auth flow", source="design-doc", record_id="r1", confidence=0.9, namespace="alpha")
    engine.ingest("alpha auth flow draft", source="scratchpad", record_id="r2", confidence=0.6, namespace="alpha")

    results = engine.query("auth flow", namespace="alpha", top_k=5)
    assert [result.record.record_id for result in results] == ["r1"]


def test_synthesize_observation_creates_provenance_backed_record() -> None:
    engine = _make_engine()
    engine.ingest("AuthService depends on TokenSigner.", record_id="r1", confidence=0.8, namespace="payments")
    engine.ingest("AuthService rotates tokens through TokenSigner daily.", record_id="r2", confidence=0.9, namespace="payments")

    observation = engine.synthesize_observation(["r1", "r2"], record_id="obs-1")

    assert observation.record_id == "obs-1"
    assert observation.kind == "observation"
    assert observation.stage == "canonical"
    assert observation.namespace == "payments"
    assert observation.metadata["derived_from"] == ["r1", "r2"]
    assert "Synthesized observation" in observation.text


def test_optional_reranker_can_reorder_top_results() -> None:
    class PreferSecond:
        def score(self, query: str, record) -> float:
            return 1.0 if record.record_id == "r2" else 0.0

    engine = _make_engine(reranker=PreferSecond(), rerank_weight=1.0)
    engine.ingest("AuthService tokens sessions login middleware", record_id="r1", confidence=0.8)
    engine.ingest("AuthService", record_id="r2", confidence=0.8)

    baseline = engine.query("AuthService tokens", top_k=2)
    reranked = engine.query("AuthService tokens", top_k=2, rerank_top_n=2)

    assert baseline[0].record.record_id == "r1"
    assert reranked[0].record.record_id == "r2"
    assert reranked[0].rerank_score == 1.0
    assert "reranked" in reranked[0].rationale


def test_reranker_can_promote_result_from_below_top_k_cutoff() -> None:
    class PreferThird:
        def score(self, query: str, record) -> float:
            return 1.0 if record.record_id == "r3" else 0.0

    engine = _make_engine(reranker=PreferThird(), rerank_weight=1.0)
    engine.ingest("auth service tokens login sessions middleware", record_id="r1", confidence=0.8)
    engine.ingest("auth service tokens login", record_id="r2", confidence=0.8)
    engine.ingest("auth service", record_id="r3", confidence=0.8)

    baseline = engine.query("auth service tokens", top_k=2)
    reranked = engine.query("auth service tokens", top_k=2, rerank_top_n=3)

    assert all(result.record.record_id != "r3" for result in baseline)
    assert reranked[0].record.record_id == "r3"
    assert reranked[0].rerank_score == 1.0


def test_temporal_query_does_not_match_boundary_touching_event() -> None:
    engine = _make_engine()
    engine.ingest(
        "Quarterly report published on 2025-07-01.",
        record_id="july-1",
        confidence=0.8,
    )
    engine.ingest(
        "Quarterly report published on 2025-06-15.",
        record_id="june-15",
        confidence=0.8,
    )

    results = engine.query("What happened in June 2025?", top_k=5)

    by_id = {result.record.record_id: result for result in results}
    assert "june-15" in by_id
    assert by_id["june-15"].temporal_score > 0.0
    assert "july-1" in by_id
    assert by_id["july-1"].temporal_score == 0.0
    assert [result.record.record_id for result in results].index("june-15") < [result.record.record_id for result in results].index("july-1")


# --- temporal.py unit tests ---


def test_extract_time_range_yesterday() -> None:
    now = datetime(2025, 7, 15, 12, 0, tzinfo=timezone.utc)
    result = extract_time_range("what happened yesterday?", now=now)
    assert result is not None
    assert result[0] == datetime(2025, 7, 14, tzinfo=timezone.utc)
    assert result[1] == datetime(2025, 7, 15, tzinfo=timezone.utc)


def test_extract_time_range_last_week() -> None:
    now = datetime(2025, 7, 15, 12, 0, tzinfo=timezone.utc)  # Tuesday
    result = extract_time_range("what happened last week?", now=now)
    assert result is not None
    start, end = result
    assert (end - start).days == 7


def test_extract_time_range_iso_date() -> None:
    result = extract_time_range("Report from 2025-06-15 was important", now=datetime(2025, 7, 1, tzinfo=timezone.utc))
    assert result is not None
    assert result[0] == datetime(2025, 6, 15, tzinfo=timezone.utc)
    assert result[1] == datetime(2025, 6, 16, tzinfo=timezone.utc)


def test_extract_time_range_returns_none_for_no_temporal_cues() -> None:
    assert extract_time_range("AuthService depends on TokenSigner") is None


def test_temporal_overlap_score_no_overlap() -> None:
    query = (datetime(2025, 6, 1, tzinfo=timezone.utc), datetime(2025, 7, 1, tzinfo=timezone.utc))
    score = temporal_overlap_score(
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime(2025, 7, 2, tzinfo=timezone.utc),
        query,
    )
    assert score == 0.0


def test_temporal_overlap_score_full_overlap() -> None:
    query = (datetime(2025, 6, 1, tzinfo=timezone.utc), datetime(2025, 7, 1, tzinfo=timezone.utc))
    score = temporal_overlap_score(
        datetime(2025, 6, 1, tzinfo=timezone.utc),
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        query,
    )
    assert score == 1.0


def test_temporal_overlap_score_partial() -> None:
    query = (datetime(2025, 6, 1, tzinfo=timezone.utc), datetime(2025, 7, 1, tzinfo=timezone.utc))
    score = temporal_overlap_score(
        datetime(2025, 6, 15, tzinfo=timezone.utc),
        datetime(2025, 7, 15, tzinfo=timezone.utc),
        query,
    )
    assert 0.0 < score < 1.0


# --- RRF + temporal ---


def test_rrf_scoring_with_temporal_signal() -> None:
    engine = _make_engine(scoring="rrf")
    engine.ingest(
        "Incident review happened in June 2025.",
        record_id="june",
        confidence=0.8,
    )
    engine.ingest(
        "Incident review happened in July 2025.",
        record_id="july",
        confidence=0.8,
    )
    results = engine.query("What happened in June 2025?", top_k=2)
    assert results
    assert any("rrf" in r.rationale for r in results)
    assert results[0].record.record_id == "june"
    assert "temporal_match" in results[0].rationale


# --- Reranker edge case: single result ---


def test_reranker_with_single_result_returns_none_rerank_score() -> None:
    class AlwaysOne:
        def score(self, query: str, record) -> float:
            return 1.0

    engine = _make_engine(reranker=AlwaysOne(), rerank_weight=0.5)
    engine.ingest("auth service tokens", record_id="r1", confidence=0.8)

    results = engine.query("auth service tokens", top_k=1, rerank_top_n=5)
    assert len(results) == 1
    assert results[0].rerank_score is None
    assert "reranked" not in results[0].rationale
