from datetime import datetime, timezone

from src.context_fabrica.storage.sqlite import SQLiteRecordStore
from src.context_fabrica.models import KnowledgeRecord
from src.context_fabrica import HybridMemoryStore, HashEmbedder


def test_sqlite_bootstrap_creates_tables(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    tables = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = [t[0] for t in tables]
    assert "memory_records" in names
    assert "memory_chunks" in names
    assert "memory_relations" in names
    assert "memory_promotions" in names
    store.close()


def test_sqlite_upsert_and_fetch(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()

    record = KnowledgeRecord(
        record_id="r1",
        text="AuthService depends on TokenSigner.",
        source="design-doc",
        domain="platform",
        confidence=0.9,
    )
    store.upsert_record(record)
    fetched = store.fetch_record("r1")

    assert fetched is not None
    assert fetched.record_id == "r1"
    assert fetched.text == "AuthService depends on TokenSigner."
    assert fetched.source == "design-doc"
    assert fetched.confidence == 0.9
    store.close()


def test_sqlite_upsert_overwrites_existing(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()

    record = KnowledgeRecord(record_id="r1", text="v1", source="doc", confidence=0.5)
    store.upsert_record(record)
    record.text = "v2"
    record.confidence = 0.9
    store.upsert_record(record)

    fetched = store.fetch_record("r1")
    assert fetched is not None
    assert fetched.text == "v2"
    assert fetched.confidence == 0.9
    store.close()


def test_sqlite_semantic_search(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    embedder = HashEmbedder(dimensions=64)

    r1 = KnowledgeRecord(record_id="r1", text="authentication login tokens", source="doc", domain="platform", confidence=0.8)
    r2 = KnowledgeRecord(record_id="r2", text="billing invoices payments", source="doc", domain="billing", confidence=0.8)

    store.upsert_record(r1)
    store.upsert_record(r2)
    store.replace_chunks("r1", [("authentication login tokens", embedder.embed(r1.text), 0)])
    store.replace_chunks("r2", [("billing invoices payments", embedder.embed(r2.text), 0)])

    query_vec = embedder.embed("authentication login")
    results = store.semantic_search(query_vec, top_k=2)

    assert len(results) >= 1
    assert results[0].record.record_id == "r1"
    store.close()


def test_sqlite_semantic_search_excludes_staged(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    embedder = HashEmbedder(dimensions=64)

    r1 = KnowledgeRecord(record_id="r1", text="staged draft note", source="doc", stage="staged", confidence=0.3)
    store.upsert_record(r1)
    store.replace_chunks("r1", [("staged draft note", embedder.embed(r1.text), 0)])

    results = store.semantic_search(embedder.embed("draft note"), top_k=5)
    assert all(r.record.record_id != "r1" for r in results)
    store.close()


def test_sqlite_semantic_search_filters_by_domain(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    embedder = HashEmbedder(dimensions=64)

    r1 = KnowledgeRecord(record_id="r1", text="auth service", source="doc", domain="platform", confidence=0.8)
    r2 = KnowledgeRecord(record_id="r2", text="auth gateway", source="doc", domain="infra", confidence=0.8)
    store.upsert_record(r1)
    store.upsert_record(r2)
    store.replace_chunks("r1", [("auth service", embedder.embed(r1.text), 0)])
    store.replace_chunks("r2", [("auth gateway", embedder.embed(r2.text), 0)])

    results = store.semantic_search(embedder.embed("auth"), domain="platform", top_k=5)
    assert all(r.record.domain == "platform" for r in results)
    store.close()


def test_sqlite_replace_relations(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()

    record = KnowledgeRecord(record_id="r1", text="test", source="doc", confidence=0.8)
    store.upsert_record(record)
    store.replace_relations("r1", [
        ("r1", "auth_service", "DEPENDS_ON", "token_signer", 1.0),
        ("r1", "auth_service", "CALLS", "key_store", 1.0),
    ])

    rows = store.conn.execute("SELECT * FROM memory_relations WHERE record_id = 'r1'").fetchall()
    assert len(rows) == 2
    store.close()


def test_sqlite_promotion(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()

    record = KnowledgeRecord(record_id="r1", text="draft", source="doc", stage="staged", confidence=0.4)
    store.upsert_record(record)

    from datetime import datetime, timezone
    store.record_promotion("r1", "r1", "reviewed", datetime.now(tz=timezone.utc))

    rows = store.conn.execute("SELECT * FROM memory_promotions").fetchall()
    assert len(rows) == 1
    store.close()


def test_sqlite_end_to_end_via_hybrid_store(tmp_path) -> None:
    db_path = str(tmp_path / "e2e.db")
    embedder = HashEmbedder(dimensions=64)
    store = HybridMemoryStore(store=SQLiteRecordStore(db_path), embedder=embedder)
    store.bootstrap()

    record = KnowledgeRecord(
        record_id="e2e-1",
        text="AuthService depends on TokenSigner and calls KeyStore.",
        source="design-doc",
        domain="platform",
        confidence=0.9,
    )

    store.write_text(record)

    query_vec = embedder.embed("AuthService TokenSigner")
    results = store.semantic_search(query_vec, domain="platform", top_k=3)
    assert len(results) >= 1
    assert results[0].record.record_id == "e2e-1"


def test_sqlite_delete_record_cascades(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    embedder = HashEmbedder(dimensions=64)

    record = KnowledgeRecord(record_id="r1", text="to delete", source="doc", confidence=0.8)
    store.upsert_record(record)
    store.replace_chunks("r1", [("to delete", embedder.embed("to delete"), 0)])
    store.replace_relations("r1", [("r1", "a", "DEPENDS_ON", "b", 1.0)])

    assert store.delete_record("r1")
    assert store.fetch_record("r1") is None
    assert store.conn.execute("SELECT count(*) FROM memory_chunks WHERE record_id='r1'").fetchone()[0] == 0
    assert store.conn.execute("SELECT count(*) FROM memory_relations WHERE record_id='r1'").fetchone()[0] == 0
    assert not store.delete_record("nonexistent")
    store.close()


def test_sqlite_list_records(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()

    store.upsert_record(KnowledgeRecord(record_id="r1", text="a", source="doc", domain="platform", confidence=0.8))
    store.upsert_record(KnowledgeRecord(record_id="r2", text="b", source="doc", domain="platform", stage="staged", confidence=0.4))
    store.upsert_record(KnowledgeRecord(record_id="r3", text="c", source="doc", domain="billing", confidence=0.8))

    all_records = store.list_records()
    assert len(all_records) == 3

    platform = store.list_records(domain="platform")
    assert len(platform) == 2
    assert all(r.domain == "platform" for r in platform)

    staged = store.list_records(stage="staged")
    assert len(staged) == 1
    assert staged[0].record_id == "r2"

    limited = store.list_records(limit=1)
    assert len(limited) == 1
    store.close()


def test_sqlite_batch_upsert(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()

    records = [
        KnowledgeRecord(record_id=f"batch-{i}", text=f"record {i}", source="bulk", confidence=0.8)
        for i in range(5)
    ]
    store.upsert_records(records)

    all_records = store.list_records()
    assert len(all_records) == 5
    store.close()


def test_sqlite_fetch_record_with_chunks(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    embedder = HashEmbedder(dimensions=64)

    record = KnowledgeRecord(record_id="r1", text="test with chunks", source="doc", confidence=0.8)
    store.upsert_record(record)
    emb = embedder.embed("test with chunks")
    store.replace_chunks("r1", [("chunk 0", emb, 0), ("chunk 1", emb, 1)])

    result = store.fetch_record_with_chunks("r1")
    assert result is not None
    rec, chunks = result
    assert rec.record_id == "r1"
    assert len(chunks) == 2
    assert chunks[0][0] == "chunk 0"
    assert chunks[1][2] == 1  # chunk_index

    assert store.fetch_record_with_chunks("nonexistent") is None
    store.close()


def test_sqlite_semantic_search_returns_complete_records(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    embedder = HashEmbedder(dimensions=64)

    record = KnowledgeRecord(
        record_id="r1",
        text="auth service",
        source="design-doc",
        domain="platform",
        confidence=0.9,
        tags=["auth", "critical"],
        metadata={"team": "platform", "reviewed": True},
    )
    store.upsert_record(record)
    store.replace_chunks("r1", [("auth service", embedder.embed(record.text), 0)])

    results = store.semantic_search(embedder.embed("auth"), top_k=1)
    assert len(results) == 1
    hit = results[0].record
    assert hit.tags == ["auth", "critical"]
    assert hit.metadata["team"] == "platform"
    assert hit.source == "design-doc"
    store.close()


def test_sqlite_round_trips_occurrence_fields(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    occurred_from = datetime(2025, 6, 1, tzinfo=timezone.utc)
    occurred_to = datetime(2025, 6, 30, tzinfo=timezone.utc)

    record = KnowledgeRecord(
        record_id="r1",
        text="June incident review",
        source="doc",
        confidence=0.8,
        occurred_from=occurred_from,
        occurred_to=occurred_to,
    )
    store.upsert_record(record)
    fetched = store.fetch_record("r1")

    assert fetched is not None
    assert fetched.occurred_from == occurred_from
    assert fetched.occurred_to == occurred_to
    store.close()


def test_sqlite_list_all_texts(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    r1 = KnowledgeRecord(record_id="r1", text="auth service", source="doc", confidence=0.8)
    r2 = KnowledgeRecord(record_id="r2", text="billing service", source="doc", confidence=0.8)
    store.upsert_record(r1)
    store.upsert_record(r2)

    texts = store.list_all_texts()
    assert len(texts) == 2
    ids = {rid for rid, _ in texts}
    assert ids == {"r1", "r2"}
    store.close()


def test_sqlite_list_all_texts_excludes_invalidated(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    r1 = KnowledgeRecord(record_id="r1", text="valid fact", source="doc", confidence=0.8)
    r2 = KnowledgeRecord(
        record_id="r2", text="old fact", source="doc", confidence=0.8,
        valid_to=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    store.upsert_record(r1)
    store.upsert_record(r2)

    texts = store.list_all_texts()
    assert len(texts) == 1
    assert texts[0][0] == "r1"
    store.close()


def test_sqlite_list_all_relations(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    r1 = KnowledgeRecord(record_id="r1", text="auth service", source="doc", confidence=0.8)
    store.upsert_record(r1)
    store.replace_relations("r1", [
        ("r1", "AuthService", "DEPENDS_ON", "TokenSigner", 1.0),
    ])

    relations = store.list_all_relations()
    assert len(relations) == 1
    assert relations[0][1] == "AuthService"
    assert relations[0][3] == "TokenSigner"
    store.close()


def test_sqlite_namespace_isolation(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()
    embedder = HashEmbedder(dimensions=64)

    r1 = KnowledgeRecord(record_id="r1", text="team alpha auth", source="doc", domain="platform", namespace="alpha", confidence=0.8)
    r2 = KnowledgeRecord(record_id="r2", text="team beta auth", source="doc", domain="platform", namespace="beta", confidence=0.8)
    store.upsert_record(r1)
    store.upsert_record(r2)
    store.replace_chunks("r1", [("team alpha auth", embedder.embed(r1.text), 0)])
    store.replace_chunks("r2", [("team beta auth", embedder.embed(r2.text), 0)])

    alpha = store.list_records(namespace="alpha")
    assert len(alpha) == 1
    assert alpha[0].namespace == "alpha"

    results = store.semantic_search(embedder.embed("auth"), namespace="beta", top_k=5)
    assert all(r.record.namespace == "beta" for r in results)
    store.close()


def test_sqlite_lifecycle_expire_and_purge(tmp_path) -> None:
    from datetime import timedelta
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()

    old_time = datetime.now(tz=timezone.utc) - timedelta(days=60)
    r1 = KnowledgeRecord(record_id="old", text="old record", source="doc", confidence=0.8)
    r1.created_at = old_time
    r2 = KnowledgeRecord(record_id="new", text="new record", source="doc", confidence=0.8)
    store.upsert_record(r1)
    store.upsert_record(r2)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=30)
    expired = store.expire_records(before=cutoff)
    assert expired == 1

    old = store.fetch_record("old")
    assert old is not None
    assert old.valid_to is not None

    purged = store.purge_expired()
    assert purged == 1
    assert store.fetch_record("old") is None
    assert store.fetch_record("new") is not None
    store.close()


def test_sqlite_supersession_chain(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()

    r1 = KnowledgeRecord(record_id="v1", text="timeout is 60s", source="doc", confidence=0.8)
    r2 = KnowledgeRecord(record_id="v2", text="timeout is 45s", source="doc", confidence=0.8, supersedes="v1")
    r3 = KnowledgeRecord(record_id="v3", text="timeout is 30s", source="doc", confidence=0.8, supersedes="v2")
    store.upsert_record(r1)
    store.upsert_record(r2)
    store.upsert_record(r3)

    chain = store.supersession_chain("v3")
    assert [r.record_id for r in chain] == ["v3", "v2", "v1"]
    store.close()


def test_sqlite_record_outcome(tmp_path) -> None:
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()

    record = KnowledgeRecord(record_id="r1", text="auth info", source="doc", confidence=0.5)
    store.upsert_record(record)

    store.record_outcome("r1", "how does auth work?", "useful", delta=0.1)
    store.record_outcome("r1", "what is the login flow?", "useful", delta=0.05)
    store.record_outcome("r1", "billing info?", "not_useful", delta=-0.1)

    summary = store.outcome_summary("r1")
    assert summary["useful"] == 2
    assert summary["not_useful"] == 1

    updated = store.fetch_record("r1")
    assert updated is not None
    assert abs(updated.confidence - 0.55) < 0.01  # 0.5 + 0.1 + 0.05 - 0.1
    store.close()


def test_sqlite_lifecycle_decay_confidence(tmp_path) -> None:
    from datetime import timedelta
    store = SQLiteRecordStore(str(tmp_path / "test.db"))
    store.bootstrap()

    old_time = datetime.now(tz=timezone.utc) - timedelta(days=60)
    r1 = KnowledgeRecord(record_id="old", text="old record", source="doc", confidence=0.8)
    r1.created_at = old_time
    store.upsert_record(r1)

    decayed = store.decay_confidence(older_than_days=30, decay_factor=0.5)
    assert decayed == 1

    record = store.fetch_record("old")
    assert record is not None
    assert abs(record.confidence - 0.4) < 0.01
    store.close()
