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
