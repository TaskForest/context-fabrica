from src.context_fabrica import HybridMemoryStore, HybridStoreSettings, KnowledgeRecord, KuzuSettings, PostgresSettings
from src.context_fabrica.storage.sqlite import SQLiteRecordStore
from src.context_fabrica.storage.postgres import PostgresPgvectorAdapter
from src.context_fabrica.storage.kuzu import KuzuGraphProjectionAdapter


def test_postgres_bootstrap_includes_pgvector_schema() -> None:
    adapter = PostgresPgvectorAdapter(PostgresSettings(dsn="postgresql://localhost/test"))
    bootstrap = adapter.bootstrap_statements()
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in bootstrap)
    assert any("memory_records" in s for s in bootstrap)
    assert any("memory_chunks" in s for s in bootstrap)
    assert any("memory_stage" in s for s in bootstrap)


def test_kuzu_bootstrap_includes_record_and_entity_tables() -> None:
    adapter = KuzuGraphProjectionAdapter(KuzuSettings(path="./tmp-kuzu"))
    bootstrap = adapter.bootstrap_statements()
    assert any("CREATE NODE TABLE IF NOT EXISTS MemoryRecord" in s for s in bootstrap)
    assert any("CREATE NODE TABLE IF NOT EXISTS Entity" in s for s in bootstrap)
    assert any("CREATE REL TABLE IF NOT EXISTS RELATED" in s for s in bootstrap)


def test_write_plan_generates_graph_projection() -> None:
    store = HybridMemoryStore(
        HybridStoreSettings(
            postgres=PostgresSettings(dsn="postgresql://localhost/test"),
            kuzu=KuzuSettings(path="./tmp-kuzu"),
        )
    )
    record = KnowledgeRecord(
        record_id="r1",
        text="AuthService depends on TokenSigner and calls KeyStore.",
        source="design-doc",
        domain="platform",
        confidence=0.85,
    )

    plan = store.write_plan(record)

    assert plan.record_id == "r1"
    assert plan.graph_projection.record_id == "r1"
    assert "authservice" in plan.graph_projection.entities


def test_postgres_search_statement_filters_by_domain_and_validity() -> None:
    adapter = PostgresPgvectorAdapter(PostgresSettings(dsn="postgresql://localhost/test"))
    statement = adapter.search_statement()
    assert "r.domain = %s" in statement
    assert "r.valid_from <= %s" in statement
    assert "r.valid_to IS NULL OR r.valid_to >= %s" in statement
    assert "r.memory_stage <> 'staged'" in statement


def test_write_record_calls_store_write_methods(mocker) -> None:
    store = HybridMemoryStore(
        HybridStoreSettings(
            postgres=PostgresSettings(dsn="postgresql://localhost/test"),
            kuzu=KuzuSettings(path="./tmp-kuzu"),
        )
    )
    upsert = mocker.patch.object(store.store, "upsert_record")
    replace_chunks = mocker.patch.object(store.store, "replace_chunks")
    replace_relations = mocker.patch.object(store.store, "replace_relations")
    enqueue_projection = mocker.patch.object(store.store, "enqueue_projection")

    record = KnowledgeRecord(
        record_id="r-live",
        text="AuthService depends on TokenSigner.",
        source="design-doc",
        domain="platform",
        confidence=0.8,
    )

    store.write_record(record, chunks=[("AuthService depends on TokenSigner.", [0.1] * 1536, 0)])

    upsert.assert_called_once_with(record)
    replace_chunks.assert_called_once()
    replace_relations.assert_called_once()
    enqueue_projection.assert_called_once_with(record.record_id)


def test_write_text_generates_chunks_with_default_embedder(mocker) -> None:
    store = HybridMemoryStore(
        HybridStoreSettings(
            postgres=PostgresSettings(dsn="postgresql://localhost/test"),
            kuzu=KuzuSettings(path="./tmp-kuzu"),
        )
    )
    write_record = mocker.patch.object(store, "write_record")
    record = KnowledgeRecord(record_id="r2", text="alpha " * 300, source="design-doc", confidence=0.8)

    store.write_text(record)

    assert write_record.called
    chunks = write_record.call_args.kwargs["chunks"]
    assert len(chunks) > 1


def test_promote_record_persists_promotion_provenance(mocker) -> None:
    store = HybridMemoryStore(
        HybridStoreSettings(
            postgres=PostgresSettings(dsn="postgresql://localhost/test"),
            kuzu=KuzuSettings(path="./tmp-kuzu"),
        )
    )
    record = KnowledgeRecord(record_id="draft-promote", text="Draft TODO", stage="staged", kind="note")
    fetch = mocker.patch.object(store.store, "fetch_record", return_value=record)
    upsert = mocker.patch.object(store.store, "upsert_record")
    record_promotion = mocker.patch.object(store.store, "record_promotion")
    enqueue = mocker.patch.object(store.store, "enqueue_projection")

    promoted = store.promote_record("draft-promote", reason="reviewed")

    fetch.assert_called_once_with("draft-promote")
    upsert.assert_called_once_with(promoted)
    record_promotion.assert_called_once()
    enqueue.assert_called_once_with("draft-promote")


def test_settings_based_construction_creates_postgres_and_kuzu() -> None:
    store = HybridMemoryStore(
        HybridStoreSettings(
            postgres=PostgresSettings(dsn="postgresql://localhost/test"),
            kuzu=KuzuSettings(path="./tmp-kuzu"),
        )
    )
    assert isinstance(store.store, PostgresPgvectorAdapter)
    assert isinstance(store.graph, KuzuGraphProjectionAdapter)


def test_protocol_based_construction_with_sqlite(tmp_path) -> None:
    db_path = str(tmp_path / "test.db")
    sqlite_store = SQLiteRecordStore(db_path)
    store = HybridMemoryStore(store=sqlite_store)

    assert store.store is sqlite_store
    assert store.graph is None


def test_graph_is_optional_write_skips_projection(mocker, tmp_path) -> None:
    db_path = str(tmp_path / "test.db")
    sqlite_store = SQLiteRecordStore(db_path)
    sqlite_store.bootstrap()
    store = HybridMemoryStore(store=sqlite_store)

    record = KnowledgeRecord(
        record_id="r-no-graph",
        text="AuthService depends on TokenSigner.",
        source="design-doc",
        domain="platform",
        confidence=0.8,
    )

    # Should not raise even though there's no graph store
    plan = store.write_record(record)
    assert plan.record_id == "r-no-graph"
