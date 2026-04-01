from src.context_fabrica import HybridMemoryStore, HybridStoreSettings, KnowledgeRecord, KuzuSettings, PostgresSettings


def test_postgres_bootstrap_includes_pgvector_schema() -> None:
    store = HybridMemoryStore(
        HybridStoreSettings(
            postgres=PostgresSettings(dsn="postgresql://localhost/test"),
            kuzu=KuzuSettings(path="./tmp-kuzu"),
        )
    )

    bootstrap = store.bootstrap_plan()["postgres"]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in statement for statement in bootstrap)
    assert any("memory_records" in statement for statement in bootstrap)
    assert any("memory_chunks" in statement for statement in bootstrap)


def test_kuzu_bootstrap_includes_record_and_entity_tables() -> None:
    store = HybridMemoryStore(
        HybridStoreSettings(
            postgres=PostgresSettings(dsn="postgresql://localhost/test"),
            kuzu=KuzuSettings(path="./tmp-kuzu"),
        )
    )

    bootstrap = store.bootstrap_plan()["kuzu"]
    assert any("CREATE NODE TABLE IF NOT EXISTS MemoryRecord" in statement for statement in bootstrap)
    assert any("CREATE NODE TABLE IF NOT EXISTS Entity" in statement for statement in bootstrap)
    assert any("CREATE REL TABLE IF NOT EXISTS RELATED" in statement for statement in bootstrap)


def test_write_plan_generates_postgres_payload_and_graph_projection() -> None:
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

    assert plan.postgres_record_payload[0] == "r1"
    assert plan.graph_projection.record_id == "r1"
    assert "authservice" in plan.graph_projection.entities
    assert any(row[2] == "DEPENDS_ON" for row in plan.postgres_relation_rows)
    assert any("MERGE (r:MemoryRecord" in statement for statement in plan.kuzu_projection_statements)


def test_postgres_search_statement_filters_by_domain_and_validity() -> None:
    store = HybridMemoryStore(
        HybridStoreSettings(
            postgres=PostgresSettings(dsn="postgresql://localhost/test"),
            kuzu=KuzuSettings(path="./tmp-kuzu"),
        )
    )

    statement = store.postgres.search_statement()
    assert "r.domain = $2" in statement
    assert "r.valid_from <= $3" in statement
    assert "r.valid_to IS NULL OR r.valid_to >= $3" in statement
