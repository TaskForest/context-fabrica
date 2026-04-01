from __future__ import annotations

import os
from typing import cast

import pytest

from src.context_fabrica import HybridMemoryStore, HybridStoreSettings, KuzuSettings, KnowledgeRecord, PostgresSettings


TEST_DSN = os.environ.get("CONTEXT_FABRICA_TEST_DSN")


@pytest.mark.skipif(not TEST_DSN, reason="CONTEXT_FABRICA_TEST_DSN not set")
def test_live_postgres_write_and_query() -> None:
    store = HybridMemoryStore(
        HybridStoreSettings(
            postgres=PostgresSettings(dsn=cast(str, TEST_DSN)),
            kuzu=KuzuSettings(path="./tmp-kuzu"),
        )
    )
    store.bootstrap_postgres()

    record = KnowledgeRecord(
        record_id="live-auth-1",
        text="AuthService depends on TokenSigner and calls KeyStore.",
        source="integration-test",
        domain="platform",
        confidence=0.95,
    )
    embedding = [0.01] * 1536
    store.write_record(record, chunks=[(record.text, embedding, 0)])

    fetched = store.postgres.fetch_record("live-auth-1")
    assert fetched is not None
    assert fetched.record_id == "live-auth-1"
    assert fetched.stage == "canonical"

    results = store.semantic_search(embedding, domain="platform", top_k=3)
    assert any(hit.record.record_id == "live-auth-1" for hit in results)
