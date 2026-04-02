from __future__ import annotations

from context_fabrica import HybridMemoryStore, HybridStoreSettings, KuzuSettings, PostgresSettings
from context_fabrica.models import KnowledgeRecord


def main() -> None:
    store = HybridMemoryStore(
        HybridStoreSettings(
            postgres=PostgresSettings(dsn="postgresql:///context_fabrica"),
            kuzu=KuzuSettings(path="./var/context-fabrica-graph"),
        )
    )
    store.bootstrap_postgres()
    record = KnowledgeRecord(
        record_id="example-live-1",
        text="AuthService depends on TokenSigner and calls KeyStore.",
        source="example",
        domain="platform",
        confidence=0.95,
        tags=["design-doc"],
    )
    store.write_text(record)
    hits = store.semantic_search(store.embedder.embed(record.text), domain="platform", top_k=3)
    print(hits)


if __name__ == "__main__":
    main()
