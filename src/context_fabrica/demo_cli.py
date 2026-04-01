from __future__ import annotations

import argparse

from .config import HybridStoreSettings, KuzuSettings, PostgresSettings
from .models import KnowledgeRecord
from .storage.hybrid import HybridMemoryStore
from .storage.kuzu import KuzuGraphProjectionAdapter
from .storage.postgres import PostgresPgvectorAdapter
from .storage.projector import GraphProjectionWorker


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap, ingest, query, and optionally project one demo record")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--kuzu-path", default="./var/context-fabrica-graph")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--record-id", default="demo-auth-1")
    args = parser.parse_args()

    store = HybridMemoryStore(
        HybridStoreSettings(
            postgres=PostgresSettings(dsn=args.dsn),
            kuzu=KuzuSettings(path=args.kuzu_path),
        )
    )
    store.bootstrap_postgres()

    record = KnowledgeRecord(
        record_id=args.record_id,
        text="AuthService depends on TokenSigner and calls KeyStore. Platform owns AuthService.",
        source="demo-cli",
        domain="platform",
        confidence=0.95,
        tags=["design-doc"],
        metadata={"repo": "context-fabrica", "kind": "demo"},
    )
    store.write_text(record)
    embedding = store.embedder.embed(record.text)
    hits = store.semantic_search(embedding, domain="platform", top_k=3)

    print(f"Wrote record: {record.record_id}")
    print(f"Top hit count: {len(hits)}")
    for hit in hits:
        print(f"- {hit.record.record_id}: {hit.score:.3f} ({','.join(hit.rationale)})")

    if args.project:
        worker = GraphProjectionWorker(
            PostgresPgvectorAdapter(store.settings.postgres),
            KuzuGraphProjectionAdapter(store.settings.kuzu),
        )
        results = worker.process_pending(limit=10)
        print(f"Projection results: {results}")


if __name__ == "__main__":
    main()
