from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import HybridStoreSettings, KuzuSettings, PostgresSettings
from .storage.kuzu import KuzuGraphProjectionAdapter
from .storage.postgres import PostgresPgvectorAdapter
from .storage.projector import GraphProjectionWorker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run context-fabrica graph projection worker")
    parser.add_argument("--dsn", default=os.environ.get("CONTEXT_FABRICA_TEST_DSN", "postgresql:///context_fabrica"))
    parser.add_argument("--kuzu-path", default="./var/context-fabrica-graph")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    Path(args.kuzu_path).parent.mkdir(parents=True, exist_ok=True)
    settings = HybridStoreSettings(
        postgres=PostgresSettings(dsn=args.dsn),
        kuzu=KuzuSettings(path=args.kuzu_path),
    )
    worker = GraphProjectionWorker(
        PostgresPgvectorAdapter(settings.postgres),
        KuzuGraphProjectionAdapter(settings.kuzu),
    )
    if args.once:
        print(worker.process_pending(limit=args.batch_size))
        return
    worker.run_forever(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
