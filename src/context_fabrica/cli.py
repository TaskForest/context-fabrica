from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .storage.hybrid import HybridMemoryStore
from .storage.sqlite import SQLiteRecordStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="context-fabrica CLI")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to JSONL file with records")
    parser.add_argument("--query", required=True, help="Natural-language query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--namespace", default=None, help="Filter results to a namespace")
    return parser


def _parse_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    store = HybridMemoryStore(store=SQLiteRecordStore(":memory:"))
    store.bootstrap()
    for line in args.dataset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        store.ingest(
            payload["text"],
            source=payload.get("source", "unknown"),
            domain=payload.get("domain", "global"),
            namespace=payload.get("namespace", "default"),
            confidence=float(payload.get("confidence", 0.6)),
            tags=payload.get("tags", []),
            metadata=payload.get("metadata", {}),
            record_id=payload.get("record_id"),
            occurred_from=_parse_iso(payload.get("occurred_from")),
            occurred_to=_parse_iso(payload.get("occurred_to")),
        )

    results = store.query(args.query, top_k=args.top_k, namespace=args.namespace)
    for idx, item in enumerate(results, start=1):
        print(f"{idx}. {item.record.record_id} score={item.score:.3f} rationale={','.join(item.rationale)}")
        print(f"   {item.record.text[:200]}")


if __name__ == "__main__":
    main()
