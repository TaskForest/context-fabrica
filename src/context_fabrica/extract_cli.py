"""CLI for extracting knowledge from source files into context-fabrica.

Usage:
    context-fabrica-extract ./src
    context-fabrica-extract ./src --db ./memory.db --namespace myproject
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .extractors import PythonASTExtractor, TypeScriptASTExtractor
from .storage.hybrid import HybridMemoryStore
from .storage.sqlite import SQLiteRecordStore

_EXTRACTORS = {
    "python": PythonASTExtractor,
    "typescript": TypeScriptASTExtractor,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract knowledge from source files and ingest into context-fabrica memory",
    )
    parser.add_argument("path", type=Path, help="File or directory to extract from")
    parser.add_argument("--db", default="./context-fabrica-memory.db", help="SQLite database path")
    parser.add_argument("--namespace", default="default", help="Namespace for ingested records")
    parser.add_argument("--domain", default="code", help="Domain label for extracted knowledge")
    parser.add_argument(
        "--language",
        choices=sorted(_EXTRACTORS),
        default="python",
        help="Built-in extractor to use",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} does not exist")
        raise SystemExit(1)

    store = HybridMemoryStore(store=SQLiteRecordStore(args.db))
    store.bootstrap()

    extractor = _EXTRACTORS[args.language](domain=args.domain)
    records = store.extract_and_ingest(args.path, extractor, namespace=args.namespace)

    print(f"Extracted and ingested {len(records)} records from {args.path}")
    for record in records:
        entities_count = len(record.metadata.get("classes", [])) + len(record.metadata.get("functions", []))
        print(f"  {record.source}: {entities_count} entities, confidence={record.confidence:.2f}")


if __name__ == "__main__":
    main()
