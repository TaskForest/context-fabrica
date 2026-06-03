"""Zero-dependency MCP server for context-fabrica.

Implements the Model Context Protocol over stdio using JSON-RPC 2.0.
No external dependencies beyond context-fabrica itself.

Memories are persisted to SQLite (or Postgres via --dsn) and survive
server restarts. BM25 and graph indexes bootstrap lazily on first query.

Usage:
    context-fabrica-mcp --db ./memory.db
    context-fabrica-mcp --db ./memory.db --namespace myproject
    context-fabrica-mcp --dsn postgresql:///context_fabrica
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from .embedding import build_default_embedder
from .models import KnowledgeRecord
from .storage.hybrid import HybridMemoryStore
from .storage.sqlite import SQLiteRecordStore

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "context-fabrica"
SERVER_VERSION = "1.0.2"

logging.basicConfig(level=logging.DEBUG, stream=sys.stderr, format="%(levelname)s: %(message)s")
log = logging.getLogger(SERVER_NAME)


def _tool_definitions(*, read_only: bool = False) -> list[dict[str, Any]]:
    read_tools = [
        {
            "name": "recall",
            "description": "Search memory. Concise by default; use get to expand a hit.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Max hits", "default": 3, "minimum": 1, "maximum": 20},
                    "domain": {"type": "string", "description": "Optional domain filter"},
                    "verbosity": {
                        "type": "string",
                        "enum": ["concise", "verbose"],
                        "description": "concise: [id] text; verbose: scores and metadata",
                        "default": "concise",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max text chars per hit",
                        "default": 300,
                        "minimum": 80,
                        "maximum": 2000,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get",
            "description": "Fetch one full memory record by ID.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "Record ID"},
                    "include_chunks": {"type": "boolean", "description": "Include chunk text", "default": False},
                },
                "required": ["record_id"],
            },
        },
    ]
    if read_only:
        return read_tools

    return [
        {
            "name": "remember",
            "description": "Store a fact or observation in memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The knowledge to store"},
                    "source": {"type": "string", "description": "Where this came from (e.g. 'code-review', 'user', 'investigation')", "default": "agent"},
                    "domain": {"type": "string", "description": "Knowledge domain (e.g. 'auth', 'payments', 'infra')", "default": "global"},
                    "confidence": {"type": "number", "description": "How confident you are in this fact (0.0-1.0)", "default": 0.7, "minimum": 0.0, "maximum": 1.0},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for categorization", "default": []},
                    "record_id": {"type": "string", "description": "Optional explicit ID for the record"},
                },
                "required": ["text"],
            },
        },
        *read_tools,
        {
            "name": "synthesize",
            "description": "Combine records into a provenance-backed observation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "record_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of records to synthesize into an observation",
                        "minItems": 2,
                    },
                    "record_id": {"type": "string", "description": "Optional ID for the new observation"},
                },
                "required": ["record_ids"],
            },
        },
        {
            "name": "promote",
            "description": "Promote a staged memory to canonical.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "ID of the record to promote"},
                },
                "required": ["record_id"],
            },
        },
        {
            "name": "invalidate",
            "description": "Invalidate a memory while retaining audit history.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "ID of the record to invalidate"},
                    "reason": {"type": "string", "description": "Why this record is being invalidated", "default": "obsolete"},
                },
                "required": ["record_id"],
            },
        },
        {
            "name": "supersede",
            "description": "Replace a memory with an updated version.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "old_record_id": {"type": "string", "description": "ID of the record being replaced"},
                    "new_text": {"type": "string", "description": "The updated knowledge"},
                    "reason": {"type": "string", "description": "Why the old record is being replaced", "default": "updated"},
                    "confidence": {"type": "number", "description": "Confidence in the new fact (0.0-1.0)", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["old_record_id", "new_text"],
            },
        },
        {
            "name": "related",
            "description": "Find graph-related records.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "ID of the record to find relations for"},
                    "hops": {"type": "integer", "description": "Graph traversal depth", "default": 1, "minimum": 1, "maximum": 5},
                    "top_k": {"type": "integer", "description": "Maximum related records to return", "default": 5, "minimum": 1, "maximum": 20},
                },
                "required": ["record_id"],
            },
        },
        {
            "name": "history",
            "description": "Show a record's supersession chain.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "ID of the record to trace history for"},
                },
                "required": ["record_id"],
            },
        },
    ]


class ContextFabricaMCP:
    def __init__(self, store: HybridMemoryStore, namespace: str = "default", *, read_only: bool = False) -> None:
        self._store = store
        self._namespace = namespace
        self._read_only = read_only

    def handle_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method = msg.get("method", "")
        request_id = msg.get("id")
        params = msg.get("params", {})

        if request_id is None:
            return None

        handler = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "ping": self._handle_ping,
        }.get(method)

        if handler is None:
            return _error(request_id, -32601, f"Method not found: {method}")

        try:
            result = handler(params)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except KeyError as exc:
            return _error(request_id, -32602, f"Record not found: {exc}")
        except Exception as exc:
            log.exception("Tool execution failed")
            return _error(request_id, -32603, str(exc))

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    def _handle_ping(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"tools": _tool_definitions(read_only=self._read_only)}

    def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        args = params.get("arguments", {})

        dispatch = {
            "recall": self._tool_recall,
            "get": self._tool_get,
        }
        if not self._read_only:
            dispatch.update(
                {
                    "remember": self._tool_remember,
                    "synthesize": self._tool_synthesize,
                    "promote": self._tool_promote,
                    "invalidate": self._tool_invalidate,
                    "supersede": self._tool_supersede,
                    "related": self._tool_related,
                    "history": self._tool_history,
                }
            )

        handler = dispatch.get(name)
        if handler is None:
            return _tool_error(f"Unknown tool: {name}")

        try:
            return handler(args)
        except KeyError as exc:
            return _tool_error(f"Record not found: {exc}")
        except Exception as exc:
            log.exception("Tool %s failed", name)
            return _tool_error(str(exc))

    # ── Tools ──

    def _tool_remember(self, args: dict[str, Any]) -> dict[str, Any]:
        record = self._store.ingest(
            args["text"],
            source=args.get("source", "agent"),
            domain=args.get("domain", "global"),
            namespace=self._namespace,
            confidence=args.get("confidence", 0.7),
            tags=args.get("tags", []),
            record_id=args.get("record_id"),
        )
        return _tool_result(
            f"Stored as {record.record_id} (stage={record.stage}, kind={record.kind}, confidence={record.confidence:.2f})"
        )

    def _tool_recall(self, args: dict[str, Any]) -> dict[str, Any]:
        max_chars = min(max(int(args.get("max_chars", 300)), 80), 2000)
        verbosity = args.get("verbosity", "concise")
        results = self._store.query(
            args["query"],
            top_k=args.get("top_k", 3),
            domain=args.get("domain"),
            namespace=self._namespace,
        )
        if not results:
            return _tool_result("No relevant memories found.")

        lines: list[str] = []
        for i, hit in enumerate(results, 1):
            r = hit.record
            text = _truncate(r.text, max_chars)
            if verbosity == "verbose":
                lines.append(
                    f"{i}. [{r.record_id}] score={hit.score:.3f} "
                    f"({','.join(hit.rationale)})\n"
                    f"   source={r.source} domain={r.domain} confidence={r.confidence:.2f} "
                    f"stage={r.stage}\n"
                    f"   {text}"
                )
            else:
                lines.append(f"{i}. [{r.record_id}] {text}")
        return _tool_result("\n\n".join(lines))

    def _tool_get(self, args: dict[str, Any]) -> dict[str, Any]:
        record_id = args["record_id"]
        include_chunks = bool(args.get("include_chunks", False))

        if include_chunks and hasattr(self._store.store, "fetch_record_with_chunks"):
            fetched = self._store.store.fetch_record_with_chunks(record_id)
            if fetched is None:
                raise KeyError(record_id)
            record, chunks = fetched
        else:
            record = self._store.store.fetch_record(record_id)
            if record is None:
                raise KeyError(record_id)
            chunks = []

        lines = [
            f"[{record.record_id}]",
            f"source={record.source} domain={record.domain} namespace={record.namespace} "
            f"confidence={record.confidence:.2f} stage={record.stage} kind={record.kind}",
            record.text,
        ]
        if include_chunks:
            lines.append("chunks:")
            for chunk_text, _embedding, chunk_index in chunks:
                lines.append(f"{chunk_index}. {chunk_text}")
        return _tool_result("\n".join(lines))

    def _tool_synthesize(self, args: dict[str, Any]) -> dict[str, Any]:
        observation = self._store.synthesize_observation(
            args["record_ids"],
            record_id=args.get("record_id"),
        )
        return _tool_result(
            f"Synthesized observation {observation.record_id}\n"
            f"  derived_from={observation.metadata['derived_from']}\n"
            f"  {observation.text[:300]}"
        )

    def _tool_promote(self, args: dict[str, Any]) -> dict[str, Any]:
        record = self._store.promote_record(args["record_id"])
        return _tool_result(f"Promoted {record.record_id} to stage={record.stage}")

    def _tool_invalidate(self, args: dict[str, Any]) -> dict[str, Any]:
        self._store.invalidate_record(
            args["record_id"],
            reason=args.get("reason", "obsolete"),
        )
        return _tool_result(f"Invalidated {args['record_id']}")

    def _tool_supersede(self, args: dict[str, Any]) -> dict[str, Any]:
        new = self._store.supersede_record_by_text(
            args["old_record_id"],
            args["new_text"],
            reason=args.get("reason", "updated"),
            confidence=args.get("confidence"),
        )
        return _tool_result(
            f"Superseded {args['old_record_id']} with {new.record_id}\n"
            f"  {new.text[:300]}"
        )

    def _tool_related(self, args: dict[str, Any]) -> dict[str, Any]:
        related = self._store.related_records(
            args["record_id"],
            hops=args.get("hops", 1),
            top_k=args.get("top_k", 5),
        )
        if not related:
            return _tool_result("No related records found.")

        lines: list[str] = []
        for i, r in enumerate(related, 1):
            lines.append(
                f"{i}. [{r.record_id}] source={r.source} domain={r.domain} "
                f"confidence={r.confidence:.2f}\n"
                f"   {r.text[:300]}"
            )
        return _tool_result("\n\n".join(lines))

    def _tool_history(self, args: dict[str, Any]) -> dict[str, Any]:
        chain = self._store.supersession_chain(args["record_id"])
        if not chain:
            return _tool_result("Record not found.")
        if len(chain) == 1:
            return _tool_result(f"No supersession history — [{chain[0].record_id}] is the original record.")

        lines: list[str] = []
        for i, r in enumerate(chain):
            marker = "(current)" if i == 0 else "(superseded)"
            lines.append(
                f"{i + 1}. [{r.record_id}] {marker} stage={r.stage} "
                f"confidence={r.confidence:.2f}\n"
                f"   {r.text[:200]}"
            )
        return _tool_result("\n\n".join(lines))


# ── JSON-RPC helpers ──

def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _tool_error(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Error: {text}"}], "isError": True}


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


# ── Entry point ──

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="context-fabrica MCP server")
    parser.add_argument("--db", default=None, help="Path to SQLite database file (default: ./context-fabrica-memory.db)")
    parser.add_argument("--dsn", default=None, help="PostgreSQL connection string")
    parser.add_argument("--namespace", default="default", help="Default namespace for this server instance")
    parser.add_argument(
        "--embedder",
        choices=["auto", "fastembed", "sentence-transformers", "hash"],
        default="auto",
        help="Embedding backend for semantic recall (default: auto, prefers FastEmbed/MiniLM)",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="Local embedding model name for fastembed or sentence-transformers",
    )
    parser.add_argument(
        "--embed-dimensions",
        type=int,
        default=384,
        help="Vector dimensions for hash fallback and Postgres schema when using hash (default: 384)",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Expose only recall and get tools to reduce MCP schema overhead",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    embedder = build_default_embedder(
        dimensions=args.embed_dimensions,
        embedder=args.embedder,
        model_name=args.embed_model,
    )

    if args.dsn:
        from .config import PostgresSettings
        from .storage.postgres import PostgresPgvectorAdapter
        record_store = PostgresPgvectorAdapter(
            PostgresSettings(dsn=args.dsn, embedding_dimensions=embedder.dimensions)
        )
        backend_label = args.dsn
    else:
        db_path = args.db or "./context-fabrica-memory.db"
        record_store = SQLiteRecordStore(db_path)
        backend_label = db_path

    store = HybridMemoryStore(store=record_store, embedder=embedder)
    store.bootstrap()

    server = ContextFabricaMCP(store, namespace=args.namespace, read_only=args.read_only)
    log.info(
        "context-fabrica MCP server started (backend=%s, namespace=%s, embedder=%s, dimensions=%s, read_only=%s)",
        backend_label,
        args.namespace,
        embedder.__class__.__name__,
        embedder.dimensions,
        args.read_only,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log.warning("Ignoring malformed JSON: %s", line[:100])
            continue

        response = server.handle_message(msg)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
