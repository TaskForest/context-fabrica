"""Tests for the zero-dependency MCP server."""
from __future__ import annotations

from typing import Any

from src.context_fabrica.mcp_server import ContextFabricaMCP
from src.context_fabrica.storage.hybrid import HybridMemoryStore
from src.context_fabrica.storage.sqlite import SQLiteRecordStore


def _make_server(tmp_path) -> ContextFabricaMCP:
    db = str(tmp_path / "test.db")
    store = HybridMemoryStore(store=SQLiteRecordStore(db))
    store.bootstrap()
    return ContextFabricaMCP(store, namespace="test")


def _call(server: ContextFabricaMCP, method: str, params: dict[str, Any] | None = None, msg_id: int = 1) -> dict[str, Any]:
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}
    result = server.handle_message(msg)
    assert result is not None
    return result


def _call_tool(server: ContextFabricaMCP, name: str, arguments: dict[str, Any], msg_id: int = 1) -> dict[str, Any]:
    return _call(server, "tools/call", {"name": name, "arguments": arguments}, msg_id)


def test_initialize(tmp_path) -> None:
    server = _make_server(tmp_path)
    resp = _call(server, "initialize")
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "context-fabrica"


def test_ping(tmp_path) -> None:
    server = _make_server(tmp_path)
    resp = _call(server, "ping")
    assert resp["result"] == {}


def test_tools_list(tmp_path) -> None:
    server = _make_server(tmp_path)
    resp = _call(server, "tools/list")
    tool_names = {t["name"] for t in resp["result"]["tools"]}
    assert tool_names == {"remember", "recall", "synthesize", "promote", "invalidate", "supersede", "related", "history"}


def test_notification_returns_none(tmp_path) -> None:
    server = _make_server(tmp_path)
    result = server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert result is None


def test_unknown_method_returns_error(tmp_path) -> None:
    server = _make_server(tmp_path)
    resp = _call(server, "nonexistent/method")
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_unknown_tool_returns_error(tmp_path) -> None:
    server = _make_server(tmp_path)
    resp = _call_tool(server, "nonexistent_tool", {})
    assert resp["result"]["isError"] is True


def test_remember_and_recall(tmp_path) -> None:
    server = _make_server(tmp_path)
    remember_resp = _call_tool(server, "remember", {
        "text": "AuthService depends on TokenSigner",
        "source": "test",
        "confidence": 0.9,
    })
    assert remember_resp["result"]["isError"] is False
    assert "Stored as" in remember_resp["result"]["content"][0]["text"]

    recall_resp = _call_tool(server, "recall", {"query": "AuthService"})
    assert recall_resp["result"]["isError"] is False
    assert "AuthService" in recall_resp["result"]["content"][0]["text"]


def test_remember_with_explicit_id(tmp_path) -> None:
    server = _make_server(tmp_path)
    resp = _call_tool(server, "remember", {
        "text": "Token rotation happens daily",
        "record_id": "my-id-1",
    })
    assert "my-id-1" in resp["result"]["content"][0]["text"]


def test_recall_empty_memory(tmp_path) -> None:
    server = _make_server(tmp_path)
    resp = _call_tool(server, "recall", {"query": "anything"})
    assert "No relevant memories found" in resp["result"]["content"][0]["text"]


def test_synthesize(tmp_path) -> None:
    server = _make_server(tmp_path)
    _call_tool(server, "remember", {"text": "AuthService depends on TokenSigner.", "record_id": "r1"})
    _call_tool(server, "remember", {"text": "TokenSigner rotates keys daily.", "record_id": "r2"})

    resp = _call_tool(server, "synthesize", {"record_ids": ["r1", "r2"], "record_id": "obs-1"})
    assert resp["result"]["isError"] is False
    assert "obs-1" in resp["result"]["content"][0]["text"]
    assert "derived_from" in resp["result"]["content"][0]["text"]


def test_promote(tmp_path) -> None:
    server = _make_server(tmp_path)
    _call_tool(server, "remember", {"text": "Draft observation", "record_id": "d1", "confidence": 0.3})

    resp = _call_tool(server, "promote", {"record_id": "d1"})
    assert resp["result"]["isError"] is False
    assert "Promoted" in resp["result"]["content"][0]["text"]


def test_invalidate(tmp_path) -> None:
    server = _make_server(tmp_path)
    _call_tool(server, "remember", {"text": "Old fact", "record_id": "old-1", "confidence": 0.8})

    resp = _call_tool(server, "invalidate", {"record_id": "old-1", "reason": "wrong"})
    assert resp["result"]["isError"] is False
    assert "Invalidated" in resp["result"]["content"][0]["text"]


def test_supersede(tmp_path) -> None:
    server = _make_server(tmp_path)
    _call_tool(server, "remember", {"text": "timeout is 60s", "record_id": "v1", "confidence": 0.8})

    resp = _call_tool(server, "supersede", {
        "old_record_id": "v1",
        "new_text": "timeout is 30s",
        "reason": "corrected",
    })
    assert resp["result"]["isError"] is False
    assert "Superseded" in resp["result"]["content"][0]["text"]


def test_related(tmp_path) -> None:
    server = _make_server(tmp_path)
    _call_tool(server, "remember", {"text": "AuthService uses TokenSigner and depends on KeyStore.", "record_id": "a"})
    _call_tool(server, "remember", {"text": "TokenSigner rotates keys from KeyStore daily.", "record_id": "b"})

    resp = _call_tool(server, "related", {"record_id": "a"})
    assert resp["result"]["isError"] is False


def test_history_no_chain(tmp_path) -> None:
    server = _make_server(tmp_path)
    _call_tool(server, "remember", {"text": "standalone fact", "record_id": "solo"})
    resp = _call_tool(server, "history", {"record_id": "solo"})
    assert "original record" in resp["result"]["content"][0]["text"]


def test_persistence_survives_restart(tmp_path) -> None:
    """Critical test: memories must survive server restart."""
    db = str(tmp_path / "persist.db")

    # Session 1: store a record
    store1 = HybridMemoryStore(store=SQLiteRecordStore(db))
    store1.bootstrap()
    server1 = ContextFabricaMCP(store1, namespace="test")
    _call_tool(server1, "remember", {"text": "AuthService depends on TokenSigner", "record_id": "r1", "confidence": 0.9})

    # Session 2: new store, same db — should recover via lazy bootstrap
    store2 = HybridMemoryStore(store=SQLiteRecordStore(db))
    server2 = ContextFabricaMCP(store2, namespace="test")

    resp = _call_tool(server2, "recall", {"query": "AuthService"})
    text = resp["result"]["content"][0]["text"]
    assert "r1" in text
    assert "AuthService" in text


def test_sqlite_migration_old_db(tmp_path) -> None:
    """Test that bootstrap adds occurred_from/occurred_to to an old database."""
    import sqlite3

    db_path = str(tmp_path / "old.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE memory_records (
            record_id TEXT PRIMARY KEY,
            text_content TEXT NOT NULL,
            source TEXT NOT NULL,
            domain TEXT NOT NULL DEFAULT 'global',
            namespace TEXT NOT NULL DEFAULT 'default',
            confidence REAL NOT NULL,
            memory_stage TEXT NOT NULL DEFAULT 'canonical',
            memory_kind TEXT NOT NULL DEFAULT 'fact',
            tags TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            supersedes TEXT,
            reviewed_at TEXT
        );
    """)
    conn.close()

    store = SQLiteRecordStore(db_path)
    store.bootstrap()

    from datetime import datetime, timezone
    from src.context_fabrica.models import KnowledgeRecord

    record = KnowledgeRecord(
        record_id="migrated-1",
        text="test migration",
        source="test",
        confidence=0.8,
        occurred_from=datetime(2025, 6, 1, tzinfo=timezone.utc),
        occurred_to=datetime(2025, 6, 30, tzinfo=timezone.utc),
    )
    store.upsert_record(record)
    fetched = store.fetch_record("migrated-1")
    assert fetched is not None
    assert fetched.occurred_from == datetime(2025, 6, 1, tzinfo=timezone.utc)
    assert fetched.occurred_to == datetime(2025, 6, 30, tzinfo=timezone.utc)
    store.close()
