"""Tests for the TypeScriptASTExtractor and extract_and_ingest flow."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_typescript")

from src.context_fabrica.extractors.typescript_ast import TypeScriptASTExtractor
from src.context_fabrica.storage.hybrid import HybridMemoryStore
from src.context_fabrica.storage.sqlite import SQLiteRecordStore


SAMPLE_TYPESCRIPT = '''\
/** Auth service module. */

import { sign } from "./crypto";

export class AuthService extends BaseService {
  /** Authenticate a user and return a session token. */
  async login(username: string, password: string): Promise<string> {
    const token = sign(username);
    return token;
  }
}

export function createSession(userId: string): string {
  return saveSession(userId);
}

export interface SessionStore {
  persist(token: string): void;
}

export type Status = "active" | "inactive";
'''

SAMPLE_TSX = '''\
import React from "react";

export const App = () => {
  return <button onClick={() => setCount(count + 1)}>Increment</button>;
};
'''

SAMPLE_SYNTAX_ERROR = "export const broken = ( =>\n"


def _write_sample(tmp_path: Path, filename: str, content: str) -> Path:
    f = tmp_path / filename
    f.write_text(content, encoding="utf-8")
    return f


def test_extract_single_typescript_file(tmp_path) -> None:
    f = _write_sample(tmp_path, "auth.ts", SAMPLE_TYPESCRIPT)
    extractor = TypeScriptASTExtractor()
    results = extractor.extract(f)

    assert len(results) == 1
    result = results[0]
    assert "AuthService" in result.entities
    assert "AuthService.login" in result.entities
    assert "createSession" in result.entities
    assert "SessionStore" in result.entities
    assert "Status" in result.entities
    assert any(r.relation == "inherits" and r.source_entity == "AuthService" and r.target_entity == "BaseService" for r in result.relations)
    assert result.confidence == 0.9
    assert "typescript" in result.tags


def test_extract_typescript_imports_and_calls(tmp_path) -> None:
    f = _write_sample(tmp_path, "auth.ts", SAMPLE_TYPESCRIPT)
    extractor = TypeScriptASTExtractor()
    result = extractor.extract(f)[0]

    assert "sign" in result.entities
    assert any(r.relation == "imports" and r.target_entity == "sign" for r in result.relations)
    assert any(r.relation == "calls" and r.source_entity == "AuthService.login" and r.target_entity == "sign" for r in result.relations)
    assert any(r.relation == "calls" and r.source_entity == "createSession" and r.target_entity == "saveSession" for r in result.relations)


def test_extract_tsx_arrow_function(tmp_path) -> None:
    f = _write_sample(tmp_path, "App.tsx", SAMPLE_TSX)
    extractor = TypeScriptASTExtractor()
    result = extractor.extract(f)[0]

    assert "App" in result.entities
    assert any(r.relation == "calls" and r.source_entity == "App" and r.target_entity == "setCount" for r in result.relations)
    assert result.metadata["language"] == "tsx"


def test_extract_directory(tmp_path) -> None:
    _write_sample(tmp_path, "auth.ts", SAMPLE_TYPESCRIPT)
    _write_sample(tmp_path, "App.tsx", SAMPLE_TSX)
    _write_sample(tmp_path, "ignore.js", "export const nope = true;\n")

    extractor = TypeScriptASTExtractor()
    results = extractor.extract(tmp_path)

    assert len(results) == 2


def test_extract_skips_syntax_error(tmp_path) -> None:
    _write_sample(tmp_path, "broken.ts", SAMPLE_SYNTAX_ERROR)
    extractor = TypeScriptASTExtractor()
    results = extractor.extract(tmp_path)

    assert results == []


def test_extract_and_ingest_end_to_end(tmp_path) -> None:
    (tmp_path / "frontend").mkdir()
    _write_sample(tmp_path / "frontend", "auth.ts", SAMPLE_TYPESCRIPT)

    db = str(tmp_path / "memory.db")
    store = HybridMemoryStore(store=SQLiteRecordStore(db))
    store.bootstrap()

    records = store.extract_and_ingest(tmp_path / "frontend", TypeScriptASTExtractor(), namespace="frontend")

    assert len(records) == 1
    assert records[0].namespace == "frontend"
    assert "typescript" in records[0].tags

    results = store.query("AuthService saveSession", namespace="frontend", top_k=3)
    assert results
    assert any("AuthService" in r.record.text for r in results)
