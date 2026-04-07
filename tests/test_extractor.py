"""Tests for the PythonASTExtractor and extract_and_ingest flow."""
from __future__ import annotations

from pathlib import Path

from src.context_fabrica.extractors.python_ast import PythonASTExtractor
from src.context_fabrica.storage.hybrid import HybridMemoryStore
from src.context_fabrica.storage.sqlite import SQLiteRecordStore


SAMPLE_PYTHON = '''\
"""Auth service module."""

import hashlib
from datetime import datetime

class TokenSigner:
    """Signs and verifies JWT tokens."""

    def sign(self, payload: dict, secret: str) -> str:
        """Create a signed token."""
        return hashlib.sha256(str(payload).encode()).hexdigest()

    def verify(self, token: str, secret: str) -> bool:
        return True


class AuthService(TokenSigner):
    """Handles authentication and session management."""

    def login(self, username: str, password: str) -> str:
        """Authenticate a user and return a session token."""
        token = self.sign({"user": username}, password)
        return token

    def logout(self, token: str) -> None:
        pass
'''

SAMPLE_EMPTY = "# empty file\n"

SAMPLE_SYNTAX_ERROR = "def broken(:\n"


def _write_sample(tmp_path: Path, filename: str, content: str) -> Path:
    f = tmp_path / filename
    f.write_text(content, encoding="utf-8")
    return f


def test_extract_single_file(tmp_path) -> None:
    f = _write_sample(tmp_path, "auth.py", SAMPLE_PYTHON)
    extractor = PythonASTExtractor()
    results = extractor.extract(f)

    assert len(results) == 1
    result = results[0]
    assert "TokenSigner" in result.entities
    assert "AuthService" in result.entities
    assert any(r.relation == "inherits" and r.source_entity == "AuthService" and r.target_entity == "TokenSigner" for r in result.relations)
    assert result.confidence == 0.9
    assert "python" in result.tags


def test_extract_finds_methods(tmp_path) -> None:
    f = _write_sample(tmp_path, "auth.py", SAMPLE_PYTHON)
    extractor = PythonASTExtractor()
    results = extractor.extract(f)
    result = results[0]

    assert "AuthService.login" in result.entities
    assert "TokenSigner.sign" in result.entities
    assert any(r.relation == "has_method" for r in result.relations)


def test_extract_finds_imports(tmp_path) -> None:
    f = _write_sample(tmp_path, "auth.py", SAMPLE_PYTHON)
    extractor = PythonASTExtractor()
    results = extractor.extract(f)
    result = results[0]

    assert "hashlib" in result.entities
    assert "datetime" in result.entities
    assert any(r.relation == "imports" for r in result.relations)


def test_extract_finds_calls(tmp_path) -> None:
    f = _write_sample(tmp_path, "auth.py", SAMPLE_PYTHON)
    extractor = PythonASTExtractor()
    results = extractor.extract(f)
    result = results[0]

    assert any(
        r.relation == "calls" and "login" in r.source_entity and "sign" in r.target_entity
        for r in result.relations
    )


def test_extract_directory(tmp_path) -> None:
    _write_sample(tmp_path, "a.py", SAMPLE_PYTHON)
    _write_sample(tmp_path, "b.py", "class Foo:\n    pass\n")
    _write_sample(tmp_path, "readme.txt", "not python")

    extractor = PythonASTExtractor()
    results = extractor.extract(tmp_path)
    assert len(results) == 2


def test_extract_skips_empty_file(tmp_path) -> None:
    _write_sample(tmp_path, "empty.py", SAMPLE_EMPTY)
    extractor = PythonASTExtractor()
    results = extractor.extract(tmp_path)
    assert len(results) == 0


def test_extract_skips_syntax_error(tmp_path) -> None:
    _write_sample(tmp_path, "broken.py", SAMPLE_SYNTAX_ERROR)
    extractor = PythonASTExtractor()
    results = extractor.extract(tmp_path)
    assert len(results) == 0


def test_extract_includes_docstrings(tmp_path) -> None:
    f = _write_sample(tmp_path, "auth.py", SAMPLE_PYTHON)
    extractor = PythonASTExtractor()
    results = extractor.extract(f)
    text = results[0].text

    assert "Auth service module" in text
    assert "Signs and verifies JWT tokens" in text


def test_extract_metadata_has_source_info(tmp_path) -> None:
    f = _write_sample(tmp_path, "auth.py", SAMPLE_PYTHON)
    extractor = PythonASTExtractor()
    results = extractor.extract(f)

    meta = results[0].metadata
    assert meta["language"] == "python"
    assert "auth.py" in meta["source_file"]
    assert "TokenSigner" in meta["classes"]
    assert "AuthService" in meta["classes"]


def test_extract_custom_domain(tmp_path) -> None:
    f = _write_sample(tmp_path, "auth.py", SAMPLE_PYTHON)
    extractor = PythonASTExtractor(domain="auth")
    results = extractor.extract(f)
    assert results[0].domain == "auth"


def test_extract_and_ingest_end_to_end(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    _write_sample(tmp_path / "src", "auth.py", SAMPLE_PYTHON)

    db = str(tmp_path / "memory.db")
    store = HybridMemoryStore(store=SQLiteRecordStore(db))
    store.bootstrap()

    extractor = PythonASTExtractor()
    records = store.extract_and_ingest(tmp_path / "src", extractor, namespace="myproject")

    assert len(records) == 1
    assert records[0].namespace == "myproject"
    assert "python" in records[0].tags

    # Query should find the extracted knowledge
    results = store.query("TokenSigner authentication", namespace="myproject", top_k=3)
    assert results
    assert any("TokenSigner" in r.record.text for r in results)


def test_extract_and_ingest_persists_across_restart(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    _write_sample(tmp_path / "src", "auth.py", SAMPLE_PYTHON)
    db = str(tmp_path / "memory.db")

    # Session 1: extract and ingest
    store1 = HybridMemoryStore(store=SQLiteRecordStore(db))
    store1.bootstrap()
    store1.extract_and_ingest(tmp_path / "src", PythonASTExtractor())

    # Session 2: new store, same db — query should work
    store2 = HybridMemoryStore(store=SQLiteRecordStore(db))
    results = store2.query("AuthService TokenSigner", top_k=3)
    assert results
    assert any("AuthService" in r.record.text for r in results)
