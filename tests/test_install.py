"""Tests for the multi-platform installer."""
from __future__ import annotations

from pathlib import Path

from src.context_fabrica.install_cli import (
    install_claude,
    install_codex,
    install_droid,
    install_opencode,
)


def test_install_claude_creates_mcp_json(tmp_path) -> None:
    actions = install_claude(tmp_path)
    assert (tmp_path / ".mcp.json").exists()
    assert any(".mcp.json" in a for a in actions)


def test_install_claude_creates_commands(tmp_path) -> None:
    install_claude(tmp_path)
    commands_dir = tmp_path / ".claude" / "commands"
    assert (commands_dir / "remember.md").exists()
    assert (commands_dir / "recall.md").exists()
    assert (commands_dir / "synthesize.md").exists()
    assert (commands_dir / "memory-status.md").exists()


def test_install_claude_skips_existing_mcp_json(tmp_path) -> None:
    (tmp_path / ".mcp.json").write_text("{}", encoding="utf-8")
    actions = install_claude(tmp_path)
    assert any("Skipped" in a and ".mcp.json" in a for a in actions)
    assert (tmp_path / ".mcp.json").read_text() == "{}"


def test_install_codex_creates_agents_md(tmp_path) -> None:
    actions = install_codex(tmp_path)
    assert (tmp_path / "AGENTS.md").exists()
    content = (tmp_path / "AGENTS.md").read_text()
    assert "context-fabrica" in content


def test_install_opencode_creates_config(tmp_path) -> None:
    actions = install_opencode(tmp_path)
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "opencode.json").exists()
    import json
    config = json.loads((tmp_path / "opencode.json").read_text())
    assert "context-fabrica" in config["mcp"]["servers"]


def test_install_droid_creates_droid_file(tmp_path) -> None:
    actions = install_droid(tmp_path)
    droid_path = tmp_path / ".factory" / "droids" / "context-fabrica.md"
    assert droid_path.exists()
    content = droid_path.read_text()
    assert "context-fabrica" in content
    assert "tools:" in content


def test_install_idempotent(tmp_path) -> None:
    install_claude(tmp_path)
    actions = install_claude(tmp_path)
    assert all("Skipped" in a or "Created" not in a for a in actions)


def test_install_all_platforms(tmp_path) -> None:
    """All five platform installers run without error on a clean directory."""
    from src.context_fabrica.install_cli import install_claw
    install_claude(tmp_path)
    install_codex(tmp_path)
    install_opencode(tmp_path)
    install_claw(tmp_path)
    install_droid(tmp_path)
    assert (tmp_path / ".mcp.json").exists()
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "opencode.json").exists()
    assert (tmp_path / ".factory" / "droids" / "context-fabrica.md").exists()
