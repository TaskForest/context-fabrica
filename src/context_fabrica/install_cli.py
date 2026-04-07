"""Multi-platform installer for context-fabrica.

Detects the agent platform and writes the appropriate configuration files
to register the context-fabrica MCP server and agent instructions.

Usage:
    context-fabrica install                  # auto-detect platform
    context-fabrica install --platform codex # explicit platform
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PLATFORMS = ("claude", "codex", "opencode", "claw", "droid")

MCP_JSON = {
    "mcpServers": {
        "context-fabrica": {
            "command": "context-fabrica-mcp",
            "args": ["--db", "./context-fabrica-memory.db", "--namespace", "default"],
        }
    }
}

CODEX_TOML = """\
[mcp_servers.context-fabrica]
command = "context-fabrica-mcp"
args = ["--db", "./context-fabrica-memory.db", "--namespace", "default"]
"""

OPENCODE_JSON = {
    "mcp": {
        "servers": {
            "context-fabrica": {
                "command": "context-fabrica-mcp",
                "args": ["--db", "./context-fabrica-memory.db", "--namespace", "default"],
            }
        }
    }
}

OPENCLAW_JSON = {
    "mcp": {
        "servers": {
            "context-fabrica": {
                "command": "context-fabrica-mcp",
                "args": ["--db", "./context-fabrica-memory.db", "--namespace", "default"],
            }
        }
    }
}


def _detect_platform() -> str:
    if Path(".claude").is_dir() or Path(".mcp.json").exists():
        return "claude"
    if Path(".codex").is_dir() or Path(os.path.expanduser("~/.codex")).is_dir():
        return "codex"
    if Path("opencode.json").exists() or Path(".opencode").is_dir():
        return "opencode"
    if Path(os.path.expanduser("~/.openclaw")).is_dir():
        return "claw"
    if Path(".factory").is_dir():
        return "droid"
    return "claude"


def _agents_md_source() -> Path:
    """Path to the bundled AGENTS.md template."""
    return Path(__file__).parent.parent.parent / "AGENTS.md"


def _copy_agents_md(dest: Path) -> None:
    src = _agents_md_source()
    if src.exists():
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        # Fallback: minimal AGENTS.md
        dest.write_text(
            "# context-fabrica\n\n"
            "This project uses context-fabrica for governed long-term memory.\n"
            "Run `context-fabrica install` for setup instructions.\n",
            encoding="utf-8",
        )


def install_claude(project_root: Path) -> list[str]:
    """Install for Claude Code: .mcp.json + .claude/commands/."""
    actions: list[str] = []

    mcp_path = project_root / ".mcp.json"
    if not mcp_path.exists():
        mcp_path.write_text(json.dumps(MCP_JSON, indent=2) + "\n", encoding="utf-8")
        actions.append(f"Created {mcp_path}")
    else:
        actions.append(f"Skipped {mcp_path} (already exists)")

    commands_dir = project_root / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    commands = {
        "remember.md": "Store knowledge in context-fabrica long-term memory.\n\nUse the `remember` tool to store: $ARGUMENTS\n",
        "recall.md": "Search context-fabrica memory.\n\nUse the `recall` tool to search for: $ARGUMENTS\n",
        "synthesize.md": "Synthesize facts into an observation.\n\nUse the `synthesize` tool with: $ARGUMENTS\n",
        "memory-status.md": "Show context-fabrica memory status.\n\nUse the `recall` tool with a broad query to summarize stored knowledge.\n",
    }
    for name, content in commands.items():
        cmd_path = commands_dir / name
        if not cmd_path.exists():
            cmd_path.write_text(content, encoding="utf-8")
            actions.append(f"Created {cmd_path}")

    return actions


def install_codex(project_root: Path) -> list[str]:
    """Install for Codex: AGENTS.md + ~/.codex/config.toml."""
    actions: list[str] = []

    agents_path = project_root / "AGENTS.md"
    if not agents_path.exists():
        _copy_agents_md(agents_path)
        actions.append(f"Created {agents_path}")

    codex_dir = Path(os.path.expanduser("~/.codex"))
    codex_dir.mkdir(parents=True, exist_ok=True)
    config_path = codex_dir / "config.toml"

    if config_path.exists():
        existing = config_path.read_text(encoding="utf-8")
        if "context-fabrica" not in existing:
            with config_path.open("a", encoding="utf-8") as f:
                f.write("\n" + CODEX_TOML)
            actions.append(f"Appended MCP config to {config_path}")
        else:
            actions.append(f"Skipped {config_path} (context-fabrica already configured)")
    else:
        config_path.write_text(CODEX_TOML, encoding="utf-8")
        actions.append(f"Created {config_path}")

    return actions


def install_opencode(project_root: Path) -> list[str]:
    """Install for OpenCode: AGENTS.md + opencode.json."""
    actions: list[str] = []

    agents_path = project_root / "AGENTS.md"
    if not agents_path.exists():
        _copy_agents_md(agents_path)
        actions.append(f"Created {agents_path}")

    config_path = project_root / "opencode.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(OPENCODE_JSON, indent=2) + "\n", encoding="utf-8")
        actions.append(f"Created {config_path}")
    else:
        actions.append(f"Skipped {config_path} (already exists — add MCP config manually)")

    return actions


def install_claw(project_root: Path) -> list[str]:
    """Install for OpenClaw: AGENTS.md + config hint."""
    actions: list[str] = []

    agents_path = project_root / "AGENTS.md"
    if not agents_path.exists():
        _copy_agents_md(agents_path)
        actions.append(f"Created {agents_path}")

    claw_dir = Path(os.path.expanduser("~/.openclaw"))
    claw_dir.mkdir(parents=True, exist_ok=True)
    config_path = claw_dir / "config.json"

    if config_path.exists():
        existing = config_path.read_text(encoding="utf-8")
        if "context-fabrica" not in existing:
            actions.append(f"Add MCP config to {config_path} manually (existing config detected)")
        else:
            actions.append(f"Skipped {config_path} (context-fabrica already configured)")
    else:
        config_path.write_text(json.dumps(OPENCLAW_JSON, indent=2) + "\n", encoding="utf-8")
        actions.append(f"Created {config_path}")

    return actions


def install_droid(project_root: Path) -> list[str]:
    """Install for Factory Droid: .factory/droids/context-fabrica.md + AGENTS.md."""
    actions: list[str] = []

    agents_path = project_root / "AGENTS.md"
    if not agents_path.exists():
        _copy_agents_md(agents_path)
        actions.append(f"Created {agents_path}")

    droid_dir = project_root / ".factory" / "droids"
    droid_dir.mkdir(parents=True, exist_ok=True)
    droid_path = droid_dir / "context-fabrica.md"

    if not droid_path.exists():
        # Copy from bundled template
        bundled = Path(__file__).parent.parent.parent / ".factory" / "droids" / "context-fabrica.md"
        if bundled.exists():
            droid_path.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            droid_path.write_text(
                "---\nname: context-fabrica\ndescription: Governed long-term project memory\n"
                "model: inherit\ntools: \"mcp\"\n---\n\n"
                "You manage project memory via context-fabrica MCP tools.\n"
                "Use `recall` before answering, `remember` when you learn something.\n",
                encoding="utf-8",
            )
        actions.append(f"Created {droid_path}")

    return actions


INSTALLERS = {
    "claude": install_claude,
    "codex": install_codex,
    "opencode": install_opencode,
    "claw": install_claw,
    "droid": install_droid,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install context-fabrica for your agent platform")
    parser.add_argument(
        "--platform",
        choices=PLATFORMS,
        default=None,
        help="Agent platform (auto-detected if omitted)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Install for all supported platforms",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    project_root = args.root.resolve()

    if args.all:
        platforms = list(PLATFORMS)
    elif args.platform:
        platforms = [args.platform]
    else:
        detected = _detect_platform()
        platforms = [detected]
        print(f"Detected platform: {detected}")

    all_actions: list[str] = []
    for platform in platforms:
        print(f"\nInstalling for {platform}...")
        installer = INSTALLERS[platform]
        actions = installer(project_root)
        all_actions.extend(actions)
        for action in actions:
            print(f"  {action}")

    print(f"\nDone. {len(all_actions)} actions completed.")
    print("\nNext steps:")
    print("  1. Restart your agent (Claude Code, Codex, etc.) to pick up the new config")
    print("  2. Try: /recall or ask your agent to 'remember' something")


if __name__ == "__main__":
    main()
