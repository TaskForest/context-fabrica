from __future__ import annotations

from pathlib import Path
import argparse
import json


REGISTRY_DIR = ".context_fabrica"
REGISTRY_FILE = "registry.json"
MEMORY_ROOT = "memory"
CANONICAL_DIRS = ("staging", "canonical", "patterns")


def bootstrap(root: Path) -> dict[str, object]:
    registry_dir = root / REGISTRY_DIR
    registry_dir.mkdir(parents=True, exist_ok=True)
    memory_root = root / MEMORY_ROOT
    memory_root.mkdir(parents=True, exist_ok=True)
    for directory in CANONICAL_DIRS:
        (memory_root / directory).mkdir(parents=True, exist_ok=True)

    payload = {
        "project_root": str(root.resolve()),
        "memory_root": str(memory_root.resolve()),
        "tiers": list(CANONICAL_DIRS),
        "status": "bootstrapped",
    }
    (registry_dir / REGISTRY_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def status(root: Path) -> dict[str, object]:
    registry_path = root / REGISTRY_DIR / REGISTRY_FILE
    if not registry_path.exists():
        return {"status": "missing", "project_root": str(root.resolve())}

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    memory_root = Path(payload["memory_root"])
    payload["counts"] = {
        name: sum(1 for path in (memory_root / name).glob("*.md") if path.is_file())
        for name in CANONICAL_DIRS
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="context-fabrica project-memory helper")
    parser.add_argument("command", choices=["bootstrap", "status"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    if args.command == "bootstrap":
        payload = bootstrap(args.root)
    else:
        payload = status(args.root)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
