# Contributing

Thanks for helping improve `context-fabrica`.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Project Standards

- keep changes small and focused
- preserve the canonical-store-first design
- prefer explicit provenance over hidden heuristics
- keep the first-run UX simple for outside users

## Before Opening a PR

Run:

```bash
python -m pytest
python -m pip install .
```

If your change touches the live Postgres path, also verify:

```bash
PYTHONPATH=src python -m context_fabrica.bootstrap_cli --root . --dsn "postgresql:///context_fabrica"
PYTHONPATH=src python -m context_fabrica.demo_cli --dsn "postgresql:///context_fabrica" --project
```

## Good First Contributions

- better promotion policies
- projector observability
- stronger install docs
- more runnable examples
