# Getting Started

## 1. Install the package

```bash
python -m pip install "context-fabrica[postgres,kuzu,fastembed]"
```

If you are running from a local clone instead of PyPI:

```bash
python -m pip install .
python -m pip install -r requirements-v2.txt
```

## 2. Quick start with the in-memory engine

No database setup required — great for trying things out:

```python
from context_fabrica import HybridMemoryStore, SQLiteRecordStore

store = HybridMemoryStore(store=SQLiteRecordStore(":memory:"))
store.bootstrap()

store.ingest(
    "PaymentsService depends on LedgerAdapter and calls RiskGateway.",
    source="design-doc",
    domain="fintech",
    confidence=0.9,
)

results = store.query("How does PaymentsService interact with LedgerAdapter?", top_k=3)
for hit in results:
    print(f"{hit.record.record_id}  score={hit.score:.3f}  {hit.rationale}")
```

## 3. MCP server for agent integration

The MCP server gives any MCP-compatible client (Claude Code, Cursor, etc.) access to persistent long-term memory:

```bash
# Add to your project's .mcp.json
cat > .mcp.json << 'EOF'
{
  "mcpServers": {
    "context-fabrica": {
      "command": "context-fabrica-mcp",
      "args": ["--db", "./memory.db", "--namespace", "myproject"]
    }
  }
}
EOF
```

Once configured, agents can use `remember`, `recall`, `synthesize`, `promote`, `invalidate`, `supersede`, `related`, and `history` tools.

## 4. Temporal queries

Records can carry occurrence windows for time-scoped recall:

```python
store.ingest(
    "Quarterly incident review happened in June 2025.",
    source="incident",
    domain="platform",
    confidence=0.9,
)

results = store.query("What happened in June 2025?", top_k=3)
# The June record will score higher via temporal_match
```

## 5. Namespace policies

Use namespace policies when different teams or agents need different retrieval rules:

```python
from context_fabrica import HybridMemoryStore, NamespacePolicy, SQLiteRecordStore

store = HybridMemoryStore(
    store=SQLiteRecordStore("./memory.db"),
    namespace_policies={
        "production-ops": NamespacePolicy(
            min_confidence=0.8,
            source_allowlist=("runbook", "incident", "design-doc"),
            default_hops=1,
        )
    }
)
```

## 6. Postgres setup (production)

```bash
context-fabrica-bootstrap --dsn "postgresql:///context_fabrica"
context-fabrica-doctor --dsn "postgresql:///context_fabrica"
context-fabrica-demo --dsn "postgresql:///context_fabrica" --project
```

## 7. Inspect the projector queue

```bash
context-fabrica-projector --status
```
