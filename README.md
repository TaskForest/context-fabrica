<div align="center">

# context-fabrica

**A governed memory layer built specifically for coding knowledge—giving agents durable recall, relation awareness, and evidence they can explain.**

Hybrid retrieval, graph reasoning, temporal recall, provenance-backed synthesis, and policy controls in one composable library.

[![CI](https://github.com/TaskForest/context-fabrica/actions/workflows/ci.yml/badge.svg)](https://github.com/TaskForest/context-fabrica/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Getting Started](docs/getting-started.md) | [Architecture](docs/architecture.md) | [Examples](examples/) | [Contributing](CONTRIBUTING.md) | [Releasing](RELEASING.md)

</div>

---

## The Problem

Flat vector memory fails autonomous coding agents. Without the ability to trace provenance, relate concepts across a codebase, or understand temporal validity, agents cannot distinguish between deeply vetted architectural rules and outdated drafts. Session recall isn't enough—agents need memory they can govern, trust, and safely update over time.

## What context-fabrica Does

`context-fabrica` is a composable library that combines semantic retrieval, graph traversal, and temporal recall so agents can genuinely reason about their memory. By treating provenance, validity, curation stage, and supersession as first-class data rather than afterthought metadata, it enables agents to justify their answers rather than just retrieving vaguely similar text.

```
Query: "How does PaymentsService interact with LedgerAdapter?"

  Semantic score ──── 0.72  (embedding similarity + BM25 lexical boost)
  Graph score ─────── 0.85  (2-hop traversal: PaymentsService → depends_on → LedgerAdapter)
  Temporal score ──── 0.00  (not a time-scoped query)
  Recency score ───── 0.91  (ingested 3 hours ago)
  Confidence score ── 0.80  (from design-doc source)
                      ────
  Final score ─────── 0.66  (hybrid weighted fusion)
  Rationale: [semantic_match, graph_relation, recent, high_confidence]
```

Every query returns **scored results with full breakdowns** — your agents can reason about *why* a memory was relevant, not just *that* it was.

---

## Perfect for Self-Learning & Self-Improving Agents

`context-fabrica` was designed to give coding agents the scaffolding they need to reason about their own growth and incrementally improve over time:

- **Observation Synthesis:** Agents can piece together raw, disparate facts over time and explicitly *synthesize* them into foundational insights (e.g., "The auth framework in this repo is unstable around token refresh").
- **Memory Tiers & Promotion:** "I saw something once" is not "This is a factual rule." Agents can log low-confidence observations in a `staged` tier. If the observation proves true repeatedly, it gets promoted to `canonical` or extracted as a reusable `pattern` applied to future tasks.
- **Supersession & Error Correction:** When an agent realizes a past assumption was incorrect, it uses supersession chains (soft invalidation). This means it remembers both what it *previously* thought and *why* it updated its understanding, rather than destructively overwriting past knowledge.
- **Temporal Recall:** Agents can explicitly time-scope memories to avoid confusing legacy system knowledge with the current state of a codebase.

---

## Core vs Extensible

context-fabrica separates **what is core** (the retrieval model, memory semantics, and governance) from **what is pluggable** (storage backends, embedders, and entity extraction).

```
  ┌──────────────────────────────────────────────────────────┐
  │                      CORE (fixed)                        │
  │                                                          │
  │  HybridMemoryStore         Hybrid scoring formula         │
  │  KnowledgeRecord model     Memory tiers & promotion      │
  │  Validity windows          Provenance tracking           │
  │  Temporal recall           Namespace policies            │
  │  BM25 lexical index        Knowledge graph traversal     │
  └──────────────────────────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
  ┌──────────────┐ ┌───────────┐ ┌──────────────┐
  │ RecordStore  │ │ Embedder  │ │ GraphStore   │
  │  (protocol)  │ │ (protocol)│ │  (protocol)  │
  └──────┬───────┘ └─────┬─────┘ └──────┬───────┘
         │               │              │
   ┌─────┴─────┐   ┌─────┴─────┐  ┌────┴─────┐
   │  SQLite   │   │   Hash    │  │   Kuzu   │
   │  Postgres │   │ FastEmbed │  │  Neo4j*  │
   │  Custom   │   │ Sentence  │  │  Custom  │
   └───────────┘   │ Transformr│  └──────────┘
                   │  Custom   │    * planned
                   └───────────┘
```

**Core** — the retrieval model, ranking formula, memory lifecycle, and governance primitives. These define what context-fabrica *is* and are not meant to be swapped out.

**Extensible** — storage backends, embedding providers, and graph stores are pluggable via Python protocols. Implement the interface, pass it in.

---

## Storage Options

Pick the backend that matches your scale. No code changes needed — the `HybridMemoryStore` API is the same regardless of backend.

| Backend | Dependencies | Server required? | Best for |
|---------|-------------|-----------------|----------|
| **SQLite** (built-in) | None (stdlib) | No | Local dev, single-agent, getting started |
| **Postgres + pgvector** | `psycopg`, `pgvector` | Yes | Production, multi-agent, teams |
| **Kuzu** (optional add-on) | `kuzu` | No | Graph-heavy traversal at scale |
| **Custom** | You decide | You decide | Bring your own (LanceDB, DuckDB, etc.) |

### SQLite — zero setup, no server

```bash
pip install context-fabrica
```

```python
from context_fabrica import HybridMemoryStore, SQLiteRecordStore

store = HybridMemoryStore(store=SQLiteRecordStore("./memory.db"))
store.bootstrap()

# Same API as Postgres — write, query, promote, search
store.write_text(record)
results = store.semantic_search(query_embedding, top_k=5)
```

SQLite stores records, chunks, embeddings, relations, and promotions in a single file. Semantic search uses brute-force cosine similarity — fast enough for local dev and single-agent workloads up to ~50k records.

### Postgres + pgvector — production scale

```bash
pip install "context-fabrica[postgres,kuzu,fastembed]"
```

If you are working from a local clone instead of PyPI:

```bash
python -m pip install .
python -m pip install -r requirements-v2.txt
```

```python
from context_fabrica import HybridMemoryStore, HybridStoreSettings, PostgresSettings, KuzuSettings

store = HybridMemoryStore(
    HybridStoreSettings(
        postgres=PostgresSettings(dsn="postgresql:///context_fabrica"),
        kuzu=KuzuSettings(path="./var/graph"),
    )
)
store.bootstrap()
```

Postgres handles records, chunks, HNSW-indexed vector search, validity windows, and provenance. Kuzu is optional — if you don't need multi-hop graph traversal at scale, skip it.

### Postgres without Kuzu

```python
from context_fabrica import HybridMemoryStore
from context_fabrica.storage.postgres import PostgresPgvectorAdapter

store = HybridMemoryStore(
    store=PostgresPgvectorAdapter(PostgresSettings(dsn="postgresql:///context_fabrica"))
)
store.bootstrap()
# No graph projection — relations still stored in Postgres, just no Kuzu traversal
```

### Bring your own backend

Implement the `RecordStore` protocol and pass it in:

```python
from context_fabrica.adapters import RecordStore

class MyLanceDBStore:
    """Implements RecordStore protocol."""
    def bootstrap(self) -> None: ...
    def upsert_record(self, record: KnowledgeRecord) -> None: ...
    def fetch_record(self, record_id: str) -> KnowledgeRecord | None: ...
    def replace_chunks(self, record_id: str, chunks: list) -> None: ...
    def replace_relations(self, record_id: str, relations: list) -> None: ...
    def record_promotion(self, source_id: str, target_id: str, reason: str, promoted_at: datetime) -> None: ...
    def semantic_search(self, query_embedding: list[float], *, domain: str | None, top_k: int) -> list[QueryResult]: ...
    def enqueue_projection(self, record_id: str) -> None: ...

store = HybridMemoryStore(store=MyLanceDBStore(), graph=MyGraphStore())  # graph is optional
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid retrieval** | Embedding cosine similarity + BM25 lexical boost + graph traversal, fused into one score |
| **Temporal retrieval** | Time-aware recall for queries like "what happened in June 2025?" |
| **Knowledge graph** | Entity-relation extraction with multi-hop traversal (configurable depth) |
| **Curated memory tiers** | `staged` (draft) -> `canonical` (reviewed) -> `pattern` (reusable) |
| **Observation synthesis** | Explicitly synthesize provenance-backed observation records from multiple facts |
| **Soft invalidation** | Validity windows (`valid_from`/`valid_to`) instead of hard deletes |
| **Promotion provenance** | Track when, why, and by whom records were promoted |
| **Namespace policies** | Per-namespace retrieval controls for hops, confidence floor, source allowlists, and reranking |
| **Knowledge extraction** | Built-in Python AST extractor (zero-dep); pluggable `Extractor` protocol for custom or third-party extractors |
| **Caller-provided extraction** | Pass your own entities and relations from an upstream LLM — or use built-in heuristics |
| **Optional reranking** | Add a second-stage reranker on top of hybrid or RRF retrieval when precision matters |
| **Scoring modes** | `hybrid` (default), `embedding`-only, `bm25`-only, or `rrf` |
| **Pluggable storage** | SQLite (built-in), Postgres + pgvector, or bring your own via `RecordStore` protocol |
| **Pluggable embedders** | HashEmbedder (zero-dep), FastEmbed, SentenceTransformers, or bring your own via `Embedder` protocol |
| **Optional graph store** | Kuzu ships as default, but graph projection is fully optional |
| **Framework-agnostic** | Not locked to LangChain, CrewAI, or any orchestrator |

## Quick Start

Install from PyPI:

```bash
python -m pip install "context-fabrica[postgres,kuzu,fastembed]"
```

Then bootstrap and verify:

```bash
context-fabrica-bootstrap --dsn "postgresql:///context_fabrica"
context-fabrica-doctor --dsn "postgresql:///context_fabrica"
context-fabrica-demo --dsn "postgresql:///context_fabrica" --project
```

```python
from context_fabrica import HybridMemoryStore, NamespacePolicy, ScoringWeights, SQLiteRecordStore, TokenOverlapReranker
from context_fabrica.models import Relation

store = HybridMemoryStore(
    store=SQLiteRecordStore("./memory.db"),
    reranker=TokenOverlapReranker(),
    namespace_policies={
        "payments": NamespacePolicy(
            min_confidence=0.75,
            source_allowlist=("design-doc", "runbook"),
            rerank_top_n=5,
        )
    },
)
store.bootstrap()

# Ingest with automatic entity/relation extraction
store.ingest(
    "PaymentsService depends on LedgerAdapter and calls RiskGateway.",
    source="design-doc",
    domain="fintech",
    confidence=0.8,
)

# Or provide your own entities/relations (e.g. from an upstream LLM)
store.ingest(
    "The auth service validates tokens before routing to the API gateway.",
    source="architecture-review",
    domain="platform",
    confidence=0.9,
    entities=["auth_service", "api_gateway", "token_validator"],
    relations=[
        Relation("auth_service", "calls", "api_gateway"),
        Relation("auth_service", "uses", "token_validator"),
    ],
)

# Query with full score breakdown
results = store.query("How does PaymentsService interact with LedgerAdapter?", top_k=3)
for hit in results:
    print(f"{hit.record.record_id}  score={hit.score:.2f}  {hit.rationale}")

# Temporal recall — occurrence window is inferred from text
incident = store.ingest(
    "Quarterly incident review happened in June 2025.",
    source="incident",
    domain="platform",
    confidence=0.9,
    record_id="incident-june",
)
time_scoped = store.query("What happened in June 2025?", top_k=3)

# Observation synthesis — combine multiple facts into one provenance-backed record
store.ingest("AuthService depends on TokenSigner.", record_id="f1", confidence=0.8)
store.ingest("TokenSigner rotates keys daily.", record_id="f2", confidence=0.9)
observation = store.synthesize_observation(["f1", "f2"], record_id="obs-1")
assert observation.metadata["derived_from"] == ["f1", "f2"]
```

## Architecture

```
                    +------------------+
                    |   Agent / CLI    |
                    +--------+---------+
                             |
                    +--------v---------+
                    | HybridMemoryStore |
                    | (unified engine)  |
                    +--------+---------+
                             |
         +----------+--------+--------+----------+
         |          |                 |           |
  +------v---+ +---v--------+ +-----v------+ +--v--------+
  | Embedding| | BM25       | | Knowledge  | | Temporal  |
  | (store)  | | (in-memory)| | Graph      | | Overlap   |
  +------+---+ +------------+ | (in-memory)| +--+--------+
         |                     +-----+------+    |
   SQLite or                  multi-hop BFS  occurrence
   pgvector                    with decay    windows
```

**Scoring formula (default weights, normalized to sum to 1.0):**
`0.42 * semantic + 0.25 * graph + 0.15 * temporal + 0.10 * recency + 0.07 * confidence`

Where semantic = `0.70 * embedding + 0.30 * BM25` in hybrid mode.
Weights are always normalized at query time, so custom values don't need to sum to 1.0.

Temporal scoring is neutral unless the query or record carries time information.

### Namespace Policies

Use namespace policies when one team or agent needs stricter retrieval than another without forking the engine:

```python
from context_fabrica import HybridMemoryStore, NamespacePolicy, ScoringWeights, SQLiteRecordStore

store = HybridMemoryStore(
    store=SQLiteRecordStore("./memory.db"),
    namespace_policies={
        "production-ops": NamespacePolicy(
            weights=ScoringWeights(semantic=0.45, graph=0.25, temporal=0.25, recency=0.10, confidence=0.10),
            min_confidence=0.8,
            source_allowlist=("runbook", "incident", "design-doc"),
            default_hops=1,
            rerank_top_n=8,
        )
    }
)
```

### Persistent Storage

```
  Agent
    |
    v
  HybridMemoryStore ─────── same API regardless of backend
    |
    ├── RecordStore (protocol)
    │     ├── SQLiteRecordStore     ← zero setup, single file
    │     ├── PostgresPgvectorAdapter ← production, HNSW indexing
    │     └── YourCustomAdapter     ← implement the protocol
    │
    └── GraphStore (protocol, optional)
          ├── KuzuGraphProjectionAdapter ← embedded graph
          └── YourCustomGraph            ← implement the protocol
```

When using Postgres, the projection worker uses **LISTEN/NOTIFY** for low-latency graph projection job pickup with polling fallback.

## Memory Tiers

Not every agent output deserves canonical memory. context-fabrica models three tiers:

```
raw observation ──> staged ──> reviewed ──> canonical
repeated pattern ──> mined ──> pattern
```

| Tier | Purpose | In default retrieval? |
|------|---------|----------------------|
| `staged` | Draft notes, low-confidence observations | No |
| `canonical` | Reviewed facts, trusted knowledge | Yes |
| `pattern` | Reusable templates and extracted patterns | Yes |

```python
# Low-confidence notes are auto-staged
draft = store.ingest("TODO: investigate flaky auth refresh", confidence=0.4)
assert draft.stage == "staged"  # excluded from default queries

# Promote after review
store.promote_record(draft.record_id)  # now canonical, queryable
```

## Embedder Options

| Embedder | Dimensions | Dependencies | Quality |
|----------|-----------|-------------|---------|
| `HashEmbedder` (default) | 1536 | None | Deterministic hashing, good for dev/testing |
| `FastEmbedEmbedder` | 384 | `fastembed` | Lightweight ML, good balance |
| `SentenceTransformerEmbedder` | 384+ | `sentence-transformers` | Production-quality semantic similarity |

```python
from context_fabrica import HybridMemoryStore, SQLiteRecordStore, SentenceTransformerEmbedder

# Production setup with real embeddings
store = HybridMemoryStore(
    store=SQLiteRecordStore("./memory.db"),
    embedder=SentenceTransformerEmbedder(),
    scoring="hybrid",
)
store.bootstrap()
```

## Agent Platform Support

context-fabrica works with any MCP-compatible agent platform. One command sets up everything:

```bash
pip install context-fabrica
context-fabrica-install                      # auto-detects your platform
context-fabrica-install --platform codex     # or specify explicitly
context-fabrica-install --all                # install for all platforms
```

| Platform | What gets created | Status |
|----------|------------------|--------|
| **Claude Code** | `.mcp.json` + `.claude/commands/` slash commands | Supported |
| **Codex (OpenAI)** | `AGENTS.md` + `~/.codex/config.toml` MCP config | Supported |
| **OpenCode** | `AGENTS.md` + `opencode.json` MCP config | Supported |
| **OpenClaw** | `AGENTS.md` + `~/.openclaw/config.json` MCP config | Supported |
| **Factory Droid** | `AGENTS.md` + `.factory/droids/context-fabrica.md` | Supported |

All platforms use the same `context-fabrica-mcp` server over stdio. The installer writes the platform-specific config so the agent discovers the tools automatically.

## MCP Server (Model Context Protocol)

The MCP server runs locally as a subprocess — no hosting required. It exposes 8 tools over JSON-RPC 2.0 via stdio.

You can also configure it manually if you prefer:

```json
{
  "mcpServers": {
    "context-fabrica": {
      "command": "context-fabrica-mcp",
      "args": ["--db", "./memory.db", "--namespace", "myproject"]
    }
  }
}
```

**Available tools:**

| Tool | Description |
|------|-------------|
| `remember` | Store a fact, observation, or insight in long-term memory |
| `recall` | Search memory for relevant knowledge with scored results |
| `synthesize` | Combine multiple facts into a provenance-backed observation |
| `promote` | Promote a staged draft memory to canonical status |
| `invalidate` | Soft-delete a memory that is no longer valid |
| `supersede` | Replace an existing memory with an updated version |
| `related` | Find graph-connected records to explore how concepts link together |
| `history` | Show the supersession chain — how a fact evolved over time |

Once configured, your agent can use these tools directly:

```
Agent: "Let me check if I already know about the auth service architecture."
→ calls recall("auth service architecture")

Agent: "I learned that TokenSigner rotates keys daily. Let me remember that."
→ calls remember("TokenSigner rotates keys daily", source="code-review", confidence=0.9)
```

## Claude Code Skills

If you're using Claude Code, context-fabrica includes slash commands that wrap the MCP tools:

| Command | Description |
|---------|-------------|
| `/remember <text>` | Store knowledge in memory |
| `/recall <query>` | Search memory for relevant facts |
| `/synthesize <ids or topic>` | Synthesize an observation from multiple facts |
| `/memory-status` | Overview of what's stored in memory |

These commands are defined in `.claude/commands/` and work automatically when the MCP server is configured.

## Knowledge Extraction

context-fabrica can automatically extract knowledge from source code and ingest it into governed memory. Extracted knowledge is persisted, queryable with full multi-signal scoring, and evolves over time via supersession chains.

### Built-in Python extractor

```bash
# Extract from a codebase into persistent memory
context-fabrica-extract ./src --db ./memory.db --namespace myproject
```

The built-in `PythonASTExtractor` uses the stdlib `ast` module (zero external dependencies) to extract:
- Classes with inheritance, decorators, and methods
- Functions with parameters, docstrings, and call targets
- Import relationships
- Call graphs (function A calls function B)

All extraction is deterministic and free — no LLM tokens consumed.

```python
from context_fabrica import HybridMemoryStore, PythonASTExtractor, SQLiteRecordStore

store = HybridMemoryStore(store=SQLiteRecordStore("./memory.db"))
store.bootstrap()

# Extract and ingest — knowledge is persisted and queryable
records = store.extract_and_ingest("./src", PythonASTExtractor(), namespace="myproject")

# Agent queries the extracted knowledge with multi-signal scoring
results = store.query("How does AuthService use TokenSigner?", namespace="myproject", top_k=3)
```

### Building a custom extractor

Implement the `Extractor` protocol to support any language, document format, or extraction strategy:

```python
from pathlib import Path
from context_fabrica import Extractor, ExtractionResult
from context_fabrica.models import Relation

class TypeScriptExtractor:
    """Custom extractor for TypeScript projects."""

    def extract(self, path: Path) -> list[ExtractionResult]:
        results = []
        for ts_file in path.rglob("*.ts"):
            # Your extraction logic here (tree-sitter, regex, LLM, etc.)
            results.append(ExtractionResult(
                text="UserService handles authentication and session management.",
                source=str(ts_file),
                entities=["UserService", "SessionManager"],
                relations=[Relation("UserService", "depends_on", "SessionManager")],
                confidence=0.85,
                domain="auth",
                tags=["typescript", "ast-extracted"],
                metadata={"language": "typescript", "source_file": str(ts_file)},
            ))
        return results

# Use it exactly like the built-in extractor
store.extract_and_ingest("./src", TypeScriptExtractor(), namespace="frontend")
```

### Using third-party extractors

Any tool that can produce `ExtractionResult` objects works. For example, a Graphify adapter:

```python
import json
from pathlib import Path
from context_fabrica import ExtractionResult
from context_fabrica.models import Relation

class GraphifyAdapter:
    """Import knowledge from Graphify's graph.json output."""

    def extract(self, path: Path) -> list[ExtractionResult]:
        graph_data = json.loads((path / "graphify-out" / "graph.json").read_text())
        # Convert Graphify nodes/edges to ExtractionResults
        ...
```

## CLI

```bash
# Install for your agent platform
context-fabrica-install                      # auto-detect
context-fabrica-install --platform codex     # explicit
context-fabrica-install --all                # all platforms

# Extract knowledge from source code
context-fabrica-extract ./src --db ./memory.db --namespace myproject

# Query from JSONL dataset
context-fabrica --dataset records.jsonl --query "How is TokenSigner connected?" --top-k 5

# MCP server (usually started by the client, but can run manually)
context-fabrica-mcp --db ./memory.db --namespace myproject

# Postgres operations
context-fabrica-bootstrap --dsn "postgresql:///context_fabrica"
context-fabrica-doctor --dsn "postgresql:///context_fabrica"
context-fabrica-demo --dsn "postgresql:///context_fabrica" --project

# Projection worker
context-fabrica-projector --once            # process pending jobs
context-fabrica-projector --status          # queue summary
context-fabrica-projector --retry-failed    # requeue failed jobs

# Project memory bootstrap
context-fabrica-project-memory bootstrap --root .
```

## Where It Fits

**Good fit:**
- Coding agents that need durable codebase/domain memory
- Multi-agent systems that share a canonical knowledge layer
- Orchestration systems wanting inspectable, auditable memory
- Control-plane UIs that need evidence, freshness, and relation visibility

**Not a fit:**
- Pure chatbot session memory
- Replacement for your agent runtime/orchestrator
- Generic BI or human-only knowledge portal

## Governance Primitives

| Primitive | Purpose |
|-----------|---------|
| `valid_from` / `valid_to` | Temporal validity windows, enables as-of queries |
| `occurred_from` / `occurred_to` | Event-time windows for time-scoped recall |
| `invalidate_record()` | Soft deletion with reason tracking |
| `stage` / `kind` | Promotion routing and curated retrieval |
| `reviewed_at` | Promotion auditability |
| `confidence` | Trust prior in ranking |
| `source` / `metadata` | Provenance for policy gates |
| `namespace` / `NamespacePolicy` | Tenant isolation plus namespace-specific retrieval controls |
| `supersedes` | Record replacement chains |

## Project Structure

```
src/context_fabrica/
  models.py          # KnowledgeRecord, Relation, QueryResult (core)
  adapters.py        # RecordStore, GraphStore, Embedder, Reranker protocols (core)
  scoring.py         # ScoringPipeline: BM25 + graph + temporal + reranking (core)
  policy.py          # Memory tier routing and promotion (core)
  temporal.py        # Time-range extraction and temporal overlap scoring
  synthesis.py       # Provenance-backed observation synthesis
  reranking.py       # Optional second-stage rerankers
  mcp_server.py      # Zero-dependency MCP server over stdio
  entity.py          # Entity/relation extraction heuristics (core, bypassable)
  index.py           # BM25 lexical index (core)
  graph.py           # In-memory knowledge graph with BFS traversal (core)
  embedding.py       # Embedder adapters: Hash, FastEmbed, SentenceTransformer (pluggable)
  extract_cli.py     # CLI for knowledge extraction from source files
  install_cli.py     # Multi-platform installer (Claude Code, Codex, OpenCode, OpenClaw, Factory Droid)
  extractors/
    python_ast.py    # Built-in Python AST extractor (zero deps, stdlib only)
  storage/
    sqlite.py        # SQLite record store — zero deps (pluggable)
    postgres.py      # Postgres + pgvector adapter with LISTEN/NOTIFY (pluggable)
    kuzu.py          # Kuzu graph projection adapter (pluggable, optional)
    hybrid.py        # HybridMemoryStore — orchestrates any RecordStore + GraphStore
    projector.py     # Background projection worker
AGENTS.md            # Agent instructions for Codex, OpenCode, OpenClaw
.claude/commands/    # Claude Code slash commands (/remember, /recall, /synthesize, /memory-status)
.mcp.json            # MCP server configuration for Claude Code
.factory/droids/     # Factory Droid agent definition
tests/               # pytest suite, including live Postgres coverage
docs/                # Architecture docs and getting-started guide
examples/            # Runnable usage examples
sql/                 # Postgres bootstrap and smoke test SQL
```

## Development

```bash
git clone https://github.com/TaskForest/context-fabrica.git
cd context-fabrica
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## References

- [GraphRAG](https://microsoft.github.io/graphrag/index/architecture/) — pipeline architecture for graph-enhanced retrieval
- [Graphiti](https://github.com/getzep/graphiti) — hybrid retrieval with temporal edges
- [Neo4j GraphRAG](https://github.com/neo4j/neo4j-graphrag-python) — hybrid graph retriever
- [Mem0](https://github.com/mem0ai/mem0) — soft-invalidation patterns
- [Elasticsearch RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) — reciprocal rank fusion

## License

MIT
