<div align="center">

# context-fabrica

**A hybrid memory substrate for AI agents that need durable, queryable knowledge.**

Semantic retrieval + knowledge graph traversal + curated memory tiers — in one library.

[![CI](https://github.com/TaskForest/context-fabrica/actions/workflows/ci.yml/badge.svg)](https://github.com/TaskForest/context-fabrica/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Getting Started](docs/getting-started.md) | [Architecture](docs/architecture.md) | [Examples](examples/) | [Contributing](CONTRIBUTING.md)

</div>

---

## The Problem

Most agent memory is flat vector search. That works for "find similar text" but fails when agents need to reason about **how concepts connect** — service dependencies, ownership chains, architectural decisions and their downstream effects.

Agents also need to know **where a fact came from**, **whether it's still valid**, and **how confident they should be** in it. Session recall isn't enough.

## What context-fabrica Does

```
Query: "How does PaymentsService interact with LedgerAdapter?"

  Semantic score ──── 0.72  (embedding similarity + BM25 lexical boost)
  Graph score ─────── 0.85  (2-hop traversal: PaymentsService → depends_on → LedgerAdapter)
  Recency score ───── 0.91  (ingested 3 hours ago)
  Confidence score ── 0.80  (from design-doc source)
                      ────
  Final score ─────── 0.81  (hybrid weighted fusion)
  Rationale: [semantic_match, graph_relation, recent, high_confidence]
```

Every query returns **scored results with full breakdowns** — your agents can reason about *why* a memory was relevant, not just *that* it was.

---

## Core vs Extensible

context-fabrica separates **what is core** (the retrieval model, memory semantics, and governance) from **what is pluggable** (storage backends, embedders, and entity extraction).

```
  ┌──────────────────────────────────────────────────────────┐
  │                      CORE (fixed)                        │
  │                                                          │
  │  DomainMemoryEngine        Hybrid scoring formula        │
  │  KnowledgeRecord model     Memory tiers & promotion      │
  │  Validity windows          Provenance tracking           │
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
pip install .
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
pip install -r requirements-v2.txt
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
| **Knowledge graph** | Entity-relation extraction with multi-hop traversal (configurable depth) |
| **Curated memory tiers** | `staged` (draft) -> `canonical` (reviewed) -> `pattern` (reusable) |
| **Soft invalidation** | Validity windows (`valid_from`/`valid_to`) instead of hard deletes |
| **Promotion provenance** | Track when, why, and by whom records were promoted |
| **Caller-provided extraction** | Pass your own entities and relations from an upstream LLM — or use built-in heuristics |
| **Scoring modes** | `hybrid` (default), `embedding`-only, or `bm25`-only |
| **Pluggable storage** | SQLite (built-in), Postgres + pgvector, or bring your own via `RecordStore` protocol |
| **Pluggable embedders** | HashEmbedder (zero-dep), FastEmbed, SentenceTransformers, or bring your own via `Embedder` protocol |
| **Optional graph store** | Kuzu ships as default, but graph projection is fully optional |
| **Framework-agnostic** | Not locked to LangChain, CrewAI, or any orchestrator |

## Quick Start

```python
from context_fabrica import DomainMemoryEngine
from context_fabrica.models import Relation

engine = DomainMemoryEngine()  # or DomainMemoryEngine(scoring="embedding")

# Ingest with automatic entity/relation extraction
engine.ingest(
    "PaymentsService depends on LedgerAdapter and calls RiskGateway.",
    source="design-doc",
    domain="fintech",
    confidence=0.8,
)

# Or provide your own entities/relations (e.g. from an upstream LLM)
engine.ingest(
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
results = engine.query("How does PaymentsService interact with LedgerAdapter?", top_k=3)
for hit in results:
    print(f"{hit.record.record_id}  score={hit.score:.2f}  {hit.rationale}")
```

## Architecture

```
                    +------------------+
                    |   Agent / CLI    |
                    +--------+---------+
                             |
                    +--------v---------+
                    | DomainMemoryEngine|
                    |  (in-process)     |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v------+  +----v-------+
     | Embedding  |  | BM25 Lexical|  | Knowledge  |
     | Similarity |  | Index       |  | Graph      |
     +------+-----+  +-------------+  +-----+------+
            |                                |
       (pluggable)                     multi-hop BFS
                                       with decay
```

**Scoring formula:**
`0.50 * semantic + 0.30 * graph + 0.12 * recency + 0.08 * confidence`

Where semantic = `0.70 * embedding + 0.30 * BM25` in hybrid mode.

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
draft = engine.ingest("TODO: investigate flaky auth refresh", confidence=0.4)
assert draft.stage == "staged"  # excluded from queries

# Promote after review
engine.promote_record(draft.record_id)  # now canonical, queryable
```

## Embedder Options

| Embedder | Dimensions | Dependencies | Quality |
|----------|-----------|-------------|---------|
| `HashEmbedder` (default) | 1536 | None | Deterministic hashing, good for dev/testing |
| `FastEmbedEmbedder` | 384 | `fastembed` | Lightweight ML, good balance |
| `SentenceTransformerEmbedder` | 384+ | `sentence-transformers` | Production-quality semantic similarity |

```python
from context_fabrica import DomainMemoryEngine, SentenceTransformerEmbedder

# Production setup with real embeddings
engine = DomainMemoryEngine(
    embedder=SentenceTransformerEmbedder(),
    scoring="hybrid",
)
```

## CLI

```bash
# Query from JSONL dataset
context-fabrica --dataset records.jsonl --query "How is TokenSigner connected?" --top-k 5

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
| `invalidate_record()` | Soft deletion with reason tracking |
| `stage` / `kind` | Promotion routing and curated retrieval |
| `reviewed_at` | Promotion auditability |
| `confidence` | Trust prior in ranking |
| `source` / `metadata` | Provenance for policy gates |
| `supersedes` | Record replacement chains |

## Project Structure

```
src/context_fabrica/
  engine.py          # In-process hybrid retrieval engine (core)
  models.py          # KnowledgeRecord, Relation, QueryResult (core)
  adapters.py        # RecordStore, GraphStore, Embedder protocols (core)
  policy.py          # Memory tier routing and promotion (core)
  entity.py          # Entity/relation extraction heuristics (core, bypassable)
  index.py           # BM25 lexical index (core)
  graph.py           # In-memory knowledge graph with BFS traversal (core)
  embedding.py       # Embedder adapters: Hash, FastEmbed, SentenceTransformer (pluggable)
  storage/
    sqlite.py        # SQLite record store — zero deps (pluggable)
    postgres.py      # Postgres + pgvector adapter with LISTEN/NOTIFY (pluggable)
    kuzu.py          # Kuzu graph projection adapter (pluggable, optional)
    hybrid.py        # HybridMemoryStore — orchestrates any RecordStore + GraphStore
    projector.py     # Background projection worker
tests/               # pytest suite (37 tests)
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

## Roadmap

- [x] Pluggable storage backends via `RecordStore` protocol
- [x] SQLite adapter for zero-setup persistent storage
- [x] Optional graph projection (Kuzu not required)
- [x] Caller-provided entity/relation extraction
- [x] Configurable scoring modes (hybrid, embedding, bm25)
- [x] Pluggable storage backends via `RecordStore` protocol
- [x] SQLite adapter for zero-setup persistent storage
- [x] Optional graph projection (Kuzu not required)
- [x] Caller-provided entity/relation extraction
- [x] Configurable scoring modes (hybrid, embedding, bm25, rrf)
- [x] Configurable hybrid ranking weights via `ScoringWeights`
- [x] Multi-tenant namespaces (per agent/team isolation)
- [x] Memory lifecycle policies (TTL, decay, purge)
- [x] Embedding dimension bootstrap migration (auto-resize on dimension change)
- [x] Conflict handling (supersession chains, `supersede_record()`)
- [x] Reciprocal Rank Fusion (RRF) scoring mode
- [x] Feedback loop primitives (`record_outcome()`, `outcome_summary()`)
- [ ] Pluggable graph adapters (Neo4j, Memgraph)
- [ ] Additional vector stores (LanceDB, FAISS)

## References

- [GraphRAG](https://microsoft.github.io/graphrag/index/architecture/) — pipeline architecture for graph-enhanced retrieval
- [Graphiti](https://github.com/getzep/graphiti) — hybrid retrieval with temporal edges
- [Neo4j GraphRAG](https://github.com/neo4j/neo4j-graphrag-python) — hybrid graph retriever
- [Mem0](https://github.com/mem0ai/mem0) — soft-invalidation patterns
- [Elasticsearch RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) — reciprocal rank fusion

## License

MIT
