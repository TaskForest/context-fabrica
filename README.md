<div align="center">

# context-fabrica

**A hybrid memory substrate for AI agents that need durable, queryable knowledge.**

Semantic retrieval + knowledge graph traversal + curated memory tiers — in one library.

[![CI](https://github.com/context-fabrica/context-fabrica/actions/workflows/ci.yml/badge.svg)](https://github.com/context-fabrica/context-fabrica/actions)
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
| **Zero mandatory deps** | Core engine runs on pure Python with `HashEmbedder`; plug in sentence-transformers or fastembed for real semantic similarity |
| **Framework-agnostic** | Not locked to LangChain, CrewAI, or any orchestrator |

## Quick Start

### Install

```bash
pip install .
```

For the full Postgres + Kuzu storage layer:

```bash
pip install -r requirements-v2.txt
```

### Basic Usage

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

### Persistent Storage (Postgres + Kuzu)

```python
from context_fabrica import HybridMemoryStore, HybridStoreSettings, KuzuSettings, PostgresSettings
from context_fabrica.models import KnowledgeRecord

store = HybridMemoryStore(
    HybridStoreSettings(
        postgres=PostgresSettings(dsn="postgresql:///context_fabrica"),
        kuzu=KuzuSettings(path="./var/graph"),
    )
)
store.bootstrap_postgres()

record = KnowledgeRecord(
    record_id="adr-12",
    text="AuthService depends on TokenSigner and calls KeyStore.",
    source="adr",
    domain="platform",
    confidence=0.9,
)

# Auto-chunks text, embeds, stores in Postgres, enqueues graph projection
store.write_text(record)
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
     +------------+  +-------------+  +-----+------+
                                             |
                                      multi-hop BFS
                                      with decay
```

**Scoring formula:**
`0.50 * semantic + 0.30 * graph + 0.12 * recency + 0.08 * confidence`

Where semantic = `0.70 * embedding + 0.30 * BM25` in hybrid mode.

### Production Storage (v2)

```
  Write path                          Read path
  ─────────                           ─────────
  Agent                               Agent
    |                                   |
    v                                   v
  Postgres + pgvector ──────────> Semantic search
    |  (source of truth)                |
    |                                   v
    +──> projection_jobs ──>  Kuzu (graph projection)
              |                         |
              v                         v
         LISTEN/NOTIFY ──>        Multi-hop traversal
         Projection Worker
```

**Postgres** is the single source of truth for records, chunks, embeddings, validity windows, and provenance. **Kuzu** is an optional read-optimized graph projection for relation-heavy traversals. The projection worker uses **LISTEN/NOTIFY** for low-latency job pickup with polling fallback.

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
  engine.py          # In-process hybrid retrieval engine
  models.py          # KnowledgeRecord, Relation, QueryResult
  policy.py          # Memory tier routing and promotion
  entity.py          # Entity/relation extraction (heuristic)
  index.py           # BM25 lexical index
  graph.py           # In-memory knowledge graph with BFS traversal
  embedding.py       # Embedder adapters (Hash, FastEmbed, SentenceTransformer)
  storage/
    postgres.py      # Postgres + pgvector adapter with LISTEN/NOTIFY
    kuzu.py          # Kuzu graph projection adapter
    hybrid.py        # Orchestrates Postgres + Kuzu writes
    projector.py     # Background projection worker
tests/               # pytest suite (25 tests)
docs/                # Architecture docs and getting-started guide
examples/            # Runnable usage examples
sql/                 # Postgres bootstrap and smoke test SQL
```

## Development

```bash
git clone https://github.com/context-fabrica/context-fabrica.git
cd context-fabrica
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## Roadmap

- [ ] Configurable hybrid ranking weights via settings
- [ ] Multi-tenant namespaces (per agent/team isolation)
- [ ] Pluggable graph adapters (Neo4j, Memgraph)
- [ ] Pluggable vector stores (LanceDB, FAISS)
- [ ] Memory lifecycle policies (TTL, decay, archival)
- [ ] Conflict handling (contradiction sets, supersession chains)
- [ ] Weighted-RRF and calibrated fusion modes
- [ ] Continuous learning loops from agent outcomes

## References

- [GraphRAG](https://microsoft.github.io/graphrag/index/architecture/) — pipeline architecture for graph-enhanced retrieval
- [Graphiti](https://github.com/getzep/graphiti) — hybrid retrieval with temporal edges
- [Neo4j GraphRAG](https://github.com/neo4j/neo4j-graphrag-python) — hybrid graph retriever
- [Mem0](https://github.com/mem0ai/mem0) — soft-invalidation patterns
- [Elasticsearch RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) — reciprocal rank fusion

## License

MIT
