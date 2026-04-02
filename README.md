# context-fabrica

`context-fabrica` is an open-source memory substrate for agents that need durable knowledge of a codebase or work domain.

It combines:
- semantic retrieval (BM25-like lexical baseline),
- entity-relation graph traversal, and
- ranking policies (recency + confidence + relation proximity)

so agents can reason about both **what is relevant** and **how concepts are connected**.

## Status

- `v0.1`: in-process prototype engine is working and tested
- `v2 executable`: `Postgres + pgvector` write-authority model now supports live bootstrap, write, fetch, semantic search, promotion provenance, and projection job enqueueing
- local Postgres bootstrap and smoke test are working

This repo is currently best understood as an **architecture-first working prototype**: the core retrieval model works, the storage split is real, and the next step is wiring full live ingestion/query execution end to end.

## Current Direction

`context-fabrica` now ships two layers:
- an in-process prototype engine for local experimentation, and
- a v2 storage architecture that treats `Postgres + pgvector` as write authority and `Kuzu` as a read-optimized graph projection.

This keeps durable memory, provenance, freshness, and embeddings in Postgres while letting relation-heavy traversals move into a graph store when needed.

The Postgres side is now executable, not just declarative: the adapter can bootstrap schema, upsert records, replace chunks, replace relations, fetch records, and run semantic search against a live local Postgres instance.

Recent improvement inspired by `claude-scholar`:
- memory is now treated as **curated in layers**, not as one flat pool
- `staged` notes are excluded from default retrieval until promoted
- reusable extracted templates/patterns can live in a dedicated `pattern` tier
- a deterministic project-memory bootstrap/status script now exists for low-freedom repo setup

## Why This Exists

Engineering memory is not only about nearest text chunks. In real work, agents need relation-aware context:

- service dependencies,
- ownership and interfaces,
- runbook to code links,
- architectural decisions and their downstream effects.

Pure embedding search often misses this node-graph structure.

More importantly, agents need more than session recall. They need a memory layer that can answer:

- what is true about this system,
- where that fact came from,
- how it connects to other facts,
- and whether it is still valid.

`context-fabrica` is built around that distinction.

## What This Is Not

- not just a vector database wrapper
- not just chat history search
- not just user-personalization memory
- not a graph database bet disguised as an agent framework

The design keeps the canonical memory model separate from any one backend so the system can evolve without rewriting the agent-facing API.

## Design Principles

- `write authority first`: one canonical place to store facts, provenance, and freshness
- `relations are logical, not mandatory physical graph infra`: model relations early, choose graph storage only when needed
- `curation over accumulation`: not every agent output deserves canonical memory
- `promotion over raw accumulation`: durable knowledge should be curated, not just appended forever
- `freshness beats elegance`: stale facts are more damaging than imperfect ranking
- `agent-facing, not human-facing`: optimize for reliable autonomous reasoning, not just dashboards

## Quickstart

Verified package install:

```bash
python -m pip install .
```

Verified source-first setup for active development:

```bash
cd context-fabrica
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
pytest
```

Runtime dependencies for the full v2 path:

```bash
python -m pip install -r requirements-v2.txt
```

Single happy-path demo:

```bash
context-fabrica-demo --dsn "postgresql:///context_fabrica" --project
```

See also:
- `docs/getting-started.md`
- `examples/basic_memory.py`
- `examples/live_postgres_demo.py`

This command will:
- bootstrap the Postgres schema
- ingest one demo record with automatic chunking + embedding
- run semantic search
- optionally project pending jobs into Kuzu

Default embedding behavior:
- no extra dependency required -> dimension-safe local `HashEmbedder`
- if you configure `embedding_dimensions=384` and install `fastembed` manually -> `FastEmbed` is used automatically
- if you explicitly want sentence-transformers -> install it separately and pass your own embedder instance

For CLI users, the current practical split is:
- `python -m pip install .` -> base package + installed console scripts
- `python -m pip install -r requirements-v2.txt` -> Postgres/Kuzu runtime dependencies

## Python Usage

```python
from context_fabrica.engine import DomainMemoryEngine

engine = DomainMemoryEngine()
engine.ingest(
    "PaymentsService depends on LedgerAdapter and calls RiskGateway.",
    source="design-doc",
    domain="fintech",
    confidence=0.8,
)
engine.ingest(
    "LedgerAdapter writes transactions to event store.",
    source="runbook",
    domain="fintech",
)

results = engine.query("How does PaymentsService interact with LedgerAdapter?", top_k=3)
for hit in results:
    print(hit.record.record_id, hit.score, hit.rationale)
```

## CLI Usage

Input JSONL example:

```json
{"record_id":"r1","text":"AuthService uses TokenSigner and depends on KeyStore.","domain":"platform","source":"design"}
{"record_id":"r2","text":"TokenSigner rotates keys from KeyStore daily.","domain":"platform","source":"runbook"}
```

Run query:

```bash
context-fabrica --dataset sample.jsonl --query "How is TokenSigner connected to AuthService?" --top-k 5
```

## Where It Fits

`context-fabrica` is the knowledge plane.

It is a good fit underneath:
- coding agents that need durable codebase/domain memory
- orchestration systems that want a canonical memory backend
- control-plane UIs that need to inspect evidence, freshness, and relations

It is a poor fit as:
- a pure hosted chatbot memory feature
- a replacement for your agent runtime/orchestrator
- a generic BI or human-only knowledge portal

## Support Matrix

Currently verified in this repo:

- Python: `3.9`
- OS: `macOS` (local verification environment)
- Postgres: local Postgres with `pgvector` extension available
- Graph projection: local `Kuzu` runtime installed and projector exercised successfully

Known current boundaries:

- installed package path is verified for the base CLI surface
- full v2 runtime still expects `requirements-v2.txt` dependencies to be installed
- compatibility outside the verified environment should be treated as best-effort until broader CI coverage is added

## Method

See `docs/architecture.md` for full design.

Default hybrid score:

`0.50 * semantic + 0.30 * graph + 0.12 * recency + 0.08 * confidence`

## V2 Storage Architecture

- `Postgres + pgvector`: source of truth for records, chunks, embeddings, provenance, validity windows, and typed relation rows
- `Kuzu`: projected graph of entities and relation edges for multi-hop traversal
- `HybridMemoryStore`: composes both stores and emits bootstrap/write plans

Install runtime dependencies for the v2 stack:

```bash
python -m pip install -r requirements-v2.txt
```

Bootstrap and verify the local Postgres write-authority schema:

```bash
bash scripts/verify_local_postgres.sh
```

Python example:

```python
from context_fabrica import HybridMemoryStore, HybridStoreSettings, KuzuSettings, PostgresSettings
from context_fabrica.models import KnowledgeRecord

store = HybridMemoryStore(
    HybridStoreSettings(
        postgres=PostgresSettings(dsn="postgresql://localhost/context_fabrica"),
        kuzu=KuzuSettings(path="./var/context-fabrica-graph"),
    )
)

record = KnowledgeRecord(
    record_id="adr-12",
    text="AuthService depends on TokenSigner and calls KeyStore.",
    source="adr",
    domain="platform",
    confidence=0.9,
)

plan = store.write_plan(record)
```

Live Postgres example:

```python
from context_fabrica import HybridMemoryStore, HybridStoreSettings, KuzuSettings, PostgresSettings
from context_fabrica.models import KnowledgeRecord

store = HybridMemoryStore(
    HybridStoreSettings(
        postgres=PostgresSettings(dsn="postgresql:///context_fabrica"),
        kuzu=KuzuSettings(path="./var/context-fabrica-graph"),
    )
)

store.bootstrap_postgres()

record = KnowledgeRecord(
    record_id="auth-live-1",
    text="AuthService depends on TokenSigner and calls KeyStore.",
    source="design-doc",
    domain="platform",
    confidence=0.9,
)

embedding = [0.01] * 1536
store.write_record(record, chunks=[(record.text, embedding, 0)])
hits = store.semantic_search(embedding, domain="platform", top_k=3)
```

Auto chunking + embedding example:

```python
from context_fabrica import HybridMemoryStore, HybridStoreSettings, KuzuSettings, PostgresSettings
from context_fabrica.models import KnowledgeRecord

store = HybridMemoryStore(
    HybridStoreSettings(
        postgres=PostgresSettings(dsn="postgresql:///context_fabrica", embedding_dimensions=1536),
        kuzu=KuzuSettings(path="./var/context-fabrica-graph"),
    )
)

store.bootstrap_postgres()
store.write_text(
    KnowledgeRecord(
        record_id="service-auth-1",
        text="AuthService depends on TokenSigner and calls KeyStore. The service is owned by Platform.",
        source="design-doc",
        domain="platform",
        confidence=0.9,
    )
)
```

Promotion provenance example:

```python
store.promote_record("draft-note-1", reason="reviewed-by-agent")
```

Projection worker examples:

```bash
context-fabrica-projector --once
context-fabrica-projector --status
context-fabrica-projector --retry-failed
context-fabrica-projector --requeue-record live-auth-1
context-fabrica-projector --requeue-all-canonical
context-fabrica-projector --requeue-domain platform
```

Installed console-script examples (if your user script directory is on `PATH`):

```bash
context-fabrica-doctor --dsn "postgresql:///context_fabrica"
context-fabrica-bootstrap --dsn "postgresql:///context_fabrica"
context-fabrica-demo --dsn "postgresql:///context_fabrica" --project
context-fabrica-projector --status
```

Source-first equivalents:

```bash
PYTHONPATH=src python -m context_fabrica.bootstrap_cli --root . --dsn "postgresql:///context_fabrica"
PYTHONPATH=src python -m context_fabrica.doctor_cli --dsn "postgresql:///context_fabrica"
```

See `docs/v2-architecture.md` for the exact split between the databases.

Current v2 components in the repo:
- `HybridMemoryStore` for write-plan composition
- `PostgresPgvectorAdapter` for canonical storage schema and live execution methods
- `KuzuGraphProjectionAdapter` for relation projection statements
- `GraphProjectionWorker` for queued Postgres -> graph projection
- `policy.py` for staged/canonical/pattern routing and promotion decisions
- local Postgres bootstrap in `sql/postgres_bootstrap.sql`
- local smoke verification in `scripts/verify_local_postgres.sh`
- deterministic repo bootstrap/status tool in `scripts/project_memory.py` and `context-fabrica-project-memory`

## Curated Memory Tiers

The system now models three memory tiers:

- `staged`: draft or low-confidence notes that should not influence normal retrieval yet
- `canonical`: reviewed facts and workflow knowledge safe for agent-facing retrieval
- `pattern`: reusable mined patterns/templates worth preserving separately from ordinary facts

Typical lifecycle:

```text
raw observation -> staged -> reviewed/prompted -> canonical
repeated reusable structure -> mined -> pattern
```

The in-memory engine and Postgres search path both respect this distinction by filtering out staged memories by default.

Promotion provenance is stored in Postgres so canonicalization can be replayed and audited.

## Deterministic Project Memory Tooling

Inspired by claude-scholar's low-freedom project-memory scripts, `context-fabrica` now includes:

```bash
python scripts/project_memory.py bootstrap --root .
python scripts/project_memory.py status --root .
PYTHONPATH=src python -m context_fabrica.project_memory_cli bootstrap --root .
```

This creates a repo-local structure for:
- `memory/staging/`
- `memory/canonical/`
- `memory/patterns/`

and tracks it in `.context_fabrica/registry.json`.

The project ships with a portable baseline inspired by proven patterns used across open-source memory systems:
- weighted hybrid retrieval with RRF-style robustness,
- relation-aware graph expansion (1-2 hops),
- soft invalidation instead of hard deletes,
- recency and trust priors in the final rank.

## What Makes This Generic

- No dependency on one vector DB or one graph DB.
- In-memory defaults for easy local use and testing.
- Clear adapters path for LanceDB/FAISS/pgvector and Neo4j/Memgraph/Kuzu.
- Works for software, data, infra, research, and ops domains.

## Project Files For Launch

- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `docs/getting-started.md`
- `.github/workflows/ci.yml`

## Current Repository Layout

- `src/context_fabrica/engine.py`: in-memory prototype engine
- `src/context_fabrica/storage/`: v2 storage adapters and hybrid write-plan logic
- `sql/`: executable Postgres bootstrap and smoke-test SQL
- `docs/architecture.md`: baseline hybrid retrieval method
- `docs/v2-architecture.md`: canonical-store + projection architecture
- `tests/`: prototype and storage-plan tests

## Governance Primitives Included

- `valid_from` / `valid_to` per memory record
- `invalidate_record()` for soft deletion and supersession
- `stage` / `kind` for promotion routing and curated retrieval
- `reviewed_at` for promotion auditability
- `confidence` as a trust prior
- provenance fields (`source`, `metadata`) for future policy gates

## Verification

Current checks that pass in this repo:

- `python3 -m pytest`
- local `Postgres 18 + pgvector` schema bootstrap
- local smoke insert/query for records, chunks, and relation rows
- live Python integration test for bootstrap + write + fetch + semantic search
- chunking and embedding unit tests
- projection worker unit tests
- staged-memory promotion policy tests
- deterministic project-memory bootstrap/status script tests
- static diagnostics with zero Python errors in `src/` and `tests/`

## Research References That Shaped v0.1

- GraphRAG pipeline architecture: `https://microsoft.github.io/graphrag/index/architecture/`
- Graphiti hybrid retrieval + temporal edges: `https://github.com/getzep/graphiti`
- Neo4j hybrid retriever: `https://github.com/neo4j/neo4j-graphrag-python`
- Mem0 soft-invalidation patterns: `https://github.com/mem0ai/mem0`
- Elasticsearch RRF reference: `https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion`
- NetworkX personalized PageRank: `https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html`

## Next Milestones

1. Wire real Postgres read/write execution into the package API
2. Add projector state observability and retry controls
3. Add promotion review queues and agent-assisted conflict handling
4. Add conflict handling (`supersedes`, contradiction sets, as-of queries)
5. Add weighted-RRF and calibrated fusion modes
6. Add tenant-aware namespaces and memory lifecycle policies (TTL/decay/archival)

## Roadmap

- Pluggable embedding adapters (LanceDB, FAISS, pgvector)
- Pluggable graph adapters (Neo4j, Memgraph)
- Memory governance (staleness, contradiction resolution, source trust)
- Continuous learning loops from agent outcomes

## License

MIT
