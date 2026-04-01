# context-fabrica v2 Architecture

## Decision

`context-fabrica` v2 uses:
- `Postgres + pgvector` as the only write authority
- `Kuzu` as a graph projection for traversal-heavy reads

This keeps ingestion, provenance, freshness, invalidation, and embeddings in one system while letting graph reasoning specialize where it helps.

## Why This Split

### Postgres owns

- durable memory records
- semantic chunks + embeddings
- validity windows
- curation stage (`staged` / `canonical` / `pattern`)
- provenance and metadata
- typed relation rows
- promotion/invalidation history

### Kuzu owns

- projected entity graph
- relation traversal
- neighborhood expansion
- impact and dependency path queries

## Write Flow

1. Agent writes or updates a `KnowledgeRecord`.
2. A promotion policy classifies it into `staged`, `canonical`, or `pattern`.
3. Postgres stores the canonical record state and chunk embeddings.
4. Projection extracts entities and typed relations.
5. Kuzu receives projected `MemoryRecord`, `Entity`, `HAS_ENTITY`, and `RELATED` edges.

## Read Flow

1. Semantic retrieval starts in Postgres via `pgvector`.
2. `staged` records are filtered out unless explicitly requested.
3. Matching records expose entities/relations.
4. Kuzu expands nearby entities or paths when relation reasoning matters.
5. `context-fabrica` fuses semantic, graph, recency, and confidence signals.

## Why Postgres Remains Canonical

- one authoritative write path
- easier invalidation and replay
- easier backups and migrations
- avoids dual-write drift between vector and graph stores

## Kuzu Deferred Responsibilities

Kuzu is not the source of truth for raw text, provenance, or lifecycle policy. It is a projection optimized for graph reads.

## Initial Schema Contract

### Postgres tables

- `memory_records`
- `memory_chunks`
- `memory_relations`

Important fields added for curation:

- `memory_stage`
- `memory_kind`
- `reviewed_at`

### Kuzu tables

- `MemoryRecord`
- `Entity`
- `HAS_ENTITY`
- `RELATED`

## When This Architecture Is Worth It

- agents must retrieve relevant codebase knowledge and also explain impact chains
- multiple sessions need durable memory with governance
- relation-heavy questions are important enough to justify projection

## When To Avoid It

- semantic retrieval alone solves most of the problem
- you do not yet have stable entity or relation extraction
- you are still validating whether graph traversal matters in the product
