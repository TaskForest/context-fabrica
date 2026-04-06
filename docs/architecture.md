# context-fabrica Architecture

## Goal

Provide a generic memory method for domain-specific agents that must reason over:
- semantic similarity (what text is relevant), and
- relation topology (how engineering concepts connect), and
- temporal scope (when a fact occurred, not just when it was ingested).

## Core Method

The MVP uses a **hybrid retrieval pipeline**:
1. **Semantic retrieval** via lexical BM25-like scoring (`LexicalSemanticIndex`).
2. **Graph retrieval** via multi-hop entity relation traversal (`KnowledgeGraph`).
3. **Temporal retrieval** via inferred or caller-provided occurrence windows.
4. **Policy ranking** to combine semantic + graph + temporal + recency + confidence.

Final score:

`score = 0.42 * semantic + 0.25 * graph + 0.15 * temporal + 0.10 * recency + 0.07 * confidence`

These are the default weights after normalization (raw defaults: 0.50, 0.30, 0.18, 0.12, 0.08).
Weights are always normalized at query time so custom values don't need to sum to 1.0.

Alternative production rankers (planned):

- Weighted RRF style fusion:
  `S(d)=sum(w_m/(k+rank_m(d))) + wt*temporal + wr*recency + wc*confidence`
- Calibrated additive fusion:
  `p(d)=sigmoid(b0 + bv*vec + bl*lex + bg*graph + bt*temporal + br*recency + bc*confidence)`

## Data Model

- `KnowledgeRecord`: memory unit (text, source, domain, confidence, tags, metadata, timestamp, optional occurrence window)
- `Relation`: directed typed edge (`source_entity`, `relation`, `target_entity`, `weight`)

## Ingestion Flow

1. Insert/Update record in semantic index.
2. Extract entities from text (capitalized tokens, snake_case, numeric/technical tokens).
3. Extract weak relations from co-occurrence and cues (`depends_on`, `uses`, `implements`, `owns`).
4. Infer an occurrence window when the text contains simple temporal cues, or accept caller-provided occurrence fields.
5. Attach entities to record and add graph edges.
6. Persist validity window (`valid_from`, optional `valid_to`) for future as-of queries.

Optional explicit synthesis:

- Combine related facts into a provenance-backed `observation` record.
- Store derived record IDs in metadata so observations remain inspectable and reversible.

## Query Flow

1. Score prompt against semantic index.
2. Extract query entities and traverse graph up to `hops`.
3. Infer a query time range when the prompt is time-scoped.
4. Normalize semantic/graph/temporal scores.
5. Add recency/confidence priors.
6. Apply namespace policy constraints when configured: hops, confidence floor, source allowlist, staged visibility.
7. Optionally rerank the top results with a second-stage reranker.
8. Filter invalid/superseded memories for current-time queries.
9. Return top-k records with rationale tags.

## Service-Oriented Blueprint (for OSS evolution)

- `ingest-service`: append records, normalize metadata, emit extraction jobs
- `extract-service`: entities, relations, contradiction candidates
- `retrieval-service`: lexical/vector retrieval + graph expansion + fusion
- `policy-service`: trust thresholds, source allowlists, namespace policies, conflict policy
- `synthesis-service`: promote repeated facts into observations or patterns with provenance
- `memory-api`: write/query/invalidate/as-of query endpoints

This package currently implements the core logic in-process to stay lightweight, but interfaces are shaped to split into services later.

## Why This Method Generalizes

- Works without a specific vector DB or graph DB.
- Can be upgraded to embeddings and persistent graph backends with no API changes.
- Separation of concerns allows environment-specific storage adapters.

## MCP Server

The MCP server (`context-fabrica-mcp`) exposes `HybridMemoryStore` over JSON-RPC 2.0 via stdio:

```
Agent (Claude Code, Cursor, etc.)
  | stdin/stdout (JSON-RPC 2.0)
context-fabrica-mcp
  | in-process calls
HybridMemoryStore (unified engine)
  | embedding search             | BM25 + graph (in-memory)
SQLite or Postgres (durable)     ScoringPipeline (lazy bootstrap)
```

BM25 and graph indexes bootstrap lazily from the store on first query.
Embedding search delegates to the store (SQLite cosine or pgvector HNSW).
Every mutation persists immediately — no hydration step needed.

Tools: `remember`, `recall`, `synthesize`, `promote`, `invalidate`, `supersede`, `related`, `history`.

## Extension Points

- Replace lexical index with embedding store (FAISS/LanceDB/pgvector).
- Replace in-memory graph with Neo4j, Memgraph, or NetworkX-backed persistence.
- Replace deterministic reranking with a cross-encoder or hosted reranker.
- Add richer temporal extraction beyond the current lightweight parser.
- Add source verification and conflict detection for trust calibration.
- Add memory lifecycle policies (decay, consolidation, archival).

## References

- GraphRAG architecture: `https://microsoft.github.io/graphrag/index/architecture/`
- Graphiti repository: `https://github.com/getzep/graphiti`
- Neo4j GraphRAG hybrid retriever: `https://github.com/neo4j/neo4j-graphrag-python`
- Mem0 graph memory patterns: `https://github.com/mem0ai/mem0`
- Elasticsearch RRF docs: `https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion`
