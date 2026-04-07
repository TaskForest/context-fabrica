# Changelog

## 1.0.0

### Breaking Changes
- **`DomainMemoryEngine` removed.** Use `HybridMemoryStore` directly instead. Migration: replace `DomainMemoryEngine()` with `HybridMemoryStore(store=SQLiteRecordStore(":memory:"))` (or a persistent path). All methods (`ingest`, `query`, `related_records`, `invalidate_record`, `supersede_record`, `synthesize_observation`, `promote_record`, `supersession_chain`) are now on `HybridMemoryStore`.
- **Unified architecture**: `HybridMemoryStore` is now the single entry point for both storage and scoring. BM25 and graph indexes live in-memory but bootstrap lazily from the persistent store. Embedding search delegates to the store backend (SQLite cosine or pgvector HNSW).

### Added
- `Extractor` protocol and `ExtractionResult` model for pluggable knowledge extraction
- `PythonASTExtractor` — zero-dep Python AST extractor (classes, functions, imports, calls, inheritance)
- `HybridMemoryStore.extract_and_ingest()` for extraction-to-memory pipeline
- `context-fabrica-extract` CLI for extracting knowledge from source code
- `context-fabrica-install` CLI with multi-platform support: Claude Code, Codex, OpenCode, OpenClaw, Factory Droid
- `AGENTS.md` for Codex/OpenCode/OpenClaw agent instructions
- `.factory/droids/context-fabrica.md` for Factory Droid
- `ScoringPipeline` class extracted from engine for composable multi-signal scoring
- `HybridMemoryStore.ingest()` and `HybridMemoryStore.query()` with full scoring pipeline
- `HybridMemoryStore.related_records()`, `invalidate_record()`, `supersede_record_by_text()`, `synthesize_observation()`
- `list_all_texts()` and `list_all_relations()` on `RecordStore` protocol for BM25/graph bootstrap
- MCP server `--dsn` flag for Postgres backends
- zero-dependency MCP server (`context-fabrica-mcp`) for agent tool discovery over stdio
- `related` and `history` MCP tools for graph exploration and supersession chain inspection
- Claude Code slash commands: `/remember`, `/recall`, `/synthesize`, `/memory-status`
- `.mcp.json` project config for drop-in Claude Code integration

### Changed
- MCP server uses `HybridMemoryStore` directly — no hydration step, lazy BM25/graph bootstrap
- All scoring features (temporal, reranking, namespace policies) now work with any storage backend

## 0.4.0

### Added
- temporal occurrence fields on `KnowledgeRecord` plus time-scoped retrieval
- explicit provenance-backed observation synthesis via `synthesize_observation()`
- namespace retrieval policies with confidence floors, source allowlists, hop defaults, and rerank controls
- opt-in second-stage reranking with a default deterministic `TokenOverlapReranker`
- self-contained test support via local `mocker` fixture fallback

### Changed
- hybrid scoring now includes a temporal signal when a query or record carries time information
- SQLite and Postgres adapters persist occurrence windows
- pytest path configuration now works from a plain repository checkout

## 0.3.0

### Added
- in-memory hybrid retrieval engine
- Postgres + pgvector canonical store
- Kuzu projection worker and queue controls
- staged/canonical/pattern memory tiers
- bootstrap, demo, doctor, projector, and project-memory CLIs
- verified package install path and source-first runtime path
- examples, getting-started guide, CI workflow, and release-facing docs
- projector replay controls and health checks
