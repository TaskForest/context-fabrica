# context-fabrica — Agent Memory

This project uses [context-fabrica](https://github.com/TaskForest/context-fabrica) for governed long-term memory. A persistent memory store is available via MCP tools.

## Available Tools

| Tool | When to use |
|------|------------|
| `remember` | Store a fact, architectural decision, debugging insight, or pattern you've learned |
| `recall` | Search memory before making assumptions — check if you already know something |
| `synthesize` | Combine multiple facts into a single provenance-backed observation |
| `promote` | Promote a draft/staged memory to canonical status after verifying it's true |
| `invalidate` | Soft-delete a memory that turned out to be wrong |
| `supersede` | Replace a memory with an updated version, preserving the correction history |
| `related` | Explore graph connections — find how concepts link together |
| `history` | See how a fact evolved over time via its supersession chain |

## Guidelines

- **Check before assuming.** Before answering questions about this codebase, use `recall` to search memory first. You may already know the answer.
- **Store what you learn.** When you discover something non-obvious (architecture decisions, gotchas, debugging insights), use `remember` to store it for future sessions.
- **Use confidence scores.** Set `confidence` lower (0.3-0.5) for uncertain observations and higher (0.8-1.0) for verified facts. Low-confidence records are auto-staged as drafts.
- **Correct mistakes.** If you find a stored fact is wrong, use `supersede` to replace it — don't just ignore it.
- **Synthesize patterns.** When you notice the same insight appearing across multiple facts, use `synthesize` to combine them into a single observation.

## Setup

If the MCP server is not already configured, run:

```bash
pip install context-fabrica
context-fabrica install
```

This registers the `context-fabrica-mcp` server with your agent platform automatically.
