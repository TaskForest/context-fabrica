from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

MemoryStage = Literal["staged", "canonical", "pattern"]
MemoryKind = Literal["fact", "workflow", "pattern", "note"]


@dataclass
class KnowledgeRecord:
    record_id: str
    text: str
    source: str = "unknown"
    domain: str = "global"
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    confidence: float = 0.6
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    valid_from: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    valid_to: datetime | None = None
    supersedes: str | None = None
    stage: MemoryStage = "canonical"
    kind: MemoryKind = "fact"
    reviewed_at: datetime | None = None


@dataclass
class Relation:
    source_entity: str
    relation: str
    target_entity: str
    weight: float = 1.0


@dataclass
class QueryResult:
    record: KnowledgeRecord
    score: float
    semantic_score: float
    graph_score: float
    recency_score: float
    confidence_score: float
    rationale: list[str]
