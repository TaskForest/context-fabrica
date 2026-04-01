from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .models import KnowledgeRecord, MemoryKind, MemoryStage


@dataclass(frozen=True)
class PromotionDecision:
    stage: MemoryStage
    kind: MemoryKind
    rationale: str


def decide_memory_tier(record: KnowledgeRecord) -> PromotionDecision:
    text = record.text.lower()
    tags = {tag.lower() for tag in record.tags}
    metadata_keys = {key.lower() for key in record.metadata}

    if "template" in tags or "pattern" in tags or record.source in {"paper-miner", "pattern-miner"}:
        return PromotionDecision(
            stage="pattern",
            kind="pattern",
            rationale="explicit reusable pattern signal",
        )

    if record.confidence >= 0.75 and (
        {"adr", "design-doc", "runbook", "incident", "decision"} & tags
        or {"provenance", "owner", "repo", "file_path"} & metadata_keys
    ):
        return PromotionDecision(
            stage="canonical",
            kind="fact",
            rationale="high-confidence domain fact with provenance",
        )

    if any(token in text for token in ("todo", "draft", "scratch", "wip", "temporary")):
        return PromotionDecision(
            stage="staged",
            kind="note",
            rationale="needs promotion review before becoming durable memory",
        )

    return PromotionDecision(
        stage="canonical",
        kind="workflow",
        rationale="stable operational knowledge",
    )


def promote_record(record: KnowledgeRecord, *, reviewed_at: datetime | None = None) -> KnowledgeRecord:
    record.stage = "canonical"
    record.reviewed_at = reviewed_at or datetime.now(tz=timezone.utc)
    if record.kind == "note":
        record.kind = "fact"
    return record
