from __future__ import annotations

from dataclasses import dataclass

from .entity import extract_entities, extract_relations
from .models import KnowledgeRecord, Relation


@dataclass(frozen=True)
class GraphProjection:
    record_id: str
    entities: list[str]
    relations: list[Relation]


def build_graph_projection(
    record: KnowledgeRecord,
    *,
    entities: list[str] | None = None,
    relations: list[Relation] | None = None,
) -> GraphProjection:
    resolved_entities = entities if entities is not None else extract_entities(record.text)
    if relations is not None:
        resolved_relations = relations
    else:
        resolved_relations = [
            Relation(source_entity=left, relation=rel.upper(), target_entity=right, weight=1.0)
            for left, rel, right in extract_relations(record.text, resolved_entities)
        ]
    return GraphProjection(record_id=record.record_id, entities=resolved_entities, relations=resolved_relations)
