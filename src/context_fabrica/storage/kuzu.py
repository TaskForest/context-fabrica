from __future__ import annotations

from contextlib import suppress
from importlib import import_module
from typing import Any, cast

from ..config import KuzuSettings
from ..models import Relation
from ..projection import GraphProjection


class KuzuGraphProjectionAdapter:
    def __init__(self, settings: KuzuSettings) -> None:
        self.settings = settings

    def bootstrap_statements(self) -> list[str]:
        return [
            "CREATE NODE TABLE IF NOT EXISTS MemoryRecord(record_id STRING PRIMARY KEY, domain STRING, source STRING);",
            "CREATE NODE TABLE IF NOT EXISTS Entity(name STRING PRIMARY KEY);",
            "CREATE REL TABLE IF NOT EXISTS HAS_ENTITY(FROM MemoryRecord TO Entity, weight DOUBLE);",
            "CREATE REL TABLE IF NOT EXISTS RELATED(FROM Entity TO Entity, relation_type STRING, weight DOUBLE);",
        ]

    def project_statements(self, projection: GraphProjection, domain: str, source: str) -> list[str]:
        statements = [
            "MERGE (r:MemoryRecord {record_id: $record_id}) SET r.domain = $domain, r.source = $source"
        ]
        for entity in projection.entities:
            statements.append(
                "MERGE (e:Entity {name: $entity_name})"
            )
            statements.append(
                "MATCH (r:MemoryRecord {record_id: $record_id}), (e:Entity {name: $entity_name}) "
                "MERGE (r)-[:HAS_ENTITY {weight: 1.0}]->(e)"
            )
        for relation in projection.relations:
            statements.append(self._relation_statement(relation))
        return statements

    def neighbor_query(self) -> str:
        max_hops = self.settings.max_hops
        return (
            "MATCH (seed:Entity) WHERE seed.name IN $entities "
            f"MATCH p=(seed)-[:RELATED*1..{max_hops}]-(neighbor:Entity) "
            "RETURN neighbor.name AS entity_name, count(*) AS path_count ORDER BY path_count DESC LIMIT $limit"
        )

    def connect(self) -> Any:
        with suppress(ModuleNotFoundError):
            kuzu = import_module("kuzu")
            database = kuzu.Database(self.settings.path)
            return kuzu.Connection(database)
        raise ModuleNotFoundError("Install context-fabrica[kuzu] to use the Kuzu adapter")

    def bootstrap(self) -> None:
        conn = cast(Any, self.connect())
        for statement in self.bootstrap_statements():
            conn.execute(statement)

    def project(self, projection: GraphProjection, *, domain: str, source: str) -> None:
        conn = cast(Any, self.connect())
        conn.execute(
            "MERGE (r:MemoryRecord {record_id: $record_id}) SET r.domain = $domain, r.source = $source",
            {
                "record_id": projection.record_id,
                "domain": domain,
                "source": source,
            },
        )
        for entity in projection.entities:
            conn.execute("MERGE (e:Entity {name: $entity_name})", {"entity_name": entity})
            conn.execute(
                "MATCH (r:MemoryRecord {record_id: $record_id}), (e:Entity {name: $entity_name}) MERGE (r)-[:HAS_ENTITY {weight: 1.0}]->(e)",
                {"record_id": projection.record_id, "entity_name": entity},
            )
        for relation in projection.relations:
            conn.execute(
                self._relation_statement(relation),
                {
                    "source_entity": relation.source_entity,
                    "target_entity": relation.target_entity,
                    "relation_type": relation.relation,
                    "weight": relation.weight,
                },
            )

    def _relation_statement(self, relation: Relation) -> str:
        return (
            "MERGE (left:Entity {name: $source_entity}) "
            "MERGE (right:Entity {name: $target_entity}) "
            "MERGE (left)-[:RELATED {relation_type: $relation_type, weight: $weight}]->(right)"
        )
