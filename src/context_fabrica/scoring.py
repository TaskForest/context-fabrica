"""Scoring pipeline for multi-signal retrieval.

Owns the in-memory BM25 index and knowledge graph. Designed to be
owned by HybridMemoryStore and bootstrapped lazily from the persistent
store on first query.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from typing import Literal

from .adapters import Reranker
from .config import NamespacePolicy, ScoringWeights
from .entity import extract_entities, extract_relations
from .graph import KnowledgeGraph
from .index import LexicalSemanticIndex
from .models import KnowledgeRecord, QueryResult, Relation
from .temporal import extract_time_range, temporal_overlap_score

ScoringMode = Literal["hybrid", "embedding", "bm25", "rrf"]


class ScoringPipeline:
    """Stateful scoring pipeline with in-memory BM25 index and knowledge graph.

    The BM25 index and graph are maintained in-memory for speed, but
    bootstrapped lazily from the persistent store on first query.
    """

    def __init__(
        self,
        *,
        scoring: ScoringMode = "hybrid",
        weights: ScoringWeights | None = None,
        reranker: Reranker | None = None,
        rerank_weight: float = 0.15,
        namespace_policies: dict[str, NamespacePolicy] | None = None,
    ) -> None:
        self._index = LexicalSemanticIndex()
        self._graph = KnowledgeGraph()
        self._scoring = scoring
        self._scoring_weights = weights or ScoringWeights()
        self._weights = self._weight_map(self._scoring_weights)
        self._reranker = reranker
        self._rerank_weight = max(0.0, min(rerank_weight, 1.0))
        self._namespace_policies = dict(namespace_policies or {})
        self._bootstrapped = False

    @property
    def bootstrapped(self) -> bool:
        return self._bootstrapped

    @property
    def graph(self) -> KnowledgeGraph:
        return self._graph

    @property
    def index(self) -> LexicalSemanticIndex:
        return self._index

    # ── Index maintenance ──

    def index_record(self, record_id: str, text: str, entities: list[str], relations: list[Relation]) -> None:
        """Incrementally add a record to BM25 index and knowledge graph."""
        self._index.upsert(record_id, text)
        self._graph.attach_record_entities(record_id, entities)
        for rel in relations:
            self._graph.add_relation(rel)

    def bootstrap_from_store(
        self,
        texts: list[tuple[str, str]],
        relations: list[tuple[str, str, str, str, float]],
    ) -> None:
        """Bulk-load BM25 index and graph from persistent store data."""
        for record_id, text in texts:
            self._index.upsert(record_id, text)
            entities = extract_entities(text)
            self._graph.attach_record_entities(record_id, entities)
        for _record_id, source_entity, relation_type, target_entity, weight in relations:
            self._graph.add_relation(Relation(source_entity, relation_type, target_entity, weight))
        self._bootstrapped = True

    # ── Scoring ──

    def score_candidates(
        self,
        prompt: str,
        candidates: dict[str, KnowledgeRecord],
        embedding_scores: dict[str, float],
        bm25_scores: dict[str, float],
        graph_scores: dict[str, float],
        *,
        top_k: int = 5,
        domain: str | None = None,
        namespace: str | None = None,
        now: datetime | None = None,
        as_of: datetime | None = None,
        include_staged: bool | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        rerank_top_n: int | None = None,
    ) -> list[QueryResult]:
        """Full scoring pipeline over a set of candidate records."""
        ref_now = now or datetime.now(tz=timezone.utc)
        point_in_time = as_of or ref_now
        policy = self._namespace_policies.get(namespace) if namespace is not None else None
        scoring_weights = policy.weights if policy and policy.weights is not None else self._scoring_weights
        weights = self._weight_map(scoring_weights)
        effective_include_staged = (
            include_staged
            if include_staged is not None
            else (policy.include_staged if policy and policy.include_staged is not None else False)
        )
        effective_rerank_top_n = (
            rerank_top_n
            if rerank_top_n is not None
            else (policy.rerank_top_n if policy is not None else 0)
        )
        candidate_limit = max(top_k, effective_rerank_top_n)

        resolved_time_range = time_range or extract_time_range(prompt, now=ref_now)

        all_ids = set(candidates)
        temporal = self._temporal_scores(candidates, resolved_time_range)
        all_ids = self._filter_candidates(
            all_ids,
            candidates=candidates,
            point_in_time=point_in_time,
            domain=domain,
            namespace=namespace,
            include_staged=effective_include_staged,
            policy=policy,
        )

        semantic = self._fuse_semantic(
            bm25_scores, embedding_scores, all_ids,
            scoring_weights=scoring_weights,
        )

        if self._scoring == "rrf":
            rrf_results = self._score_rrf(
                all_ids, candidates, semantic, graph_scores, temporal,
                ref_now, candidate_limit, weights=weights,
            )
            reranked_rrf = self._apply_reranker(prompt, rrf_results, effective_rerank_top_n)
            return reranked_rrf[:top_k]

        sem_max = max(semantic.values(), default=1.0)
        graph_max = max(graph_scores.values(), default=1.0)
        temporal_max = max(temporal.values(), default=1.0)

        ranked: list[QueryResult] = []
        for rid in all_ids:
            record = candidates[rid]
            sem_norm = semantic.get(rid, 0.0) / sem_max
            graph_norm = graph_scores.get(rid, 0.0) / graph_max
            temporal_norm = temporal.get(rid, 0.0) / temporal_max

            age_hours = max((ref_now - record.created_at).total_seconds() / 3600.0, 0.0)
            recency_score = 1.0 / (1.0 + age_hours / 24.0)
            confidence_score = record.confidence

            final_score = (
                weights["semantic"] * sem_norm
                + weights["graph"] * graph_norm
                + weights["temporal"] * temporal_norm
                + weights["recency"] * recency_score
                + weights["confidence"] * confidence_score
            )

            rationale: list[str] = []
            if sem_norm > 0.0:
                rationale.append("semantic_match")
            if graph_norm > 0.0:
                rationale.append("graph_relation")
            if temporal_norm > 0.0:
                rationale.append("temporal_match")
            if recency_score > 0.5:
                rationale.append("recent")
            if confidence_score > 0.7:
                rationale.append("high_confidence")

            ranked.append(
                QueryResult(
                    record=record,
                    score=final_score,
                    semantic_score=sem_norm,
                    graph_score=graph_norm,
                    temporal_score=temporal_norm,
                    recency_score=recency_score,
                    confidence_score=confidence_score,
                    rationale=rationale,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        reranked = self._apply_reranker(prompt, ranked[:candidate_limit], effective_rerank_top_n)
        return reranked[:top_k]

    # ── Internal helpers ──

    @staticmethod
    def _is_valid(record: KnowledgeRecord, at: datetime) -> bool:
        if at < record.valid_from:
            return False
        if record.valid_to is not None and at > record.valid_to:
            return False
        return True

    @staticmethod
    def _weight_map(weights: ScoringWeights) -> dict[str, float]:
        raw = {
            "semantic": weights.semantic,
            "graph": weights.graph,
            "temporal": weights.temporal,
            "recency": weights.recency,
            "confidence": weights.confidence,
        }
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    @staticmethod
    def _filter_candidates(
        candidate_ids: set[str],
        *,
        candidates: dict[str, KnowledgeRecord],
        point_in_time: datetime,
        domain: str | None = None,
        namespace: str | None = None,
        include_staged: bool,
        policy: NamespacePolicy | None,
    ) -> set[str]:
        result = {rid for rid in candidate_ids if rid in candidates}
        if domain:
            result = {rid for rid in result if candidates[rid].domain == domain}
        if namespace:
            result = {rid for rid in result if candidates[rid].namespace == namespace}
        result = {rid for rid in result if ScoringPipeline._is_valid(candidates[rid], point_in_time)}
        if not include_staged:
            result = {rid for rid in result if candidates[rid].stage != "staged"}
        if policy is not None and policy.min_confidence is not None:
            result = {rid for rid in result if candidates[rid].confidence >= policy.min_confidence}
        if policy is not None and policy.source_allowlist:
            allowed_sources = set(policy.source_allowlist)
            result = {rid for rid in result if candidates[rid].source in allowed_sources}
        return result

    @staticmethod
    def _temporal_scores(
        candidates: dict[str, KnowledgeRecord],
        query_range: tuple[datetime, datetime] | None,
    ) -> dict[str, float]:
        if query_range is None:
            return {}
        scores: dict[str, float] = {}
        for rid, record in candidates.items():
            score = temporal_overlap_score(record.occurred_from, record.occurred_to, query_range)
            if score > 0.0:
                scores[rid] = score
        return scores

    def _fuse_semantic(
        self,
        bm25: dict[str, float],
        embedding: dict[str, float],
        candidates: set[str],
        *,
        scoring_weights: ScoringWeights,
    ) -> dict[str, float]:
        if self._scoring == "bm25":
            return {rid: bm25.get(rid, 0.0) for rid in candidates if bm25.get(rid, 0.0) > 0}
        if self._scoring == "embedding":
            return {rid: embedding.get(rid, 0.0) for rid in candidates if embedding.get(rid, 0.0) > 0}
        emb_weight = scoring_weights.semantic_embedding
        bm25_weight = scoring_weights.semantic_bm25
        fused: dict[str, float] = {}
        for rid in candidates:
            emb = embedding.get(rid, 0.0)
            bm = bm25.get(rid, 0.0)
            score = emb_weight * emb + bm25_weight * bm
            if score > 0.0:
                fused[rid] = score
        return fused

    def _score_rrf(
        self,
        candidate_ids: set[str],
        candidates: dict[str, KnowledgeRecord],
        semantic: dict[str, float],
        graph: dict[str, float],
        temporal: dict[str, float],
        ref_now: datetime,
        top_k: int,
        *,
        weights: dict[str, float],
        k: int = 60,
    ) -> list[QueryResult]:
        sem_ranked = sorted(candidate_ids, key=lambda rid: semantic.get(rid, 0.0), reverse=True)
        graph_ranked = sorted(candidate_ids, key=lambda rid: graph.get(rid, 0.0), reverse=True)
        temporal_ranked = sorted(candidate_ids, key=lambda rid: temporal.get(rid, 0.0), reverse=True)
        recency_ranked = sorted(candidate_ids, key=lambda rid: candidates[rid].created_at, reverse=True)
        confidence_ranked = sorted(candidate_ids, key=lambda rid: candidates[rid].confidence, reverse=True)

        signal_ranks = [
            (sem_ranked, weights["semantic"]),
            (graph_ranked, weights["graph"]),
            (temporal_ranked, weights["temporal"]),
            (recency_ranked, weights["recency"]),
            (confidence_ranked, weights["confidence"]),
        ]

        rrf_scores: dict[str, float] = defaultdict(float)
        for ranked_list, weight in signal_ranks:
            for rank, rid in enumerate(ranked_list):
                rrf_scores[rid] += weight * (1.0 / (k + rank + 1))

        sem_max = max(semantic.values(), default=1.0)
        graph_max = max(graph.values(), default=1.0)
        temporal_max = max(temporal.values(), default=1.0)

        results: list[QueryResult] = []
        for rid in candidate_ids:
            record = candidates[rid]
            sem_norm = semantic.get(rid, 0.0) / sem_max
            graph_norm = graph.get(rid, 0.0) / graph_max
            temporal_norm = temporal.get(rid, 0.0) / temporal_max
            age_hours = max((ref_now - record.created_at).total_seconds() / 3600.0, 0.0)
            recency_score = 1.0 / (1.0 + age_hours / 24.0)

            rationale: list[str] = ["rrf"]
            if sem_norm > 0.0:
                rationale.append("semantic_match")
            if graph_norm > 0.0:
                rationale.append("graph_relation")
            if temporal_norm > 0.0:
                rationale.append("temporal_match")

            results.append(
                QueryResult(
                    record=record,
                    score=rrf_scores[rid],
                    semantic_score=sem_norm,
                    graph_score=graph_norm,
                    temporal_score=temporal_norm,
                    recency_score=recency_score,
                    confidence_score=record.confidence,
                    rationale=rationale,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _apply_reranker(
        self,
        query: str,
        results: list[QueryResult],
        rerank_top_n: int,
    ) -> list[QueryResult]:
        if self._reranker is None or rerank_top_n <= 1 or len(results) <= 1:
            return results
        top_n = min(rerank_top_n, len(results))
        reranked: list[QueryResult] = []
        for result in results[:top_n]:
            rerank_score = max(min(self._reranker.score(query, result.record), 1.0), 0.0)
            adjusted = replace(
                result,
                score=((1.0 - self._rerank_weight) * result.score) + (self._rerank_weight * rerank_score),
                rerank_score=rerank_score,
                rationale=result.rationale + ["reranked"],
            )
            reranked.append(adjusted)
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked + results[top_n:]
