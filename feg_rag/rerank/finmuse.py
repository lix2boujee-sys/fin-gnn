"""FinMUSE: reliability-guided multi-hop evidence set reranking.

FinMUSE keeps the evaluation surface fair by reranking only the same top-N
candidate passages used by the other rerankers.  Unlike the earlier prototype,
it does not consume cached R-GCN rankings.  Its passage backbone is a typed
financial graph propagation score over the candidate-local evidence graph,
combined with retrieval prior and query-entity consistency.  The final passage
ranking is retrieval-preserved: graph and set evidence can only apply a bounded
correction to the initial retrieval score.  Evidence sets are still built for
RAG context selection and diagnostics, but they no longer force companion
passages to the top of the passage ranking.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from feg_rag.rerank.path_encoder import (
    canonical_metric,
    canonical_metric_set,
    expand_years_from_text,
)


@dataclass
class EvidenceSetBreakdown:
    relevance: float = 0.0
    coverage: float = 0.0
    coherence: float = 0.0
    redundancy_penalty: float = 0.0
    conflict_penalty: float = 0.0
    size_penalty: float = 0.0
    final_score: float = 0.0


@dataclass
class EvidenceSet:
    passage_ids: List[str]
    seed_id: str
    breakdown: EvidenceSetBreakdown
    coverage: Dict[str, float] = field(default_factory=dict)
    conflicts: Dict[str, int] = field(default_factory=dict)
    companion_reasons: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ChunkProfile:
    chunk_id: str
    companies: set[str] = field(default_factory=set)
    years: set[str] = field(default_factory=set)
    metrics: set[str] = field(default_factory=set)
    filings: set[str] = field(default_factory=set)
    sections: set[str] = field(default_factory=set)


def _as_nx_graph(graph: Any) -> nx.MultiDiGraph:
    return graph.graph if hasattr(graph, "graph") and isinstance(graph.graph, nx.MultiDiGraph) else graph


def _node_type(graph: Any, node_id: str) -> str:
    node_types = getattr(graph, "node_types", {})
    if node_id in node_types:
        return node_types[node_id]
    nxg = _as_nx_graph(graph)
    return str(nxg.nodes.get(node_id, {}).get("node_type", "chunk"))


def _edge_type(data: Any) -> str:
    return str(data.get("edge_type", "")) if isinstance(data, dict) else ""


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {_norm(v) for v in value if _norm(v)}
    text = _norm(value)
    return {text} if text else set()


def _minmax_score_map(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _entity_value(node_id: str) -> str:
    return _norm(node_id.split("::", 1)[1] if "::" in node_id else node_id)


def _company_from_filing(nxg: nx.MultiDiGraph, filing_id: str) -> str:
    attrs = nxg.nodes.get(filing_id, {})
    if attrs.get("company"):
        return _norm(attrs.get("company"))
    raw = str(filing_id).split("::", 1)[-1]
    return _norm(raw.split("_", 1)[0])


def normalise_query_entities(query: str, entities: Optional[Dict[str, Any]]) -> Dict[str, set[str]]:
    entities = entities or {}
    metrics = _norm_set(entities.get("metric")) | _norm_set(entities.get("metrics"))
    metrics |= _metric_aliases_in_query(query)
    return {
        "company": _norm_set(entities.get("company")) | _norm_set(entities.get("companies")),
        "year": _norm_set(entities.get("year")) | _norm_set(entities.get("years")) | expand_years_from_text(query),
        "metric": canonical_metric_set(metrics),
        "filing": _norm_set(entities.get("filing_type")) | _norm_set(entities.get("filing_types")),
        "section": _norm_set(entities.get("section_hint")) | _norm_set(entities.get("sections")),
    }


def _metric_aliases_in_query(query: str) -> set[str]:
    norm = _norm(query).replace("_", " ")
    candidates = [
        "revenue", "revenues", "sales", "net sales", "rev",
        "net income", "earnings", "profit", "operating income",
        "eps", "ebitda", "cash flow", "assets", "liabilities",
        "equity", "debt", "expenses",
    ]
    return {c for c in candidates if re.search(rf"\b{re.escape(c)}\b", norm)}


def _relation_weight(edge_type: str) -> float:
    weights = {
        "chunk-mentions-metric": 1.00,
        "chunk-mentions-year": 0.95,
        "chunk-belongs-to-filing": 0.75,
        "section-has-chunk": 0.70,
        "filing-has-section": 0.60,
        "company-has-filing": 0.70,
        "semantic-similar": 0.35,
        "same-company": 0.55,
        "same-year": 0.45,
        "same-metric": 0.60,
    }
    return weights.get(edge_type, 0.25)


class FinMUSESetReranker:
    """Reliability-guided multi-hop evidence set reranker."""

    def __init__(
        self,
        max_set_size: int = 5,
        seed_top_k: int = 10,
        companion_pool_k: int = 50,
        min_reliability: float = 0.15,
        semantic_min_overlap: int = 1,
        delta_cap: float = 0.15,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.max_set_size = max_set_size
        self.seed_top_k = seed_top_k
        self.companion_pool_k = companion_pool_k
        self.min_reliability = min_reliability
        self.semantic_min_overlap = semantic_min_overlap
        self.delta_cap = float(delta_cap)
        self.weights = weights or {
            "relevance": 0.25,
            "coverage": 0.40,
            "coherence": 0.20,
            "redundancy": 0.10,
            "conflict": 0.35,
            "size": 0.02,
        }

    def rerank(
        self,
        query: str,
        candidate_ids: Sequence[str],
        retrieval_scores: Sequence[float],
        graph: Any,
        query_entities: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], EvidenceSet, List[EvidenceSet]]:
        if len(candidate_ids) != len(retrieval_scores):
            raise ValueError(
                f"FinMUSE input mismatch: {len(candidate_ids)} ids vs {len(retrieval_scores)} scores"
            )
        if not candidate_ids:
            empty = EvidenceSet([], "", EvidenceSetBreakdown())
            return [], empty, []

        q_entities = normalise_query_entities(query, query_entities)
        nxg = _as_nx_graph(graph)
        candidate_pool = list(candidate_ids[: self.companion_pool_k])
        profiles = {cid: self.chunk_profile(graph, cid) for cid in candidate_pool}
        retrieval_map = _minmax_score_map({cid: float(s) for cid, s in zip(candidate_ids, retrieval_scores)})
        rank_prior = {
            cid: 1.0 - (idx / max(len(candidate_ids) - 1, 1))
            for idx, cid in enumerate(candidate_ids)
        }
        graph_score_map = self.score_passages(candidate_pool, profiles, q_entities, nxg, retrieval_map, rank_prior)

        sets: List[EvidenceSet] = []
        seed_candidates = self._rank_seed_candidates(candidate_pool, profiles, q_entities, nxg, graph_score_map, rank_prior)
        for seed in seed_candidates[: self.seed_top_k]:
            selected, reasons = self._construct_set(seed, candidate_pool, profiles, q_entities, nxg, graph_score_map, rank_prior)
            sets.append(self.score_set(selected, seed, profiles, q_entities, nxg, graph_score_map, rank_prior, reasons))

        best = max(sets, key=lambda s: s.breakdown.final_score) if sets else EvidenceSet([], "", EvidenceSetBreakdown())
        ranked = self._stable_passage_ranking(candidate_pool, best, profiles, q_entities, nxg, graph_score_map, retrieval_map, rank_prior)
        return ranked, best, sets

    def score_passages(
        self,
        candidate_pool: Sequence[str],
        profiles: Dict[str, ChunkProfile],
        q_entities: Dict[str, set[str]],
        nxg: nx.MultiDiGraph,
        retrieval_map: Dict[str, float],
        rank_prior: Dict[str, float],
    ) -> Dict[str, float]:
        graph_scores = self._typed_graph_propagation_scores(candidate_pool, profiles, q_entities, nxg, rank_prior)
        combined: Dict[str, float] = {}
        for cid in candidate_pool:
            profile = profiles[cid]
            coverage = self._coverage_ratio(self._covered_entities([profile], q_entities), q_entities)
            conflict = self._conflict_score([profile], q_entities)
            consistency = max(0.0, coverage - conflict)
            combined[cid] = (
                0.35 * retrieval_map.get(cid, rank_prior.get(cid, 0.0))
                + 0.45 * graph_scores.get(cid, 0.0)
                + 0.20 * consistency
            )
        return _minmax_score_map(combined)

    def _passage_relevance(self, cid: str, score_map: Dict[str, float], rank_prior: Dict[str, float]) -> float:
        """FinMUSE graph-backed passage relevance, with retrieval only as prior."""

        return 0.80 * score_map.get(cid, 0.0) + 0.20 * rank_prior.get(cid, 0.0)

    def _typed_graph_propagation_scores(
        self,
        candidate_pool: Sequence[str],
        profiles: Dict[str, ChunkProfile],
        q_entities: Dict[str, set[str]],
        nxg: nx.MultiDiGraph,
        rank_prior: Dict[str, float],
    ) -> Dict[str, float]:
        seeds = self._query_seed_nodes(nxg, q_entities)
        if not candidate_pool:
            return {}
        if not seeds:
            return {cid: rank_prior.get(cid, 0.0) for cid in candidate_pool}

        nodes: set[str] = set(candidate_pool) | set(seeds)
        for cid in candidate_pool:
            if cid not in nxg:
                continue
            nodes.update(nxg.predecessors(cid))
            nodes.update(nxg.successors(cid))

        sub = nx.DiGraph()
        for n in nodes:
            sub.add_node(n)
        for u, v, data in nxg.edges(data=True):
            if u not in nodes or v not in nodes:
                continue
            etype = _edge_type(data)
            w = _relation_weight(etype)
            if etype == "semantic-similar":
                u_prof = profiles.get(u)
                v_prof = profiles.get(v)
                if u_prof is not None and v_prof is not None:
                    overlap = len((u_prof.metrics | v_prof.metrics) & q_entities["metric"])
                    overlap += len((u_prof.years | v_prof.years) & q_entities["year"])
                    if overlap < self.semantic_min_overlap:
                        w *= 0.25
            if sub.has_edge(u, v):
                sub[u][v]["weight"] = max(sub[u][v]["weight"], w)
            else:
                sub.add_edge(u, v, weight=w)
            if sub.has_edge(v, u):
                sub[v][u]["weight"] = max(sub[v][u]["weight"], w)
            else:
                sub.add_edge(v, u, weight=w)

        personalization = {n: 0.0 for n in sub.nodes}
        for s in seeds:
            if s in personalization:
                personalization[s] += 1.0
        for cid in candidate_pool[:5]:
            if cid in personalization:
                personalization[cid] += 0.15 * rank_prior.get(cid, 0.0)
        total = sum(personalization.values())
        if total <= 0:
            return {cid: rank_prior.get(cid, 0.0) for cid in candidate_pool}
        personalization = {k: v / total for k, v in personalization.items()}
        try:
            pr = nx.pagerank(
                sub,
                alpha=0.85,
                personalization=personalization,
                max_iter=100,
                tol=1e-6,
                weight="weight",
            )
        except nx.PowerIterationFailedConvergence:
            return {cid: rank_prior.get(cid, 0.0) for cid in candidate_pool}
        return _minmax_score_map({cid: pr.get(cid, 0.0) for cid in candidate_pool})

    def _query_seed_nodes(self, nxg: nx.MultiDiGraph, q_entities: Dict[str, set[str]]) -> set[str]:
        seeds: set[str] = set()
        wanted_by_type = {
            "company": q_entities.get("company", set()),
            "metric": q_entities.get("metric", set()),
            "year": q_entities.get("year", set()),
        }
        for node, attrs in nxg.nodes(data=True):
            ntype = str(attrs.get("node_type", ""))
            wanted = wanted_by_type.get(ntype)
            if not wanted:
                continue
            values = {
                _entity_value(str(node)),
                _norm(attrs.get("name")),
                _norm(attrs.get("company")),
            }
            if ntype == "metric":
                values = canonical_metric_set(values)
            if values & wanted:
                seeds.add(node)
        for m in q_entities.get("metric", set()):
            node = f"metric::{m}"
            if node in nxg:
                seeds.add(node)
        for y in q_entities.get("year", set()):
            node = f"year::{y}"
            if node in nxg:
                seeds.add(node)
        for c in q_entities.get("company", set()):
            node = f"company::{c}"
            if node in nxg:
                seeds.add(node)
        return seeds

    def _rank_seed_candidates(
        self,
        candidate_pool: Sequence[str],
        profiles: Dict[str, ChunkProfile],
        q_entities: Dict[str, set[str]],
        nxg: nx.MultiDiGraph,
        score_map: Dict[str, float],
        bge_rank_map: Dict[str, float],
    ) -> List[str]:
        scored: List[Tuple[float, str]] = []
        for cid in candidate_pool:
            profile = profiles[cid]
            covered = self._covered_entities([profile], q_entities)
            coverage = self._coverage_ratio(covered, q_entities)
            conflict = self._conflict_score([profile], q_entities)
            coherence_potential = self._candidate_connectivity(cid, candidate_pool, nxg)
            relevance = self._passage_relevance(cid, score_map, bge_rank_map)
            seed_score = (
                0.25 * relevance
                + 0.45 * coverage
                + 0.20 * coherence_potential
                - 0.35 * conflict
            )
            scored.append((seed_score, cid))
        return [cid for _score, cid in sorted(scored, reverse=True)]

    def _construct_set(
        self,
        seed: str,
        candidate_pool: Sequence[str],
        profiles: Dict[str, ChunkProfile],
        q_entities: Dict[str, set[str]],
        nxg: nx.MultiDiGraph,
        score_map: Dict[str, float],
        bge_rank_map: Dict[str, float],
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        selected = [seed]
        reasons: Dict[str, List[str]] = {seed: ["seed"]}
        covered = self._covered_entities([profiles[seed]], q_entities)

        scored: List[Tuple[float, str, List[str]]] = []
        for cid in candidate_pool:
            if cid == seed:
                continue
            reliability, why = self._set_companion_reliability(selected, cid, profiles, q_entities, nxg)
            if reliability < self.min_reliability:
                continue
            gain = self._coverage_gain(covered, profiles[cid], q_entities)
            trial_profiles = [profiles[x] for x in selected + [cid]]
            conflict = self._conflict_score(trial_profiles, q_entities)
            redundancy = self._redundancy_score(trial_profiles)
            coherence = self._coherence(selected + [cid], nxg)
            relevance = self._passage_relevance(cid, score_map, bge_rank_map)
            score = (
                0.15 * relevance
                + 0.45 * gain
                + 0.25 * reliability
                + 0.20 * coherence
                - 0.30 * conflict
                - 0.15 * redundancy
            )
            scored.append((score, cid, why))

        for _score, cid, why in sorted(scored, reverse=True):
            if len(selected) >= self.max_set_size:
                break
            trial_profiles = [profiles[x] for x in selected + [cid]]
            if self._conflict_score(trial_profiles, q_entities) > 0.75:
                continue
            selected.append(cid)
            reasons[cid] = why
            covered = self._covered_entities([profiles[x] for x in selected], q_entities)
            if self._coverage_ratio(covered, q_entities) >= 1.0 and len(selected) >= 2:
                break
        return selected, reasons

    def _set_companion_reliability(
        self,
        selected: Sequence[str],
        cid: str,
        profiles: Dict[str, ChunkProfile],
        q_entities: Dict[str, set[str]],
        nxg: nx.MultiDiGraph,
    ) -> Tuple[float, List[str]]:
        best_score = -1.0
        best_reasons: List[str] = []
        for seed in selected:
            score, reasons = self._companion_reliability(seed, cid, profiles, q_entities, nxg)
            if score > best_score:
                best_score = score
                best_reasons = reasons
        return best_score, best_reasons

    def _companion_reliability(
        self,
        seed: str,
        cid: str,
        profiles: Dict[str, ChunkProfile],
        q_entities: Dict[str, set[str]],
        nxg: nx.MultiDiGraph,
    ) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        score = 0.0
        sp = profiles[seed]
        cp = profiles[cid]

        if sp.filings & cp.filings:
            score += 0.30
            reasons.append("same_filing")
        if sp.sections & cp.sections:
            score += 0.15
            reasons.append("same_section")
        if sp.metrics & cp.metrics:
            score += 0.20
            reasons.append("same_metric")
        if sp.years & cp.years:
            score += 0.10
            reasons.append("same_year")
        if sp.companies & cp.companies:
            score += 0.15
            reasons.append("same_company")
        if self._has_edge(nxg, seed, cid, "semantic-similar"):
            overlap = len((cp.metrics & q_entities["metric"]) | (cp.years & q_entities["year"]))
            if overlap >= self.semantic_min_overlap:
                score += 0.15
                reasons.append("semantic_support")
            else:
                score -= 0.15
                reasons.append("semantic_no_entity_overlap")

        conflicts = self._profile_conflicts(cp, q_entities)
        if any(conflicts.values()):
            score -= 0.35 * sum(conflicts.values())
            reasons.extend([k for k, v in conflicts.items() if v])
        return float(np.clip(score, -1.0, 1.0)), reasons

    def score_set(
        self,
        passage_ids: Sequence[str],
        seed_id: str,
        profiles: Dict[str, ChunkProfile],
        q_entities: Dict[str, set[str]],
        nxg: nx.MultiDiGraph,
        score_map: Dict[str, float],
        bge_rank_map: Dict[str, float],
        reasons: Optional[Dict[str, List[str]]] = None,
    ) -> EvidenceSet:
        selected_profiles = [profiles[cid] for cid in passage_ids if cid in profiles]
        relevance = (
            float(np.mean([self._passage_relevance(cid, score_map, bge_rank_map) for cid in passage_ids]))
            if passage_ids else 0.0
        )
        covered = self._covered_entities(selected_profiles, q_entities)
        coverage = self._coverage_ratio(covered, q_entities)
        coherence = self._coherence(passage_ids, nxg)
        redundancy = self._redundancy_score(selected_profiles)
        conflict = self._conflict_score(selected_profiles, q_entities)
        size_penalty = max(0.0, (len(passage_ids) - 3) / max(self.max_set_size, 1))

        final = (
            self.weights["relevance"] * relevance
            + self.weights["coverage"] * coverage
            + self.weights["coherence"] * coherence
            - self.weights["redundancy"] * redundancy
            - self.weights["conflict"] * conflict
            - self.weights["size"] * size_penalty
        )
        conflicts = self._set_conflict_flags(selected_profiles, q_entities)
        return EvidenceSet(
            passage_ids=list(passage_ids),
            seed_id=seed_id,
            breakdown=EvidenceSetBreakdown(
                relevance=relevance,
                coverage=coverage,
                coherence=coherence,
                redundancy_penalty=redundancy,
                conflict_penalty=conflict,
                size_penalty=size_penalty,
                final_score=float(final),
            ),
            coverage=covered,
            conflicts=conflicts,
            companion_reasons=reasons or {},
        )

    def _set_aware_ranking(
        self,
        candidate_pool: Sequence[str],
        best: EvidenceSet,
        profiles: Dict[str, ChunkProfile],
        q_entities: Dict[str, set[str]],
        nxg: nx.MultiDiGraph,
        score_map: Dict[str, float],
        bge_rank_map: Dict[str, float],
    ) -> List[str]:
        selected = set(best.passage_ids)
        selected_profiles = [profiles[cid] for cid in best.passage_ids if cid in profiles]
        selected_covered = self._covered_entities(selected_profiles, q_entities)
        passage_scores: Dict[str, float] = {}
        for cid in candidate_pool:
            profile = profiles[cid]
            single_coverage = self._coverage_ratio(self._covered_entities([profile], q_entities), q_entities)
            marginal_gain = self._coverage_gain(selected_covered, profile, q_entities)
            reliability, _why = self._set_companion_reliability(best.passage_ids or [cid], cid, profiles, q_entities, nxg)
            conflict = self._conflict_score([profile], q_entities)
            redundancy = self._redundancy_score(selected_profiles + [profile]) if cid not in selected else 0.0
            set_bonus = 0.35 if cid in selected else 0.0
            passage_scores[cid] = (
                0.25 * self._passage_relevance(cid, score_map, bge_rank_map)
                + 0.25 * single_coverage
                + 0.20 * marginal_gain
                + 0.15 * max(reliability, 0.0)
                + set_bonus
                - 0.30 * conflict
                - 0.10 * redundancy
            )

        selected_rank = sorted(best.passage_ids, key=lambda cid: passage_scores.get(cid, -1e9), reverse=True)
        remaining = sorted(
            [cid for cid in candidate_pool if cid not in selected],
            key=lambda cid: passage_scores.get(cid, -1e9),
            reverse=True,
        )
        return selected_rank + remaining

    def _stable_passage_ranking(
        self,
        candidate_pool: Sequence[str],
        best: EvidenceSet,
        profiles: Dict[str, ChunkProfile],
        q_entities: Dict[str, set[str]],
        nxg: nx.MultiDiGraph,
        graph_score_map: Dict[str, float],
        retrieval_map: Dict[str, float],
        rank_prior: Dict[str, float],
    ) -> List[str]:
        """Rank passages with bounded graph/set correction.

        This is the stable FinMUSE ranking path. Retrieval remains the anchor;
        typed graph propagation and evidence-set membership can only move a
        passage within a small score band. This prevents companion passages
        from displacing strong first-hop evidence just because they improve
        set coverage.
        """

        selected = set(best.passage_ids)
        selected_profiles = [profiles[cid] for cid in best.passage_ids if cid in profiles]
        final_scores: Dict[str, float] = {}
        for cid in candidate_pool:
            profile = profiles[cid]
            retrieval_anchor = 0.75 * retrieval_map.get(cid, rank_prior.get(cid, 0.0)) + 0.25 * rank_prior.get(cid, 0.0)
            graph_gain = graph_score_map.get(cid, 0.0) - retrieval_map.get(cid, rank_prior.get(cid, 0.0))
            coverage = self._coverage_ratio(self._covered_entities([profile], q_entities), q_entities)
            conflict = self._conflict_score([profile], q_entities)
            reliability = 0.0
            if best.passage_ids:
                reliability, _why = self._set_companion_reliability(best.passage_ids, cid, profiles, q_entities, nxg)
            set_bonus = 0.04 if cid in selected else 0.0
            redundancy = self._redundancy_score(selected_profiles + [profile]) if cid not in selected else 0.0
            raw_delta = (
                0.45 * graph_gain
                + 0.12 * coverage
                + 0.08 * max(reliability, 0.0)
                + set_bonus
                - 0.22 * conflict
                - 0.03 * redundancy
            )
            clipped_delta = float(np.clip(raw_delta, -self.delta_cap, self.delta_cap))
            final_scores[cid] = retrieval_anchor + clipped_delta
        return [cid for cid, _score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)]

    def chunk_profile(self, graph: Any, cid: str) -> ChunkProfile:
        nxg = _as_nx_graph(graph)
        prof = ChunkProfile(chunk_id=cid)
        if cid not in nxg:
            return prof
        attrs = nxg.nodes.get(cid, {})
        if attrs.get("company"):
            prof.companies.add(_norm(attrs.get("company")))
        if attrs.get("filing_year"):
            prof.years.add(_norm(attrs.get("filing_year")))
        if attrs.get("section"):
            prof.sections.add(_norm(attrs.get("section")))

        for _, dst, data in nxg.out_edges(cid, data=True):
            etype = _edge_type(data)
            if etype == "chunk-mentions-metric" and _node_type(graph, dst) == "metric":
                prof.metrics.add(canonical_metric(_entity_value(dst)))
            elif etype == "chunk-mentions-year" and _node_type(graph, dst) == "year":
                prof.years.add(_entity_value(dst))
            elif etype == "chunk-belongs-to-filing":
                prof.filings.add(_norm(dst))
                company = _company_from_filing(nxg, dst)
                if company:
                    prof.companies.add(company)
                fattrs = nxg.nodes.get(dst, {})
                if fattrs.get("filing_type"):
                    prof.filings.add(_norm(fattrs.get("filing_type")))
                if fattrs.get("filing_year"):
                    prof.years.add(_norm(fattrs.get("filing_year")))

        for src, _, data in nxg.in_edges(cid, data=True):
            etype = _edge_type(data)
            if etype == "section-has-chunk":
                prof.sections.add(_entity_value(src))
                for filing, _, fdata in nxg.in_edges(src, data=True):
                    if _edge_type(fdata) == "filing-has-section":
                        prof.filings.add(_norm(filing))
                        company = _company_from_filing(nxg, filing)
                        if company:
                            prof.companies.add(company)
        return prof

    def _has_edge(self, nxg: nx.MultiDiGraph, a: str, b: str, edge_type: str) -> bool:
        for u, v in ((a, b), (b, a)):
            if u not in nxg or v not in nxg:
                continue
            for _src, _dst, data in nxg.edges(u, data=True):
                if _dst == v and _edge_type(data) == edge_type:
                    return True
        return False

    def _candidate_connectivity(self, cid: str, candidate_pool: Sequence[str], nxg: nx.MultiDiGraph) -> float:
        if cid not in nxg or len(candidate_pool) <= 1:
            return 0.0
        connected = 0
        total = 0
        for other in candidate_pool:
            if other == cid:
                continue
            total += 1
            connected += int(nxg.has_edge(cid, other) or nxg.has_edge(other, cid))
        return connected / total if total else 0.0

    def _covered_entities(self, profiles: Sequence[ChunkProfile], q_entities: Dict[str, set[str]]) -> Dict[str, float]:
        merged = self._merge_profiles(profiles)
        return {
            "company": self._entity_coverage(q_entities["company"], merged.companies),
            "year": self._entity_coverage(q_entities["year"], merged.years),
            "metric": self._entity_coverage(q_entities["metric"], merged.metrics),
            "filing": self._entity_coverage(q_entities["filing"], merged.filings),
            "section": self._entity_coverage(q_entities["section"], merged.sections),
        }

    def _coverage_ratio(self, covered: Dict[str, float], q_entities: Dict[str, set[str]]) -> float:
        active = [k for k, v in q_entities.items() if v]
        if not active:
            return 0.0
        return float(np.mean([covered.get(k, 0.0) for k in active]))

    def _coverage_gain(self, covered: Dict[str, float], profile: ChunkProfile, q_entities: Dict[str, set[str]]) -> float:
        new_covered = self._covered_entities([profile], q_entities)
        gains = []
        for key, required in q_entities.items():
            if not required:
                continue
            gains.append(max(0.0, new_covered.get(key, 0.0) - covered.get(key, 0.0)))
        return float(np.mean(gains)) if gains else 0.0

    @staticmethod
    def _entity_coverage(required: set[str], observed: set[str]) -> float:
        if not required:
            return 0.0
        return len(required & observed) / len(required)

    @staticmethod
    def _merge_profiles(profiles: Sequence[ChunkProfile]) -> ChunkProfile:
        merged = ChunkProfile("set")
        for p in profiles:
            merged.companies |= p.companies
            merged.years |= p.years
            merged.metrics |= p.metrics
            merged.filings |= p.filings
            merged.sections |= p.sections
        return merged

    def _profile_conflicts(self, profile: ChunkProfile, q_entities: Dict[str, set[str]]) -> Dict[str, int]:
        return {
            "wrong_company": int(bool(q_entities["company"] and profile.companies and not (profile.companies & q_entities["company"]))),
            "wrong_year": int(bool(q_entities["year"] and profile.years and not (profile.years & q_entities["year"]))),
            "wrong_metric": int(bool(q_entities["metric"] and profile.metrics and not (profile.metrics & q_entities["metric"]))),
        }

    def _set_conflict_flags(self, profiles: Sequence[ChunkProfile], q_entities: Dict[str, set[str]]) -> Dict[str, int]:
        merged = {"wrong_company": 0, "wrong_year": 0, "wrong_metric": 0}
        for p in profiles:
            c = self._profile_conflicts(p, q_entities)
            for k, v in c.items():
                merged[k] = int(merged[k] or v)
        return merged

    def _conflict_score(self, profiles: Sequence[ChunkProfile], q_entities: Dict[str, set[str]]) -> float:
        if not profiles:
            return 0.0
        vals = []
        for p in profiles:
            flags = self._profile_conflicts(p, q_entities)
            vals.append(sum(flags.values()) / 3.0)
        return float(np.mean(vals))

    def _redundancy_score(self, profiles: Sequence[ChunkProfile]) -> float:
        if len(profiles) < 2:
            return 0.0
        sims = []
        for a, b in itertools.combinations(profiles, 2):
            same = 0
            total = 0
            for attr in ("companies", "years", "metrics", "filings", "sections"):
                av = getattr(a, attr)
                bv = getattr(b, attr)
                if av or bv:
                    total += 1
                    same += int(bool(av & bv))
            sims.append(same / total if total else 0.0)
        return float(np.mean(sims)) if sims else 0.0

    def _coherence(self, passage_ids: Sequence[str], nxg: nx.MultiDiGraph) -> float:
        if len(passage_ids) < 2:
            return 0.0
        edges = 0
        total = 0
        for a, b in itertools.combinations(passage_ids, 2):
            total += 1
            connected = False
            if a in nxg and b in nxg:
                connected = nxg.has_edge(a, b) or nxg.has_edge(b, a)
            if connected:
                edges += 1
        return edges / total if total else 0.0


def generate_conflict_negative_sets(
    positive_set: EvidenceSet,
    candidate_ids: Sequence[str],
    profiles: Dict[str, ChunkProfile],
    q_entities: Dict[str, set[str]],
    max_size: int = 5,
) -> Dict[str, List[str]]:
    """Generate lightweight hard-negative evidence sets for ablation/training."""

    negatives: Dict[str, List[str]] = {}
    base = list(positive_set.passage_ids[:max_size])
    for name, key in [
        ("wrong_year", "wrong_year"),
        ("wrong_metric", "wrong_metric"),
        ("wrong_company", "wrong_company"),
    ]:
        for cid in candidate_ids:
            flags = FinMUSESetReranker()._profile_conflicts(profiles.get(cid, ChunkProfile(cid)), q_entities)
            if flags.get(key):
                negatives[name] = [cid] + [x for x in base if x != cid][: max_size - 1]
                break
    if base:
        negatives["redundant"] = (base + base[:1])[:max_size]
    return negatives


def evidence_set_metrics(
    items: Sequence[Dict[str, Any]],
    selected_sets: Sequence[EvidenceSet],
) -> Dict[str, float]:
    gold_cov = []
    entity_cov = []
    conflict = []
    redundancy = []
    for item, eset in zip(items, selected_sets):
        gold = set(item.get("gold_evidence_ids", []))
        selected = set(eset.passage_ids)
        gold_cov.append(len(gold & selected) / len(gold) if gold else 0.0)
        entity_cov.append(eset.breakdown.coverage)
        conflict.append(float(any(eset.conflicts.values())))
        redundancy.append(eset.breakdown.redundancy_penalty)
    return {
        "evidence_set_gold_coverage": float(np.mean(gold_cov)) if gold_cov else 0.0,
        "query_entity_coverage": float(np.mean(entity_cov)) if entity_cov else 0.0,
        "conflict_rate": float(np.mean(conflict)) if conflict else 0.0,
        "redundancy_rate": float(np.mean(redundancy)) if redundancy else 0.0,
    }
