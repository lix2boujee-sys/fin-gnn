"""Financial Evidence Graph builder.

Constructs a heterogeneous graph with nodes {chunk, metric, year} and edges
representing structural relationships between them.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from feg_rag.data.chunker import Chunk
from feg_rag.graph.entities import ExtractedEntities, extract_entities


# ═════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═════════════════════════════════════════════════════════════════════════════

class FinancialEvidenceGraph:
    """Heterogeneous graph connecting chunks, metrics, years, and more.

    Built on top of networkx for flexibility; can be exported to DGL/PyG for
    GNN training.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        # node_id → node_type
        self.node_types: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        chunks: List[Chunk],
        entity_map: Optional[Dict[str, ExtractedEntities]] = None,
        chunk_embeddings: Optional[Dict[str, np.ndarray]] = None,
        semantic_threshold: float = 0.7,
        max_semantic_edges_per_node: int = 10,
        add_semantic_edges: bool = True,
        use_edge_weights: bool = False,
        edge_weight_map: Optional[Dict[str, float]] = None,
        add_company_nodes: bool = True,
        add_filing_nodes: bool = True,
        add_section_nodes: bool = True,
        add_same_entity_edges: bool = False,
        max_same_entity_edges: int = 20,
    ) -> "FinancialEvidenceGraph":
        """Build the full evidence graph from chunks.

        Args:
            chunks: All text/table chunks.
            entity_map: Pre-extracted entities per chunk (auto-extracted if None).
            chunk_embeddings: Embedding vectors per chunk_id for semantic edges.
            semantic_threshold: Cosine similarity threshold for semantic edges.
            max_semantic_edges_per_node: Cap per-node semantic edges.
            add_semantic_edges: Whether to include semantic similarity edges.
            use_edge_weights: If True, store edge weights for weighted PPR.
            edge_weight_map: Dict from edge_type → weight (defaults used if None).
            add_company_nodes: Whether to create Company nodes.
            add_filing_nodes: Whether to create Filing nodes.
            add_section_nodes: Whether to create Section nodes.
        """
        if entity_map is None:
            entity_map = extract_entities(chunks)

        # Default edge weights per edge type
        if edge_weight_map is None:
            edge_weight_map = {
                "chunk-mentions-metric": 0.8,
                "chunk-mentions-year": 0.8,
                "company-has-filing": 0.7,
                "filing-has-section": 0.6,
                "section-has-chunk": 0.6,
                "chunk-belongs-to-filing": 0.5,
                "same-company": 0.5,
                "same-filing-year": 0.5,
                "same-metric": 0.5,
                "same-year": 0.5,
                "same-section": 0.5,
                "semantic-similar": 0.3,
            }

        # — chunk nodes —
        for c in chunks:
            self._add_node(c.chunk_id, "chunk", text=c.text, doc_id=c.doc_id)

        # Collect global entities
        all_metrics: Dict[str, List[str]] = defaultdict(list)
        all_years: Dict[str, List[str]] = defaultdict(list)
        all_companies: Dict[str, List[str]] = defaultdict(list)
        all_filing_types: Dict[str, List[str]] = defaultdict(list)
        all_sections: Dict[str, List[str]] = defaultdict(list)

        # Track chunk metadata for filing nodes
        chunk_doc_info: Dict[str, Dict[str, str]] = {}
        filing_set: Set[Tuple[str, str, str]] = set()  # (company, filing_type, year)

        for c in chunks:
            ent = entity_map.get(c.chunk_id)
            if c.company:
                chunk_doc_info.setdefault("company", {})[c.chunk_id] = c.company
                all_companies[c.company].append(c.chunk_id)
            if c.filing_type and c.filing_year:
                key = (c.company or "unknown", c.filing_type, c.filing_year)
                filing_set.add(key)
                chunk_doc_info.setdefault("filing_key", {})[c.chunk_id] = "|".join(key)
            if c.section:
                all_sections[c.section].append(c.chunk_id)
            if c.filing_type:
                all_filing_types[c.filing_type].append(c.chunk_id)

            if ent is None:
                continue
            for m in ent.metrics:
                all_metrics[m].append(c.chunk_id)
            for y in ent.years:
                all_years[y].append(c.chunk_id)
            for comp in ent.companies:
                all_companies[comp].append(c.chunk_id)

        # — company nodes —
        if add_company_nodes:
            for comp_name in all_companies:
                self._add_node(f"company::{comp_name}", "company", name=comp_name)

        # — filing nodes —
        if add_filing_nodes:
            for comp, ftype, fyear in filing_set:
                fid = f"filing::{comp}_{ftype}_{fyear}"
                self._add_node(
                    fid, "filing", company=comp, filing_type=ftype, filing_year=fyear
                )
                # company-has-filing
                comp_node = f"company::{comp}"
                if comp_node in self.graph:
                    self._add_edge(comp_node, fid, "company-has-filing",
                                   weight=edge_weight_map.get("company-has-filing", 0.7))

        # — section nodes —
        if add_section_nodes:
            for sec_name in all_sections:
                self._add_node(f"section::{sec_name}", "section", name=sec_name)

        # — metric/year nodes —
        for metric_name in all_metrics:
            self._add_node(f"metric::{metric_name}", "metric", name=metric_name)
        for year_val in all_years:
            self._add_node(f"year::{year_val}", "year", name=year_val)

        # — edges —
        # chunk-mentions-metric
        for metric_name, cids in all_metrics.items():
            m_node = f"metric::{metric_name}"
            for cid in cids:
                self._add_edge(cid, m_node, "chunk-mentions-metric",
                               weight=edge_weight_map.get("chunk-mentions-metric", 0.8))

        # chunk-mentions-year
        for year_val, cids in all_years.items():
            y_node = f"year::{year_val}"
            for cid in cids:
                self._add_edge(cid, y_node, "chunk-mentions-year",
                               weight=edge_weight_map.get("chunk-mentions-year", 0.8))

        # section-has-chunk
        if add_section_nodes:
            for sec_name, cids in all_sections.items():
                s_node = f"section::{sec_name}"
                for cid in cids:
                    self._add_edge(s_node, cid, "section-has-chunk",
                                   weight=edge_weight_map.get("section-has-chunk", 0.6))

        # filing-has-section: connect filing nodes to section nodes by shared chunks
        if add_filing_nodes and add_section_nodes:
            # Build filing → sections mapping via chunks
            filing_sections: Dict[str, Set[str]] = defaultdict(set)
            for c in chunks:
                fkey = chunk_doc_info.get("filing_key", {}).get(c.chunk_id, "")
                if fkey and c.section:
                    fid = f"filing::{fkey.replace('|', '_')}"
                    filing_sections[fid].add(c.section)
            for fid, secs in filing_sections.items():
                for sec in secs:
                    s_node = f"section::{sec}"
                    if s_node in self.graph:
                        self._add_edge(fid, s_node, "filing-has-section",
                                       weight=edge_weight_map.get("filing-has-section", 0.6))

        # chunk-belongs-to-filing
        if add_filing_nodes:
            for c in chunks:
                fkey = chunk_doc_info.get("filing_key", {}).get(c.chunk_id, "")
                if fkey:
                    fid = f"filing::{fkey.replace('|', '_')}"
                    if fid in self.graph:
                        self._add_edge(c.chunk_id, fid, "chunk-belongs-to-filing",
                                       weight=edge_weight_map.get("chunk-belongs-to-filing", 0.5))

        # same-metric (capped to prevent O(n²) blowup)
        if add_same_entity_edges:
            _add_capped_same_edges(self, all_metrics, "same-metric",
                                   weight=edge_weight_map.get("same-metric", 0.5),
                                   max_edges=max_same_entity_edges)

        # same-year
        if add_same_entity_edges:
            _add_capped_same_edges(self, all_years, "same-year",
                                   weight=edge_weight_map.get("same-year", 0.5),
                                   max_edges=max_same_entity_edges)

        # same-company
        if add_same_entity_edges and add_company_nodes:
            _add_capped_same_edges(self, all_companies, "same-company",
                                   weight=edge_weight_map.get("same-company", 0.5),
                                   max_edges=max_same_entity_edges)

        # same-filing-year
        if add_same_entity_edges and add_filing_nodes:
            year_chunks: Dict[str, List[str]] = defaultdict(list)
            for c in chunks:
                if c.filing_year:
                    year_chunks[c.filing_year].append(c.chunk_id)
            _add_capped_same_edges(self, year_chunks, "same-filing-year",
                                   weight=edge_weight_map.get("same-filing-year", 0.5),
                                   max_edges=max_same_entity_edges)

        # semantic-similar (optional, expensive)
        if add_semantic_edges and chunk_embeddings:
            self._add_semantic_edges(
                chunks, chunk_embeddings, semantic_threshold, max_semantic_edges_per_node,
                weight=edge_weight_map.get("semantic-similar", 0.3),
            )

        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_chunk_neighbors(self, chunk_id: str, max_hops: int = 2) -> Set[str]:
        """Return all node ids within *max_hops* of a chunk."""
        if chunk_id not in self.graph:
            return set()
        nodes: Set[str] = {chunk_id}
        frontier = {chunk_id}
        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for n in frontier:
                for _, neighbor in self.graph.edges(n):
                    if neighbor not in nodes:
                        nodes.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        return nodes

    def get_subgraph(
        self, chunk_ids: List[str], max_hops: int = 2
    ) -> nx.MultiDiGraph:
        """Extract the induced subgraph around a set of chunk nodes."""
        keep: Set[str] = set()
        for cid in chunk_ids:
            keep |= self.get_chunk_neighbors(cid, max_hops)
        return self.graph.subgraph(keep).copy()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dgl(self):
        """Convert to a DGL heterograph (requires dgl)."""
        import dgl

        # Build per-edge-type (src, dst) lists
        edge_data: Dict[str, Tuple[List[int], List[int]]] = defaultdict(
            lambda: ([], [])
        )
        # Assign integer ids per node
        node_list = list(self.graph.nodes())
        node2idx = {n: i for i, n in enumerate(node_list)}

        for u, v, key, etype in self.graph.edges(keys=True, data="edge_type"):
            srcs, dsts = edge_data[etype]
            srcs.append(node2idx[u])
            dsts.append(node2idx[v])

        data_dict = {}
        for etype, (srcs, dsts) in edge_data.items():
            data_dict[("node", etype, "node")] = (
                np.array(srcs, dtype=np.int64),
                np.array(dsts, dtype=np.int64),
            )

        g = dgl.heterograph(data_dict)
        g.ndata["_id"] = node_list
        return g

    def to_pyg(self):
        """Convert to a PyG HeteroData (requires torch_geometric)."""
        from torch_geometric.data import HeteroData

        data = HeteroData()
        node_list = list(self.graph.nodes())
        node2idx = {n: i for i, n in enumerate(node_list)}

        # Group edges by type
        edge_dict: Dict[str, Tuple[List[int], List[int]]] = defaultdict(
            lambda: ([], [])
        )
        for u, v, key, etype in self.graph.edges(keys=True, data="edge_type"):
            edge_dict[etype][0].append(node2idx[u])
            edge_dict[etype][1].append(node2idx[v])

        for etype, (srcs, dsts) in edge_dict.items():
            data[("node", etype, "node")].edge_index = np.array(
                [srcs, dsts], dtype=np.int64
            )

        return data

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"graph": self.graph, "node_types": self.node_types}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "FinancialEvidenceGraph":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls()
        obj.graph = data["graph"]
        obj.node_types = data["node_types"]
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def edge_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for _u, _v, _k, etype in self.graph.edges(keys=True, data="edge_type"):
            counts[etype] += 1
        return dict(counts)

    def __repr__(self) -> str:
        return (
            f"FinancialEvidenceGraph(nodes={self.num_nodes}, edges={self.num_edges})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_node(self, node_id: str, node_type: str, **attrs) -> None:
        self.graph.add_node(node_id, node_type=node_type, **attrs)
        self.node_types[node_id] = node_type

    def _add_edge(self, u: str, v: str, edge_type: str, weight: float = 1.0) -> None:
        self.graph.add_edge(u, v, edge_type=edge_type, weight=weight)

    def _add_semantic_edges(
        self,
        chunks: List[Chunk],
        embeddings: Dict[str, np.ndarray],
        threshold: float,
        max_edges: int,
        weight: float = 0.3,
    ) -> None:
        cids = [c.chunk_id for c in chunks if c.chunk_id in embeddings]
        if len(cids) < 2:
            return
        embs = np.stack([embeddings[cid] for cid in cids])
        sim = cosine_similarity(embs)
        n = len(cids)
        for i in range(n):
            row = sim[i]
            # get top-k most similar (excluding self)
            top = np.argsort(row)[::-1][1 : max_edges + 1]
            for j in top:
                if row[j] >= threshold:
                    self._add_edge(cids[i], cids[j], "semantic-similar", weight=weight)


# ═════════════════════════════════════════════════════════════════════════════
# Helper: capped same-entity edges
# ═════════════════════════════════════════════════════════════════════════════

def _add_capped_same_edges(
    graph: "FinancialEvidenceGraph",
    entity_map: Dict[str, List[str]],
    edge_type: str,
    weight: float = 0.5,
    max_edges: int = 20,
) -> None:
    """Add same-entity edges with a per-entity cap to prevent O(n²) blowup.

    For each entity value, at most *max_edges* edges are added by connecting
    chunks in a sparse structure (ring/tree) rather than a full clique.
    """
    import random
    for cids in entity_map.values():
        n = len(cids)
        if n < 2:
            continue
        # Cap: connect each chunk to at most its 2 nearest neighbors in list,
        # plus a few random connections for long-range information flow.
        # This gives O(n) edges instead of O(n²).
        for i in range(n):
            # local connections (sliding window of 2)
            for j in range(i + 1, min(i + 3, n)):
                graph._add_edge(cids[i], cids[j], edge_type, weight=weight)
        # Add a few random bridges for long-range connectivity
        if n > 3 and max_edges > 0:
            extra = min(max_edges, n // 2)
            sampled_pairs = set()
            for _ in range(extra):
                a, b = random.randint(0, n - 1), random.randint(0, n - 1)
                if a != b:
                    key = (min(a, b), max(a, b))
                    if key not in sampled_pairs:
                        sampled_pairs.add(key)
                        graph._add_edge(cids[a], cids[b], edge_type, weight=weight)


# ═════════════════════════════════════════════════════════════════════════════
# Convenience function
# ═════════════════════════════════════════════════════════════════════════════

def build_financial_evidence_graph(
    chunks: List[Chunk],
    chunk_embeddings: Optional[Dict[str, np.ndarray]] = None,
    use_edge_weights: bool = False,
    add_same_entity_edges: bool = False,
    max_same_entity_edges: int = 20,
    **kwargs,
) -> FinancialEvidenceGraph:
    """One-liner to build the financial evidence graph."""
    g = FinancialEvidenceGraph()
    g.build(chunks, chunk_embeddings=chunk_embeddings,
            use_edge_weights=use_edge_weights,
            add_same_entity_edges=add_same_entity_edges,
            max_same_entity_edges=max_same_entity_edges,
            **kwargs)
    return g
