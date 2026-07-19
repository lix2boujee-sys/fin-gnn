"""Evidence reranking: PPR, GNN (GraphSAGE, GATv2), R-GCN, R-GCN Lite, DCF-GNN, C2-DCF-GNN, QFE-RGCN, FinPath-RGCN, MonoT5, ListT5, QCE-Graph Lite, and constraint-aware fusion."""

from feg_rag.rerank.ppr import ppr_rerank
from feg_rag.rerank.gnn import (
    GraphSAGEReranker,
    GNNFusionReranker,
    RerankDataset,
    DenseGATv2Layer,
    GATv2Reranker,
)
from feg_rag.rerank.rgcn import RGCNReranker, RGCNFusionReranker, RGCNRerankDataset
from feg_rag.rerank.rgcn import LiteRGCNLayer, LiteRGCNReranker
from feg_rag.rerank.dcf_gnn import (
    DCFGNNReranker,
    DCFGNNFusionReranker,
    DCFRerankDataset,
    split_relation_channels,
    infer_query_type_features,
)
from feg_rag.rerank.path_encoder import FinancialPath, FinancialPathExtractor
from feg_rag.rerank.finpath_rgcn import FinPathRGCNReranker
from feg_rag.rerank.finmuse import FinMUSESetReranker
from feg_rag.rerank.qfe_rgcn import (
    QFERGCNLayer,
    QFERGCNReranker,
    QFERGCNRerankDataset,
    QFERGCNFusionReranker,
    EntityGatedScoringHead,
    QUERY_EMBED_DIM,
    derive_query_vector,
    build_query_embedding_cache,
)
from feg_rag.rerank.c2_dcf_gnn import (
    C2DCFGNNReranker,
    C2DCFFusionReranker,
    C2DCFDataset,
    QUERY_TYPES,
    infer_query_entity_features,
    build_retrieval_features,
    build_conflict_features,
    write_c2_diagnostics,
)
from feg_rag.rerank.fusion import ConstraintScorer, FusionScorer
from feg_rag.rerank.query_features import (
    QUERY_FEATURE_DIM,
    build_query_augmented_features,
)
from feg_rag.rerank.scoring import normalise_score_map
from feg_rag.rerank.mono_t5 import MonoT5Reranker, run_mono_t5_reranking
from feg_rag.rerank.list_t5 import ListT5Reranker, run_list_t5_reranking

# QCE-Graph Lite
from feg_rag.rerank.qce_expansion import (
    RELATION_NAMES,
    NUM_RELATIONS,
    GraphExpansionIndex,
    BudgetedGraphExpander,
    ExpandedCandidate,
    compute_expansion_diagnostics,
)
from feg_rag.rerank.qce_features import (
    QUERY_FEATURE_DIM_QCE,
    SUPPORT_FEATURE_DIM,
    CONFLICT_FEATURE_DIM,
    build_qce_query_features,
    extract_support_features,
    extract_conflict_features,
)
from feg_rag.rerank.qce_graph import (
    QueryRelationRouter,
    CounterfactualEvidenceScorer,
    QCEGraphLiteReranker,
    QCEInferencePipeline,
    QCEFixedCandidatePipeline,
    save_qce_checkpoint,
    load_qce_checkpoint,
)
from feg_rag.rerank.qce_dataset import (
    QCERerankDataset,
    build_qce_training_candidates,
    build_qce_rerank_candidates,
)
