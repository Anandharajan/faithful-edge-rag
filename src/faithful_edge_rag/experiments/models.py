from dataclasses import dataclass
from enum import StrEnum


class Condition(StrEnum):
    CENTRALIZED = "centralized_rag"
    LONG_CONTEXT = "long_context_rag"
    EDGE_ONLY = "edge_only_rag"
    PROPOSED = "edge_cloud_faithful_rag"
    PROPOSED_NO_CONFLICT = "edge_cloud_no_conflict_detection"
    PROPOSED_NO_BUDGET = "edge_cloud_no_context_budget"


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    topic: str
    text: str
    claim: str
    edge_node_id: str
    version: int
    authority: int
    is_private: bool
    is_summary: bool
    token_count: int


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    topic: str
    question: str
    edge_node_id: str
    expected_claim: str
    relevant_chunk_ids: tuple[str, ...]
    conflict_expected: bool
    requires_private_evidence: bool


@dataclass(frozen=True)
class RetrievalHit:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class AnswerTrace:
    condition: Condition
    query_id: str
    selected_claim: str | None
    cited_chunk_ids: tuple[str, ...]
    context_chunk_ids: tuple[str, ...]
    conflict_detected: bool
    tokens_used: int
    raw_private_bytes_moved: int
    edge_to_central_bytes: int
    retrieval_latency_units: int
    tool_calls: int


@dataclass(frozen=True)
class MetricRow:
    condition: Condition
    queries: int
    answer_accuracy: float
    recall_at_5: float
    mrr: float
    citation_correctness: float
    citation_faithfulness: float
    conflict_f1: float
    avg_tokens_used: float
    avg_raw_private_bytes_moved: float
    avg_edge_to_central_bytes: float
    avg_latency_units: float
    avg_tool_calls: float


@dataclass(frozen=True)
class AggregateMetricRow:
    condition: Condition
    seeds: int
    queries_per_seed: int
    answer_accuracy_mean: float
    answer_accuracy_ci95: float
    recall_at_5_mean: float
    recall_at_5_ci95: float
    mrr_mean: float
    mrr_ci95: float
    citation_faithfulness_mean: float
    citation_faithfulness_ci95: float
    conflict_f1_mean: float
    conflict_f1_ci95: float
    avg_tokens_used_mean: float
    avg_tokens_used_ci95: float
    avg_raw_private_bytes_moved_mean: float
    avg_raw_private_bytes_moved_ci95: float
    avg_edge_to_central_bytes_mean: float
    avg_edge_to_central_bytes_ci95: float
    avg_latency_units_mean: float
    avg_latency_units_ci95: float


@dataclass(frozen=True)
class RetrievalBenchmarkRow:
    dataset: str
    queries: int
    corpus_documents: int
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
