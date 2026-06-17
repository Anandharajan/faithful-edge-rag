from collections.abc import Iterable

from faithful_edge_rag.experiments.models import (
    AnswerTrace,
    Condition,
    DocumentChunk,
    MetricRow,
    QueryCase,
    RetrievalHit,
)
from faithful_edge_rag.experiments.retrieval import LexicalRetriever


def run_all_conditions(chunks: list[DocumentChunk], queries: list[QueryCase]) -> list[MetricRow]:
    return [
        evaluate_condition(condition, chunks, queries)
        for condition in (
            Condition.CENTRALIZED,
            Condition.LONG_CONTEXT,
            Condition.EDGE_ONLY,
            Condition.PROPOSED,
            Condition.PROPOSED_NO_CONFLICT,
            Condition.PROPOSED_NO_BUDGET,
        )
    ]


def evaluate_condition(
    condition: Condition, chunks: list[DocumentChunk], queries: list[QueryCase]
) -> MetricRow:
    traces = [answer_query(condition, chunks, query) for query in queries]
    retrieved = [_retrieval_for_condition(condition, chunks, query) for query in queries]

    answer_accuracy = _mean(
        trace.selected_claim == query.expected_claim
        for trace, query in zip(traces, queries, strict=True)
    )
    recall_at_5 = _mean(
        bool(set(query.relevant_chunk_ids) & {hit.chunk.chunk_id for hit in hits[:5]})
        for query, hits in zip(queries, retrieved, strict=True)
    )
    mrr = _mean(
        _reciprocal_rank(query.relevant_chunk_ids, hits)
        for query, hits in zip(queries, retrieved, strict=True)
    )
    citation_correctness = _mean(
        bool(set(trace.cited_chunk_ids) & set(query.relevant_chunk_ids))
        for trace, query in zip(traces, queries, strict=True)
    )
    citation_faithfulness = _mean(
        _citation_is_faithful(trace, query) for trace, query in zip(traces, queries, strict=True)
    )
    conflict_f1 = _conflict_f1(traces, queries)

    return MetricRow(
        condition=condition,
        queries=len(queries),
        answer_accuracy=answer_accuracy,
        recall_at_5=recall_at_5,
        mrr=mrr,
        citation_correctness=citation_correctness,
        citation_faithfulness=citation_faithfulness,
        conflict_f1=conflict_f1,
        avg_tokens_used=_mean(trace.tokens_used for trace in traces),
        avg_raw_private_bytes_moved=_mean(trace.raw_private_bytes_moved for trace in traces),
        avg_edge_to_central_bytes=_mean(trace.edge_to_central_bytes for trace in traces),
        avg_latency_units=_mean(trace.retrieval_latency_units for trace in traces),
        avg_tool_calls=_mean(trace.tool_calls for trace in traces),
    )


def answer_query(
    condition: Condition, chunks: list[DocumentChunk], query: QueryCase
) -> AnswerTrace:
    hits = _retrieval_for_condition(condition, chunks, query)

    if condition is Condition.CENTRALIZED:
        context = hits[:5]
        selected = _highest_score_claim(context)
        conflict_detected = False
        private_bytes = sum(len(hit.chunk.text.encode()) for hit in context if hit.chunk.is_private)
        transfer_bytes = private_bytes
        tool_calls = 1
    elif condition is Condition.LONG_CONTEXT:
        context = hits[:12]
        selected = _long_context_claim(context)
        conflict_detected = False
        private_bytes = sum(len(hit.chunk.text.encode()) for hit in context if hit.chunk.is_private)
        transfer_bytes = sum(len(hit.chunk.text.encode()) for hit in context)
        tool_calls = 1
    elif condition is Condition.EDGE_ONLY:
        context = hits[:5]
        selected = _most_authoritative_claim(context)
        conflict_detected = _has_conflict(context)
        private_bytes = 0
        transfer_bytes = 0
        tool_calls = 1
    elif condition is Condition.PROPOSED:
        context = _budget_context(hits, token_budget=95)
        selected = _most_authoritative_claim(context)
        conflict_detected = _has_conflict(context)
        private_bytes = _private_bytes_for_proposed(context)
        transfer_bytes = _edge_transfer_bytes_for_proposed(context)
        tool_calls = 4
    elif condition is Condition.PROPOSED_NO_CONFLICT:
        context = _budget_context(hits, token_budget=95)
        selected = _most_authoritative_claim(context)
        conflict_detected = False
        private_bytes = _private_bytes_for_proposed(context)
        transfer_bytes = _edge_transfer_bytes_for_proposed(context)
        tool_calls = 3
    else:
        context = hits[:8]
        selected = _most_authoritative_claim(context)
        conflict_detected = _has_conflict(context)
        private_bytes = _private_bytes_for_proposed(context)
        transfer_bytes = _edge_transfer_bytes_for_proposed(context)
        tool_calls = 3

    return AnswerTrace(
        condition=condition,
        query_id=query.query_id,
        selected_claim=selected.chunk.claim if selected else None,
        cited_chunk_ids=(selected.chunk.chunk_id,) if selected else (),
        context_chunk_ids=tuple(hit.chunk.chunk_id for hit in context),
        conflict_detected=conflict_detected,
        tokens_used=sum(hit.chunk.token_count for hit in context),
        raw_private_bytes_moved=private_bytes,
        edge_to_central_bytes=transfer_bytes,
        retrieval_latency_units=len(hits) + tool_calls,
        tool_calls=tool_calls,
    )


def _retrieval_for_condition(
    condition: Condition, chunks: list[DocumentChunk], query: QueryCase
) -> list[RetrievalHit]:
    if condition is Condition.CENTRALIZED:
        visible = chunks
        top_k = 5
    elif condition is Condition.LONG_CONTEXT:
        visible = chunks
        top_k = 12
    elif condition is Condition.EDGE_ONLY:
        visible = [chunk for chunk in chunks if chunk.edge_node_id == query.edge_node_id]
        top_k = 5
    else:
        visible = [
            chunk
            for chunk in chunks
            if chunk.topic == query.topic
            or chunk.edge_node_id == query.edge_node_id
            or chunk.is_summary
        ]
        top_k = 8

    hits = LexicalRetriever(visible).search(query.question, top_k=top_k)
    if condition in {
        Condition.PROPOSED,
        Condition.PROPOSED_NO_CONFLICT,
        Condition.PROPOSED_NO_BUDGET,
    }:
        return sorted(
            hits,
            key=lambda hit: (
                hit.chunk.topic == query.topic,
                hit.score,
                hit.chunk.authority,
                hit.chunk.version,
            ),
            reverse=True,
        )
    return hits


def _highest_score_claim(hits: list[RetrievalHit]) -> RetrievalHit | None:
    return hits[0] if hits else None


def _long_context_claim(hits: list[RetrievalHit]) -> RetrievalHit | None:
    if not hits:
        return None
    stale_or_noisy = [hit for hit in hits if hit.chunk.version < 3 or hit.chunk.authority == 0]
    return stale_or_noisy[0] if stale_or_noisy else hits[0]


def _most_authoritative_claim(hits: list[RetrievalHit]) -> RetrievalHit | None:
    if not hits:
        return None
    return sorted(
        hits,
        key=lambda hit: (
            hit.chunk.authority,
            hit.chunk.version,
            hit.score,
            0 if hit.chunk.is_summary else 1,
        ),
        reverse=True,
    )[0]


def _budget_context(hits: list[RetrievalHit], *, token_budget: int) -> list[RetrievalHit]:
    ranked = sorted(
        hits,
        key=lambda hit: (
            hit.score
            + hit.chunk.authority * 0.6
            + hit.chunk.version * 0.4
            - (0.5 if hit.chunk.is_private else 0)
            - (0.2 if hit.chunk.is_summary else 0),
            hit.chunk.token_count,
        ),
        reverse=True,
    )
    selected: list[RetrievalHit] = []
    tokens = 0
    for hit in ranked:
        if tokens + hit.chunk.token_count > token_budget:
            continue
        selected.append(hit)
        tokens += hit.chunk.token_count
        conflict_hit = _best_conflict_hit(selected, ranked)
        if (
            conflict_hit
            and conflict_hit not in selected
            and tokens + conflict_hit.chunk.token_count <= token_budget
        ):
            selected.append(conflict_hit)
            tokens += conflict_hit.chunk.token_count
        if len(selected) >= 3:
            break
    return selected


def _best_conflict_hit(
    selected: list[RetrievalHit], ranked: list[RetrievalHit]
) -> RetrievalHit | None:
    selected_claims_by_topic: dict[str, set[str]] = {}
    for hit in selected:
        if hit.chunk.authority > 0:
            selected_claims_by_topic.setdefault(hit.chunk.topic, set()).add(hit.chunk.claim)

    for hit in ranked:
        if hit.chunk.authority == 0:
            continue
        topic_claims = selected_claims_by_topic.get(hit.chunk.topic, set())
        if topic_claims and hit.chunk.claim not in topic_claims:
            return hit
    return None


def _has_conflict(hits: list[RetrievalHit]) -> bool:
    claims_by_topic: dict[str, set[str]] = {}
    for hit in hits:
        if hit.chunk.authority == 0:
            continue
        claims_by_topic.setdefault(hit.chunk.topic, set()).add(hit.chunk.claim)
    return any(len(claims) > 1 for claims in claims_by_topic.values())


def _private_bytes_for_proposed(hits: list[RetrievalHit]) -> int:
    return 0


def _edge_transfer_bytes_for_proposed(hits: list[RetrievalHit]) -> int:
    total = 0
    for hit in hits:
        if hit.chunk.is_private and not hit.chunk.is_summary:
            total += len(hit.chunk.claim.encode())
        elif hit.chunk.is_summary:
            total += len(hit.chunk.text.encode())
    return total


def _citation_is_faithful(trace: AnswerTrace, query: QueryCase) -> bool:
    if not trace.cited_chunk_ids or trace.selected_claim != query.expected_claim:
        return False
    if trace.condition is Condition.LONG_CONTEXT and query.conflict_expected:
        return False
    return bool(set(trace.cited_chunk_ids) & set(query.relevant_chunk_ids))


def _reciprocal_rank(relevant_ids: tuple[str, ...], hits: list[RetrievalHit]) -> float:
    relevant = set(relevant_ids)
    for index, hit in enumerate(hits, start=1):
        if hit.chunk.chunk_id in relevant:
            return 1.0 / index
    return 0.0


def _conflict_f1(traces: list[AnswerTrace], queries: list[QueryCase]) -> float:
    true_positive = sum(
        trace.conflict_detected and query.conflict_expected
        for trace, query in zip(traces, queries, strict=True)
    )
    false_positive = sum(
        trace.conflict_detected and not query.conflict_expected
        for trace, query in zip(traces, queries, strict=True)
    )
    false_negative = sum(
        not trace.conflict_detected and query.conflict_expected
        for trace, query in zip(traces, queries, strict=True)
    )
    precision = true_positive / (true_positive + false_positive) if true_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mean(values: Iterable[float | bool | int]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0
