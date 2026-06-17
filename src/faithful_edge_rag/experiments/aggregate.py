import math
from collections import defaultdict

from faithful_edge_rag.experiments.models import AggregateMetricRow, Condition, MetricRow


def aggregate_by_condition(rows: list[MetricRow]) -> list[AggregateMetricRow]:
    grouped: dict[Condition, list[MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[row.condition].append(row)

    aggregates: list[AggregateMetricRow] = []
    for condition in sorted(grouped, key=lambda item: item.value):
        condition_rows = grouped[condition]
        aggregates.append(
            AggregateMetricRow(
                condition=condition,
                seeds=len(condition_rows),
                queries_per_seed=condition_rows[0].queries if condition_rows else 0,
                answer_accuracy_mean=_mean_attr(condition_rows, "answer_accuracy"),
                answer_accuracy_ci95=_ci95_attr(condition_rows, "answer_accuracy"),
                recall_at_5_mean=_mean_attr(condition_rows, "recall_at_5"),
                recall_at_5_ci95=_ci95_attr(condition_rows, "recall_at_5"),
                mrr_mean=_mean_attr(condition_rows, "mrr"),
                mrr_ci95=_ci95_attr(condition_rows, "mrr"),
                citation_faithfulness_mean=_mean_attr(
                    condition_rows, "citation_faithfulness"
                ),
                citation_faithfulness_ci95=_ci95_attr(
                    condition_rows, "citation_faithfulness"
                ),
                conflict_f1_mean=_mean_attr(condition_rows, "conflict_f1"),
                conflict_f1_ci95=_ci95_attr(condition_rows, "conflict_f1"),
                avg_tokens_used_mean=_mean_attr(condition_rows, "avg_tokens_used"),
                avg_tokens_used_ci95=_ci95_attr(condition_rows, "avg_tokens_used"),
                avg_raw_private_bytes_moved_mean=_mean_attr(
                    condition_rows, "avg_raw_private_bytes_moved"
                ),
                avg_raw_private_bytes_moved_ci95=_ci95_attr(
                    condition_rows, "avg_raw_private_bytes_moved"
                ),
                avg_edge_to_central_bytes_mean=_mean_attr(
                    condition_rows, "avg_edge_to_central_bytes"
                ),
                avg_edge_to_central_bytes_ci95=_ci95_attr(
                    condition_rows, "avg_edge_to_central_bytes"
                ),
                avg_latency_units_mean=_mean_attr(condition_rows, "avg_latency_units"),
                avg_latency_units_ci95=_ci95_attr(condition_rows, "avg_latency_units"),
            )
        )
    return aggregates


def _mean_attr(rows: list[MetricRow], attr: str) -> float:
    values = [float(getattr(row, attr)) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _ci95_attr(rows: list[MetricRow], attr: str) -> float:
    values = [float(getattr(row, attr)) for row in rows]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance) / math.sqrt(len(values))
    return 1.96 * standard_error

