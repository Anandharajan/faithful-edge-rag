import csv
import json
from dataclasses import asdict
from pathlib import Path

from faithful_edge_rag.experiments.aggregate import aggregate_by_condition
from faithful_edge_rag.experiments.models import AggregateMetricRow, MetricRow
from faithful_edge_rag.experiments.runner import run_all_conditions
from faithful_edge_rag.experiments.seeded import build_seeded_corpus


def run_publication_track(
    *, seeds: int, topics: int, edge_nodes: int
) -> tuple[list[MetricRow], list[AggregateMetricRow]]:
    rows: list[MetricRow] = []
    for seed in range(seeds):
        chunks, queries = build_seeded_corpus(seed=seed, topics=topics, edge_nodes=edge_nodes)
        rows.extend(run_all_conditions(chunks, queries))
    return rows, aggregate_by_condition(rows)


def write_publication_results(
    per_seed_rows: list[MetricRow],
    aggregate_rows: list[AggregateMetricRow],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(per_seed_rows, output_dir / "per_seed_metrics.json")
    _write_csv(per_seed_rows, output_dir / "per_seed_metrics.csv")
    _write_aggregate_json(aggregate_rows, output_dir / "aggregate_metrics.json")
    _write_aggregate_csv(aggregate_rows, output_dir / "aggregate_metrics.csv")
    _write_summary(aggregate_rows, output_dir / "summary.md")


def _write_json(rows: list[MetricRow], path: Path) -> None:
    payload = []
    for index, row in enumerate(rows):
        item = asdict(row)
        item["condition"] = row.condition.value
        item["seed_index"] = index // 6
        payload.append(item)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(rows: list[MetricRow], path: Path) -> None:
    fieldnames = ["seed_index", *list(asdict(rows[0]).keys())] if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            item = asdict(row)
            item["condition"] = row.condition.value
            item["seed_index"] = index // 6
            writer.writerow(item)


def _write_aggregate_json(rows: list[AggregateMetricRow], path: Path) -> None:
    payload = []
    for row in rows:
        item = asdict(row)
        item["condition"] = row.condition.value
        payload.append(item)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_aggregate_csv(rows: list[AggregateMetricRow], path: Path) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = asdict(row)
            item["condition"] = row.condition.value
            writer.writerow(item)


def _write_summary(rows: list[AggregateMetricRow], path: Path) -> None:
    lines = [
        "# Publication-Track Experiment Results",
        "",
        "These results aggregate seeded synthetic conflict scenarios. Values are reported "
        "as mean +/- 95% confidence interval across seeds.",
        "",
        "| Condition | Accuracy | Citation Faithfulness | Conflict F1 | Avg Tokens | "
        "Private Bytes | Transfer Bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.condition.value} | "
            f"{row.answer_accuracy_mean:.3f} +/- {row.answer_accuracy_ci95:.3f} | "
            f"{row.citation_faithfulness_mean:.3f} +/- "
            f"{row.citation_faithfulness_ci95:.3f} | "
            f"{row.conflict_f1_mean:.3f} +/- {row.conflict_f1_ci95:.3f} | "
            f"{row.avg_tokens_used_mean:.1f} +/- {row.avg_tokens_used_ci95:.1f} | "
            f"{row.avg_raw_private_bytes_moved_mean:.1f} +/- "
            f"{row.avg_raw_private_bytes_moved_ci95:.1f} | "
            f"{row.avg_edge_to_central_bytes_mean:.1f} +/- "
            f"{row.avg_edge_to_central_bytes_ci95:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Reading the Table",
            "",
            "- `edge_cloud_no_conflict_detection` tests the contribution of conflict detection.",
            "- `edge_cloud_no_context_budget` tests the contribution of bounded context selection.",
            "- Confidence intervals are across seeded scenario generations, not across "
            "real-world sites.",
            "",
            "## Remaining Publication Gap",
            "",
            "This benchmark is larger and statistically summarized, but it still uses synthetic "
            "claims and deterministic retrieval. SCI-grade submission requires public datasets, "
            "open-source embedding/vector-store experiments, hardware measurements, and human "
            "or expert adjudication for sampled faithfulness labels.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
