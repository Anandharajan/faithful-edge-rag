import csv
import json
from dataclasses import asdict
from pathlib import Path

from faithful_edge_rag.experiments.models import MetricRow


def write_results(rows: list[MetricRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(rows, output_dir / "metrics.json")
    write_csv(rows, output_dir / "metrics.csv")
    write_markdown(rows, output_dir / "summary.md")


def write_json(rows: list[MetricRow], path: Path) -> None:
    payload = []
    for row in rows:
        item = asdict(row)
        item["condition"] = row.condition.value
        payload.append(item)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(rows: list[MetricRow], path: Path) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = asdict(row)
            item["condition"] = row.condition.value
            writer.writerow(item)


def write_markdown(rows: list[MetricRow], path: Path) -> None:
    lines = [
        "# Initial Experiment Results",
        "",
        "These results come from the deterministic synthetic benchmark included in the "
        "repository. They are intended as a reproducible engineering baseline, not a final "
        "SCI-grade empirical claim.",
        "",
        "| Condition | Accuracy | Recall@5 | MRR | Citation Faithfulness | Conflict F1 | "
        "Avg Tokens | Avg Private Bytes | Avg Transfer Bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.condition.value} | {row.answer_accuracy:.3f} | {row.recall_at_5:.3f} | "
            f"{row.mrr:.3f} | {row.citation_faithfulness:.3f} | {row.conflict_f1:.3f} | "
            f"{row.avg_tokens_used:.1f} | {row.avg_raw_private_bytes_moved:.1f} | "
            f"{row.avg_edge_to_central_bytes:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Centralized RAG is a strong retrieval baseline but moves private raw evidence.",
            "- Long-context stuffing consumes more context and is intentionally stress-tested "
            "against stale/conflicting evidence.",
            "- Edge-only RAG minimizes transfer but cannot represent collaborative "
            "edge-cloud behavior.",
            "- The proposed condition adds conflict detection and context budgeting while limiting "
            "private evidence movement.",
            "",
            "## Next Step for Publication-Grade Results",
            "",
            "Replace the deterministic lexical retriever with open-source embedding models and "
            "Qdrant/LanceDB indexes, run larger public and synthetic conflict datasets, "
            "add confidence intervals, and report hardware-normalized latency and energy "
            "or compute measurements.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
