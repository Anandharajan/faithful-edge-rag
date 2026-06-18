import argparse
import csv
import json
from pathlib import Path
from typing import Any


def collect_result_rows(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metrics_path in sorted(results_dir.glob("**/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict):
                continue
            if "dataset" not in record or "ndcg_at_10" not in record:
                continue
            rows.append(_normalize_row(record, metrics_path.parent))
    return rows


def write_comparison(rows: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row["dataset"],
            row["source"],
            row["retriever"],
            row["model_name"],
        ),
    )
    write_csv(sorted_rows, output_dir / "retrieval_comparison.csv")
    write_markdown(sorted_rows, output_dir / "retrieval_comparison.md")


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
        "source",
        "dataset",
        "retriever",
        "model_name",
        "queries",
        "corpus_documents",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "index_seconds",
        "query_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        "# Retrieval Benchmark Comparison",
        "",
        "This table aggregates available real-world retrieval benchmark outputs. "
        "Add Colab outputs under `results/colab/` and rerun the comparison command "
        "to include dense and hybrid GPU results.",
        "",
        "| Source | Dataset | Retriever | Model | Queries | Docs | Recall@5 | "
        "Recall@10 | MRR@10 | nDCG@10 | Index s | Query s |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | {row['dataset']} | {row['retriever']} | "
            f"{row['model_name']} | {row['queries']} | {row['corpus_documents']} | "
            f"{row['recall_at_5']} | {row['recall_at_10']} | {row['mrr_at_10']} | "
            f"{row['ndcg_at_10']} | {row['index_seconds']} | {row['query_seconds']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_row(record: dict[str, Any], source_dir: Path) -> dict[str, str]:
    retriever = str(record.get("retriever") or "lexical")
    model_name = str(record.get("model_name") or "bm25_lexical")
    return {
        "source": source_dir.as_posix(),
        "dataset": str(record.get("dataset", "")),
        "retriever": retriever,
        "model_name": model_name,
        "queries": _value(record, "queries"),
        "corpus_documents": _value(record, "corpus_documents"),
        "recall_at_5": _number(record, "recall_at_5"),
        "recall_at_10": _number(record, "recall_at_10"),
        "mrr_at_10": _number(record, "mrr_at_10"),
        "ndcg_at_10": _number(record, "ndcg_at_10"),
        "index_seconds": _number(record, "index_seconds"),
        "query_seconds": _number(record, "query_seconds"),
    }


def _value(record: dict[str, Any], key: str) -> str:
    value = record.get(key, "")
    return str(value)


def _number(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval benchmark result files.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "comparison")
    args = parser.parse_args()

    rows = collect_result_rows(args.results_dir)
    write_comparison(rows, args.output_dir)
    print(f"Wrote {len(rows)} comparison rows to {args.output_dir}")


if __name__ == "__main__":
    main()
