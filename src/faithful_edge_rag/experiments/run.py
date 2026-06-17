import argparse
from pathlib import Path

from faithful_edge_rag.experiments.reporting import write_results
from faithful_edge_rag.experiments.runner import run_all_conditions
from faithful_edge_rag.experiments.synthetic import build_synthetic_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic edge-cloud RAG experiments.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "initial",
        help="Directory where metrics.json, metrics.csv, and summary.md will be written.",
    )
    args = parser.parse_args()

    chunks, queries = build_synthetic_corpus()
    rows = run_all_conditions(chunks, queries)
    write_results(rows, args.output_dir)
    for row in rows:
        print(
            f"{row.condition.value}: accuracy={row.answer_accuracy:.3f}, "
            f"faithfulness={row.citation_faithfulness:.3f}, conflict_f1={row.conflict_f1:.3f}, "
            f"avg_tokens={row.avg_tokens_used:.1f}"
        )


if __name__ == "__main__":
    main()
