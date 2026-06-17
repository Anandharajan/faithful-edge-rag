import argparse
from pathlib import Path

from faithful_edge_rag.experiments.publication import (
    run_publication_track,
    write_publication_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run publication-track seeded edge-cloud RAG experiments."
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--topics", type=int, default=120)
    parser.add_argument("--edge-nodes", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "publication")
    args = parser.parse_args()

    per_seed_rows, aggregate_rows = run_publication_track(
        seeds=args.seeds,
        topics=args.topics,
        edge_nodes=args.edge_nodes,
    )
    write_publication_results(per_seed_rows, aggregate_rows, args.output_dir)
    for row in aggregate_rows:
        print(
            f"{row.condition.value}: accuracy={row.answer_accuracy_mean:.3f} "
            f"+/- {row.answer_accuracy_ci95:.3f}, faithfulness="
            f"{row.citation_faithfulness_mean:.3f} +/- "
            f"{row.citation_faithfulness_ci95:.3f}, conflict_f1="
            f"{row.conflict_f1_mean:.3f} +/- {row.conflict_f1_ci95:.3f}"
        )


if __name__ == "__main__":
    main()

