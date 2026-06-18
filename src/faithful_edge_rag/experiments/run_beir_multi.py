import argparse
from pathlib import Path

from faithful_edge_rag.experiments.beir import (
    DEFAULT_BEIR_DATASETS,
    evaluate_beir_datasets,
    write_benchmark_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lexical BEIR benchmarks.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "benchmarks")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "beir-multi")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_BEIR_DATASETS))
    parser.add_argument("--max-corpus-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    rows = evaluate_beir_datasets(
        args.datasets,
        args.data_dir,
        max_corpus_docs=args.max_corpus_docs,
        max_queries=args.max_queries,
    )
    write_benchmark_results(rows, args.output_dir, title="Multi-Dataset BEIR Benchmark Results")
    for row in rows:
        print(
            f"{row.dataset}: queries={row.queries}, docs={row.corpus_documents}, "
            f"recall@10={row.recall_at_10:.4f}, mrr@10={row.mrr_at_10:.4f}, "
            f"ndcg@10={row.ndcg_at_10:.4f}"
        )


if __name__ == "__main__":
    main()

