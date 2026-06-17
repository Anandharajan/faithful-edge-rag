import argparse
from pathlib import Path

from faithful_edge_rag.experiments.beir import (
    download_scifact,
    evaluate_scifact,
    write_benchmark_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BEIR SciFact retrieval benchmark.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "benchmarks")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "beir-scifact")
    parser.add_argument("--max-corpus-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    dataset_dir = download_scifact(args.data_dir)
    row = evaluate_scifact(
        dataset_dir,
        max_corpus_docs=args.max_corpus_docs,
        max_queries=args.max_queries,
    )
    write_benchmark_result(row, args.output_dir)
    print(
        f"{row.dataset}: queries={row.queries}, docs={row.corpus_documents}, "
        f"recall@10={row.recall_at_10:.4f}, mrr@10={row.mrr_at_10:.4f}, "
        f"ndcg@10={row.ndcg_at_10:.4f}"
    )


if __name__ == "__main__":
    main()

