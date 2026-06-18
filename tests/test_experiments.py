from pathlib import Path

from faithful_edge_rag.experiments.beir import (
    evaluate_beir_dataset,
    evaluate_scifact,
    write_benchmark_result,
    write_benchmark_results,
)
from faithful_edge_rag.experiments.compare import collect_result_rows, write_comparison
from faithful_edge_rag.experiments.dense import TextEmbedder, evaluate_dense_scifact
from faithful_edge_rag.experiments.models import Condition
from faithful_edge_rag.experiments.publication import run_publication_track
from faithful_edge_rag.experiments.reporting import write_results
from faithful_edge_rag.experiments.runner import run_all_conditions
from faithful_edge_rag.experiments.seeded import build_seeded_corpus
from faithful_edge_rag.experiments.synthetic import build_synthetic_corpus


def test_synthetic_corpus_has_queries_and_conflicts() -> None:
    chunks, queries = build_synthetic_corpus()

    assert len(chunks) == 32
    assert len(queries) == 8
    assert all(query.conflict_expected for query in queries)
    assert any(chunk.is_private for chunk in chunks)
    assert any(chunk.is_summary for chunk in chunks)


def test_experiment_conditions_return_metrics() -> None:
    chunks, queries = build_synthetic_corpus()

    rows = run_all_conditions(chunks, queries)

    assert {row.condition for row in rows} == set(Condition)
    proposed = next(row for row in rows if row.condition is Condition.PROPOSED)
    centralized = next(row for row in rows if row.condition is Condition.CENTRALIZED)
    long_context = next(row for row in rows if row.condition is Condition.LONG_CONTEXT)

    assert proposed.conflict_f1 >= centralized.conflict_f1
    assert proposed.citation_faithfulness >= long_context.citation_faithfulness
    assert proposed.avg_tokens_used < long_context.avg_tokens_used


def test_results_are_written(tmp_path: Path) -> None:
    chunks, queries = build_synthetic_corpus()
    rows = run_all_conditions(chunks, queries)

    write_results(rows, tmp_path)

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "summary.md").exists()


def test_seeded_corpus_is_large_and_varied() -> None:
    chunks, queries = build_seeded_corpus(seed=7, topics=25, edge_nodes=3)

    assert len(queries) == 25
    assert len(chunks) > 75
    assert {query.edge_node_id for query in queries} == {"edge-0", "edge-1", "edge-2"}
    assert any(query.conflict_expected for query in queries)
    assert any(not query.conflict_expected for query in queries)


def test_publication_track_aggregates_seeded_runs() -> None:
    per_seed_rows, aggregate_rows = run_publication_track(seeds=3, topics=20, edge_nodes=2)

    assert len(per_seed_rows) == 3 * len(Condition)
    assert {row.condition for row in aggregate_rows} == set(Condition)
    proposed = next(row for row in aggregate_rows if row.condition is Condition.PROPOSED)
    no_conflict = next(
        row for row in aggregate_rows if row.condition is Condition.PROPOSED_NO_CONFLICT
    )

    assert proposed.conflict_f1_mean >= no_conflict.conflict_f1_mean
    assert proposed.seeds == 3
    assert proposed.queries_per_seed == 20


def test_beir_fixture_evaluates_retrieval_metrics(tmp_path: Path) -> None:
    dataset_dir = _write_tiny_scifact(tmp_path)

    row = evaluate_scifact(dataset_dir)
    write_benchmark_result(row, tmp_path / "out")

    assert row.queries == 1
    assert row.corpus_documents == 3
    assert row.recall_at_10 == 1.0
    assert (tmp_path / "out" / "summary.md").exists()


def test_generic_beir_fixture_writes_multiple_rows(tmp_path: Path) -> None:
    dataset_dir = _write_tiny_scifact(tmp_path)

    rows = [
        evaluate_beir_dataset("toy-a", dataset_dir),
        evaluate_beir_dataset("toy-b", dataset_dir),
    ]
    write_benchmark_results(rows, tmp_path / "multi", title="Toy Multi BEIR")

    assert [row.dataset for row in rows] == ["BEIR toy-a", "BEIR toy-b"]
    assert (tmp_path / "multi" / "metrics.json").exists()
    assert "BEIR toy-a" in (tmp_path / "multi" / "summary.md").read_text(encoding="utf-8")


def test_dense_scifact_fixture_evaluates_metrics(tmp_path: Path) -> None:
    dataset_dir = _write_tiny_scifact(tmp_path)

    row = evaluate_dense_scifact(dataset_dir, ToyEmbedder())

    assert row.retriever == "dense"
    assert row.queries == 1
    assert row.recall_at_10 == 1.0


def test_compare_collects_metric_files(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "results" / "colab" / "beir-scifact-bge"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(
        """[
  {
    "dataset": "BEIR SciFact",
    "retriever": "dense",
    "model_name": "BAAI/bge-small-en-v1.5",
    "queries": 300,
    "corpus_documents": 5183,
    "recall_at_5": 0.7,
    "recall_at_10": 0.8,
    "mrr_at_10": 0.6,
    "ndcg_at_10": 0.65,
    "index_seconds": 12.0,
    "query_seconds": 3.0
  }
]""",
        encoding="utf-8",
    )

    rows = collect_result_rows(tmp_path / "results")
    write_comparison(rows, tmp_path / "results" / "comparison")

    assert len(rows) == 1
    assert rows[0]["retriever"] == "dense"
    assert (tmp_path / "results" / "comparison" / "retrieval_comparison.md").exists()


class ToyEmbedder(TextEmbedder):
    model_name = "toy-embedder"

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("platelet" in lowered or "aggregation" in lowered),
                    float("insulin" in lowered or "glucose" in lowered),
                    float("astronomy" in lowered),
                ]
            )
        return vectors


def _write_tiny_scifact(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "scifact"
    qrels_dir = dataset_dir / "qrels"
    qrels_dir.mkdir(parents=True)
    (dataset_dir / "corpus.jsonl").write_text(
        "\n".join(
            [
                (
                    '{"_id": "d1", "title": "Aspirin", '
                    '"text": "Aspirin reduces platelet aggregation."}'
                ),
                '{"_id": "d2", "title": "Insulin", "text": "Insulin regulates blood glucose."}',
                (
                    '{"_id": "d3", "title": "Noise", '
                    '"text": "This document discusses unrelated astronomy."}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "queries.jsonl").write_text(
        '{"_id": "q1", "text": "What reduces platelet aggregation?"}\n',
        encoding="utf-8",
    )
    (qrels_dir / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\t0\td1\t1\n",
        encoding="utf-8",
    )
    return dataset_dir
