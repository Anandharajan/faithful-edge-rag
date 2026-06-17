from pathlib import Path

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
