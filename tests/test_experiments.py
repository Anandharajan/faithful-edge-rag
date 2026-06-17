from pathlib import Path

from faithful_edge_rag.experiments.models import Condition
from faithful_edge_rag.experiments.reporting import write_results
from faithful_edge_rag.experiments.runner import run_all_conditions
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
