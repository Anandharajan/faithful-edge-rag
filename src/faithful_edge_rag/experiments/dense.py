import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from faithful_edge_rag.experiments.beir import (
    _load_corpus,
    _load_qrels,
    _load_queries,
    _mrr_at_k,
    _ndcg_at_k,
    _recall_at_k,
    download_scifact,
)
from faithful_edge_rag.experiments.beir import evaluate_scifact as evaluate_lexical_scifact
from faithful_edge_rag.experiments.models import DocumentChunk
from faithful_edge_rag.experiments.retrieval import LexicalRetriever

MetricValue: TypeAlias = str | int | float
MetricDict: TypeAlias = dict[str, MetricValue]


class TextEmbedder(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts into dense vectors."""
        ...


@dataclass(frozen=True)
class DenseBenchmarkRow:
    dataset: str
    retriever: str
    model_name: str
    queries: int
    corpus_documents: int
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    index_seconds: float
    query_seconds: float


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, *, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised in Colab/runtime envs
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval. "
                "Install with: pip install -e \".[rag]\""
            ) from exc

        self.model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return [list(map(float, vector)) for vector in vectors]


def evaluate_dense_scifact(
    dataset_dir: Path,
    embedder: TextEmbedder,
    *,
    max_corpus_docs: int | None = None,
    max_queries: int | None = None,
    hybrid_alpha: float | None = None,
) -> DenseBenchmarkRow:
    chunks = _load_corpus(dataset_dir / "corpus.jsonl", max_docs=max_corpus_docs)
    queries = _load_queries(dataset_dir / "queries.jsonl", max_queries=max_queries)
    qrels = _load_qrels(dataset_dir / "qrels" / "test.tsv")

    corpus_texts = [chunk.text for chunk in chunks]
    start_index = time.perf_counter()
    corpus_vectors = [_normalize(vector) for vector in embedder.encode(corpus_texts)]
    index_seconds = time.perf_counter() - start_index

    lexical_hits_by_query: dict[str, list[str]] = {}
    lexical_scores_by_query: dict[str, dict[str, float]] = {}
    if hybrid_alpha is not None:
        lexical = LexicalRetriever(chunks)
        for query_id, query_text in queries.items():
            hits = lexical.search(query_text, top_k=50)
            lexical_hits_by_query[query_id] = [hit.chunk.chunk_id for hit in hits]
            lexical_scores_by_query[query_id] = {
                hit.chunk.chunk_id: hit.score for hit in hits
            }

    query_ids = list(queries)
    query_texts = [queries[query_id] for query_id in query_ids]
    start_query = time.perf_counter()
    query_vectors = [_normalize(vector) for vector in embedder.encode(query_texts)]

    recall_5: list[float] = []
    recall_10: list[float] = []
    mrr_10: list[float] = []
    ndcg_10: list[float] = []
    evaluated_queries = 0
    available_doc_ids = {chunk.chunk_id for chunk in chunks}

    for query_id, query_vector in zip(query_ids, query_vectors, strict=True):
        relevant = {
            doc_id
            for doc_id, score in qrels.get(query_id, {}).items()
            if score > 0 and doc_id in available_doc_ids
        }
        if not relevant:
            continue
        scored = _score_dense(chunks, corpus_vectors, query_vector)
        if hybrid_alpha is not None:
            scored = _score_hybrid(
                scored,
                lexical_hits_by_query.get(query_id, []),
                lexical_scores_by_query.get(query_id, {}),
                alpha=hybrid_alpha,
            )
        hit_ids = [doc_id for doc_id, _score in scored[:10]]
        recall_5.append(_recall_at_k(hit_ids, relevant, k=5))
        recall_10.append(_recall_at_k(hit_ids, relevant, k=10))
        mrr_10.append(_mrr_at_k(hit_ids, relevant, k=10))
        ndcg_10.append(_ndcg_at_k(hit_ids, relevant, k=10))
        evaluated_queries += 1

    query_seconds = time.perf_counter() - start_query
    retriever = "hybrid" if hybrid_alpha is not None else "dense"
    return DenseBenchmarkRow(
        dataset="BEIR SciFact",
        retriever=retriever,
        model_name=embedder.model_name,
        queries=evaluated_queries,
        corpus_documents=len(chunks),
        recall_at_5=_mean(recall_5),
        recall_at_10=_mean(recall_10),
        mrr_at_10=_mean(mrr_10),
        ndcg_at_10=_mean(ndcg_10),
        index_seconds=index_seconds,
        query_seconds=query_seconds,
    )


def run_scifact_comparison(
    *,
    data_dir: Path,
    output_dir: Path,
    embedding_model: str,
    max_corpus_docs: int | None,
    max_queries: int | None,
    hybrid_alpha: float,
) -> list[MetricDict]:
    dataset_dir = download_scifact(data_dir)
    lexical = evaluate_lexical_scifact(
        dataset_dir,
        max_corpus_docs=max_corpus_docs,
        max_queries=max_queries,
    )
    embedder = SentenceTransformerEmbedder(embedding_model)
    dense = evaluate_dense_scifact(
        dataset_dir,
        embedder,
        max_corpus_docs=max_corpus_docs,
        max_queries=max_queries,
    )
    hybrid = evaluate_dense_scifact(
        dataset_dir,
        embedder,
        max_corpus_docs=max_corpus_docs,
        max_queries=max_queries,
        hybrid_alpha=hybrid_alpha,
    )
    lexical_row = cast(MetricDict, asdict(lexical))
    rows: list[MetricDict] = [
        {"retriever": "lexical", "model_name": "bm25_lexical", **lexical_row},
        cast(MetricDict, asdict(dense)),
        cast(MetricDict, asdict(hybrid)),
    ]
    write_dense_results(rows, output_dir)
    return rows


def write_dense_results(rows: list[MetricDict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    headers = [
        "retriever",
        "model_name",
        "dataset",
        "queries",
        "corpus_documents",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "index_seconds",
        "query_seconds",
    ]
    csv_lines = [",".join(headers)]
    for row in rows:
        csv_lines.append(",".join(str(row.get(header, "")) for header in headers))
    (output_dir / "metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(_summary(rows), encoding="utf-8")


def _score_dense(
    chunks: Sequence[DocumentChunk],
    corpus_vectors: Sequence[Sequence[float]],
    query_vector: Sequence[float],
) -> list[tuple[str, float]]:
    scored = [
        (chunk.chunk_id, _dot(query_vector, vector))
        for chunk, vector in zip(chunks, corpus_vectors, strict=True)
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)


def _score_hybrid(
    dense_scores: list[tuple[str, float]],
    lexical_hit_ids: list[str],
    lexical_scores: dict[str, float],
    *,
    alpha: float,
) -> list[tuple[str, float]]:
    dense_normalized = _normalize_scores(dict(dense_scores[:50]))
    lexical_normalized = _normalize_scores(lexical_scores)
    candidate_ids = set(dense_normalized) | set(lexical_hit_ids)
    scored = [
        (
            doc_id,
            alpha * dense_normalized.get(doc_id, 0.0)
            + (1 - alpha) * lexical_normalized.get(doc_id, 0.0),
        )
        for doc_id in candidate_ids
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if math.isclose(maximum, minimum):
        return {key: 1.0 for key in scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summary(rows: list[MetricDict]) -> str:
    lines = [
        "# Dense and Hybrid SciFact Benchmark",
        "",
        "| Retriever | Model | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | "
        "Index Seconds | Query Seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['retriever']} | {row['model_name']} | "
            f"{_metric_float(row, 'recall_at_5'):.4f} | "
            f"{_metric_float(row, 'recall_at_10'):.4f} | "
            f"{_metric_float(row, 'mrr_at_10'):.4f} | "
            f"{_metric_float(row, 'ndcg_at_10'):.4f} | "
            f"{_metric_float(row, 'index_seconds'):.2f} | "
            f"{_metric_float(row, 'query_seconds'):.2f} |"
        )
    lines.extend(
        [
            "",
            "These results compare the existing lexical baseline with dense and hybrid "
            "retrieval using an open-source sentence-transformer embedding model.",
        ]
    )
    return "\n".join(lines) + "\n"


def _metric_float(row: MetricDict, key: str) -> float:
    value = row.get(key, 0.0)
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense/hybrid BEIR SciFact benchmark.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "benchmarks")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "beir-dense")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--max-corpus-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--hybrid-alpha", type=float, default=0.65)
    args = parser.parse_args()

    rows = run_scifact_comparison(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        max_corpus_docs=args.max_corpus_docs,
        max_queries=args.max_queries,
        hybrid_alpha=args.hybrid_alpha,
    )
    for row in rows:
        print(
            f"{row['retriever']} ({row['model_name']}): "
            f"recall@10={_metric_float(row, 'recall_at_10'):.4f}, "
            f"mrr@10={_metric_float(row, 'mrr_at_10'):.4f}, "
            f"ndcg@10={_metric_float(row, 'ndcg_at_10'):.4f}"
        )


if __name__ == "__main__":
    main()
