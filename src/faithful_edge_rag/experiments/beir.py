import json
import math
import shutil
import ssl
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

import certifi

from faithful_edge_rag.experiments.models import DocumentChunk, RetrievalBenchmarkRow
from faithful_edge_rag.experiments.retrieval import LexicalRetriever

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DEFAULT_BEIR_DATASETS = ("scifact", "nfcorpus", "fiqa", "trec-covid")


def dataset_url(dataset: str) -> str:
    return f"{BEIR_BASE_URL}/{dataset}.zip"


def download_scifact(data_dir: Path) -> Path:
    return download_beir_dataset("scifact", data_dir)


def download_beir_dataset(dataset: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_dir / dataset
    if (dataset_dir / "corpus.jsonl").exists():
        return dataset_dir

    archive_path = data_dir / f"{dataset}.zip"
    if not archive_path.exists():
        _download_file(dataset_url(dataset), archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(data_dir)
    return dataset_dir


def _download_file(url: str, path: Path) -> None:
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context) as response, path.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)


def evaluate_scifact(
    dataset_dir: Path,
    *,
    max_corpus_docs: int | None = None,
    max_queries: int | None = None,
) -> RetrievalBenchmarkRow:
    return evaluate_beir_dataset(
        "scifact",
        dataset_dir,
        max_corpus_docs=max_corpus_docs,
        max_queries=max_queries,
    )


def evaluate_beir_dataset(
    dataset: str,
    dataset_dir: Path,
    *,
    max_corpus_docs: int | None = None,
    max_queries: int | None = None,
) -> RetrievalBenchmarkRow:
    chunks = _load_corpus(dataset_dir / "corpus.jsonl", max_docs=max_corpus_docs)
    queries = _load_queries(dataset_dir / "queries.jsonl", max_queries=max_queries)
    qrels = _load_qrels(_find_qrels_path(dataset_dir))

    retriever = LexicalRetriever(chunks)
    recall_5: list[float] = []
    recall_10: list[float] = []
    mrr_10: list[float] = []
    ndcg_10: list[float] = []

    evaluated_queries = 0
    available_doc_ids = {chunk.chunk_id for chunk in chunks}
    for query_id, query_text in queries.items():
        relevant = {
            doc_id
            for doc_id, score in qrels.get(query_id, {}).items()
            if score > 0 and doc_id in available_doc_ids
        }
        if not relevant:
            continue
        hits = retriever.search(query_text, top_k=10)
        hit_ids = [hit.chunk.chunk_id for hit in hits]
        recall_5.append(_recall_at_k(hit_ids, relevant, k=5))
        recall_10.append(_recall_at_k(hit_ids, relevant, k=10))
        mrr_10.append(_mrr_at_k(hit_ids, relevant, k=10))
        ndcg_10.append(_ndcg_at_k(hit_ids, relevant, k=10))
        evaluated_queries += 1

    return RetrievalBenchmarkRow(
        dataset=f"BEIR {dataset}",
        queries=evaluated_queries,
        corpus_documents=len(chunks),
        recall_at_5=_mean(recall_5),
        recall_at_10=_mean(recall_10),
        mrr_at_10=_mean(mrr_10),
        ndcg_at_10=_mean(ndcg_10),
    )


def evaluate_beir_datasets(
    datasets: list[str],
    data_dir: Path,
    *,
    max_corpus_docs: int | None = None,
    max_queries: int | None = None,
) -> list[RetrievalBenchmarkRow]:
    rows: list[RetrievalBenchmarkRow] = []
    for dataset in datasets:
        dataset_dir = download_beir_dataset(dataset, data_dir)
        rows.append(
            evaluate_beir_dataset(
                dataset,
                dataset_dir,
                max_corpus_docs=max_corpus_docs,
                max_queries=max_queries,
            )
        )
    return rows


def write_benchmark_result(row: RetrievalBenchmarkRow, output_dir: Path) -> None:
    write_benchmark_results([row], output_dir, title=f"{row.dataset} Benchmark Results")


def write_benchmark_results(
    rows: list[RetrievalBenchmarkRow], output_dir: Path, *, title: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    json_payload = payload[0] if len(payload) == 1 else payload
    (output_dir / "metrics.json").write_text(
        json.dumps(json_payload, indent=2) + "\n", encoding="utf-8"
    )
    header = "dataset,queries,corpus_documents,recall_at_5,recall_at_10,mrr_at_10,ndcg_at_10"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.dataset},{row.queries},{row.corpus_documents},{row.recall_at_5},"
            f"{row.recall_at_10},{row.mrr_at_10},{row.ndcg_at_10}"
        )
    (output_dir / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(_summary(rows, title=title), encoding="utf-8")


def _load_corpus(path: Path, *, max_docs: int | None) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_docs is not None and index >= max_docs:
                break
            record = json.loads(line)
            title = record.get("title", "")
            text = record.get("text", "")
            body = f"{title}. {text}".strip()
            chunks.append(
                DocumentChunk(
                    chunk_id=str(record["_id"]),
                    topic="beir",
                    text=body,
                    claim=body[:240],
                    edge_node_id="central",
                    version=1,
                    authority=1,
                    is_private=False,
                    is_summary=False,
                    token_count=max(1, len(body.split())),
                )
            )
    return chunks


def _load_queries(path: Path, *, max_queries: int | None) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_queries is not None and index >= max_queries:
                break
            record = json.loads(line)
            queries[str(record["_id"])] = str(record["text"])
    return queries


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                query_id, corpus_id, score = parts
            elif len(parts) == 4:
                query_id, _iteration, corpus_id, score = parts
            else:
                continue
            qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return qrels


def _find_qrels_path(dataset_dir: Path) -> Path:
    preferred = dataset_dir / "qrels" / "test.tsv"
    if preferred.exists():
        return preferred
    qrels_paths = sorted((dataset_dir / "qrels").glob("*.tsv"))
    if not qrels_paths:
        raise FileNotFoundError(f"No qrels TSV file found in {dataset_dir / 'qrels'}")
    return qrels_paths[0]


def _recall_at_k(hit_ids: list[str], relevant: set[str], *, k: int) -> float:
    return len(set(hit_ids[:k]) & relevant) / len(relevant)


def _mrr_at_k(hit_ids: list[str], relevant: set[str], *, k: int) -> float:
    for rank, doc_id in enumerate(hit_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(hit_ids: list[str], relevant: set[str], *, k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(hit_ids[:k], start=1):
        if doc_id in relevant:
            dcg += 1 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summary(rows: list[RetrievalBenchmarkRow], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "This real-world public retrieval benchmark uses the repository's deterministic "
        "BM25-style lexical retriever.",
        "",
        "| Dataset | Queries | Corpus Docs | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.dataset} | {row.queries} | {row.corpus_documents} | "
            f"{row.recall_at_5:.4f} | {row.recall_at_10:.4f} | "
            f"{row.mrr_at_10:.4f} | {row.ndcg_at_10:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This establishes public benchmark baselines. Dense, hybrid, and reranked "
            "retrieval should be compared against this lexical baseline under the same "
            "metric schema.",
        ]
    )
    return "\n".join(lines) + "\n"

