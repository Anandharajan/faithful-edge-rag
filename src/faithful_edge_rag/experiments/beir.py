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

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def download_scifact(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_dir / "scifact"
    if (dataset_dir / "corpus.jsonl").exists():
        return dataset_dir

    archive_path = data_dir / "scifact.zip"
    if not archive_path.exists():
        _download_file(SCIFACT_URL, archive_path)

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
    chunks = _load_corpus(dataset_dir / "corpus.jsonl", max_docs=max_corpus_docs)
    queries = _load_queries(dataset_dir / "queries.jsonl", max_queries=max_queries)
    qrels = _load_qrels(dataset_dir / "qrels" / "test.tsv")

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
        dataset="BEIR SciFact",
        queries=evaluated_queries,
        corpus_documents=len(chunks),
        recall_at_5=_mean(recall_5),
        recall_at_10=_mean(recall_10),
        mrr_at_10=_mean(mrr_10),
        ndcg_at_10=_mean(ndcg_10),
    )


def write_benchmark_result(row: RetrievalBenchmarkRow, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(row)
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.csv").write_text(
        "dataset,queries,corpus_documents,recall_at_5,recall_at_10,mrr_at_10,ndcg_at_10\n"
        f"{row.dataset},{row.queries},{row.corpus_documents},{row.recall_at_5},"
        f"{row.recall_at_10},{row.mrr_at_10},{row.ndcg_at_10}\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_summary(row), encoding="utf-8")


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
                    topic="scifact",
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


def _summary(row: RetrievalBenchmarkRow) -> str:
    return (
        "# BEIR SciFact Benchmark Results\n\n"
        "This is the first real-world public retrieval benchmark result for the project. "
        "It uses BEIR SciFact and the repository's deterministic BM25-style lexical retriever.\n\n"
        "| Dataset | Queries | Corpus Docs | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| {row.dataset} | {row.queries} | {row.corpus_documents} | "
        f"{row.recall_at_5:.4f} | {row.recall_at_10:.4f} | "
        f"{row.mrr_at_10:.4f} | {row.ndcg_at_10:.4f} |\n\n"
        "## Interpretation\n\n"
        "This establishes a public benchmark baseline. The next comparison should add "
        "open-source embedding retrieval with Qdrant and a reranker, then compare against "
        "this lexical baseline under the same metric schema.\n"
    )
