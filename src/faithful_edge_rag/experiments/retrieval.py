import math
import re
from collections import Counter
from collections.abc import Iterable

from faithful_edge_rag.experiments.models import DocumentChunk, RetrievalHit

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class LexicalRetriever:
    """Small BM25-style retriever used for deterministic local experiments."""

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self._doc_terms = [Counter(tokenize(chunk.text)) for chunk in self._chunks]
        self._doc_freq: Counter[str] = Counter()
        for terms in self._doc_terms:
            self._doc_freq.update(terms.keys())
        self._avg_len = sum(sum(terms.values()) for terms in self._doc_terms) / max(
            len(self._doc_terms), 1
        )

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        query_terms = tokenize(query)
        hits: list[RetrievalHit] = []
        for chunk, terms in zip(self._chunks, self._doc_terms, strict=True):
            score = self._score(query_terms, terms)
            if score > 0:
                hits.append(RetrievalHit(chunk=chunk, score=score))
        return sorted(
            hits,
            key=lambda hit: (hit.score, hit.chunk.authority, hit.chunk.version),
            reverse=True,
        )[:top_k]

    def _score(self, query_terms: list[str], doc_terms: Counter[str]) -> float:
        if not query_terms:
            return 0.0
        k1 = 1.2
        b = 0.75
        doc_len = sum(doc_terms.values())
        score = 0.0
        corpus_size = len(self._chunks)
        for term in query_terms:
            freq = doc_terms[term]
            if freq == 0:
                continue
            doc_freq = self._doc_freq[term]
            idf = math.log(1 + (corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / self._avg_len)
            score += idf * (freq * (k1 + 1) / denom)
        return score
