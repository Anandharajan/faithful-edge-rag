# BEIR SciFact Benchmark Results

This is the first real-world public retrieval benchmark result for the project. It uses BEIR SciFact and the repository's deterministic BM25-style lexical retriever.

| Dataset | Queries | Corpus Docs | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BEIR SciFact | 300 | 5183 | 0.7243 | 0.7843 | 0.6302 | 0.6622 |

## Interpretation

This establishes a public benchmark baseline. The next comparison should add open-source embedding retrieval with Qdrant and a reranker, then compare against this lexical baseline under the same metric schema.
