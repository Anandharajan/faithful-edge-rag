# Multi-Dataset BEIR Benchmark Results

This real-world public retrieval benchmark uses the repository's deterministic BM25-style lexical retriever.

| Dataset | Queries | Corpus Docs | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BEIR scifact | 300 | 5183 | 0.7243 | 0.7843 | 0.6302 | 0.6622 |
| BEIR nfcorpus | 323 | 3633 | 0.1196 | 0.1486 | 0.5138 | 0.3058 |
| BEIR fiqa | 648 | 57638 | 0.2327 | 0.3009 | 0.2949 | 0.2372 |
| BEIR trec-covid | 50 | 171332 | 0.0084 | 0.0156 | 0.7906 | 0.6453 |

## Interpretation

This establishes public benchmark baselines. Dense, hybrid, and reranked retrieval should be compared against this lexical baseline under the same metric schema.
