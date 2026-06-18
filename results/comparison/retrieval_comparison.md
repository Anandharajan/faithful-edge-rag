# Retrieval Benchmark Comparison

This table aggregates available real-world retrieval benchmark outputs. Add Colab outputs under `results/colab/` and rerun the comparison command to include dense and hybrid GPU results.

| Source | Dataset | Retriever | Model | Queries | Docs | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Index s | Query s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| results/beir-scifact | BEIR SciFact | lexical | bm25_lexical | 300 | 5183 | 0.7243 | 0.7843 | 0.6302 | 0.6622 |  |  |
