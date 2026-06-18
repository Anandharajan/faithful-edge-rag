# Retrieval Benchmark Comparison

This table aggregates available real-world retrieval benchmark outputs. Add Colab outputs under `results/colab/` and rerun the comparison command to include dense and hybrid GPU results.

| Source | Dataset | Retriever | Model | Queries | Docs | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Index s | Query s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| results/beir-scifact | BEIR SciFact | lexical | bm25_lexical | 300 | 5183 | 0.7243 | 0.7843 | 0.6302 | 0.6622 |  |  |
| results/colab/beir-scifact-bge-small | BEIR SciFact | dense | BAAI/bge-small-en-v1.5 | 300 | 5183 | 0.7753 | 0.8396 | 0.6891 | 0.7215 | 43.0205 | 42.7566 |
| results/colab/beir-scifact-bge-small | BEIR SciFact | hybrid | BAAI/bge-small-en-v1.5 | 300 | 5183 | 0.7833 | 0.8579 | 0.7061 | 0.7397 | 46.0536 | 42.8193 |
| results/colab/beir-scifact-bge-small | BEIR SciFact | lexical | bm25_lexical | 300 | 5183 | 0.7243 | 0.7843 | 0.6302 | 0.6622 |  |  |
| results/colab/beir-scifact-e5-small | BEIR SciFact | dense | intfloat/e5-small-v2 | 300 | 5183 | 0.7442 | 0.8096 | 0.6393 | 0.6764 | 52.2431 | 41.3591 |
| results/colab/beir-scifact-e5-small | BEIR SciFact | hybrid | intfloat/e5-small-v2 | 300 | 5183 | 0.7996 | 0.8362 | 0.6994 | 0.7293 | 51.5054 | 41.2419 |
| results/colab/beir-scifact-e5-small | BEIR SciFact | lexical | bm25_lexical | 300 | 5183 | 0.7243 | 0.7843 | 0.6302 | 0.6622 |  |  |
| results/colab/beir-scifact-lexical | BEIR SciFact | lexical | bm25_lexical | 300 | 5183 | 0.7243 | 0.7843 | 0.6302 | 0.6622 |  |  |
