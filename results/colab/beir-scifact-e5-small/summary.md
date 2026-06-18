# Dense and Hybrid SciFact Benchmark

| Retriever | Model | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Index Seconds | Query Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical | bm25_lexical | 0.7243 | 0.7843 | 0.6302 | 0.6622 | 0.00 | 0.00 |
| dense | intfloat/e5-small-v2 | 0.7442 | 0.8096 | 0.6393 | 0.6764 | 52.24 | 41.36 |
| hybrid | intfloat/e5-small-v2 | 0.7996 | 0.8362 | 0.6994 | 0.7293 | 51.51 | 41.24 |

These results compare the existing lexical baseline with dense and hybrid retrieval using an open-source sentence-transformer embedding model.
