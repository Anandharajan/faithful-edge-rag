# Dense and Hybrid SciFact Benchmark

| Retriever | Model | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Index Seconds | Query Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical | bm25_lexical | 0.7243 | 0.7843 | 0.6302 | 0.6622 | 0.00 | 0.00 |
| dense | BAAI/bge-small-en-v1.5 | 0.7753 | 0.8396 | 0.6891 | 0.7215 | 43.02 | 42.76 |
| hybrid | BAAI/bge-small-en-v1.5 | 0.7833 | 0.8579 | 0.7061 | 0.7397 | 46.05 | 42.82 |

These results compare the existing lexical baseline with dense and hybrid retrieval using an open-source sentence-transformer embedding model.
