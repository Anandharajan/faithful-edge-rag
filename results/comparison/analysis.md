# Retrieval Benchmark Analysis

## Summary

The Colab GPU run adds dense and hybrid retrieval results on the public BEIR SciFact benchmark. Compared with the lexical baseline, open-source embedding retrieval improves all primary retrieval metrics.

## Best Results

| Metric | Best Method | Score |
| --- | --- | ---: |
| Recall@5 | Hybrid E5 small | 0.7996 |
| Recall@10 | Hybrid BGE small | 0.8579 |
| MRR@10 | Hybrid BGE small | 0.7061 |
| nDCG@10 | Hybrid BGE small | 0.7397 |

## Relative Improvement Over Lexical Baseline

Lexical baseline:

- Recall@5: 0.7243
- Recall@10: 0.7843
- MRR@10: 0.6302
- nDCG@10: 0.6622

Hybrid BGE small:

- Recall@5: 0.7833, +8.15%
- Recall@10: 0.8579, +9.38%
- MRR@10: 0.7061, +12.04%
- nDCG@10: 0.7397, +11.70%

Hybrid E5 small:

- Recall@5: 0.7996, +10.39%
- Recall@10: 0.8362, +6.62%
- MRR@10: 0.6994, +10.98%
- nDCG@10: 0.7293, +10.13%

## Interpretation for Paper

These results support the claim that the project should use hybrid retrieval as the real-world retrieval baseline for subsequent RAG and edge-cloud experiments. Dense retrieval alone improves over lexical retrieval, but score fusion with lexical retrieval gives the strongest overall performance.

The next experiment should evaluate whether the improved retrieval quality transfers into grounded answer generation and citation faithfulness.

## Multi-Dataset Lexical Baselines

The project now includes lexical BEIR baselines beyond SciFact:

| Dataset | Queries | Docs | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BEIR scifact | 300 | 5,183 | 0.7843 | 0.6302 | 0.6622 |
| BEIR nfcorpus | 323 | 3,633 | 0.1486 | 0.5138 | 0.3058 |
| BEIR fiqa | 648 | 57,638 | 0.3009 | 0.2949 | 0.2372 |
| BEIR trec-covid | 50 | 171,332 | 0.0156 | 0.7906 | 0.6453 |

These baselines make the retrieval section more credible, but dense/hybrid retrieval must now be run across all four datasets to support a journal-grade retrieval claim.
