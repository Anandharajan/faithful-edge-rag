# Initial Experiment Results

These results come from the deterministic synthetic benchmark included in the repository. They are intended as a reproducible engineering baseline, not a final SCI-grade empirical claim.

| Condition | Accuracy | Recall@5 | MRR | Citation Faithfulness | Conflict F1 | Avg Tokens | Avg Private Bytes | Avg Transfer Bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| centralized_rag | 0.875 | 1.000 | 0.938 | 0.875 | 0.000 | 167.0 | 41.9 | 41.9 |
| long_context_rag | 0.000 | 1.000 | 0.938 | 0.000 | 0.000 | 386.4 | 207.5 | 1965.6 |
| edge_only_rag | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 167.0 | 0.0 | 0.0 |
| edge_cloud_faithful_rag | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 89.0 | 0.0 | 300.1 |

## Interpretation

- Centralized RAG is a strong retrieval baseline but moves private raw evidence.
- Long-context stuffing consumes more context and is intentionally stress-tested against stale/conflicting evidence.
- Edge-only RAG minimizes transfer but cannot represent collaborative edge-cloud behavior.
- The proposed condition adds conflict detection and context budgeting while limiting private evidence movement.

## Next Step for Publication-Grade Results

Replace the deterministic lexical retriever with open-source embedding models and Qdrant/LanceDB indexes, run larger public and synthetic conflict datasets, add confidence intervals, and report hardware-normalized latency and energy or compute measurements.
