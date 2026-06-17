# Publication-Track Experiment Results

These results aggregate seeded synthetic conflict scenarios. Values are reported as mean +/- 95% confidence interval across seeds.

| Condition | Accuracy | Citation Faithfulness | Conflict F1 | Avg Tokens | Private Bytes | Transfer Bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| centralized_rag | 0.838 +/- 0.018 | 0.838 +/- 0.018 | 0.000 +/- 0.000 | 213.5 +/- 1.7 | 247.5 +/- 21.5 | 247.5 +/- 21.5 |
| edge_cloud_faithful_rag | 0.853 +/- 0.021 | 0.853 +/- 0.021 | 0.241 +/- 0.033 | 85.2 +/- 0.5 | 0.0 +/- 0.0 | 30.5 +/- 2.0 |
| edge_cloud_no_conflict_detection | 0.853 +/- 0.021 | 0.853 +/- 0.021 | 0.000 +/- 0.000 | 85.2 +/- 0.5 | 0.0 +/- 0.0 | 30.5 +/- 2.0 |
| edge_cloud_no_context_budget | 0.885 +/- 0.021 | 0.885 +/- 0.021 | 0.270 +/- 0.034 | 333.3 +/- 2.6 | 0.0 +/- 0.0 | 197.7 +/- 10.9 |
| edge_only_rag | 0.853 +/- 0.020 | 0.853 +/- 0.020 | 0.307 +/- 0.047 | 201.3 +/- 1.6 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| long_context_rag | 0.797 +/- 0.027 | 0.211 +/- 0.032 | 0.000 +/- 0.000 | 503.0 +/- 3.4 | 574.4 +/- 39.6 | 1811.6 +/- 7.5 |

## Reading the Table

- `edge_cloud_no_conflict_detection` tests the contribution of conflict detection.
- `edge_cloud_no_context_budget` tests the contribution of bounded context selection.
- Confidence intervals are across seeded scenario generations, not across real-world sites.

## Remaining Publication Gap

This benchmark is larger and statistically summarized, but it still uses synthetic claims and deterministic retrieval. SCI-grade submission requires public datasets, open-source embedding/vector-store experiments, hardware measurements, and human or expert adjudication for sampled faithfulness labels.
