# Paper Results Tracking

This file tracks which experimental evidence is ready for the paper draft.

## Completed

- Deterministic synthetic baseline: `results/initial/`
- Seeded synthetic benchmark with ablations and confidence intervals: `results/publication/`
- Public BEIR SciFact lexical baseline: `results/beir-scifact/`

## Waiting for Colab Output

Place Colab-generated outputs under:

```text
results/colab/
```

Then run:

```bash
python -m faithful_edge_rag.experiments.compare \
  --results-dir results \
  --output-dir results/comparison
```

The comparison table should then be used in the retrieval benchmark section of the paper.

## Next Evidence Needed

- Dense retrieval result on BEIR SciFact.
- Hybrid retrieval result on BEIR SciFact.
- At least one additional embedding model result.
- Runtime details from Colab: GPU type, indexing time, query time.
- Follow-up Qdrant or FAISS index experiment for approximate nearest-neighbor retrieval.

