# Experiments

The repository includes a deterministic synthetic benchmark for initial engineering validation.

Run:

```bash
python -m faithful_edge_rag.experiments.run --output-dir results/initial
```

Outputs:

- `results/initial/metrics.json`
- `results/initial/metrics.csv`
- `results/initial/summary.md`

## Conditions

1. `centralized_rag`: all documents are centrally visible.
2. `long_context_rag`: many chunks are packed into the context window.
3. `edge_only_rag`: retrieval is limited to the local edge node.
4. `edge_cloud_faithful_rag`: local retrieval, central summaries, policy-aware context budgeting, and conflict detection.

## Metrics

- Answer accuracy.
- Recall@5.
- MRR.
- Citation correctness.
- Citation faithfulness.
- Conflict F1.
- Average context tokens.
- Average raw private bytes moved.
- Average edge-to-central bytes.
- Average latency units.
- Average tool calls.

## Publication-Grade Extension Plan

The deterministic benchmark is a controlled starting point. For SCI-grade publication, extend it with:

- Public retrieval datasets and larger synthetic conflict corpora.
- Open-source embedding models such as BGE, E5, GTE, or Nomic Embed.
- Qdrant and LanceDB/FAISS experiments.
- Self-hosted LLM generation through vLLM and llama.cpp/Ollama.
- Multiple random seeds, confidence intervals, and statistical tests.
- Hardware-normalized latency, GPU/CPU time, memory, and energy measurements.
- Human or expert adjudication for a sampled subset of conflict and citation-faithfulness cases.

## Publication-Track Synthetic Run

Run the larger seeded benchmark with ablations and confidence intervals:

```bash
python -m faithful_edge_rag.experiments.run_publication \
  --seeds 10 \
  --topics 120 \
  --edge-nodes 4 \
  --output-dir results/publication
```

Outputs:

- `results/publication/per_seed_metrics.json`
- `results/publication/per_seed_metrics.csv`
- `results/publication/aggregate_metrics.json`
- `results/publication/aggregate_metrics.csv`
- `results/publication/summary.md`

Additional ablations:

- `edge_cloud_no_conflict_detection`
- `edge_cloud_no_context_budget`

## Real-World Public Benchmark: BEIR SciFact

Run the public SciFact retrieval benchmark:

```bash
python -m faithful_edge_rag.experiments.run_beir \
  --data-dir data/benchmarks \
  --output-dir results/beir-scifact
```

The command downloads BEIR SciFact into `data/benchmarks` if it is not already cached.

Outputs:

- `results/beir-scifact/metrics.json`
- `results/beir-scifact/metrics.csv`
- `results/beir-scifact/summary.md`

This benchmark currently evaluates the deterministic BM25-style lexical retriever. The next SCI-grade step is to add Qdrant plus open-source embedding models and compare dense, lexical, and hybrid retrieval on the same public dataset.

## Dense and Hybrid SciFact Retrieval

Run locally or in Colab:

```bash
python -m faithful_edge_rag.experiments.run_dense_beir \
  --data-dir data/benchmarks \
  --output-dir results/beir-dense \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --hybrid-alpha 0.65
```

The command writes lexical, dense, and hybrid metrics to:

- `results/beir-dense/metrics.json`
- `results/beir-dense/metrics.csv`
- `results/beir-dense/summary.md`

For GPU execution, open the notebook directly in Colab:

[![Open Dense Retrieval Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anandharajan/faithful-edge-rag/blob/main/notebooks/colab_dense_retrieval_scifact.ipynb)

If the repository is private, open Colab first, connect GitHub under `File -> Open notebook -> GitHub`, authorize access, then paste `Anandharajan/faithful-edge-rag`.

## Compare Colab Results

After running the Colab notebook, download `colab-results.zip`, extract it into the repository root so that files land under `results/colab/`, then run:

```bash
python -m faithful_edge_rag.experiments.compare \
  --results-dir results \
  --output-dir results/comparison
```

Outputs:

- `results/comparison/retrieval_comparison.csv`
- `results/comparison/retrieval_comparison.md`
