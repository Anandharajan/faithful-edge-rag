# Open-Source Faithful Edge-Cloud RAG

Research prototype for a fully open-source edge-cloud retrieval-augmented generation system that answers questions over distributed, private, and conflicting knowledge sources.

[![Open Dense Retrieval Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anandharajan/faithful-edge-rag/blob/main/notebooks/colab_dense_retrieval_scifact.ipynb)

The research problem is:

> How can a fully open-source edge-cloud RAG system answer questions over distributed private knowledge while optimizing latency, compute cost, privacy exposure, and citation faithfulness under ambiguous or conflicting evidence?

See [PRD.md](PRD.md) for the literature-grounded product and research design.

## Goals

- Compare centralized RAG, long-context RAG, edge-only RAG, and edge-cloud faithful RAG.
- Measure answer quality, retrieval quality, citation faithfulness, conflict handling, privacy exposure, latency, and compute cost.
- Run without proprietary managed AI services.
- Provide a reproducible open-source stack for local and distributed experiments.

## Default Stack

- API: FastAPI, Uvicorn, Pydantic
- Agents: LangGraph
- LLM serving: vLLM for central inference, llama.cpp or Ollama for edge inference
- Vector search: Qdrant centrally, LanceDB/FAISS/SQLite-vec at the edge
- Metadata: PostgreSQL
- Object storage: MinIO
- Queues: Redis
- Observability: OpenTelemetry, Prometheus, Grafana, Loki, Jaeger
- Deployment: Docker Compose for local reproducibility, K3s/Kubernetes for distributed experiments

## Repository Layout

```text
src/faithful_edge_rag/   Python package
tests/                   Unit and integration test scaffolding
docs/                    Research and architecture notes
.github/                 CI and GitHub templates
PRD.md                   Research PRD
docker-compose.yml       Local development services
```

## Local Development

Prerequisites:

- Python 3.11+
- Docker and Docker Compose

Create a virtual environment and install development dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Run checks:

```bash
ruff check .
mypy src
pytest
```

Or run the same checks through `make`:

```bash
make check
```

Start local infrastructure:

```bash
docker compose up -d postgres qdrant minio redis prometheus grafana
```

Run the API:

```bash
uvicorn faithful_edge_rag.api.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## Current Status

This repository currently contains the research PRD, software skeleton, API endpoints, and a deterministic experimental harness for the first baseline comparison.

Run the initial experiments:

```bash
python -m faithful_edge_rag.experiments.run --output-dir results/initial
```

Run the seeded publication-track synthetic benchmark:

```bash
python -m faithful_edge_rag.experiments.run_publication \
  --seeds 10 \
  --topics 120 \
  --edge-nodes 4 \
  --output-dir results/publication
```

Run the BEIR SciFact public retrieval benchmark:

```bash
python -m faithful_edge_rag.experiments.run_beir \
  --data-dir data/benchmarks \
  --output-dir results/beir-scifact
```

Run dense and hybrid retrieval, preferably on Colab GPU:

```bash
python -m faithful_edge_rag.experiments.run_dense_beir \
  --data-dir data/benchmarks \
  --output-dir results/beir-dense \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --hybrid-alpha 0.65
```

Colab notebook:

[![Open Dense Retrieval Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anandharajan/faithful-edge-rag/blob/main/notebooks/colab_dense_retrieval_scifact.ipynb)

If the repository is private, open Colab first, connect GitHub under `File -> Open notebook -> GitHub`, authorize access, then paste `Anandharajan/faithful-edge-rag`.

See [docs/experiments.md](docs/experiments.md) for the current metric definitions and publication-grade extension plan.

## License

Apache License 2.0. See [LICENSE](LICENSE).
