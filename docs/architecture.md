# Architecture

The research prototype uses a central node and one or more edge nodes.

## Central Node

- FastAPI serves ingestion, retrieval, answer generation, edge sync, and evaluation endpoints.
- LangGraph coordinates retrieval, routing, conflict detection, context budgeting, and answer generation.
- Qdrant stores central embeddings and summary indexes.
- PostgreSQL stores metadata, traces, claims, evaluation results, and run state.
- MinIO stores raw artifacts that are allowed to leave edge nodes.
- vLLM or TGI serves the central open-source LLM.
- OpenTelemetry exports traces and metrics to Prometheus, Grafana, Loki, and Jaeger.

## Edge Node

- The edge collector watches local directories or device feeds.
- Local preprocessing performs redaction, deduplication, chunking, summarization, and embedding.
- LanceDB, FAISS, SQLite-vec, or local Qdrant stores private/local vectors.
- llama.cpp or Ollama serves a quantized local model for offline operation.
- The sync client sends only policy-approved summaries, metadata, embeddings, or selected chunks.

## Research Baselines

1. Centralized RAG.
2. Long-context RAG.
3. Edge-only RAG.
4. Proposed edge-cloud faithful RAG.

