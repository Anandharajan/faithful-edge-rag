# SCI-Grade Publication Plan

The current repository now has an executable deterministic benchmark. This is an engineering baseline, not yet a sufficient SCI/SCIE journal result.

## Minimum Evidence Needed Before Submission

1. Larger datasets:
   - Public retrieval/QA benchmark subset.
   - At least 100 synthetic conflict scenarios.
   - At least 3 simulated edge sites with heterogeneous privacy policies.

2. Open-source model experiments:
   - Embeddings: BGE, E5, or GTE.
   - Vector stores: Qdrant centrally and LanceDB/FAISS at the edge.
   - LLMs: one central model served by vLLM and one quantized edge model served by llama.cpp/Ollama.

3. Baselines:
   - Centralized RAG.
   - Long-context RAG.
   - Edge-only RAG.
   - Proposed edge-cloud faithful RAG.
   - Proposed system without conflict detection.
   - Proposed system without context budgeting.

4. Statistical analysis:
   - Multiple random seeds for synthetic scenario generation.
   - Confidence intervals for primary metrics.
   - Paired significance tests for accuracy, faithfulness, and conflict handling.

5. Systems analysis:
   - Hardware profile.
   - Latency distribution.
   - CPU/GPU time.
   - Memory use.
   - Edge-to-central bytes transferred.
   - Raw private evidence exposure.

6. Reproducibility:
   - Versioned datasets.
   - Model checkpoint hashes.
   - Prompt versions.
   - Docker image digests.
   - Exact experiment commands.

## Candidate Journal Positioning

- Future Generation Computer Systems: emphasize edge-cloud architecture and systems evaluation.
- Engineering Applications of Artificial Intelligence: emphasize practical AI application and empirical performance.
- Information Fusion: emphasize conflicting evidence, source fusion, and attribution faithfulness.
- IEEE Internet of Things Journal: emphasize edge/IoT deployment and telemetry scenarios.

## Immediate Next Milestone

The seeded synthetic benchmark with ablations and confidence intervals is now the first publication-track layer. The next milestone is the embedding/vector-store experiment path:

```bash
python -m faithful_edge_rag.experiments.run_vector --output-dir results/vector
```

That milestone should replace lexical retrieval with open-source embeddings, Qdrant, and a local edge vector index while preserving the same metric schema used by `results/initial`.
