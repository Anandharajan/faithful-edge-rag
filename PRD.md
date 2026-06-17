# Research PRD: Open-Source Faithful Edge-Cloud RAG for Distributed and Conflicting Knowledge

## 1. Research Plan

### 1.1 Working Title

Open-Source Faithful Edge-Cloud Retrieval-Augmented Generation for Distributed, Private, and Conflicting Knowledge Sources

### 1.2 Defined Research Problem

Production RAG systems are increasingly used to answer questions over enterprise and operational knowledge. However, real deployments rarely have a single clean, centralized corpus. Knowledge is distributed across cloud repositories, edge devices, field logs, private local documents, and time-varying telemetry. Literature shows four related but insufficiently unified problems:

- Centralized RAG improves factuality but can increase privacy exposure, retrieval cost, and latency when knowledge is distributed across edge/cloud environments.
- Long-context models do not eliminate RAG; they can degrade under very large contexts and exhibit distinct long-context failure modes.
- RAG systems struggle when retrieved evidence is ambiguous, stale, noisy, or contradictory.
- Citation correctness is not enough; generated answers may cite apparently correct documents without genuinely relying on them, creating attribution post-rationalization.

The research problem is therefore:

> How can a fully open-source edge-cloud RAG system answer questions over distributed private knowledge while optimizing latency, compute cost, privacy exposure, and citation faithfulness under ambiguous or conflicting evidence?

### 1.3 Core Hypothesis

A context-budgeted, policy-aware, open-source edge-cloud RAG architecture can improve answer faithfulness and operational efficiency compared with centralized cloud RAG and naive long-context RAG by:

- Keeping sensitive or high-volume data at the edge when possible.
- Routing only summaries, embeddings, provenance metadata, or selected evidence to the cloud.
- Detecting conflict and ambiguity before generation.
- Allocating context using evidence quality, recency, source authority, privacy policy, and token budget.
- Evaluating both citation correctness and citation faithfulness.
- Using reproducible open-source models, vector stores, orchestration, deployment, and observability.

### 1.4 Research Questions

RQ1. Does hierarchical edge-cloud retrieval reduce latency, inference tokens, GPU/CPU time, and data movement compared with centralized cloud-only RAG?

RQ2. Can conflict-aware retrieval and generation improve answer faithfulness when sources are stale, noisy, or contradictory?

RQ3. Does context budgeting improve RAG performance compared with naive top-k retrieval and long-context stuffing?

RQ4. Can citation faithfulness metrics reveal failures that citation correctness alone misses?

RQ5. What tradeoffs emerge between privacy, retrieval quality, answer quality, and system latency across edge-only, cloud-only, and collaborative edge-cloud modes?

RQ6. Can the complete system be reproduced with open-source software on commodity servers and edge hardware?

### 1.5 Method Summary

The project will implement and evaluate a research prototype using only open-source infrastructure and models where feasible. It will compare four systems:

1. Cloud-only RAG: all documents centralized and retrieved from a self-hosted vector database.
2. Long-context baseline: retrieved or aggregated documents packed into a large open-source LLM context.
3. Edge-only RAG: local retrieval and local or constrained generation on edge hardware.
4. Proposed Edge-Cloud Faithful RAG: local retrieval, cloud summary index, policy-aware routing, conflict detection, context budgeting, and citation faithfulness evaluation.

### 1.6 Expected Research Contribution

The project contributes:

- A reproducible open-source edge-cloud RAG architecture for distributed private knowledge.
- A conflict-aware context budgeting method for RAG.
- An evaluation design that jointly measures answer quality, citation faithfulness, privacy exposure, latency, compute cost, and edge-cloud freshness.
- A deployable reference implementation using open models, open vector stores, open observability, and self-hosted infrastructure.

## 2. Literature Analysis and Gap

### 2.1 RAG Is Useful but Operationally Incomplete

RAG surveys describe retrieval, augmentation, reranking, grounding, and evaluation as core components for improving factuality and domain adaptation. They also identify unresolved challenges: robustness, retrieval quality, scalability, stale evidence, noisy evidence, evaluation, and real-world deployment complexity.

Implication for this project:

- RAG alone is not the contribution. The research contribution must be in how RAG is orchestrated across distributed private stores and evaluated under realistic conflict and edge constraints.

### 2.2 Edge-Cloud RAG Is Emerging but Still Narrow

DGRAG proposes distributed graph-based RAG where edge devices keep local knowledge and share summaries with the cloud. CoEdge-RAG studies hierarchical scheduling for collaborative edge RAG. RAGdb proposes a compact local-first RAG stack for edge and air-gapped settings. These papers validate the importance of decentralized RAG, but they leave room for a system-level study that combines edge-cloud routing with attribution faithfulness, conflict handling, context management, governance metrics, and fully reproducible open-source deployment.

Gap:

- Existing edge-RAG work focuses heavily on distributed retrieval, scheduling, or local storage efficiency. Less attention is given to faithful citation, ambiguous/conflicting evidence, reproducible open-source implementation, and practical governance metrics.

### 2.3 Long Context Does Not Remove the Need for Retrieval

The long-context RAG study across many LLMs shows that adding more context can help, but only some recent models remain accurate above large context sizes, and long-context settings introduce failure modes. This weakens the common assumption that large context windows make retrieval engineering unnecessary.

Gap:

- A research project should compare context stuffing against context budgeting, especially when edge and cloud sources contain conflicting or stale evidence.

### 2.4 Conflicting Evidence Is a Core Real-World Failure Mode

The 2025 work on RAG with conflicting evidence shows that ambiguity, misinformation, and noise jointly create hard RAG scenarios. It proposes RAMDocs and a multi-agent approach, but also reports that a substantial gap remains, especially with imbalanced support and misinformation.

Gap:

- Enterprise and edge deployments commonly contain version drift: old manuals, updated policies, local overrides, partial telemetry, and contradictory field notes. The project should explicitly evaluate this rather than treating retrieval as clean top-k lookup.

### 2.5 Citation Correctness Is Not Citation Faithfulness

Work on RAG attributions distinguishes citation correctness from citation faithfulness. A citation may support a claim, but the model may not have actually relied on that citation. This creates false trust in grounded answers.

Gap:

- Most RAG demos include citations but do not test whether the answer genuinely depends on cited evidence. This project should include counterfactual citation faithfulness tests.

### 2.6 Agentic Orchestration Needs Measurable Tool Discipline

Agent literature frames tool use, planning, and feedback as central paradigms. However, agentic systems can increase latency, cost, and failure surfaces. For this project, agents are not included for novelty alone; they are used where planning and tool choice are necessary: deciding edge/cloud routing, selecting retrieval strategy, resolving conflict, querying structured data, and running evaluation.

Gap:

- The research must measure when agentic orchestration helps versus when direct retrieval is better.

## 3. Product and Research Objective

Build and evaluate a fully open-source edge-cloud RAG research prototype that answers operational and research questions over distributed documents and telemetry while preserving privacy and improving faithfulness under conflicting evidence.

The system should be production-shaped, not merely a notebook:

- Async Python APIs for ingestion, retrieval, chat, evaluation, and edge sync.
- Self-hosted API, agent, retrieval, model-serving, storage, and observability services.
- Open-source LLM serving using vLLM, llama.cpp, Ollama, or Hugging Face Text Generation Inference.
- Open-source LLMs such as Llama, Qwen, Mistral, Gemma, DeepSeek, or Phi depending on hardware.
- Open-source embedding models such as BGE, E5, GTE, Nomic Embed, or multilingual alternatives.
- Open-source vector search using Qdrant, Milvus, Weaviate, LanceDB, FAISS, or SQLite vector extensions.
- Open-source agent orchestration using LangGraph as the default, with optional LlamaIndex, AutoGen, or CrewAI comparison.
- Open observability using OpenTelemetry, Prometheus, Grafana, Loki, and Jaeger.
- Edge collector for local preprocessing, redaction, local retrieval, and sync.

## 4. Fully Open-Source Stack

### 4.1 Default Reference Stack

The default implementation should use:

- API: FastAPI, Uvicorn, Pydantic.
- Async runtime: asyncio, anyio, httpx.
- Agent orchestration: LangGraph.
- Optional RAG framework: LlamaIndex for ingestion and retriever abstractions where useful.
- LLM serving: vLLM for GPU server deployment; llama.cpp or Ollama for CPU/edge deployment.
- Cloud-side LLM: Qwen2.5, Llama 3.1/3.3, Mistral, Gemma, or DeepSeek distilled model served through vLLM.
- Edge LLM: quantized GGUF model served through llama.cpp or Ollama.
- Embeddings: BAAI BGE, intfloat E5, Alibaba GTE, or Nomic Embed.
- Reranking: BGE reranker, Jina reranker, or cross-encoder reranker.
- Cloud vector store: Qdrant by default.
- Edge vector store: LanceDB, FAISS, SQLite-vec, or Qdrant local.
- Metadata store: PostgreSQL.
- Object store: MinIO.
- Analytics: DuckDB for local analytics and ClickHouse or PostgreSQL for larger telemetry.
- Queue/jobs: Redis with RQ/Arq, Celery, or Dramatiq.
- Streaming/events: NATS or Redis Streams.
- Observability: OpenTelemetry, Prometheus, Grafana, Loki, Jaeger.
- Deployment: Docker Compose for reproducibility; K3s or Kubernetes for distributed deployment.
- Security: Keycloak or Authentik for identity, OPA for policy, TLS with Caddy or Traefik, SOPS or Vault for secrets.

### 4.2 Open-Source Substitution Rules

The project must not depend on proprietary managed AI APIs for core experiments. Optional adapters may be documented separately, but the evaluated system must run without them.

Allowed substitutions:

- Qdrant can be replaced by Milvus, Weaviate, LanceDB, or FAISS.
- vLLM can be replaced by TGI or llama.cpp server.
- LangGraph can be replaced by LlamaIndex Workflows, AutoGen, or CrewAI.
- MinIO can be replaced by local filesystem storage in minimal deployments.
- Kubernetes can be replaced by Docker Compose for a single-node reproducible setup.

## 5. System Overview

### 5.1 Main Idea

The system keeps local knowledge at edge nodes where privacy, bandwidth, or latency constraints require it. The central node stores a summary index and metadata about edge-held knowledge rather than indiscriminately centralizing all raw content. For each user query, the system decides whether to answer locally, answer from central knowledge, request selected edge evidence, or combine both.

### 5.2 Proposed Architecture

1. Edge nodes ingest local documents, logs, telemetry, field notes, and device manuals.
2. Edge nodes parse, redact, chunk, summarize, embed, and cache content locally.
3. Edge nodes sync metadata, summaries, provenance, freshness markers, and policy tags to the central node.
4. FastAPI receives user queries and starts a LangGraph workflow.
5. Retrieval Agent queries Qdrant, PostgreSQL metadata, object-store manifests, local edge indexes, and the edge summary index.
6. Routing Agent decides whether local evidence is sufficient or whether selected edge evidence is needed.
7. Conflict Agent detects contradictory, stale, ambiguous, or low-authority evidence.
8. Context Budgeter assembles a bounded prompt using relevance, freshness, authority, privacy policy, conflict status, and token cost.
9. The selected open-source LLM generates an answer with source-level and claim-level citations.
10. Citation Verifier checks correctness and faithfulness signals.
11. Evaluation results, traces, token usage, compute cost, latency, and privacy exposure are stored in PostgreSQL, DuckDB, or ClickHouse.

## 6. Research Prototype Requirements

### 6.1 Functional Requirements

FR1. Distributed Knowledge Ingestion

- Ingest PDFs, markdown, text, JSON logs, CSV telemetry, and web snapshots.
- Support both central and edge ingestion.
- Preserve source lineage, timestamps, policy tags, and version metadata.
- Store raw artifacts in MinIO or local filesystem storage.

FR2. Edge Preprocessing

- Perform local parsing, deduplication, redaction, chunking, and summarization.
- Maintain a local vector index for recent or private documents.
- Sync only allowed artifacts to the central node: summaries, embeddings, metadata, or selected chunks.
- Support offline operation and later idempotent synchronization.

FR3. Hybrid Retrieval

- Support Qdrant or equivalent vector search for central retrieval.
- Support local edge retrieval over cached/private data.
- Combine dense semantic search, metadata filtering, recency filtering, and optional keyword search using PostgreSQL full-text search or Tantivy.

FR4. Conflict-Aware Evidence Selection

- Detect when retrieved evidence has conflicting claims.
- Identify stale versus recent versions.
- Surface ambiguity instead of forcing a single answer.
- Prefer authoritative and recent sources when policies define authority.

FR5. Context Budgeting

- Build prompts under a token budget.
- Allocate context by relevance, conflict coverage, freshness, authority, and privacy policy.
- Avoid naive top-k stuffing.
- Include concise summaries when full chunks are too costly or private.

FR6. Agentic Orchestration

- Use LangGraph as the default open-source orchestration framework.
- Implement Research Agent, Retrieval Agent, Routing Agent, Conflict Agent, Analytics Agent, Evaluation Agent, and Governance Agent.
- Log tool calls, inputs, outputs, failures, and latency.
- Compare direct retrieval versus agentic routing for cost and quality.

FR7. Grounded Generation

- Generate answers with citations using self-hosted open-source LLMs.
- Distinguish known facts, conflicting claims, and insufficient evidence.
- Refuse or ask for clarification when retrieval confidence is too low.
- Record model name, quantization, prompt version, decoding settings, and context tokens.

FR8. Evaluation Harness

- Run offline benchmark evaluations.
- Run scenario tests for stale evidence, noisy evidence, ambiguous queries, and conflicting documents.
- Store metrics in PostgreSQL, DuckDB, or ClickHouse.
- Export metrics to Prometheus where useful.

### 6.2 Non-Functional Requirements

Latency:

- Edge-only retrieval P50 under 2 seconds for local cached corpora.
- Central grounded answer P50 under 10 seconds for standard queries on the target model server.
- Edge-cloud collaborative answer P95 under 75 seconds for multi-agent workflows.

Privacy:

- Support local-only documents.
- Log raw-content movement from edge to central node.
- Measure privacy exposure as bytes/chunks/tokens transferred away from the edge.

Reliability:

- Edge sync must be idempotent.
- Services must expose health checks.
- Agent runs must fail with traceable status and recoverable state.
- Job queues must support retry with dead-letter handling.

Cost and Resource Use:

- Track token usage, model latency, GPU seconds, CPU seconds, memory use, vector retrieval calls, and edge-cloud transfer volume.
- Enforce per-request token and compute budget caps.

Security:

- Use self-hosted identity such as Keycloak or Authentik.
- Use OPA or equivalent policy checks for data access and routing.
- Use signed edge batches.
- Use row-level or metadata-based authorization.
- Audit document access and tool use.
- Store secrets with SOPS, Vault, or sealed Kubernetes secrets.

## 7. Research Design

### 7.1 Experimental Conditions

Condition A: Centralized RAG

- All documents are uploaded to the central node.
- Retrieval runs only over central vector search.
- Measures the standard production pattern.

Condition B: Long-Context RAG

- Large retrieved sets or document summaries are packed into long context.
- Measures whether long context reduces retrieval engineering needs.

Condition C: Edge-Only RAG

- Retrieval and optional lightweight generation happen locally.
- Measures privacy and latency benefits, plus quality loss.

Condition D: Proposed Edge-Cloud Faithful RAG

- Uses local retrieval, central summary index, routing, conflict detection, and context budgeting.
- Measures the proposed contribution.

### 7.2 Datasets

The project should construct a mixed benchmark corpus:

- Research papers on RAG, edge AI, agents, vector search, and context engineering.
- Synthetic enterprise policy documents with old and new versions.
- Simulated field manuals and local edge notes.
- IoT or operational telemetry tables stored in PostgreSQL, DuckDB, or ClickHouse.
- Conflict sets where sources disagree by version, location, authority, or timestamp.
- Ambiguous query sets where multiple valid interpretations exist.

Public evaluation anchors can include BEIR-style retrieval tasks, Natural Questions-style QA tasks, and custom RAMDocs-inspired conflict scenarios.

### 7.3 Metrics

Answer quality:

- Exact match or rubric score for controlled QA.
- Human or open-source LLM-judge relevance score.
- Refusal correctness for insufficient evidence.

Retrieval quality:

- Recall@k.
- MRR.
- nDCG.
- Retrieval latency.
- Source diversity.

Attribution:

- Citation correctness.
- Citation faithfulness.
- Claim-level support rate.
- Post-rationalization rate using citation ablation tests.

Conflict handling:

- Conflict detection precision and recall.
- Correct stale-source suppression.
- Correct multi-answer handling for ambiguous queries.

Edge-cloud performance:

- End-to-end latency.
- Inference token usage.
- GPU seconds or CPU seconds.
- Vector search latency.
- Edge-cloud bytes transferred.
- Local cache hit rate.
- Sync freshness lag.

Privacy:

- Raw chunks transferred to central node.
- Redacted entity count.
- Local-only policy violations.
- Private-source exposure rate.

Agent performance:

- Tool-call success rate.
- Plan completion rate.
- Unnecessary tool-call rate.
- Compute cost per successful answer.
- Recovery rate after tool failure.

Reproducibility:

- One-command local deployment success.
- Deterministic evaluation configuration.
- Model, embedding, index, prompt, and dataset version capture.

## 8. Implementation Scope

### 8.1 Open-Source Services

- FastAPI API service for async endpoints.
- LangGraph worker service for agent workflows.
- Ingestion worker service for document parsing and indexing.
- vLLM service for central GPU inference.
- llama.cpp or Ollama service for edge inference.
- Qdrant service for central vector search.
- LanceDB, FAISS, SQLite-vec, or local Qdrant for edge vector search.
- PostgreSQL for metadata, traces, authz metadata, evaluation records, and run state.
- MinIO for document artifacts.
- Redis, NATS, or RabbitMQ for queues and eventing.
- DuckDB or ClickHouse for telemetry and evaluation analytics.
- Prometheus, Grafana, Loki, Jaeger, and OpenTelemetry Collector for observability.

### 8.2 Python Components

- `api`: FastAPI service with async endpoints.
- `ingestion`: parsers, chunkers, metadata extractors, embedding jobs.
- `retrieval`: Qdrant retrieval, local retrieval, hybrid retrieval, reranking.
- `agents`: LangGraph agents and tools.
- `models`: wrappers for vLLM, llama.cpp, Ollama, TGI, and embedding services.
- `context`: evidence ranking, context budgeting, prompt assembly.
- `edge`: local collector, redactor, cache, sync client.
- `eval`: benchmark runner and metric calculators.
- `governance`: policy checks, authorization, audit logging.
- `observability`: OpenTelemetry traces, metrics, structured logs.

### 8.3 API Endpoints

- `POST /documents/register`
- `POST /documents/upload`
- `POST /search`
- `POST /answer`
- `POST /agents/run`
- `GET /agents/runs/{run_id}`
- `POST /edge/sync`
- `GET /edge/nodes/{node_id}/status`
- `POST /eval/run`
- `GET /metrics`
- `GET /health`

## 9. Data Model

### `documents`

- `document_id`
- `source_uri`
- `source_type`
- `owner`
- `authority_level`
- `version`
- `created_at`
- `ingested_at`
- `policy_tags`
- `checksum`

### `chunks`

- `chunk_id`
- `document_id`
- `chunk_text`
- `embedding_ref`
- `section`
- `page_number`
- `timestamp`
- `edge_node_id`
- `privacy_level`
- `metadata`

### `edge_summary_index`

- `summary_id`
- `edge_node_id`
- `topic`
- `summary_text`
- `summary_embedding_ref`
- `freshness_time`
- `available_evidence_count`
- `privacy_policy`

### `claims`

- `claim_id`
- `run_id`
- `claim_text`
- `cited_chunk_ids`
- `support_score`
- `faithfulness_score`
- `conflict_status`

### `agent_traces`

- `run_id`
- `agent_name`
- `tool_name`
- `input_hash`
- `output_hash`
- `latency_ms`
- `token_usage`
- `status`

### `evaluation_results`

- `eval_id`
- `condition`
- `dataset`
- `metric`
- `value`
- `run_id`
- `created_at`

## 10. Methodological Details

### 10.1 Conflict Detection

The system should compare retrieved claims by:

- Entity.
- Attribute.
- Timestamp.
- Source authority.
- Location or edge node.
- Version lineage.

When evidence conflicts, the answer should either:

- Present multiple valid answers with conditions.
- Prefer the most recent authoritative source.
- Refuse to decide if authority and freshness are insufficient.

### 10.2 Context Budgeting Formula

Each candidate evidence item receives a context utility score:

`U = relevance + freshness + authority + conflict_coverage - privacy_cost - token_cost - redundancy`

The implementation can start with weighted heuristics and later compare against learned ranking.

### 10.3 Citation Faithfulness Test

For sampled answers:

1. Generate the answer with citations.
2. Remove cited evidence and regenerate or score the same claim.
3. Replace cited evidence with controlled contradictory evidence.
4. Measure whether the claim changes appropriately.

If a claim remains unchanged despite removal or contradiction of cited evidence, the citation may be post-rationalized.

### 10.4 Privacy Exposure Metric

Track:

- Number of raw chunks transferred from edge to central node.
- Number of private chunks blocked.
- Number of summary-only responses.
- Tokens of private evidence sent to central generation.
- Policy violations.

This turns privacy from a vague design claim into an evaluation axis.

### 10.5 Open-Source Reproducibility Protocol

Every experiment must record:

- Git commit.
- Docker image digests.
- Model name and checkpoint hash.
- Quantization format and precision.
- Embedding model version.
- Vector index type and parameters.
- Prompt template version.
- Dataset version.
- Hardware profile.
- Decoding parameters.

## 11. Acceptance Criteria

The project is successful if the proposed system demonstrates:

- Lower raw edge-to-central data transfer than centralized RAG.
- Comparable or better answer quality than centralized RAG on distributed questions.
- Better conflict handling than naive top-k RAG.
- Lower post-rationalized citation rate than baseline attributed RAG.
- Lower token usage than long-context stuffing for similar or better answer quality.
- Clear operational traces for every answer, citation, tool call, and policy decision.
- Reproducible deployment without proprietary managed AI services.

Target thresholds for the prototype:

- At least 30% reduction in inference token usage versus long-context baseline.
- At least 40% reduction in raw private data transferred versus centralized RAG.
- At least 10% improvement in conflict-handling score versus naive top-k RAG.
- Zero known local-only policy violations in evaluation tests.
- P95 collaborative answer latency under 75 seconds for benchmark queries on the declared hardware profile.
- Full stack starts through Docker Compose on one development machine, with optional K3s deployment for distributed experiments.

## 12. Implementation Phases

### Phase 1: Literature-Grounded Design

- Finalize corpus and evaluation scenarios.
- Define privacy policies and edge/cloud deployment assumptions.
- Implement data schemas.
- Build baseline evaluation scripts.
- Create Docker Compose skeleton with PostgreSQL, Qdrant, MinIO, Redis/NATS, Prometheus, Grafana, and API service.

### Phase 2: Baseline Systems

- Implement centralized RAG using Qdrant and self-hosted model serving.
- Implement long-context baseline using the same LLM family where possible.
- Implement edge-only local retrieval baseline using LanceDB/FAISS/SQLite-vec and llama.cpp/Ollama.
- Store baseline metrics in PostgreSQL or DuckDB.

### Phase 3: Proposed Edge-Cloud RAG

- Implement edge collector and sync.
- Implement central summary index.
- Implement routing agent with LangGraph.
- Implement context budgeter.
- Implement conflict detector.

### Phase 4: Faithfulness and Governance

- Implement citation verifier.
- Implement citation faithfulness ablation tests.
- Add policy enforcement and privacy exposure metrics.
- Add OpenTelemetry traces and dashboard tables.
- Add OPA policy checks for local-only and summary-only sources.

### Phase 5: Evaluation and Research Report

- Run all experimental conditions.
- Analyze tradeoffs.
- Produce final research report with limitations and future work.
- Package deployment instructions for reproducibility.
- Publish configuration files, dataset generation scripts, and evaluation harness.

## 13. Limitations

- Synthetic conflict datasets may not fully represent enterprise complexity.
- Citation faithfulness tests are approximate and should be combined with human review.
- Edge hardware variability can affect latency and local model feasibility.
- Agentic orchestration may add overhead for simple queries.
- Open-source model quality will vary by checkpoint, quantization, and hardware.
- Reproducibility still depends on model license availability and hardware reporting.

## 14. Key References

- Khalid, A. B. "RAGdb: A Zero-Dependency, Embeddable Architecture for Multimodal Retrieval-Augmented Generation on the Edge." arXiv, 2026. https://arxiv.org/abs/2602.22217
- Hong, G. et al. "CoEdge-RAG: Optimizing Hierarchical Scheduling for Retrieval-Augmented LLMs in Collaborative Edge Computing." arXiv, 2025. https://arxiv.org/abs/2511.05915
- Zhou, W. et al. "DGRAG: Distributed Graph-based Retrieval-Augmented Generation in Edge-Cloud Systems." arXiv, 2025. https://arxiv.org/abs/2505.19847
- Ren, R. et al. "Retrieval-Augmented Generation for Mobile Edge Computing via Large Language Model." arXiv, 2024. https://arxiv.org/abs/2412.20820
- Leng, Q. et al. "Long Context RAG Performance of Large Language Models." arXiv, 2024. https://arxiv.org/abs/2411.03538
- Leto, A. et al. "Toward Optimal Search and Retrieval for RAG." arXiv, 2024. https://arxiv.org/abs/2411.07396
- Wang, H. et al. "Retrieval-Augmented Generation with Conflicting Evidence." arXiv, 2025. https://arxiv.org/abs/2504.13079
- Wallat, J. et al. "Correctness is not Faithfulness in RAG Attributions." arXiv, 2024. https://arxiv.org/abs/2412.18004
- Li, X. "A Review of Prominent Paradigms for LLM-Based Agents: Tool Use, Planning, and Feedback Learning." arXiv, 2024. https://arxiv.org/abs/2406.05804
- LangGraph documentation. https://langchain-ai.github.io/langgraph/
- LlamaIndex documentation. https://docs.llamaindex.ai/
- vLLM documentation. https://docs.vllm.ai/
- llama.cpp project. https://github.com/ggerganov/llama.cpp
- Qdrant documentation. https://qdrant.tech/documentation/
- LanceDB documentation. https://lancedb.github.io/lancedb/
- OpenTelemetry documentation. https://opentelemetry.io/docs/
- Prometheus documentation. https://prometheus.io/docs/
- Grafana documentation. https://grafana.com/docs/

