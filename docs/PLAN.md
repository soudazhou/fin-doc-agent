# Implementation Plan

This document captures the design decisions and implementation roadmap for the Financial Document Q&A Agent. It serves as both a development guide and a record of architectural thinking.

## Problem Statement

Financial institutions need to extract insights from large volumes of documents (earnings reports, 10-Ks, research notes). Manual analysis is slow and error-prone. This project builds a production-grade AI system that automates financial document analysis with verifiable, source-cited answers — and provides the tooling to evaluate, benchmark, and compare LLM providers at scale.

## What Sets This Project Apart

Most RAG demos stop at "ask a question, get an answer." This project goes further:

1. **Agentic search over naive RAG** — The agent decides what to search, evaluates retrieval quality, and refines queries autonomously (not a dumb embed→retrieve→generate pipeline)
2. **Multi-capability agents** — Q&A, summarisation, cross-document comparison, metric extraction
3. **Retrieval accuracy benchmarking** — Find the right chunk among 1000+ chunks, measured with Recall@k and MRR
4. **Chunk size optimisation** — Empirically benchmark 256/512/1024 token chunks, let data decide
5. **A/B provider comparison** — Run the same query across Claude, DeepSeek, Qwen, GLM-5; compare quality, latency, and cost
6. **Eval-first development** — Every pipeline change is measured against a golden dataset with automated quality gates and feedback loops
7. **Authorisation layer** — API key auth, document-level access control, audit logging
8. **Vector store benchmarking** — pgvector vs Chroma, measured head-to-head on the same dataset

## Architecture

```
                              ┌──────────────────────────────────┐
                              │      LangGraph Orchestrator      │
                              │  intent → capability → search    │
                              │  → evaluate → refine (loop)      │
                              └──────────┬───────────────────────┘
                                         │
               ┌─────────────────────────┼──────────────────────┐
               ▼                         ▼                      ▼
      ┌──────────────┐        ┌──────────────┐       ┌──────────────┐
      │  Agentic     │        │   Analyst     │       │  Comparator  │
      │  Search      │        │   Agent       │       │  Agent       │
      │  Agent       │        │ (LLM reason.) │       │ (A/B test)   │
      │              │        └──────┬───────┘       └──────┬───────┘
      │ • query plan │               │                      │
      │ • retrieve   │       ┌───────┴───────┐              │
      │ • evaluate   │       │  LLM Provider │              │
      │ • refine     │◀─────▶│  (pluggable)  │◀─────────────┘
      │ • re-search  │       ├───────────────┤
      └──────┬───────┘       │ • Anthropic   │
             │               │ • DeepSeek    │
             ▼               │ • Qwen/GLM-5  │
      ┌──────────────┐       └───────────────┘
      │ Vector Store │
      │ (pluggable)  │
      ├──────────────┤    ┌──────────────────────────────────────┐
      │ • pgvector   │    │           Evaluation Engine           │
      │ • Chroma     │    │  golden dataset → run → score → store│
      └──────────────┘    │  feedback loop → regression tracking  │
                          └──────────────────────────────────────┘
               │                          │
┌──────────────┤  ┌───────────────────────┤  ┌───────────────────┐
│ Celery+Redis │  │ PostgreSQL            │  │  Auth Layer       │
│ (async tasks)│  │ • docs & chunks       │  │  • API keys       │
│              │  │ • eval results        │  │  • doc ACL        │
└──────────────┘  │ • query metrics       │  │  • audit log      │
                  └───────────────────────┘  └───────────────────┘
```

## Technology Choices

| Component | Choice | Alternatives Considered | Why This One |
|-----------|--------|------------------------|-------------|
| API | FastAPI + Pydantic V2 | Django REST, Flask | Async-native, auto OpenAPI docs, V2 is 5-50x faster validation |
| Multi-agent | LangGraph | CrewAI, OpenAI Agents SDK, Claude Agent SDK | Graph-based state machines are most debuggable; conditional routing for fintech workflows |
| Vector DB | pgvector + Chroma (benchmarked) | Pinecone, Weaviate, Qdrant | Both implemented behind interface; benchmark results drive recommendation (see Vector Store Strategy below) |
| Embeddings | OpenAI text-embedding-3-small | Nomic Embed V2, Voyage AI | No GPU needed, high quality at low cost, simple API |
| LLM | Multi-provider (Claude, DeepSeek, Qwen, GLM-5) | Single-vendor lock-in | Provider-agnostic design: cheap models for dev, Claude for demo (see LLM Strategy below) |
| PDF Parsing | Docling (IBM) | pdfplumber, Camelot, PyMuPDF | Purpose-built for structured docs, understands financial tables |
| Task Queue | Celery + Redis | FastAPI BackgroundTasks, RQ | Reliability (retries, persistence), monitoring (Flower), scalability |
| Evaluation | DeepEval + custom metrics | Ragas, LangSmith | pytest-style, CI/CD ready; custom numerical accuracy metric for finance |

## RAG vs Agentic Search — Why We Go Beyond RAG

### The Problem with Naive RAG

Traditional RAG follows a rigid pipeline: embed query → retrieve top-k → generate answer. This fails in practice because:

1. **The query may not match embedding space** — "What risks does the company face?" won't match a chunk titled "Risk Factors" via cosine similarity alone
2. **Retrieved context may be insufficient** — top-5 chunks might miss critical information scattered across the document
3. **No quality feedback** — the system doesn't know if retrieval was good enough before generating an answer
4. **Single-shot** — one query, one retrieval, no refinement

### Our Approach: Agentic Search

The search agent is an autonomous loop, not a pipeline:

```
┌─────────────────────────────────────────────────────┐
│                 Agentic Search Loop                  │
│                                                     │
│  1. PLAN    → Analyse query, generate search plan   │
│               (may decompose into sub-queries)      │
│                                                     │
│  2. RETRIEVE → Execute vector search                │
│                                                     │
│  3. EVALUATE → LLM judges retrieval quality:        │
│               "Do these chunks answer the query?"   │
│               "Is anything missing?"                │
│                                                     │
│  4. DECIDE  → Sufficient? → Proceed to generation   │
│             → Insufficient? → Refine query, re-search│
│             → Max iterations? → Proceed with best   │
│                                                     │
│  (max 3 iterations to bound cost/latency)           │
└─────────────────────────────────────────────────────┘
```

**Why this matters for the role:**
The JD asks for "AI-powered systems" — not AI-assisted lookup. Agentic search demonstrates autonomous decision-making, self-evaluation, and iterative improvement, which is what production agent systems actually require.

**Key design decisions:**

- **Query decomposition** — complex queries are split into sub-queries (e.g., "Compare Q3 revenue to Q3 costs" → two separate retrievals)
- **Self-evaluation** — the LLM judges whether retrieved chunks are sufficient before generating
- **Bounded iteration** — max 3 search iterations to prevent runaway costs
- **Query rewriting** — if initial retrieval is poor, the agent rewrites the query (e.g., adding synonyms, restructuring)
- **Fallback to naive RAG** — simple factual queries skip the loop for speed

## Vector Store Strategy — pgvector vs Chroma

### Why Not Just Pick One?

Your friend's question is valid. The honest answer: different vector stores have different strengths, and a senior engineer should **measure** rather than assume.

### Our Approach: Pluggable Interface + Benchmark

We implement both pgvector and Chroma behind a common `VectorStore` protocol, then benchmark them head-to-head on the same dataset.

```
app/services/vectorstore.py
├── VectorStore (Protocol)       — search(query_embedding, top_k) → chunks
├── PgVectorStore                — PostgreSQL + pgvector (HNSW)
├── ChromaVectorStore            — Chroma (in-process or client/server)
└── get_vector_store()           — Factory, reads from config
```

### Benchmark Dimensions

| Dimension | What We Measure | Why It Matters |
|-----------|----------------|---------------|
| **Recall@5** | Does the correct chunk appear in top 5? | Core retrieval quality |
| **Recall@10** | Does it appear in top 10? | Measures near-misses |
| **MRR** (Mean Reciprocal Rank) | How high is the correct chunk ranked? | Position matters, not just presence |
| **Latency (p50/p95)** | Query response time | Production performance |
| **Ingestion throughput** | Chunks/second during bulk insert | Matters at scale |
| **Storage size** | Disk usage per 1000 chunks | Infrastructure cost |

### Trade-off Analysis (documented in benchmarks)

| Aspect | pgvector | Chroma |
|--------|----------|--------|
| **Extra infra** | None (PostgreSQL extension) | Separate service or embedded |
| **ACID transactions** | Yes (it's PostgreSQL) | No |
| **SQL alongside vectors** | Yes — JOIN, filter, aggregate | Limited filtering |
| **Scalability** | Millions of vectors with HNSW | Best for <1M vectors |
| **Ease of setup** | Need PostgreSQL | `pip install chromadb`, instant |
| **Metadata filtering** | Full SQL WHERE clauses | Basic key-value filters |

**Our hypothesis:** pgvector wins for this use case (financial data needs ACID, SQL joins, and metadata filtering), but we prove it with data rather than assumption.

## Chunk Size Strategy — Empirical, Not Arbitrary

### Why Chunk Size Matters

Chunk size is the most impactful RAG parameter that most projects hardcode without testing. Too small → fragments lose context. Too large → dilutes relevance signal. The right size depends on document type and query patterns.

### Our Approach: Benchmark Multiple Sizes

We ingest the same document at three chunk sizes and measure retrieval quality:

| Chunk Size | Token Overlap | Expected Behaviour |
|-----------|-------------|-------------------|
| 256 tokens | 25 tokens | High precision, may miss context |
| 512 tokens | 50 tokens | Balanced (our default) |
| 1024 tokens | 100 tokens | More context, lower precision |

**Benchmark process:**

1. Ingest the same financial document at all 3 chunk sizes (stored with metadata tag)
2. Run the golden dataset eval suite against each
3. Compare Recall@k, faithfulness, and answer quality
4. Document the winning size with evidence

This is captured in the eval framework — chunk size becomes a **tunable parameter** that we test, not assume.

### Retrieval Accuracy Test: Needle in a Haystack

To prove our retrieval works at scale:

1. Ingest 10+ financial documents (creating 1000+ chunks)
2. For each golden dataset question, record which chunk contains the answer
3. Measure: does the retriever find that chunk in top-k?
4. Report **Recall@5**, **Recall@10**, and **MRR** across the full dataset

This directly addresses "finding the target file among 1000+ files."

## Authorisation & Security

### Why This Matters

Financial documents are sensitive. Even in a demo, showing you've thought about access control signals production-readiness. The JD calls out "security across the platform."

### Auth Architecture

```
Client → API Key Auth → Request
              │
              ▼
      ┌───────────────┐
      │  Auth Middleware│
      │                │
      │ 1. Validate    │──▶ API key lookup (PostgreSQL)
      │    API key     │
      │                │
      │ 2. Check doc   │──▶ Document ACL check
      │    access      │    (can this key access this document?)
      │                │
      │ 3. Audit log   │──▶ Record who accessed what, when
      └───────────────┘
```

**What we implement:**

| Feature | Description | Why |
|---------|------------|-----|
| **API key auth** | Bearer token authentication for all endpoints | Basic access control |
| **Document-level ACL** | Each API key can only access specific documents | Multi-tenant data isolation |
| **Audit logging** | Every query logged with: who, what doc, when, what question | Compliance requirement in finance |
| **Rate limiting** | Per-key request limits | Prevents abuse, controls cost |

**Key decisions:**

- API key auth (not OAuth/JWT) — simple, appropriate for an API-first service
- Document-level ACL — demonstrates multi-tenancy thinking without overengineering
- Audit log in PostgreSQL — queryable, no extra infra
- Rate limiting via Redis — natural fit since Redis is already in the stack

## LLM Strategy — Multi-Provider Design

### Why Multi-Provider?

As of February 2026, Chinese LLMs have reached frontier parity with Western models at a fraction of the cost. Locking into a single provider is both expensive and limiting. Our architecture abstracts the LLM layer behind a common interface, allowing us to:

1. **Minimize development costs** — use DeepSeek V3 ($0.14/1M input) for iteration
2. **Maximize demo quality** — switch to Claude Sonnet 4.6 for interviews
3. **Demonstrate engineering maturity** — vendor abstraction is a production best practice
4. **Enable A/B comparison** — run the same query across providers to compare quality, latency, and cost

### LLM Landscape (February 2026)

| Model | Provider | SWE-bench | Input $/1M | Output $/1M | Best For |
|-------|----------|-----------|-----------|------------|----------|
| Claude Sonnet 4.6 | Anthropic | 80.8%* | $3.00 | $15.00 | Best reasoning quality, demos |
| GPT-5.2 | OpenAI | 80.0% | ~$0.80 | ~$3.20 | General tasks |
| MiniMax M2.5 | MiniMax | 80.2% | $0.20 | $1.00 | Coding tasks, open-weight |
| GLM-5 | Zhipu AI | 77.8% | $1.00 | $3.20 | MIT licensed, self-hostable |
| Kimi K2.5 | Moonshot AI | 76.8% | $0.60 | $2.50 | Agent swarms, 256K context |
| DeepSeek V3 | DeepSeek | 73.0% | $0.14 | $0.28 | **Best price/performance** |
| DeepSeek R1 | DeepSeek | — | $0.55 | $2.19 | Complex reasoning, 20-50x cheaper than o1 |
| Qwen 3.5-Plus | Alibaba | — | ~$0.11 | — | Cheapest flagship |

*\* Opus-tier models score 80.8-80.9%; Sonnet is slightly lower but best value.*

### Provider Architecture

```
app/services/llm.py
├── LLMProvider (Protocol)      — Common interface: complete(messages) → str
├── AnthropicProvider           — Claude API (native SDK)
├── OpenAICompatibleProvider    — Any OpenAI-compatible API:
│   ├── DeepSeek               — api.deepseek.com
│   ├── Qwen                   — dashscope.aliyuncs.com
│   ├── GLM-5                  — open.bigmodel.cn
│   ├── MiniMax                — api.minimax.chat
│   ├── Kimi                   — api.moonshot.cn
│   └── OpenAI                 — api.openai.com
└── get_llm_provider()         — Factory function, reads from config
```

**Key design decision:** Most Chinese LLMs expose OpenAI-compatible APIs. We only need two implementations — `AnthropicProvider` (for Claude's native SDK) and `OpenAICompatibleProvider` (for everything else). Switching providers is a single `.env` change:

```bash
# Development (cheap)
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=your-deepseek-key
LLM_MODEL=deepseek-chat

# Demo / Interview (best quality)
LLM_PROVIDER=anthropic
LLM_API_KEY=your-anthropic-key
LLM_MODEL=claude-sonnet-4-6
```

### Cost Strategy

| Phase | Model | Estimated Cost |
|-------|-------|---------------|
| Development & testing | DeepSeek V3 | ~$5-10/month |
| Evaluation runs | DeepSeek R1 or Qwen 3.5 | ~$5/month |
| Demo & interviews | Claude Sonnet 4.6 | ~$20 total |

## Implementation Phases

### Phase 1: Project Scaffolding (Complete)

Set up the foundational project structure, dependencies, Docker infrastructure, database models, and API skeleton.

- [x] pyproject.toml with all dependencies
- [x] Docker Compose (PostgreSQL+pgvector, Redis, app, Celery worker, Flower)
- [x] Dockerfile (Python 3.12, uv, non-root user)
- [x] Pydantic BaseSettings configuration
- [x] Async SQLAlchemy engine + session management
- [x] Database models: Document, Chunk (with pgvector embeddings, HNSW index)
- [x] Pydantic V2 request/response schemas
- [x] FastAPI app with health check endpoint
- [x] Celery app configuration and task skeleton

### Phase 2: Document Ingestion Pipeline

Build the full pipeline: PDF upload → parse → chunk → embed → store.

**Flow:**

1. `POST /ingest` accepts PDF upload → saves file → dispatches Celery task → returns task_id immediately
2. Celery worker executes: parse (Docling) → chunk (tiktoken) → embed (OpenAI) → store in vector store
3. `GET /ingest/{task_id}` polls task status
4. Chunk size is configurable (256/512/1024) for benchmarking

**Key decisions:**

- Token-based chunking (tiktoken) over character-based — aligns with model token limits
- **Configurable chunk size** — 256/512/1024 tokens, benchmarked empirically (not hardcoded)
- Batch embedding calls to OpenAI for efficiency
- Docling for PDF parsing — handles financial tables with multi-row headers and spanning cells
- **Pluggable vector store** — pgvector and Chroma behind common interface

**Tasks:**

- [ ] Implement Docling PDF parser with table-aware extraction
- [ ] Implement token-based chunker with configurable size + metadata preservation (page number, section)
- [ ] Implement OpenAI embedding service (batch embedding)
- [ ] Implement `VectorStore` protocol with pgvector and Chroma backends
- [ ] Wire up Celery task: parse → chunk → embed → store
- [ ] Create `/ingest` POST and `/ingest/{task_id}` GET endpoints
- [ ] Support chunk_size parameter in ingestion (for benchmarking different sizes)

### Phase 3: Agentic Search & Multi-Capability System

Build a LangGraph-based agentic search system with **multiple capabilities** and a provider-agnostic LLM layer. The search agent is an autonomous loop, not a naive RAG pipeline.

**Agent capabilities:**

| Capability | What It Does | Example Query |
|-----------|-------------|---------------|
| **Q&A** | Answer specific questions with source citations | "What was total revenue in Q3 2024?" |
| **Summarise** | Generate structured summaries of documents or sections | "Summarise the risk factors section" |
| **Compare** | Cross-document comparison on specific dimensions | "Compare revenue growth between AAPL and MSFT 10-Ks" |
| **Extract** | Pull structured financial metrics into JSON | "Extract all quarterly revenue figures as a table" |

**Agentic search flow (replaces naive RAG):**

```
User Query
    │
    ▼
┌──────────────────┐
│   Orchestrator    │──▶ Classify intent → select capability
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│            Agentic Search Loop               │
│                                              │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐  │
│  │  PLAN   │──▶│ RETRIEVE │──▶│ EVALUATE │  │
│  │ (rewrite│   │ (vector  │   │ (LLM     │  │
│  │  query) │   │  search) │   │  judges) │  │
│  └─────────┘   └──────────┘   └────┬─────┘  │
│       ▲                            │         │
│       │         Insufficient       │         │
│       └────────────────────────────┘         │
│                    Sufficient ──────────▶ EXIT│
│                  (max 3 iterations)          │
└──────────────────────────┬───────────────────┘
                           │
                           ▼
                    ┌─────────────┐     ┌───────────────┐
                    │   Analyst   │────▶│ LLM Provider  │
                    │  (generate) │     │ (configurable)│
                    └─────────────┘     └───────────────┘
                           │
                           ▼
                    Answer + Sources + Search Trace
```

**Key decisions:**

- **Agentic search over naive RAG** — agent plans, retrieves, self-evaluates, and refines in a loop
- **Query decomposition** — complex queries split into sub-queries
- **Self-evaluation** — LLM judges retrieval quality before generation
- **Bounded iteration** — max 3 loops to control cost/latency
- **Search trace in response** — shows the reasoning steps (query rewrites, retrieval scores per iteration)
- **Fallback** — simple factual queries detected by orchestrator skip the loop for speed
- LangGraph state machine makes the loop explicit and debuggable
- **LLM layer is provider-agnostic** — two implementations cover all providers
- Provider selected via `.env` config

**Files to create:**

- `app/services/llm.py` — Provider protocol, AnthropicProvider, OpenAICompatibleProvider, factory function
- `app/agents/search.py` — Agentic search loop (plan → retrieve → evaluate → refine)
- `app/agents/analyst.py` — LLM reasoning agent with capability-specific prompts
- `app/agents/orchestrator.py` — LangGraph graph with intent classification and capability routing
- `app/api/ask.py` — `/ask` endpoint

**Tasks:**

- [ ] Implement LLM provider abstraction (`LLMProvider` protocol + two implementations)
- [ ] Implement agentic search loop: plan → retrieve → evaluate → refine → re-retrieve
- [ ] Implement query decomposition for complex multi-part queries
- [ ] Implement self-evaluation: LLM judges whether retrieved chunks are sufficient
- [ ] Implement analyst agent with capability-specific system prompts (Q&A, summarise, compare, extract)
- [ ] Build LangGraph orchestrator: intent classification → capability routing → agentic search → analyst
- [ ] Include search trace in response (query rewrites, scores per iteration)
- [ ] Create `/ask` POST endpoint with `capability` field (auto-detected or explicit)

### Phase 4: A/B Provider Comparison & Performance Benchmarking

Build tooling to compare LLM providers side-by-side, benchmark retrieval accuracy, and track performance metrics over time.

**A/B Comparison flow:**

```
POST /compare
  { "question": "What was Q3 revenue?",
    "document_id": 1,
    "providers": ["anthropic/claude-sonnet-4-6", "deepseek/deepseek-chat", "zhipu/glm-5"] }

Response:
  { "results": [
      { "provider": "anthropic/claude-sonnet-4-6",
        "answer": "...",
        "latency_ms": 2340,
        "input_tokens": 1520,
        "output_tokens": 380,
        "estimated_cost_usd": 0.0103,
        "search_iterations": 1,
        "sources": [...] },
      ...
    ],
    "winner": { "quality": "anthropic/...", "speed": "deepseek/...", "cost": "deepseek/..." }
  }
```

**Retrieval accuracy benchmarking:**

```
POST /benchmark/retrieval
  { "document_id": 1, "chunk_sizes": [256, 512, 1024], "vector_stores": ["pgvector", "chroma"] }

Response:
  { "results": [
      { "chunk_size": 256, "vector_store": "pgvector",
        "recall_at_5": 0.82, "recall_at_10": 0.94, "mrr": 0.71,
        "avg_latency_ms": 12, "total_chunks": 1247 },
      { "chunk_size": 512, "vector_store": "pgvector",
        "recall_at_5": 0.88, "recall_at_10": 0.96, "mrr": 0.79,
        "avg_latency_ms": 8, "total_chunks": 634 },
      ...
    ]
  }
```

**Performance metrics tracked (per query):**

| Metric | What It Measures | Stored In |
|--------|-----------------|-----------|
| `latency_ms` | End-to-end response time | PostgreSQL `query_metrics` table |
| `retrieval_latency_ms` | Time spent on vector search | PostgreSQL `query_metrics` table |
| `llm_latency_ms` | Time spent waiting for LLM | PostgreSQL `query_metrics` table |
| `search_iterations` | How many agentic search loops were needed | PostgreSQL `query_metrics` table |
| `input_tokens` / `output_tokens` | Token usage | PostgreSQL `query_metrics` table |
| `estimated_cost_usd` | Cost per query (based on provider pricing) | PostgreSQL `query_metrics` table |
| `retrieval_scores` | Cosine similarity scores of retrieved chunks | PostgreSQL `query_metrics` table |

**Tasks:**

- [ ] Add `QueryMetric` DB model (latency, tokens, cost, provider, search iterations, timestamp)
- [ ] Implement metrics collection middleware (auto-capture on every `/ask` call)
- [ ] Implement provider pricing registry (maps provider+model to $/token)
- [ ] Create `/compare` POST endpoint — runs same query across N providers in parallel
- [ ] Create `/benchmark/retrieval` POST endpoint — measures Recall@k, MRR across chunk sizes and vector stores
- [ ] Create `/metrics` GET endpoint — aggregated stats (avg latency, cost-per-query, by provider)
- [ ] Add timing instrumentation to search agent and analyst

### Phase 5: Evaluation Framework & Feedback Loops

Build evaluation as a **core feature**, not an afterthought. Every pipeline change is measured. Evaluation results feed back into system improvement.

**Why eval-first matters for this role:**
The JD explicitly calls out "LLM evaluation, monitoring, and reliability." Most candidates can build a RAG pipeline — few can tell you whether it actually works well, prove it with data, and show how the system improves over time.

**Evaluation + feedback loop architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                    Feedback Loop                         │
│                                                         │
│  Golden Dataset ──▶ Eval Runner ──▶ Scores ──▶ Store    │
│       │                                         │       │
│       │              ┌──────────────────────────┘       │
│       │              ▼                                  │
│       │    ┌──────────────────┐                         │
│       │    │ Regression Check │──▶ Did we improve?      │
│       │    └────────┬─────────┘                         │
│       │             │                                   │
│       │    ┌────────▼─────────┐                         │
│       │    │ Failure Analysis │──▶ Which test cases      │
│       │    └────────┬─────────┘    regressed? Why?      │
│       │             │                                   │
│       │    ┌────────▼─────────┐                         │
│       └────│ Dataset Update   │──▶ Add new edge cases   │
│            └──────────────────┘    from failures        │
└─────────────────────────────────────────────────────────┘
```

**Metrics:**

| Metric | What It Measures | Why It Matters for Finance |
|--------|-----------------|--------------------------|
| Faithfulness | Is the answer grounded in context? | Prevents hallucinated financial figures |
| Answer Relevancy | Does the answer address the question? | Ensures actionable responses |
| Context Precision | Are retrieved chunks relevant? | Reduces noise in retrieval |
| Context Recall | Did we find all necessary info? | Ensures completeness |
| **Numerical Accuracy** | Do dollar amounts, percentages, dates match source? | **Custom metric for finance** |
| **Retrieval Recall@k** | Does the correct chunk appear in top-k? | Measures search quality at scale |

**Feedback loop features:**

- **Regression tracking** — every eval run stored with timestamp; `/evaluate/history` shows trends
- **Failure analysis** — when a test case fails, log the full search trace (what was retrieved, what was generated, what went wrong)
- **Automatic threshold alerts** — flag when any metric drops below configurable threshold
- **Cross-provider evaluation** — run the same eval suite against multiple providers to compare quality
- **Parameter sweep** — run evals across chunk sizes and vector stores to find optimal config

**Tasks:**

- [ ] Create golden dataset: 30+ Q&A pairs covering all 4 capabilities, with known source chunks tagged
- [ ] Add `EvalResult` DB model (metric scores, provider, chunk_size, vector_store, timestamp)
- [ ] Implement DeepEval test cases with 6 metrics (including custom numerical accuracy and retrieval recall)
- [ ] Implement feedback loop: failure analysis with full search trace logging
- [ ] Create `/evaluate` POST endpoint — runs eval suite, stores results, returns scores + regression comparison
- [ ] Create `/evaluate/history` GET endpoint — evaluation score history over time with trend analysis
- [ ] Create `/evaluate/failures` GET endpoint — detailed failure analysis for debugging
- [ ] Make evals runnable via `pytest tests/eval/` for CI/CD quality gates
- [ ] Implement cross-provider and parameter sweep evaluation

### Phase 6: Authorisation & Security

Add production-grade access control appropriate for financial data.

**Tasks:**

- [ ] Implement API key authentication middleware (Bearer token)
- [ ] Add `ApiKey` DB model with scopes and rate limits
- [ ] Implement document-level access control (which keys can access which documents)
- [ ] Add audit logging middleware (who queried what document, when, what question)
- [ ] Implement rate limiting via Redis (per-key, configurable)
- [ ] Add `/admin/keys` CRUD endpoints for managing API keys
- [ ] Ensure all sensitive data (embeddings, internal metadata) is never exposed in API responses

### Phase 7: Polish & Documentation

- [ ] Add sample financial PDFs (public company earnings reports, 10-K excerpts, 10+ docs for scale testing)
- [ ] Ensure `docker compose up` starts everything cleanly
- [ ] Add error handling and logging throughout
- [ ] Final README updates with demo walkthrough, screenshots, and benchmark results
- [ ] Add a demo script: ingest 10+ docs → ask all 4 capabilities → compare providers → run benchmarks → evaluate
- [ ] Document benchmark results (chunk sizes, vector stores, provider comparison) in `docs/BENCHMARKS.md`

## API Endpoints (Complete)

| Method | Endpoint | Description | Phase |
|--------|---------|-------------|-------|
| `GET` | `/health` | Health check | 1 |
| `POST` | `/ingest` | Upload a financial PDF for processing | 2 |
| `GET` | `/ingest/{task_id}` | Check ingestion status | 2 |
| `POST` | `/ask` | Agentic search + answer (auto-routes to capability) | 3 |
| `POST` | `/compare` | Run same query across multiple LLM providers | 4 |
| `POST` | `/benchmark/retrieval` | Benchmark retrieval accuracy (chunk sizes, vector stores) | 4 |
| `GET` | `/metrics` | Aggregated performance metrics by provider | 4 |
| `POST` | `/evaluate` | Run RAG evaluation suite | 5 |
| `GET` | `/evaluate/history` | Evaluation score history and trends | 5 |
| `GET` | `/evaluate/failures` | Detailed failure analysis | 5 |
| `POST` | `/admin/keys` | Create/manage API keys | 6 |

## Verification Plan

1. **Ingestion**: Upload 10+ financial PDFs, verify chunks with embeddings at multiple chunk sizes
2. **Agentic search**: Ask a complex query, verify the search trace shows query rewriting and multi-iteration retrieval
3. **Multi-capability**: Test all 4 capabilities — Q&A, summarise, compare, extract
4. **Retrieval accuracy**: Run `/benchmark/retrieval` with 1000+ chunks, verify Recall@5 > 0.85
5. **Chunk size comparison**: Compare 256/512/1024 token chunks, document which performs best
6. **Vector store comparison**: Compare pgvector vs Chroma on same dataset, document results
7. **Provider comparison**: Run `/compare` with 3 providers, verify side-by-side results
8. **Evaluation**: Run `/evaluate`, verify all 6 metrics return scores; verify regression tracking
9. **Feedback loop**: Run eval twice with a pipeline change, verify `/evaluate/history` shows comparison
10. **Auth**: Verify API key auth, document ACL, and audit logging
11. **Infrastructure**: `docker compose up` starts all services; Flower dashboard at `localhost:5555`
