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
| Embeddings | Provider-agnostic (OpenAI, DashScope, etc.) | Nomic Embed V2, Voyage AI | Configurable `EMBEDDING_BASE_URL` — any OpenAI-compatible embedding API |
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

### Phase 2: Document Ingestion Pipeline (Complete)

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
- Mixed sync/async interface: `add_chunks()` sync for Celery, `search()` async for FastAPI
- Lazy sync engine initialization — avoids import errors when only FastAPI is running
- ChromaDB >=1.5.1 for Pydantic v2 + Python 3.14 compatibility

**Tasks:**

- [x] Implement Docling PDF parser with table-aware extraction (`app/services/parser.py`)
- [x] Implement token-based chunker with configurable size + metadata preservation (`app/services/chunker.py`)
- [x] Implement OpenAI embedding service with batch processing (`app/services/embedder.py`)
- [x] Implement `VectorStore` protocol with pgvector and Chroma backends (`app/services/vectorstore.py`)
- [x] Add sync SQLAlchemy engine for Celery workers (`app/db/engine.py`)
- [x] Wire up Celery task: parse → chunk → embed → store (`app/workers/tasks.py`)
- [x] Create `/ingest` POST and `/ingest/{task_id}` GET endpoints (`app/api/ingest.py`)
- [x] Support chunk_size/chunk_overlap parameters in ingestion (for benchmarking)
- [x] Unit tests: 9 chunker tests + 4 ChromaDB vectorstore tests (all passing)

### Phase 3: Agentic Search & Multi-Capability System (Complete)

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

- [x] Implement LLM provider abstraction (`LLMProvider` protocol + two implementations) (`app/services/llm.py`)
- [x] Implement agentic search loop: embed → retrieve → evaluate → refine → re-retrieve (`app/agents/search.py`)
- [x] Implement self-evaluation: LLM judges whether retrieved chunks are sufficient
- [x] Implement analyst agent with capability-specific system prompts (Q&A, summarise, compare, extract) (`app/agents/analyst.py`)
- [x] Build LangGraph orchestrator: rule-based intent classification → agentic search → analyst (`app/agents/orchestrator.py`)
- [x] Include search trace in response (query rewrites, scores per iteration)
- [x] Create `/ask` POST endpoint with `capability` field (auto-detected or explicit) (`app/api/ask.py`)
- [x] Unit tests: 22 tests covering classification, context formatting, query simplification, and mock LLM analysis (all passing)

### Phase 4: A/B Provider Comparison & Performance Benchmarking (Complete)

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

- [x] Add `QueryMetric` DB model (`app/db/models.py`) — latency, tokens, cost, provider, search iterations
- [x] Implement metrics collection on every `/ask` call via BackgroundTasks (`app/api/ask.py`)
- [x] Implement provider pricing registry (`app/services/pricing.py`) — 12 models across 7 providers
- [x] Add `create_provider_from_id()` non-singleton factory (`app/services/llm.py`) — enables parallel multi-provider execution
- [x] Add `llm_override` injection to LangGraph orchestrator (`app/agents/orchestrator.py`)
- [x] Create `/compare` POST endpoint — runs same query across N providers in parallel (`app/api/benchmark.py`)
- [x] Create `/benchmark/retrieval` POST endpoint — measures search latency at different top_k and vector stores
- [x] Create `/metrics` GET endpoint — aggregated stats by provider with lookback window
- [x] Unit tests: 25 tests covering pricing, provider ID parsing, winner computation, LLM injection, and request validation (all passing)

### Phase 5: Evaluation Framework & Feedback Loops (Complete)

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

- [x] Create golden dataset: 29 Q&A pairs covering all 4 capabilities (qa, summarise, compare, extract), with numerical values and source chunk IDs
- [x] Add `EvalRun` + `EvalTestResult` DB models (metric scores, provider, run config, timestamps, JSONB for flexible metric storage)
- [x] Implement 2 custom pure-Python metrics: numerical accuracy (financial number matching) and retrieval recall@k (chunk ID set intersection)
- [x] Implement eval runner service: dataset loading, test case execution, DeepEval LLM-as-judge integration (4 metrics), custom metrics, pass/fail determination
- [x] Create `POST /evaluate` endpoint — background task pattern (like /ingest), returns run_id for polling
- [x] Create `GET /evaluate/runs/{run_id}` endpoint — full results with per-test-case details and regression comparison
- [x] Create `GET /evaluate/history` endpoint — evaluation score history with trend analysis (improving/declining/stable)
- [x] Create `GET /evaluate/failures` endpoint — detailed failure analysis with metric reasons, search traces, and most common failing metric
- [x] Make evals runnable via `pytest tests/eval/` for CI/CD quality gates — 47 eval tests, all passing without API keys
- [x] Cross-provider evaluation via provider_id parameter (reuses Phase 4's provider injection)
- [x] Unit tests: 47 new tests (24 eval metrics + 12 eval runner + 11 golden dataset validation), 107 total

### Phase 6: Authorisation & Security (Complete)

Add production-grade access control appropriate for financial data. Auth is **toggleable** — disabled by default for dev convenience (`auth_enabled=False`). When disabled, all endpoints work exactly as before (no breaking change).

**Auth architecture:**

```
Client → Authorization: Bearer sk-abc123...
              │
              ▼
      ┌───────────────────────────┐
      │  get_current_api_key()    │  ← FastAPI Depends()
      │  SHA-256 hash → DB lookup │
      │  Validate: active, !expired│
      │  Rate limit (Redis ZSET)  │
      └─────────────┬─────────────┘
                    │
          ┌─────────▼─────────┐
          │  check_scope()    │  "ask", "ingest", "evaluate", "admin"
          │  check_doc_access()│  Document-level ACL
          └─────────┬─────────┘
                    │
              ┌─────▼─────┐
              │  Endpoint  │
              └─────┬─────┘
                    │
      ┌─────────────▼─────────────┐
      │  AuditLoggingMiddleware   │  Starlette middleware
      │  → INSERT audit_logs row  │  (background, non-blocking)
      └───────────────────────────┘
```

**Key decisions:**

- SHA-256 hashing for API keys (not bcrypt) — keys are 32-byte random tokens, SHA-256 is secure and fast for high-entropy secrets
- Separate `api_key_documents` table for document ACL (not JSONB) — enables SQL JOINs and FK integrity
- Redis sliding window rate limiter (ZSET) with graceful degradation — Redis outage doesn't block the API
- Audit logging via Starlette middleware — wraps entire request lifecycle, non-blocking writes

**Tasks:**

- [x] Add `ApiKey`, `ApiKeyDocument`, `AuditLog` DB models with indexes (`app/db/models.py`)
- [x] Implement API key generation and SHA-256 hashing (`app/services/auth.py`)
- [x] Create auth dependency: `get_current_api_key()`, `check_scope()`, `check_document_access()` (`app/api/deps.py`)
- [x] Implement Redis-based sliding window rate limiter with graceful degradation (`app/services/rate_limiter.py`)
- [x] Create admin CRUD endpoints: 7 key management + audit log query (`app/api/admin.py`)
- [x] Create audit logging middleware (`app/api/audit.py`)
- [x] Integrate auth into all existing endpoints (ask, ingest, benchmark, evaluate) with scope and ACL checks
- [x] Register admin router and audit middleware (`app/main.py`)
- [x] Unit tests: 33 new tests (key gen, auth dep, scope, ACL, rate limiter, models), 140 total

### Phase 7: Polish & Documentation

- [x] Add GitHub Actions CI workflow (ruff lint + pytest on every PR)
- [x] Fix all pre-existing lint errors (12 errors across 4 files) for clean CI
- [x] Add a demo script: ingest → ask all 4 capabilities → compare providers → run benchmarks → evaluate (`scripts/demo.sh`)
- [x] Document benchmark methodology, metrics, and provider pricing in `docs/BENCHMARKS.md`
- [x] Create `.env.example` with 7 LLM provider presets (including Alibaba Cloud all-in-one) and full configuration template
- [x] Make embedding endpoint configurable (`EMBEDDING_BASE_URL`) so the full stack can run on a single Alibaba Cloud (DashScope) API key
- [x] Final README updates — mark Phase 7 complete, add demo script reference
- [x] Add sample financial PDFs — synthetic Apple Q3 2024 earnings report aligned with golden dataset (`scripts/generate_sample_pdf.py`, `data/samples/apple_q3_2024_earnings.pdf`)
- [ ] Populate `docs/BENCHMARKS.md` results tables after running benchmarks with real data
- [x] Integration smoke tests (verified 22 Feb 2026, DashScope Singapore):
  - [x] `POST /ingest` — Docling parsed 7-page PDF → 6 chunks → embedded (text-embedding-v4, 1536d) → stored in pgvector (~50s on CPU)
  - [x] `POST /ask` — all 4 capabilities verified: Q&A ($85.8B revenue), summarise (full breakdown), compare (Services +14% vs iPhone -1%), extract (geographic region JSON)
  - [x] `POST /compare` with qwen-plus vs qwen-turbo — both answered correctly (46.3% gross margin); turbo 2x faster, plus richer citations
  - [x] `GET /metrics` returns aggregated stats: 12 queries, avg 5.9s latency, $0.002 total cost

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
| `POST` | `/evaluate` | Run RAG evaluation suite (background task) | 5 |
| `GET` | `/evaluate/runs/{run_id}` | Get evaluation results + regression comparison | 5 |
| `GET` | `/evaluate/history` | Evaluation score history and trends | 5 |
| `GET` | `/evaluate/failures` | Detailed failure analysis with search traces | 5 |
| `POST` | `/admin/keys` | Create a new API key | 6 |
| `GET` | `/admin/keys` | List all API keys | 6 |
| `GET` | `/admin/keys/{key_id}` | Get API key details | 6 |
| `PATCH` | `/admin/keys/{key_id}` | Update an API key | 6 |
| `DELETE` | `/admin/keys/{key_id}` | Delete an API key | 6 |
| `POST` | `/admin/keys/{key_id}/documents` | Grant document access | 6 |
| `DELETE` | `/admin/keys/{key_id}/documents/{doc_id}` | Revoke document access | 6 |
| `GET` | `/admin/audit` | Query audit logs | 6 |

## Verification Plan

### Unit Tests (no API keys required)

All pure-logic tests run via `uv run pytest tests/` with zero external dependencies:

- Pricing registry, provider ID parsing, winner computation, Pydantic validation
- Chunker, vectorstore (Chroma in-memory), agent classification, LLM injection via mocks
- Eval metrics (numerical accuracy, retrieval recall@k), eval runner (dataset loading, test case execution, pass/fail), golden dataset validation
- Auth: key generation/hashing, auth dependency, scope checking, document ACL, rate limiter, request/response model validation
- 140 tests passing as of Phase 6 (107 from Phases 1-5 + 33 new auth tests)

### Integration Tests (require API keys + running services)

Verified on 22 Feb 2026 using Alibaba Cloud Model Studio (DashScope Singapore region), single API key for both LLM (`qwen-plus`) and embeddings (`text-embedding-v4`).

1. **Ingestion**: [x] Uploaded Apple Q3 2024 earnings PDF → Docling parsed 7 pages → 6 chunks with 1536d embeddings stored in pgvector (~50s on CPU)
2. **Agentic search**: [x] Queries trigger multi-iteration search loop with query rewriting; search trace included in responses
3. **Multi-capability**: [x] All 4 capabilities verified — Q&A ($85.8B revenue), summarise (full financial breakdown), compare (Services +14% vs iPhone -1%), extract (geographic region table as structured JSON)
4. **Provider comparison**: [x] `POST /compare` with qwen-plus vs qwen-turbo — both answered correctly (46.3% gross margin); turbo 2x faster (4.1s vs 8.1s), plus provided richer multi-source citations
5. **Retrieval benchmark**: Deferred — requires multiple ingested documents
6. **Metrics dashboard**: [x] `GET /metrics` returns aggregated stats: 12 queries tracked, avg 5.9s latency, $0.002 total cost, breakdown by provider
7. **Retrieval accuracy**: Deferred — requires 1000+ chunks from multiple documents
8. **Chunk size comparison**: Deferred — requires re-ingestion at multiple chunk sizes
9. **Vector store comparison**: Deferred — requires Chroma service running alongside pgvector
10. **Evaluation**: Deferred — requires golden dataset aligned with ingested document
11. **Feedback loop**: Deferred — requires multiple eval runs
12. **Auth**: Deferred — auth disabled by default for development
13. **Infrastructure**: [x] `docker compose up` starts all 5 services (app, celery-worker, flower, db, redis); health check OK; Flower dashboard at `localhost:5555`

#### Issues Found & Fixed During Integration Testing

| Issue | Root Cause | Fix | PR |
|-------|-----------|-----|-----|
| Build context 1.32GB | No `.dockerignore` | Created `.dockerignore` | #12 |
| `OSError: README.md not found` | hatchling needs `README.md` during `uv sync` | Added `README.md` to `COPY` line | #12 |
| `celery: executable file not found` | `uv sync` installs to `.venv`, not system PATH | Prefix all commands with `uv run` | #12 |
| Host `.venv` overwrites container `.venv` | Volume mount `.:/app` includes macOS `.venv` | Added anonymous volume `- /app/.venv` | #12 |
| Permission denied on `.venv` | Container `.venv` owned by root, app runs as `appuser` | `chown -R appuser:appuser /app` | #12 |
| `flower` command not found | `flower` not in production dependencies | `uv add flower` | #12 |
| `ConnectionRefusedError` in container | `.env` used `localhost` instead of Docker service names | Changed to `db` and `redis` | .env fix |
| `ImportError: libxcb.so.1` | Docling/OpenCV requires X11 system libraries | Added `libgl1 libglib2.0-0 libxcb1` to Dockerfile | #13 |
| `expected 1536 dimensions, not 1024` | `text-embedding-v3` defaults to 1024 dims | Switched to `text-embedding-v4`; pass `dimensions` param | Pending |
| Empty results on valid queries | Similarity threshold 0.7 too aggressive for DashScope embeddings | Lowered `RETRIEVAL_SIMILARITY_THRESHOLD` to 0.45 in `.env` | Pending |
