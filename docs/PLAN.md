# Implementation Plan

This document captures the design decisions and implementation roadmap for the Financial Document Q&A Agent. It serves as both a development guide and a record of architectural thinking.

## Problem Statement

Financial institutions need to extract insights from large volumes of documents (earnings reports, 10-Ks, research notes). Manual analysis is slow and error-prone. This project builds a production-grade AI system that automates financial document Q&A with verifiable, source-cited answers.

## Architecture

```
┌──────────────┐     ┌────────────────┐     ┌──────────────────┐
│  FastAPI API  │────▶│  Orchestrator  │────▶│  Retriever Agent │
│  /ingest      │     │  (LangGraph)   │     │  (vector search) │
│  /ask         │     │                │────▶│  Analyst Agent   │
│  /evaluate    │     │                │     │  (LLM reasoning) │
└──────┬───────┘     └────────────────┘     └──────────────────┘
       │                                           │
       ▼                                           ▼
┌──────────────┐                          ┌────────────────────┐
│ Celery Worker│                          │ PostgreSQL+pgvector│
│ + Redis      │                          │ (chunks, embeddings│
│ (async tasks)│                          │  eval results)     │
└──────────────┘                          └────────────────────┘
```

## Technology Choices

| Component | Choice | Alternatives Considered | Why This One |
|-----------|--------|------------------------|-------------|
| API | FastAPI + Pydantic V2 | Django REST, Flask | Async-native, auto OpenAPI docs, V2 is 5-50x faster validation |
| Multi-agent | LangGraph | CrewAI, OpenAI Agents SDK, Claude Agent SDK | Graph-based state machines are most debuggable; conditional routing for fintech workflows |
| Vector DB | PostgreSQL + pgvector | Pinecone, Weaviate, Chroma | No extra infra, full SQL alongside vectors, ACID transactions |
| Embeddings | OpenAI text-embedding-3-small | Nomic Embed V2, Voyage AI | No GPU needed, high quality at low cost, simple API |
| LLM | Multi-provider (Claude, DeepSeek, Qwen, GLM-5) | Single-vendor lock-in | Provider-agnostic design: cheap models for dev, Claude for demo (see LLM Strategy below) |
| PDF Parsing | Docling (IBM) | pdfplumber, Camelot, PyMuPDF | Purpose-built for structured docs, understands financial tables |
| Task Queue | Celery + Redis | FastAPI BackgroundTasks, RQ | Reliability (retries, persistence), monitoring (Flower), scalability |
| Evaluation | DeepEval | Ragas, LangSmith, custom | pytest-style, CI/CD ready, detailed metric explanations |

## LLM Strategy — Multi-Provider Design

### Why Multi-Provider?

As of February 2026, Chinese LLMs have reached frontier parity with Western models at a fraction of the cost. Locking into a single provider is both expensive and limiting. Our architecture abstracts the LLM layer behind a common interface, allowing us to:

1. **Minimize development costs** — use DeepSeek V3 ($0.14/1M input) for iteration
2. **Maximize demo quality** — switch to Claude Sonnet 4.6 for interviews
3. **Demonstrate engineering maturity** — vendor abstraction is a production best practice

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
2. Celery worker executes: parse (Docling) → chunk (tiktoken, 512 tokens) → embed (OpenAI) → store in pgvector
3. `GET /ingest/{task_id}` polls task status

**Key decisions:**

- Token-based chunking (tiktoken) over character-based — aligns with model token limits
- 512 tokens per chunk with 50 token overlap — ~1-2 paragraphs, prevents info loss at boundaries
- Batch embedding calls to OpenAI for efficiency
- Docling for PDF parsing — handles financial tables with multi-row headers and spanning cells

**Tasks:**

- [ ] Implement Docling PDF parser with table-aware extraction
- [ ] Implement token-based chunker with metadata preservation (page number, section)
- [ ] Implement OpenAI embedding service (batch embedding)
- [ ] Wire up Celery task: parse → chunk → embed → store
- [ ] Create `/ingest` POST and `/ingest/{task_id}` GET endpoints

### Phase 3: Multi-Agent RAG System

Build the LangGraph-based multi-agent system with a provider-agnostic LLM layer.

**Agent flow:**

```
User Query → Orchestrator → Retriever (pgvector search) → Analyst (LLM) → Answer + Sources
                                                              │
                                                    ┌────────┴────────┐
                                                    │  LLM Provider   │
                                                    │  (configurable) │
                                                    ├─────────────────┤
                                                    │ • Anthropic     │
                                                    │ • DeepSeek      │
                                                    │ • Qwen / GLM-5  │
                                                    │ • Any OAI-compat│
                                                    └─────────────────┘
```

**Key decisions:**

- LangGraph state machine makes agent flow explicit and testable
- Retriever returns top-5 chunks with cosine similarity scores
- **LLM layer is provider-agnostic** — two implementations cover all providers:
  - `AnthropicProvider` for Claude (native SDK, streaming support)
  - `OpenAICompatibleProvider` for everything else (DeepSeek, Qwen, GLM-5, Kimi, MiniMax, OpenAI)
- Provider is selected via `.env` config — switch between cheap dev models and Claude for demos
- Responses include source citations (page numbers) for verifiability

**Files to create:**

- `app/services/llm.py` — Provider protocol, AnthropicProvider, OpenAICompatibleProvider, factory function
- `app/agents/retriever.py` — Vector search agent
- `app/agents/analyst.py` — LLM reasoning agent (uses LLM provider, not hardcoded Claude)
- `app/agents/orchestrator.py` — LangGraph graph definition
- `app/api/ask.py` — `/ask` endpoint

**Tasks:**

- [ ] Implement LLM provider abstraction (`LLMProvider` protocol + two implementations)
- [ ] Add multi-provider config to `app/config.py` (provider, base_url, api_key, model)
- [ ] Implement retriever agent: cosine similarity search, top-5 chunks, relevance scores
- [ ] Implement analyst agent: uses provider-agnostic LLM service with financial analysis prompt
- [ ] Build LangGraph graph: orchestrator → retriever → analyst → response
- [ ] Create `/ask` POST endpoint
- [ ] Response includes answer, source chunks with page numbers, provider/model used

### Phase 4: Evaluation Framework

Build a systematic RAG evaluation pipeline using DeepEval.

**Metrics:**

| Metric | What It Measures | Why It Matters for Finance |
|--------|-----------------|--------------------------|
| Faithfulness | Is the answer grounded in context? | Prevents hallucinated financial figures |
| Answer Relevancy | Does the answer address the question? | Ensures actionable responses |
| Context Precision | Are retrieved chunks relevant? | Reduces noise in retrieval |
| Context Recall | Did we find all necessary info? | Ensures completeness |

**Tasks:**

- [ ] Create golden dataset: 20-30 Q&A pairs from sample financial docs
- [ ] Implement DeepEval test cases with 4 RAG metrics
- [ ] Create `/evaluate` POST endpoint that runs eval suite and returns scores
- [ ] Store eval results in PostgreSQL for tracking over time
- [ ] Make evals runnable via `pytest tests/eval/` for CI/CD

### Phase 5: Polish & Documentation

- [ ] Add sample financial PDF (public company earnings report)
- [ ] Ensure `docker compose up` starts everything cleanly
- [ ] Add error handling and logging throughout
- [ ] Final README updates with demo walkthrough

## Verification Plan

1. **Ingestion**: Upload a sample earnings report via `POST /ingest`, verify chunks with embeddings appear in PostgreSQL
2. **Q&A**: Ask financial questions via `POST /ask`, verify answer cites correct source pages
3. **Evaluation**: Run `POST /evaluate` and verify all 4 metrics return scores; run `pytest tests/eval/` for CI/CD validation
4. **Infrastructure**: `docker compose up` starts all services cleanly; `docker compose down` tears down cleanly
5. **Task queue**: Verify Celery processes ingestion asynchronously — API returns immediately, task completes in background (visible in Flower at `localhost:5555`)
