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
| LLM | Claude (claude-sonnet-4-6) | GPT-4o, open-source models | Strong reasoning, 200K context, excellent instruction following |
| PDF Parsing | Docling (IBM) | pdfplumber, Camelot, PyMuPDF | Purpose-built for structured docs, understands financial tables |
| Task Queue | Celery + Redis | FastAPI BackgroundTasks, RQ | Reliability (retries, persistence), monitoring (Flower), scalability |
| Evaluation | DeepEval | Ragas, LangSmith, custom | pytest-style, CI/CD ready, detailed metric explanations |

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

Build the LangGraph-based multi-agent system for question answering.

**Agent flow:**

```
User Query → Orchestrator → Retriever (pgvector search) → Analyst (Claude) → Answer + Sources
```

**Key decisions:**

- LangGraph state machine makes agent flow explicit and testable
- Retriever returns top-5 chunks with cosine similarity scores
- Analyst uses Claude with a financial analysis system prompt
- Responses include source citations (page numbers) for verifiability

**Tasks:**

- [ ] Implement retriever agent: cosine similarity search, top-5 chunks, relevance scores
- [ ] Implement analyst agent: Claude API call with retrieved context, financial analysis prompt
- [ ] Build LangGraph graph: orchestrator → retriever → analyst → response
- [ ] Create `/ask` POST endpoint
- [ ] Response includes answer, source chunks with page numbers, confidence score

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
