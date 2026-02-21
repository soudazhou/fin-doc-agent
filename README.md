# Financial Document Q&A Agent

A production-grade **multi-agent RAG system** for answering questions over financial documents (earnings reports, 10-Ks, annual reports). Built to demonstrate end-to-end AI engineering — from document ingestion to evaluation.

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

## What This Project Demonstrates

| Capability | Implementation |
|-----------|---------------|
| **RAG Pipeline** | PDF parsing (Docling) → token-based chunking (tiktoken) → embeddings (OpenAI) → pgvector storage & retrieval |
| **Multi-Agent Orchestration** | LangGraph state machine coordinating Retriever and Analyst agents |
| **LLM Evaluation** | DeepEval framework measuring faithfulness, relevancy, context precision/recall |
| **Async Task Processing** | Celery + Redis for background document ingestion with status tracking |
| **Production Patterns** | Pydantic V2 validation, async SQLAlchemy, Docker Compose, health checks |

## Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| API Framework | FastAPI + Pydantic V2 | Async-native, auto-generated OpenAPI docs, type-safe validation |
| Multi-Agent | LangGraph | Graph-based state machines for debuggable, testable agent orchestration |
| Vector Storage | PostgreSQL + pgvector | No separate vector DB needed — ACID transactions + vector search in one |
| Embeddings | OpenAI `text-embedding-3-small` | High quality, no GPU required, 1536 dimensions |
| LLM | Claude (claude-sonnet-4-6) | Strong reasoning for financial analysis, 200K context window |
| PDF Parsing | Docling (IBM) | Purpose-built for structured documents with complex tables |
| Task Queue | Celery 5.6 + Redis | Reliable background processing with monitoring (Flower) |
| Evaluation | DeepEval | pytest-style RAG evaluation with CI/CD integration |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- API keys: [Anthropic](https://console.anthropic.com/settings/keys) + [OpenAI](https://platform.openai.com/api-keys)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/soudazhou/fin-doc-agent.git
cd fin-doc-agent

# 2. Copy environment template and add your API keys
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY and OPENAI_API_KEY

# 3. Start all services
docker compose up

# 4. Verify — API docs available at:
#    http://localhost:8000/docs     (Swagger UI)
#    http://localhost:5555          (Flower — Celery monitoring)
```

### Local Development (without Docker)

```bash
# Install dependencies
uv sync --group dev

# Start PostgreSQL and Redis (still via Docker)
docker compose up db redis -d

# Run the API server
uv run uvicorn app.main:app --reload

# Run Celery worker (separate terminal)
uv run celery -A app.workers.celery_app worker --loglevel=info
```

## API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Upload a financial PDF for processing |
| `GET` | `/ingest/{task_id}` | Check ingestion status |
| `POST` | `/ask` | Ask a question about ingested documents |
| `POST` | `/evaluate` | Run RAG evaluation suite |

### Example: Ask a Question

```bash
# Ingest a document
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/samples/earnings_report.pdf"

# Response: {"document_id": 1, "task_id": "abc-123", "status": "processing"}

# Poll for completion
curl http://localhost:8000/ingest/abc-123

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total revenue in Q3?", "document_id": 1}'
```

## Project Structure

```
fin-doc-agent/
├── app/
│   ├── main.py                 # FastAPI entry point & lifespan
│   ├── config.py               # Pydantic BaseSettings (env vars)
│   ├── api/                    # Route handlers
│   │   ├── ingest.py           # Document upload & status
│   │   ├── ask.py              # Question answering
│   │   └── evaluate.py         # RAG evaluation
│   ├── agents/                 # LangGraph multi-agent system
│   │   ├── orchestrator.py     # Graph definition & state machine
│   │   ├── retriever.py        # Vector search agent
│   │   └── analyst.py          # LLM reasoning agent (Claude)
│   ├── db/                     # Database layer
│   │   ├── engine.py           # Async SQLAlchemy engine
│   │   └── models.py           # ORM models (Document, Chunk)
│   ├── models/                 # Pydantic V2 schemas
│   │   ├── requests.py         # API request validation
│   │   └── responses.py        # API response serialization
│   ├── services/               # Business logic
│   │   ├── parser.py           # PDF parsing (Docling)
│   │   ├── chunker.py          # Token-based chunking (tiktoken)
│   │   ├── embedder.py         # Embedding generation (OpenAI)
│   │   └── evaluator.py        # RAG evaluation (DeepEval)
│   └── workers/                # Background processing
│       ├── celery_app.py       # Celery configuration
│       └── tasks.py            # Ingestion pipeline task
├── tests/
│   └── eval/                   # Evaluation test suite
├── data/
│   └── samples/                # Sample financial PDFs
├── docs/
│   └── PLAN.md                 # Implementation plan & design decisions
├── docker-compose.yml          # Full local environment
├── Dockerfile                  # Python 3.12 app image
├── pyproject.toml              # Dependencies & tool config
└── .env.example                # Environment variable template
```

## Design Decisions

Key architectural decisions are documented in two places:

1. **Inline code comments** — Every file contains detailed comments explaining *why* each choice was made, not just *what* the code does.
2. **[docs/PLAN.md](docs/PLAN.md)** — The full implementation plan with architecture rationale, technology comparisons, and phase breakdown.

## Implementation Roadmap

- [x] **Phase 1**: Project scaffolding — FastAPI, Docker, DB models, config
- [ ] **Phase 2**: Document ingestion pipeline — PDF parsing, chunking, embeddings
- [ ] **Phase 3**: Multi-agent RAG — LangGraph orchestrator, retriever, analyst
- [ ] **Phase 4**: Evaluation framework — DeepEval metrics, golden dataset
- [ ] **Phase 5**: Polish — sample data, error handling, monitoring

## Running Tests

```bash
# Unit tests
uv run pytest tests/

# RAG evaluation suite
uv run pytest tests/eval/ -v
```

## License

MIT
