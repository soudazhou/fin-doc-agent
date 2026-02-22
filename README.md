# Financial Document Q&A Agent

A production-grade **agentic AI system** for financial document analysis. Goes beyond basic RAG with autonomous search, multi-capability agents, provider A/B testing, retrieval benchmarking, eval feedback loops, and access control.

## Why This Project Exists

Most RAG demos are a solved problem in 2026 — embed, retrieve, generate. This project demonstrates what **production AI engineering** actually requires:

| What Most Demos Do | What This Project Does |
|--------------------|-----------------------|
| Single Q&A capability | 4 capabilities: Q&A, summarise, compare, extract |
| Naive embed→retrieve→generate | Agentic search: plan → retrieve → self-evaluate → refine (loop) |
| One LLM, hardcoded | Multi-provider: Claude, DeepSeek, Qwen, GLM-5 — swap via config |
| No evaluation | Eval-first: golden dataset, 6 metrics, regression tracking, feedback loops |
| No benchmarking | Retrieval accuracy (Recall@k, MRR), chunk size comparison, vector store comparison |
| No access control | API key auth, document-level ACL, audit logging |

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
      │ (pluggable)  │    ┌──────────────────────────────────────┐
      ├──────────────┤    │           Evaluation Engine           │
      │ • pgvector   │    │  golden dataset → run → score → store│
      │ • Chroma     │    │  feedback loop → regression tracking  │
      └──────────────┘    └──────────────────────────────────────┘
```

## Key Features

### Agentic Search (not naive RAG)

The search agent is an autonomous loop, not a rigid pipeline:

1. **Plan** — Analyse query, decompose complex questions into sub-queries
2. **Retrieve** — Vector similarity search
3. **Self-evaluate** — LLM judges: "Do these chunks answer the query? Is anything missing?"
4. **Decide** — Sufficient → generate answer. Insufficient → refine query and re-search (max 3 iterations)

Every response includes a **search trace** showing the reasoning steps.

### Multi-Capability Agents

| Capability | Example Query |
|-----------|---------------|
| **Q&A** | "What was total revenue in Q3 2024?" |
| **Summarise** | "Summarise the risk factors section" |
| **Compare** | "Compare revenue growth between AAPL and MSFT 10-Ks" |
| **Extract** | "Extract all quarterly revenue figures as JSON" |

### Multi-Provider LLM (6+ providers)

Switch between LLM providers with a single `.env` change:

| Provider | Model | Cost (input/1M tokens) | Best For |
|----------|-------|----------------------|----------|
| Anthropic | Claude Sonnet 4.6 | $3.00 | Best quality, demos |
| DeepSeek | V3 | $0.14 | Development (20x cheaper) |
| DeepSeek | R1 | $0.55 | Complex reasoning |
| Alibaba | Qwen 3.5-Plus | $0.11 | Cheapest flagship |
| Zhipu AI | GLM-5 | $1.00 | MIT licensed, self-hostable |
| MiniMax | M2.5 | $0.20 | 3rd globally on SWE-bench |

### Retrieval Benchmarking

- **Needle in haystack**: Find the correct chunk among 1000+, measured with Recall@5, Recall@10, MRR
- **Chunk size comparison**: Empirically benchmark 256/512/1024 token chunks
- **Vector store comparison**: pgvector vs Chroma on the same dataset

### Evaluation & Feedback Loops

- 6 metrics including custom **numerical accuracy** for finance
- Regression tracking across eval runs
- Failure analysis with full search traces
- Cross-provider evaluation
- CI/CD quality gates via `pytest`

### Authorisation

- API key authentication
- Document-level access control (multi-tenant)
- Audit logging (who queried what, when)
- Rate limiting via Redis

## Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| API Framework | FastAPI + Pydantic V2 | Async-native, auto-generated OpenAPI docs, type-safe validation |
| Multi-Agent | LangGraph | Graph-based state machines for debuggable, testable agent orchestration |
| Vector Storage | pgvector + Chroma (benchmarked) | Both behind pluggable interface; benchmark results drive recommendation |
| Embeddings | OpenAI `text-embedding-3-small` | High quality, no GPU required, 1536 dimensions |
| LLM | Multi-provider | Provider-agnostic: Claude, DeepSeek, Qwen, GLM-5, MiniMax, Kimi |
| PDF Parsing | Docling (IBM) | Purpose-built for structured financial documents with complex tables |
| Task Queue | Celery 5.6 + Redis | Reliable background processing with monitoring (Flower) |
| Evaluation | DeepEval + custom metrics | pytest-style RAG evaluation with CI/CD integration |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- API keys: [Anthropic](https://console.anthropic.com/settings/keys) + [OpenAI](https://platform.openai.com/api-keys) (or any supported provider)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/soudazhou/fin-doc-agent.git
cd fin-doc-agent

# 2. Copy environment template and add your API keys
cp .env.example .env
# Edit .env — set API keys and choose your LLM provider

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
| `POST` | `/ask` | Agentic search + answer (auto-routes to capability) |
| `POST` | `/compare` | A/B test: same query across multiple LLM providers |
| `POST` | `/benchmark/retrieval` | Benchmark retrieval accuracy (chunk sizes, vector stores) |
| `GET` | `/metrics` | Aggregated performance metrics by provider |
| `POST` | `/evaluate` | Run evaluation suite against golden dataset |
| `GET` | `/evaluate/history` | Evaluation score trends over time |
| `GET` | `/evaluate/failures` | Detailed failure analysis for debugging |
| `POST` | `/admin/keys` | Manage API keys and access control |

### Example Usage

```bash
# Ingest a document
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/samples/earnings_report.pdf"

# Ask a question (agentic search)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total revenue in Q3?", "document_id": 1}'

# Compare providers side-by-side
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total revenue in Q3?", "document_id": 1,
       "providers": ["anthropic/claude-sonnet-4-6", "deepseek/deepseek-chat"]}'

# Run evaluation
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"document_id": 1}'

# Benchmark retrieval accuracy
curl -X POST http://localhost:8000/benchmark/retrieval \
  -H "Content-Type: application/json" \
  -d '{"document_id": 1, "chunk_sizes": [256, 512, 1024]}'
```

## Project Structure

```
fin-doc-agent/
├── app/
│   ├── main.py                 # FastAPI entry point & lifespan
│   ├── config.py               # Pydantic BaseSettings (env vars, multi-provider)
│   ├── api/                    # Route handlers
│   │   ├── ingest.py           # Document upload & status
│   │   ├── ask.py              # Agentic search + answer
│   │   ├── compare.py          # A/B provider comparison
│   │   ├── evaluate.py         # Evaluation suite
│   │   ├── metrics.py          # Performance metrics
│   │   └── admin.py            # API key management
│   ├── agents/                 # LangGraph multi-agent system
│   │   ├── orchestrator.py     # Intent classification & routing
│   │   ├── search.py           # Agentic search loop
│   │   └── analyst.py          # LLM reasoning (multi-capability)
│   ├── db/                     # Database layer
│   │   ├── engine.py           # Async SQLAlchemy engine
│   │   └── models.py           # ORM models (Document, Chunk, QueryMetric, EvalResult, ApiKey)
│   ├── models/                 # Pydantic V2 schemas
│   │   ├── requests.py         # API request validation
│   │   └── responses.py        # API response serialization
│   ├── services/               # Business logic
│   │   ├── llm.py              # Multi-provider LLM abstraction
│   │   ├── vectorstore.py      # Pluggable vector store (pgvector, Chroma)
│   │   ├── parser.py           # PDF parsing (Docling)
│   │   ├── chunker.py          # Configurable token-based chunking
│   │   ├── embedder.py         # Embedding generation (OpenAI)
│   │   ├── evaluator.py        # RAG evaluation (DeepEval + custom)
│   │   ├── metrics.py          # Performance metrics collection
│   │   └── pricing.py          # Provider pricing registry
│   └── workers/                # Background processing
│       ├── celery_app.py       # Celery configuration
│       └── tasks.py            # Ingestion pipeline task
├── tests/
│   ├── test_chunker.py         # Unit tests for token-based chunking
│   ├── test_vectorstore.py     # Unit tests for ChromaDB vector store
│   └── eval/                   # Evaluation test suite & golden dataset
├── data/
│   └── samples/                # Sample financial PDFs
├── docs/
│   ├── PLAN.md                 # Implementation plan & design decisions
│   └── BENCHMARKS.md           # Retrieval accuracy & provider comparison results
├── docker-compose.yml          # Full local environment
├── Dockerfile                  # Python 3.12 app image
├── pyproject.toml              # Dependencies & tool config
└── .env.example                # Environment variable template (6 provider presets)
```

## Design Decisions

Key architectural decisions are documented in two places:

1. **Inline code comments** — Every file contains detailed comments explaining *why* each choice was made, not just *what* the code does.
2. **[docs/PLAN.md](docs/PLAN.md)** — The full implementation plan with architecture rationale, technology comparisons, LLM landscape analysis, and phase breakdown.

## Implementation Roadmap

- [x] **Phase 1**: Project scaffolding — FastAPI, Docker, DB models, config
- [x] **Phase 2**: Document ingestion — PDF parsing, configurable chunking, pluggable vector store
- [ ] **Phase 3**: Agentic search & multi-capability agents — autonomous search loop, 4 capabilities
- [ ] **Phase 4**: A/B comparison & benchmarking — provider comparison, retrieval accuracy, chunk size optimisation
- [ ] **Phase 5**: Evaluation & feedback loops — golden dataset, 6 metrics, regression tracking, failure analysis
- [ ] **Phase 6**: Authorisation — API keys, document ACL, audit logging, rate limiting
- [ ] **Phase 7**: Polish — sample data, demo script, benchmark documentation

## Running Tests

```bash
# Unit tests
uv run pytest tests/

# RAG evaluation suite
uv run pytest tests/eval/ -v
```

## Acknowledgements

Architecture and scope refined with expert feedback from **Guanyi Li**.

## License

MIT
