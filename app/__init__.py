# =============================================================================
# Financial Document Q&A Agent
# =============================================================================
# A multi-agent RAG system for answering questions over financial documents.
# Uses agentic search (plan → retrieve → self-evaluate → refine) rather than
# naive single-shot RAG, with multi-capability agents routed by LangGraph.
#
# Package structure:
#   app/
#   ├── api/          → FastAPI route handlers (ingest, ask, evaluate, benchmark)
#   ├── agents/       → LangGraph multi-agent orchestration (Q&A, summarise,
#   │                    compare, extract capabilities)
#   ├── db/           → Database engine, session, and ORM models
#   ├── models/       → Pydantic V2 request/response schemas
#   ├── services/     → Business logic (parsing, chunking, embedding, eval,
#   │                    vector store abstraction)
#   └── workers/      → Celery task definitions and configuration
# =============================================================================
