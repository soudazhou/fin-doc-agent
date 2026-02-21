# =============================================================================
# Financial Document Q&A Agent
# =============================================================================
# A multi-agent RAG system for answering questions over financial documents.
#
# Package structure:
#   app/
#   ├── api/          → FastAPI route handlers (ingest, ask, evaluate)
#   ├── agents/       → LangGraph multi-agent orchestration
#   ├── db/           → Database engine, session, and ORM models
#   ├── models/       → Pydantic V2 request/response schemas
#   ├── services/     → Business logic (parsing, chunking, embedding, eval)
#   └── workers/      → Celery task definitions and configuration
# =============================================================================
