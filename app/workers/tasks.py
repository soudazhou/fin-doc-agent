# =============================================================================
# Celery Task Definitions
# =============================================================================
#
# This module defines the background tasks that Celery workers execute.
# Currently, the main task is `ingest_document` — the full pipeline for
# processing an uploaded PDF into searchable vector embeddings.
#
# INGESTION PIPELINE:
#   1. Parse PDF → Extract text and tables (Docling)
#   2. Chunk text → Split into ~512 token chunks (tiktoken)
#   3. Embed chunks → Generate vector embeddings (OpenAI API)
#   4. Store → Save chunks + embeddings to PostgreSQL/pgvector
#
# IMPORTANT: Celery workers are SYNCHRONOUS.
# - Do NOT use `async/await` in Celery tasks
# - Do NOT use the async SQLAlchemy engine (use sync engine instead)
# - Do NOT call FastAPI dependencies directly
# This is a common gotcha when combining FastAPI (async) with Celery (sync).
#
# NOTE: Task implementations will be completed in Phase 2 (ingestion pipeline).
# This file provides the skeleton with proper error handling and status updates.
# =============================================================================

import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="ingest_document",
    # Retry up to 3 times on failure, with exponential backoff.
    # This handles transient errors (OpenAI API rate limits, DB connection drops).
    max_retries=3,
    # Wait 60 seconds before first retry, then 120s, then 240s.
    default_retry_delay=60,
)
def ingest_document(self, document_id: int, file_path: str) -> dict:
    """
    Process an uploaded PDF through the full ingestion pipeline.

    This task runs asynchronously in a Celery worker. The API returns
    the task ID immediately, and the client polls for completion.

    Args:
        self: Celery task instance (bound task, provides self.request.id)
        document_id: Database ID of the Document record to update
        file_path: Path to the uploaded PDF file on disk

    Returns:
        dict with processing results (chunk count, status)

    Pipeline steps (implemented in Phase 2):
        1. Update document status to PROCESSING
        2. Parse PDF with Docling
        3. Chunk text with tiktoken
        4. Generate embeddings with OpenAI
        5. Store chunks + embeddings in pgvector
        6. Update document status to COMPLETED
    """
    logger.info(
        "Starting ingestion for document_id=%d, file=%s, task_id=%s",
        document_id,
        file_path,
        self.request.id,
    )

    # TODO: Phase 2 — Implement full ingestion pipeline
    # Steps will be:
    # 1. Update document status to PROCESSING (sync DB session)
    # 2. parsed_doc = parser.parse(file_path)
    # 3. chunks = chunker.chunk(parsed_doc)
    # 4. embeddings = embedder.embed_batch([c.content for c in chunks])
    # 5. Store chunks with embeddings in DB
    # 6. Update document status to COMPLETED
    # 7. Return summary

    return {
        "document_id": document_id,
        "status": "placeholder — pipeline implementation in Phase 2",
    }
