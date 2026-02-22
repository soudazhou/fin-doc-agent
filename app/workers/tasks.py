# =============================================================================
# Celery Task Definitions — Document Ingestion Pipeline
# =============================================================================
#
# This module defines the background tasks that Celery workers execute.
# The main task is `ingest_document` — the full pipeline for processing
# an uploaded PDF into searchable vector embeddings.
#
# INGESTION PIPELINE:
#   1. Update document status → PROCESSING
#   2. Parse PDF with Docling → extract text, tables, headings (with pages)
#   3. Chunk text with tiktoken → configurable token-based chunks
#   4. Generate embeddings with OpenAI → batch API calls
#   5. Store chunks + embeddings in vector store (pgvector or Chroma)
#   6. Update document status → COMPLETED (or FAILED on error)
#
# IMPORTANT: Celery workers are SYNCHRONOUS.
# - Do NOT use `async/await` in Celery tasks
# - Do NOT use the async SQLAlchemy engine (use sync engine instead)
# - Do NOT call FastAPI dependencies directly
# This is a common gotcha when combining FastAPI (async) with Celery (sync).
#
# RETRY STRATEGY:
# max_retries=3 with exponential backoff (60s, 120s, 240s).
# Handles transient errors: OpenAI rate limits, DB connection drops.
# Non-transient errors (corrupt PDF, empty doc) exhaust retries and stay FAILED.
# =============================================================================

import logging

from sqlalchemy import update

from app.config import settings
from app.db.engine import get_sync_session
from app.db.models import Document, DocumentStatus
from app.services.chunker import chunk_document
from app.services.embedder import embed_batch
from app.services.parser import parse_pdf
from app.services.vectorstore import get_vector_store
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _update_document_status(
    document_id: int,
    status: DocumentStatus,
    error_message: str | None = None,
    page_count: int | None = None,
) -> None:
    """
    Update document status in the database (sync).

    Extracted as a helper because it's called at multiple pipeline stages:
    - Start (PROCESSING)
    - Success (COMPLETED)
    - Failure (FAILED)

    Uses its own session to ensure status is committed immediately,
    even if the main pipeline transaction fails.
    """
    with get_sync_session() as session:
        values: dict = {"status": status}
        if error_message is not None:
            values["error_message"] = error_message
        if page_count is not None:
            values["page_count"] = page_count

        session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(**values)
        )
        session.commit()


# ---------------------------------------------------------------------------
# Ingestion Task
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="ingest_document",
    # Retry up to 3 times on failure, with exponential backoff.
    # This handles transient errors (OpenAI API rate limits, DB connection drops).
    max_retries=3,
    # Wait 60 seconds before first retry, then 120s, then 240s.
    default_retry_delay=60,
)
def ingest_document(
    self,
    document_id: int,
    file_path: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """
    Process an uploaded PDF through the full ingestion pipeline.

    This task runs asynchronously in a Celery worker. The API returns
    the task ID immediately, and the client polls for completion.

    Args:
        self: Celery task instance (bound task, provides self.request.id).
        document_id: Database ID of the Document record to update.
        file_path: Path to the uploaded PDF file on disk.
        chunk_size: Override chunk size in tokens (default: settings.chunk_size).
            Configurable per-ingestion for benchmarking different sizes.
        chunk_overlap: Override chunk overlap in tokens (default: settings.chunk_overlap).

    Returns:
        dict with processing summary (chunk count, page count, vectorstore, etc.)
    """
    task_id = self.request.id
    _chunk_size = chunk_size or settings.chunk_size
    _chunk_overlap = chunk_overlap or settings.chunk_overlap

    logger.info(
        "Starting ingestion: document_id=%d, file=%s, task_id=%s, "
        "chunk_size=%d, chunk_overlap=%d, vectorstore=%s",
        document_id, file_path, task_id,
        _chunk_size, _chunk_overlap, settings.vectorstore_type,
    )

    try:
        # --- Step 1: Mark as PROCESSING ---
        _update_document_status(document_id, DocumentStatus.PROCESSING)

        # --- Step 2: Parse PDF with Docling ---
        logger.info("[%s] Step 2/5: Parsing PDF with Docling...", task_id)
        parsed_doc = parse_pdf(file_path)
        logger.info(
            "[%s] Parsed: %d elements, %d pages",
            task_id, len(parsed_doc.elements), parsed_doc.page_count,
        )

        # --- Step 3: Chunk text with tiktoken ---
        logger.info(
            "[%s] Step 3/5: Chunking text (size=%d, overlap=%d)...",
            task_id, _chunk_size, _chunk_overlap,
        )
        chunks = chunk_document(
            parsed_doc,
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
        )
        logger.info("[%s] Created %d chunks", task_id, len(chunks))

        if not chunks:
            raise ValueError(
                "No chunks produced from document — PDF may be empty or unreadable"
            )

        # --- Step 4: Generate embeddings with OpenAI ---
        logger.info(
            "[%s] Step 4/5: Generating embeddings for %d chunks (model=%s)...",
            task_id, len(chunks), settings.embedding_model,
        )
        texts = [c.content for c in chunks]
        embeddings = embed_batch(texts, batch_size=settings.embedding_batch_size)
        logger.info("[%s] Generated %d embeddings", task_id, len(embeddings))

        # --- Step 5: Store in vector store ---
        logger.info(
            "[%s] Step 5/5: Storing chunks in %s...",
            task_id, settings.vectorstore_type,
        )
        vector_store = get_vector_store()

        contents = [c.content for c in chunks]
        metadatas = [
            {
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                **c.metadata,
            }
            for c in chunks
        ]

        chunk_ids = vector_store.add_chunks(
            document_id=document_id,
            contents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # --- Step 6: Mark as COMPLETED ---
        _update_document_status(
            document_id,
            DocumentStatus.COMPLETED,
            page_count=parsed_doc.page_count,
        )

        summary = {
            "document_id": document_id,
            "status": "completed",
            "chunk_count": len(chunks),
            "page_count": parsed_doc.page_count,
            "chunk_size": _chunk_size,
            "chunk_overlap": _chunk_overlap,
            "vectorstore": settings.vectorstore_type,
        }
        logger.info("[%s] Ingestion complete: %s", task_id, summary)
        return summary

    except Exception as exc:
        logger.exception(
            "[%s] Ingestion failed for document_id=%d: %s",
            task_id, document_id, exc,
        )

        # Mark document as FAILED with the error message.
        # Truncate to 1000 chars to prevent DB column overflow.
        _update_document_status(
            document_id,
            DocumentStatus.FAILED,
            error_message=str(exc)[:1000],
        )

        # Re-raise with retry — Celery will re-queue with exponential backoff.
        # After max_retries exhausted, the task stays FAILED permanently.
        raise self.retry(exc=exc)
