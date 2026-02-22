# =============================================================================
# Ingestion API — Document Upload and Status Tracking
# =============================================================================
#
# Provides endpoints for uploading financial documents (PDFs) and polling
# the status of asynchronous ingestion tasks.
#
# ENDPOINTS:
#   POST /ingest        — Upload PDF, dispatch Celery task, return task_id
#   GET  /ingest/{task_id} — Poll ingestion status (PENDING → SUCCESS/FAILURE)
#
# DESIGN DECISION: Async processing via Celery (not synchronous).
# Document ingestion involves multiple slow steps:
#   1. PDF parsing (CPU-bound, 5-30 seconds for large docs)
#   2. Embedding generation (network-bound, OpenAI API calls)
#   3. Database inserts (I/O-bound, bulk vector inserts)
#
# Running these synchronously would block the API server and cause
# request timeouts. Celery processes them in the background, returning
# a task_id immediately so the client can poll for status.
#
# DESIGN DECISION: 202 Accepted (not 200 OK) for POST /ingest.
# HTTP 202 signals "your request was accepted for processing, but
# the processing hasn't completed yet." This is the correct status
# code for async operations per RFC 7231.
# =============================================================================

import logging
from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import check_scope, get_current_api_key
from app.config import settings
from app.db.engine import get_async_session
from app.db.models import ApiKey, Document, DocumentStatus
from app.models.responses import DocumentResponse, IngestResponse, IngestStatusResponse
from app.workers.tasks import ingest_document

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion"])


# ---------------------------------------------------------------------------
# POST /ingest — Upload a financial document
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=202,
    summary="Upload a financial document for processing",
    description=(
        "Upload a PDF file to be parsed, chunked, embedded, and stored. "
        "Returns immediately with a task_id for polling. The document is "
        "NOT available for querying until processing completes."
    ),
)
async def ingest_document_endpoint(
    http_request: Request,
    file: UploadFile = File(
        ...,
        description="PDF file to ingest (financial report, 10-K, earnings, etc.)",
    ),
    chunk_size: int | None = Query(
        default=None,
        ge=64,
        le=4096,
        description=(
            "Override chunk size in tokens. Defaults to config value (512). "
            "Use 256/512/1024 to benchmark different chunk sizes."
        ),
    ),
    chunk_overlap: int | None = Query(
        default=None,
        ge=0,
        le=512,
        description=(
            "Override chunk overlap in tokens. Defaults to config value (50)."
        ),
    ),
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> IngestResponse:
    """
    Upload a PDF for ingestion into the document Q&A system.

    The file is saved to disk, a Document record is created in the database,
    and a Celery task is dispatched for async processing. The response
    includes a task_id for polling the processing status.
    """
    # Auth: scope check (no document ACL — creating a new doc)
    check_scope(api_key, "ingest")

    # --- Validate file type ---
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    # --- Ensure upload directory exists ---
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # --- Read and validate file content ---
    # FastAPI's UploadFile wraps a SpooledTemporaryFile. We read the entire
    # content into memory and then save to our persistent upload directory.
    # For very large files (>100MB), streaming to disk would be better,
    # but financial PDFs rarely exceed 50MB.
    file_content = await file.read()
    file_size = len(file_content)

    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --- Create Document record in database ---
    doc = Document(
        filename=file.filename,
        file_size=file_size,
        status=DocumentStatus.PENDING,
    )
    session.add(doc)
    await session.flush()  # Assigns doc.id without committing

    # --- Save file to disk ---
    # Prefix with document ID to avoid filename collisions.
    # Two users uploading "report.pdf" will get "1_report.pdf" and "2_report.pdf".
    safe_filename = f"{doc.id}_{file.filename}"
    file_path = upload_dir / safe_filename
    file_path.write_bytes(file_content)

    logger.info(
        "Saved upload: %s (%d bytes) → %s",
        file.filename, file_size, file_path,
    )

    # --- Dispatch Celery task ---
    task = ingest_document.delay(
        document_id=doc.id,
        file_path=str(file_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Store Celery task ID on the document record for cross-referencing
    doc.celery_task_id = task.id

    # The session commits automatically via get_async_session dependency
    logger.info(
        "Dispatched ingestion task: document_id=%d, task_id=%s",
        doc.id, task.id,
    )

    return IngestResponse(
        document_id=doc.id,
        task_id=task.id,
        status="processing",
        message=f"Document '{file.filename}' uploaded. Ingestion in progress.",
    )


# ---------------------------------------------------------------------------
# GET /ingest/{task_id} — Poll ingestion status
# ---------------------------------------------------------------------------


@router.get(
    "/ingest/{task_id}",
    response_model=IngestStatusResponse,
    summary="Check document ingestion status",
    description=(
        "Poll this endpoint to check if a document has finished processing. "
        "Returns the Celery task status and document details on success."
    ),
)
async def get_ingest_status(
    task_id: str,
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> IngestStatusResponse:
    """
    Check the status of a document ingestion task.

    Returns the Celery task status and, if completed, the Document details.
    Clients should poll this endpoint until status is SUCCESS or FAILURE.

    Celery task states:
    - PENDING: Task not yet picked up by a worker
    - STARTED: Worker has begun processing
    - SUCCESS: Ingestion completed successfully
    - FAILURE: Ingestion failed (check error field)
    - RETRY: Task is being retried after a failure
    """
    check_scope(api_key, "ingest")

    # --- Check Celery task status ---
    result = AsyncResult(task_id, app=ingest_document.app)
    status = result.status

    # --- Build response based on task state ---
    document_response: DocumentResponse | None = None
    error: str | None = None

    if status == "SUCCESS":
        # Task result contains the summary dict from ingest_document()
        task_result = result.result or {}
        doc_id = task_result.get("document_id")
        if doc_id:
            doc = await session.get(Document, doc_id)
            if doc:
                document_response = DocumentResponse.model_validate(doc)

    elif status == "FAILURE":
        # Extract error message from the failed task result
        error = str(result.result) if result.result else "Unknown error"

    return IngestStatusResponse(
        task_id=task_id,
        status=status,
        document=document_response,
        error=error,
    )
