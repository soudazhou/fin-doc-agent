# =============================================================================
# Database Models — SQLAlchemy ORM
# =============================================================================
#
# These models define the database schema for storing financial documents
# and their vector embeddings. They map directly to PostgreSQL tables.
#
# SCHEMA OVERVIEW:
#
# ┌──────────────┐       ┌──────────────────────────────────┐
# │  documents   │       │  chunks                          │
# ├──────────────┤       ├──────────────────────────────────┤
# │ id (PK)      │──1:N─▶│ id (PK)                          │
# │ filename     │       │ document_id (FK → documents.id)  │
# │ file_size    │       │ content (text)                   │
# │ page_count   │       │ page_number (int)                │
# │ status       │       │ chunk_index (int)                │
# │ created_at   │       │ token_count (int)                │
# │ updated_at   │       │ embedding (vector(1536))         │
# └──────────────┘       │ metadata_ (jsonb)                │
#                        │ created_at                       │
#                        └──────────────────────────────────┘
#
# DESIGN DECISIONS:
#
# 1. One-to-many relationship: A document has many chunks.
#    This is the standard RAG pattern — split documents into chunks,
#    embed each chunk, and retrieve the most relevant ones at query time.
#
# 2. pgvector `Vector(1536)` column: Stores OpenAI text-embedding-3-small
#    embeddings directly in PostgreSQL. The HNSW index on this column
#    enables fast approximate nearest neighbor search.
#
# 3. JSONB `metadata_` column: Flexible storage for chunk metadata
#    (section headers, table indicators, etc.) without schema changes.
#    The trailing underscore avoids conflict with SQLAlchemy's `.metadata`.
#
# 4. `status` field on documents: Tracks ingestion pipeline progress
#    (pending → processing → completed → failed). This allows the API
#    to report ingestion status to clients.
# =============================================================================

import enum
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings


class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base class.

    All ORM models inherit from this. It provides:
    - Automatic table name generation (optional, we set __tablename__ explicitly)
    - Metadata for schema introspection and migration (Alembic)
    - Type mapping configuration
    """

    pass


class DocumentStatus(str, enum.Enum):
    """
    Tracks the ingestion pipeline state for a document.

    State machine:
        PENDING → PROCESSING → COMPLETED
                             → FAILED

    DESIGN DECISION: Using a string enum (not integer) because:
    1. Human-readable in database queries and API responses
    2. Self-documenting — no need to look up what "status=2" means
    3. Safe to add new states without breaking existing data
    """

    PENDING = "pending"          # Uploaded, waiting for Celery to pick up
    PROCESSING = "processing"    # Celery worker is parsing/chunking/embedding
    COMPLETED = "completed"      # All chunks embedded and stored
    FAILED = "failed"            # Something went wrong (see error_message)


class Document(Base):
    """
    Represents an uploaded financial document (PDF).

    This is the parent entity. Each document is split into multiple chunks
    during the ingestion pipeline. The `status` field tracks pipeline progress
    so the API can report whether a document is ready for querying.
    """

    __tablename__ = "documents"

    # Primary key: UUID-style would be better for distributed systems,
    # but auto-increment integer is simpler for a demo project.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Original filename as uploaded by the user (e.g., "AAPL_10K_2024.pdf")
    filename: Mapped[str] = mapped_column(String(500), nullable=False)

    # File size in bytes — useful for validation and display
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)

    # Number of pages extracted from the PDF
    page_count: Mapped[int] = mapped_column(Integer, nullable=True)

    # Ingestion pipeline status (see DocumentStatus enum above)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus),
        nullable=False,
        default=DocumentStatus.PENDING,
    )

    # Error message if ingestion failed (null when status != FAILED)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Celery task ID for tracking the ingestion job
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ---------------------------------------------------------------------------
    # Relationship: Document → Chunks (one-to-many)
    # ---------------------------------------------------------------------------
    # cascade="all, delete-orphan": When a document is deleted, all its chunks
    # are also deleted. This prevents orphaned chunks in the database.
    #
    # lazy="selectin": Eagerly loads chunks in a single query when accessing
    # document.chunks. Alternative is "lazy" (default) which issues N+1 queries.
    # For this project, we rarely need all chunks, so this is mainly for
    # convenience during development.
    # ---------------------------------------------------------------------------
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', status={self.status})>"


class Chunk(Base):
    """
    Represents a chunk of text extracted from a financial document,
    along with its vector embedding for similarity search.

    This is the core entity for RAG:
    1. During ingestion: documents are split into chunks, each is embedded
    2. During retrieval: the user's query is embedded and compared against
       chunk embeddings using cosine similarity in pgvector
    3. During generation: the most similar chunks are passed as context
       to the Analyst agent (Claude) for answer synthesis
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to the parent document
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    # The actual text content of this chunk
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Which page of the PDF this chunk came from (1-indexed)
    # Used for source citation in answers ("Answer found on page 42")
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Sequential index of this chunk within the document (0-indexed)
    # Useful for reconstructing document order and showing context
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Number of tokens in this chunk (from tiktoken)
    # Stored for monitoring and debugging chunk size distribution
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # ---------------------------------------------------------------------------
    # Vector Embedding — THE KEY COLUMN FOR RAG
    # ---------------------------------------------------------------------------
    # This column stores the 1536-dimensional embedding vector generated by
    # OpenAI's text-embedding-3-small model.
    #
    # HOW VECTOR SEARCH WORKS:
    # 1. User asks a question → question is embedded into a 1536-dim vector
    # 2. pgvector finds chunks whose embeddings are most similar (cosine distance)
    # 3. Top-k chunks are returned as context for the LLM
    #
    # The Vector(1536) type comes from the pgvector SQLAlchemy integration.
    # Under the hood, it's stored as a PostgreSQL `vector(1536)` column.
    # ---------------------------------------------------------------------------
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimensions),
        nullable=True,  # Null during initial chunk creation, filled by embedder
    )

    # ---------------------------------------------------------------------------
    # Flexible Metadata — JSONB Column
    # ---------------------------------------------------------------------------
    # Stores additional chunk metadata without requiring schema changes:
    # - section_title: Header of the section this chunk belongs to
    # - is_table: Whether this chunk contains tabular data
    # - source_type: "text", "table", "header", etc.
    #
    # DESIGN DECISION: JSONB over separate columns because:
    # 1. Metadata varies by document type (10-K vs earnings report)
    # 2. We can add new metadata fields without database migrations
    # 3. PostgreSQL JSONB supports indexing and querying
    #
    # Named `metadata_` (with underscore) to avoid collision with
    # SQLAlchemy's built-in `.metadata` attribute on declarative base.
    # ---------------------------------------------------------------------------
    metadata_: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
    )

    # Timestamp for when this chunk was created
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship back to parent document
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"<Chunk(id={self.id}, doc_id={self.document_id}, "
            f"index={self.chunk_index}, tokens={self.token_count})>"
        )


# =============================================================================
# Database Indexes
# =============================================================================
#
# HNSW Index on chunk embeddings:
# HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor
# algorithm. It trades a tiny amount of accuracy for massive speed gains:
# - Exact search: O(n) — scans every vector
# - HNSW search: O(log n) — navigates a graph structure
#
# For <10K chunks (typical for a demo), exact search is fast enough.
# The HNSW index becomes essential at scale (100K+ vectors).
# We include it here to demonstrate production-readiness.
#
# `vector_cosine_ops`: Tells pgvector to use cosine similarity as the
# distance metric. Cosine similarity is standard for text embeddings
# because it measures directional similarity, ignoring magnitude.
# =============================================================================

# HNSW index for fast vector similarity search
chunk_embedding_idx = Index(
    "idx_chunk_embedding_hnsw",
    Chunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)

# B-tree index for fast document lookup by document_id
chunk_document_idx = Index(
    "idx_chunk_document_id",
    Chunk.document_id,
)


# =============================================================================
# Query Metrics — Per-Query Performance Telemetry (Phase 4)
# =============================================================================
#
# Every call to POST /ask (and each provider leg of POST /compare)
# writes one row. This enables:
# - Latency tracking over time
# - Cost accounting per provider
# - Retrieval quality monitoring (score distributions)
# - Agentic search efficiency (iterations required)
#
# DESIGN DECISION: Stored in PostgreSQL (not a time-series DB) because:
# 1. No extra infra — PostgreSQL is already in the stack
# 2. Volume is low (queries/minute, not queries/second)
# 3. We can JOIN against documents and chunks for richer analysis
# 4. Phase 5 will JOIN eval results against query metrics
#
# DESIGN DECISION: Nullable columns for fields that only exist in
# certain contexts. Regular /ask calls don't carry a provider_id label
# (they use whatever the singleton resolves to).
# =============================================================================


class QueryMetric(Base):
    """Per-query performance and cost telemetry."""

    __tablename__ = "query_metrics"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )

    # Which document was queried (null = all documents searched)
    document_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )

    # The question asked
    question: Mapped[str] = mapped_column(Text, nullable=False)

    # Capability selected (qa, summarise, compare, extract)
    capability: Mapped[str | None] = mapped_column(
        String(50), nullable=True,
    )

    # Provider identifier in "type/model" format
    # e.g. "anthropic/claude-sonnet-4-6"
    # Null for /ask calls using the global singleton
    provider_id: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
    )

    # Actual model name returned by the LLM API
    model: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
    )

    # --- Latency (milliseconds) ---
    total_latency_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )

    # --- Token counts ---
    input_tokens: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )
    output_tokens: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )

    # --- Cost (USD, estimated from pricing registry) ---
    estimated_cost_usd: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )

    # --- Agentic search stats ---
    search_iterations: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )
    retrieval_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )

    # Similarity scores (JSONB list of floats, one per retrieved chunk)
    # DESIGN DECISION: JSONB rather than a separate table — we only
    # ever read these as a batch for percentile stats.
    retrieval_scores: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
    )

    # Source: "ask" (regular /ask call) or "compare" (/compare leg)
    source: Mapped[str] = mapped_column(
        String(50), nullable=False, default="ask",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<QueryMetric(id={self.id}, provider={self.provider_id}, "
            f"latency={self.total_latency_ms}ms)>"
        )


# Supports /metrics queries: filter by provider, sort by time
query_metric_provider_idx = Index(
    "idx_query_metric_provider_created",
    QueryMetric.provider_id,
    QueryMetric.created_at,
)


# =============================================================================
# Evaluation Results — Eval Framework (Phase 5)
# =============================================================================
#
# Two tables capture evaluation data at different granularities:
#
# eval_runs: One row per POST /evaluate invocation. Stores aggregate
#   scores, pass/fail counts, and run metadata. Used by /evaluate/history
#   for trend analysis without loading per-test-case details.
#
# eval_test_results: One row per golden dataset test case per run.
#   Stores the full input/output, per-metric scores with reasons, and
#   the agentic search trace. Used by /evaluate/failures for debugging.
#
# DESIGN DECISION: Two tables rather than one because:
# 1. History endpoint needs run-level summaries efficiently (no N+1)
# 2. Failures endpoint needs per-test-case detail with search traces
# 3. JSONB on metric_results keeps the schema flexible for new metrics
# =============================================================================


class EvalRun(Base):
    """
    A single evaluation run — one invocation of POST /evaluate.

    Tracks aggregate scores and status for the entire run.
    Each run produces N EvalTestResult rows (one per golden dataset entry).
    """

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )

    # Which document was evaluated
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Which golden dataset was used (maps to filename in data/eval/)
    eval_dataset: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )

    # Provider used for this run (null = default singleton)
    provider_id: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
    )

    # Actual model name returned by the LLM API
    model: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
    )

    # Run lifecycle status
    # "running" → "completed" | "failed"
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="running",
    )

    # Aggregate metric scores: {metric_name: avg_score}
    # JSONB because the set of metrics may expand over time
    metric_scores: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
    )

    # Average across all metric scores (convenience for sorting/filtering)
    overall_score: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )

    total_test_cases: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )
    passed: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )
    failed: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )

    # Snapshot of thresholds and config used for this run
    run_config: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
    )

    # Total evaluation duration in milliseconds
    duration_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )

    # Error message if status="failed"
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship to individual test case results
    test_results: Mapped[list["EvalTestResult"]] = relationship(
        "EvalTestResult",
        back_populates="eval_run",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<EvalRun(id={self.id}, dataset='{self.eval_dataset}', "
            f"status='{self.status}', score={self.overall_score})>"
        )


class EvalTestResult(Base):
    """
    Result of evaluating a single golden dataset test case.

    Stores the full input/output, per-metric scores with LLM judge
    reasoning, and the agentic search trace for failure debugging.
    """

    __tablename__ = "eval_test_results"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )

    # Parent evaluation run
    eval_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    # ID from the golden dataset (e.g., "qa_revenue_01")
    test_case_id: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )

    # Input/output data
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    actual_answer: Mapped[str] = mapped_column(Text, nullable=False)

    # Retrieval context — list of chunk content strings
    # Used by DeepEval metrics as the retrieval_context parameter
    retrieval_context: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
    )

    # Per-metric scores: {metric_name: {"score": float, "reason": str|null}}
    # JSONB keeps this flexible — no schema change when adding metrics
    metric_results: Mapped[dict] = mapped_column(
        JSONB, nullable=False,
    )

    # Pass/fail based on threshold comparison
    passed: Mapped[bool] = mapped_column(nullable=False)

    # Full agentic search trace for debugging failures
    # Contains query rewrites, retrieval scores per iteration, evaluation reasons
    search_trace: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
    )

    # Retrieved source chunks
    # [{chunk_id, page_number, similarity_score, content_preview}]
    sources: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship back to parent run
    eval_run: Mapped["EvalRun"] = relationship(
        "EvalRun", back_populates="test_results",
    )

    def __repr__(self) -> str:
        return (
            f"<EvalTestResult(id={self.id}, case='{self.test_case_id}', "
            f"passed={self.passed})>"
        )


# Indexes for evaluation queries
eval_run_document_idx = Index(
    "idx_eval_run_document_created",
    EvalRun.document_id,
    EvalRun.created_at,
)

eval_run_provider_idx = Index(
    "idx_eval_run_provider",
    EvalRun.provider_id,
)

eval_test_result_run_idx = Index(
    "idx_eval_test_result_run",
    EvalTestResult.eval_run_id,
)

eval_test_result_passed_idx = Index(
    "idx_eval_test_result_passed",
    EvalTestResult.eval_run_id,
    EvalTestResult.passed,
)


# =============================================================================
# Authorization & Security (Phase 6)
# =============================================================================
#
# Three tables support the auth layer:
#
# api_keys: Bearer token credentials with scopes, rate limits,
#   and soft-disable. Key material is SHA-256 hashed (never plaintext).
#
# api_key_documents: Many-to-many association for document-level ACL.
#   Only consulted when ApiKey.all_documents_access is False.
#
# audit_logs: Immutable compliance trail recording every authenticated
#   API request (who, what, when, from where).
#
# DESIGN DECISION: SHA-256 for key hashing (not bcrypt). API keys are
# 32-byte random tokens — high entropy makes rainbow tables infeasible.
# SHA-256 is fast enough for per-request validation without the overhead
# of bcrypt's deliberate slowness (designed for low-entropy passwords).
#
# DESIGN DECISION: Separate api_key_documents table (not JSONB array)
# for document ACL. Enables SQL JOINs, FK integrity, and efficient
# reverse lookups ("which keys can access document X?").
# =============================================================================


class ApiKey(Base):
    """
    An API key for authenticating requests.

    Each key has a hashed secret (SHA-256), a human-readable prefix
    for log identification, optional scopes, and optional document ACL.
    The raw key is only returned once at creation time.
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )

    # Human-readable label (e.g., "frontend-app", "data-team")
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    # First 8 chars of the key for identification in logs
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False)

    # SHA-256 hash of the full key — never store plaintext
    key_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True,
    )

    # Allowed scopes as JSONB array: ["ask", "ingest", ...]
    # Null or empty = full access (all scopes)
    scopes: Mapped[list | None] = mapped_column(
        JSONB, nullable=True, default=list,
    )

    # Per-key rate limit override (requests per minute)
    # Null = use settings.rate_limit_rpm default
    rate_limit_rpm: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )

    # If True, this key can access ALL documents (bypass ACL)
    all_documents_access: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False,
    )

    # Soft-disable without deletion
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False,
    )

    # Expiration (null = never expires)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Updated on each authenticated request
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Document ACL relationship
    allowed_documents: Mapped[list["ApiKeyDocument"]] = relationship(
        "ApiKeyDocument",
        back_populates="api_key",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<ApiKey(id={self.id}, name='{self.name}', "
            f"prefix='{self.key_prefix}', active={self.is_active})>"
        )


class ApiKeyDocument(Base):
    """
    Many-to-many association: which API keys can access which documents.

    Only consulted when the parent ApiKey has all_documents_access=False.
    """

    __tablename__ = "api_key_documents"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )
    api_key_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    api_key: Mapped["ApiKey"] = relationship(
        "ApiKey", back_populates="allowed_documents",
    )


class AuditLog(Base):
    """
    Immutable audit trail for all authenticated API requests.

    Records who accessed what, when, and what they asked.
    Required for financial compliance — queryable in PostgreSQL.
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )

    # Which API key made the request (null if auth disabled)
    api_key_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("api_keys.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Key name snapshot — preserved even if key is later deleted
    api_key_name: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
    )

    # Request details
    endpoint: Mapped[str] = mapped_column(String(200), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)

    # Document accessed (extracted from request body/path)
    document_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )

    # Question asked (for /ask and /compare endpoints)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Client IP address (IPv6-safe: max 45 chars)
    client_ip: Mapped[str | None] = mapped_column(
        String(45), nullable=True,
    )

    # HTTP response status code
    status_code: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )

    # Response time in milliseconds
    response_time_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


# Indexes for auth tables
api_key_hash_idx = Index(
    "idx_api_key_hash",
    ApiKey.key_hash,
    unique=True,
)

api_key_prefix_idx = Index(
    "idx_api_key_prefix",
    ApiKey.key_prefix,
)

api_key_document_unique_idx = Index(
    "idx_api_key_document_unique",
    ApiKeyDocument.api_key_id,
    ApiKeyDocument.document_id,
    unique=True,
)

audit_log_api_key_idx = Index(
    "idx_audit_log_api_key_created",
    AuditLog.api_key_id,
    AuditLog.created_at,
)

audit_log_document_idx = Index(
    "idx_audit_log_document_created",
    AuditLog.document_id,
    AuditLog.created_at,
)
