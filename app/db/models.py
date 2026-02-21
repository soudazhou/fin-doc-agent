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
    DateTime,
    Enum,
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
