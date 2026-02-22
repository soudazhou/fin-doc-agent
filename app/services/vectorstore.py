# =============================================================================
# Vector Store Abstraction — Pluggable Backend Protocol
# =============================================================================
#
# Provides a common interface for vector similarity search, with concrete
# implementations for pgvector (PostgreSQL) and ChromaDB.
#
# DESIGN DECISION: Protocol (structural typing) over ABC (nominal typing).
# Protocol allows any class with the right methods to be used, without
# requiring explicit inheritance. This is more Pythonic and makes testing
# easier — you can create a mock that matches the protocol without
# inheriting from anything.
#
# DESIGN DECISION: Two implementations, benchmarked head-to-head.
# Most projects assume one vector store is "best." We implement both
# pgvector and Chroma, then benchmark on the same dataset with identical
# metrics (Recall@k, MRR, latency). This lets data drive the recommendation.
#
# DESIGN DECISION: Mixed sync/async interface.
# - add_chunks() is sync → called by Celery workers during ingestion
# - search() is async → called by FastAPI handlers during query
# This avoids forcing either side into an unnatural execution model.
#
# ARCHITECTURE:
#   VectorStore (Protocol)
#   ├── PgVectorStore     — PostgreSQL + pgvector extension
#   │   ├── add_chunks()  — sync via sync_session_factory (Celery)
#   │   └── search()      — async via async_session_factory (FastAPI)
#   └── ChromaVectorStore — ChromaDB (in-process or client/server)
#       ├── add_chunks()  — sync (ChromaDB client is sync)
#       └── search()      — async via asyncio.to_thread() wrapper
# =============================================================================

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol

import chromadb
from sqlalchemy import select

from app.config import settings
from app.db.engine import async_session_factory, get_sync_session
from app.db.models import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class VectorSearchResult:
    """
    A single result from vector similarity search.

    Contains the chunk data plus its similarity score, used by the
    search agent to build context for the LLM.
    """

    chunk_id: int
    content: str
    page_number: int | None
    similarity_score: float  # 0.0–1.0 (cosine similarity, higher = more relevant)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol Definition
# ---------------------------------------------------------------------------


class VectorStore(Protocol):
    """
    Protocol defining the vector store interface.

    Both pgvector and ChromaDB implementations must provide these methods.
    The protocol is checked statically by mypy — no runtime registration needed.
    """

    def add_chunks(
        self,
        document_id: int,
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> list[int]:
        """
        Store chunks with their embeddings. Sync (for Celery).

        Args:
            document_id: Parent document ID.
            contents: Chunk text contents.
            embeddings: Corresponding embedding vectors.
            metadatas: Per-chunk metadata dicts (page_number, section, etc.)

        Returns:
            List of created chunk IDs.
        """
        ...

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        Find the most similar chunks to the query. Async (for FastAPI).

        Args:
            query_embedding: The query vector (1536 dimensions).
            top_k: Number of results to return.
            document_id: Optional filter to search within a specific document.

        Returns:
            List of VectorSearchResult sorted by similarity (highest first).
        """
        ...


# ---------------------------------------------------------------------------
# Implementation 1: pgvector (PostgreSQL)
# ---------------------------------------------------------------------------


class PgVectorStore:
    """
    pgvector-backed vector store using PostgreSQL.

    DESIGN DECISION: Uses SQLAlchemy ORM for writes (type safety, bulk inserts)
    and pgvector's distance operators for reads (cosine similarity search).

    Uses sync engine for add_chunks (Celery) and async engine for search
    (FastAPI), matching each caller's execution model.
    """

    def add_chunks(
        self,
        document_id: int,
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> list[int]:
        """Store chunks in PostgreSQL with pgvector embeddings."""
        chunk_ids: list[int] = []

        with get_sync_session() as session:
            chunks = []
            for i, (content, embedding, meta) in enumerate(
                zip(contents, embeddings, metadatas, strict=True)
            ):
                chunk = Chunk(
                    document_id=document_id,
                    content=content,
                    page_number=meta.get("page_number"),
                    chunk_index=meta.get("chunk_index", i),
                    token_count=meta.get("token_count", 0),
                    embedding=embedding,
                    metadata_=meta,
                )
                session.add(chunk)
                chunks.append(chunk)

            # Flush assigns IDs without committing the transaction
            session.flush()
            chunk_ids = [c.id for c in chunks]
            session.commit()

        logger.info(
            "Stored %d chunks for document_id=%d in pgvector",
            len(chunk_ids), document_id,
        )
        return chunk_ids

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        Cosine similarity search using pgvector.

        DESIGN DECISION: pgvector's cosine_distance() returns values in [0, 2].
        We convert to similarity: 1 - distance, giving values in [-1, 1].
        For normalized embeddings (which OpenAI returns), the range is [0, 1].
        """
        async with async_session_factory() as session:
            # Build query: select chunks ordered by cosine distance (ascending)
            stmt = (
                select(
                    Chunk,
                    Chunk.embedding.cosine_distance(query_embedding).label("distance"),
                )
                .where(Chunk.embedding.is_not(None))
                .order_by(Chunk.embedding.cosine_distance(query_embedding))
                .limit(top_k)
            )

            # Optional: filter by document
            if document_id is not None:
                stmt = stmt.where(Chunk.document_id == document_id)

            result = await session.execute(stmt)
            rows = result.all()

        logger.debug(
            "Vector search returned %d rows (top_k=%d, doc_id=%s)",
            len(rows), top_k, document_id,
        )

        return [
            VectorSearchResult(
                chunk_id=chunk.id,
                content=chunk.content,
                page_number=chunk.page_number,
                similarity_score=round(1.0 - distance, 4),
                metadata=chunk.metadata_ or {},
            )
            for chunk, distance in rows
        ]


# ---------------------------------------------------------------------------
# Implementation 2: ChromaDB
# ---------------------------------------------------------------------------


class ChromaVectorStore:
    """
    ChromaDB-backed vector store.

    DESIGN DECISION: Single global collection named 'fin_doc_chunks'.
    Per-document filtering uses ChromaDB's metadata where clause.
    This enables cross-document search (the primary use case for /ask)
    while still supporting per-document queries.

    ChromaDB supports both in-process and client/server modes:
    - In-process (default): No extra infra, data stored in memory/disk
    - Client/server: Set CHROMA_URL for Docker deployment
    """

    def __init__(self) -> None:
        if settings.chroma_url:
            # Client/server mode (e.g., Docker deployment)
            self._client = chromadb.HttpClient(host=settings.chroma_url)
        else:
            # In-process mode (local development, testing)
            self._client = chromadb.Client()

        # DESIGN DECISION: Use cosine distance to match pgvector behavior.
        # This ensures fair comparison when benchmarking the two backends.
        self._collection = self._client.get_or_create_collection(
            name="fin_doc_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        document_id: int,
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> list[int]:
        """Store chunks in ChromaDB with document_id in metadata for filtering."""
        # ChromaDB requires string IDs
        ids = [
            f"doc{document_id}_chunk{meta.get('chunk_index', i)}"
            for i, meta in enumerate(metadatas)
        ]

        # Enrich metadata with document_id for per-document search filtering
        enriched_metadatas = [
            {**meta, "document_id": document_id}
            for meta in metadatas
        ]

        # Sanitise metadata: ChromaDB requires all values to be str, int,
        # float, or bool — not lists or None. Convert problematic values.
        sanitised_metadatas = [
            _sanitise_chroma_metadata(m) for m in enriched_metadatas
        ]

        # ChromaDB add() is idempotent — upserts on matching IDs
        self._collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=sanitised_metadatas,
        )

        logger.info(
            "Stored %d chunks for document_id=%d in ChromaDB",
            len(ids), document_id,
        )
        # ChromaDB doesn't return integer IDs; return sequential indices
        return list(range(len(ids)))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: int | None = None,
    ) -> list[VectorSearchResult]:
        """
        Similarity search in ChromaDB.

        DESIGN DECISION: ChromaDB's Python client is synchronous.
        We wrap it in asyncio.to_thread() to avoid blocking the
        FastAPI event loop during search.
        """

        def _sync_search() -> list[VectorSearchResult]:
            where_filter = (
                {"document_id": document_id} if document_id is not None else None
            )

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            search_results: list[VectorSearchResult] = []
            if results and results["ids"] and results["ids"][0]:
                for i, chroma_id in enumerate(results["ids"][0]):
                    distance = (
                        results["distances"][0][i]
                        if results["distances"]
                        else 0.0
                    )
                    # ChromaDB cosine distance is in [0, 2]; convert to similarity
                    similarity = round(1.0 - distance, 4)
                    metadata = (
                        results["metadatas"][0][i]
                        if results["metadatas"]
                        else {}
                    )

                    content = (
                        results["documents"][0][i]
                        if results["documents"]
                        else ""
                    )
                    search_results.append(VectorSearchResult(
                        chunk_id=hash(chroma_id) % (10**9),
                        content=content,
                        page_number=metadata.get("page_number"),
                        similarity_score=similarity,
                        metadata=metadata,
                    ))

            return search_results

        return await asyncio.to_thread(_sync_search)


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def get_vector_store(
    override_type: str | None = None,
) -> PgVectorStore | ChromaVectorStore:
    """
    Factory that returns the configured vector store backend.

    Reads `vectorstore_type` from settings:
    - "pgvector" → PgVectorStore (default, no extra infra)
    - "chroma" → ChromaVectorStore (lightweight, fast prototyping)

    Args:
        override_type: Optional type override. When set, ignores the
            config setting. Used by /benchmark/retrieval to test
            multiple backends in one request.

    DESIGN DECISION: Simple factory over DI container. For two
    implementations, a factory function is clearer and more explicit
    than a full dependency injection framework.
    """
    store_type = override_type or settings.vectorstore_type

    if store_type == "chroma":
        logger.info("Using ChromaDB vector store")
        return ChromaVectorStore()

    logger.info("Using pgvector vector store")
    return PgVectorStore()


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _sanitise_chroma_metadata(metadata: dict) -> dict:
    """
    Sanitise metadata for ChromaDB compatibility.

    ChromaDB requires all metadata values to be str, int, float, or bool.
    Lists and None values are not supported. We convert:
    - list → comma-separated string
    - None → empty string
    """
    sanitised = {}
    for key, value in metadata.items():
        if value is None:
            sanitised[key] = ""
        elif isinstance(value, list):
            sanitised[key] = ",".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            sanitised[key] = value
        else:
            sanitised[key] = str(value)
    return sanitised
