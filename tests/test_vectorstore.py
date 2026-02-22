# =============================================================================
# Unit Tests — Vector Store (ChromaDB backend)
# =============================================================================
#
# Tests ChromaDB vector store operations: add, search, metadata filtering.
# Uses ChromaDB's in-process mode (no external services needed).
# pgvector tests are skipped here — they require a running PostgreSQL instance.
# =============================================================================

import asyncio

from app.services.vectorstore import ChromaVectorStore, VectorSearchResult


def _get_event_loop():
    """Get or create an event loop for running async tests."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class TestChromaVectorStore:
    """Tests for ChromaVectorStore (in-process mode)."""

    _test_counter = 0

    def _make_store(self, dim: int = 3) -> ChromaVectorStore:
        """Create a fresh ChromaVectorStore with a unique collection per test."""
        TestChromaVectorStore._test_counter += 1
        store = ChromaVectorStore()
        # Use a unique collection name per test to avoid dimension conflicts
        collection_name = f"test_collection_{TestChromaVectorStore._test_counter}"
        store._collection = store._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return store

    def test_add_chunks_returns_sequential_ids(self):
        store = self._make_store()
        ids = store.add_chunks(
            document_id=1,
            contents=["Hello world", "Goodbye world"],
            embeddings=[[0.1] * 3, [0.2] * 3],
            metadatas=[
                {"chunk_index": 0, "page_number": 1},
                {"chunk_index": 1, "page_number": 1},
            ],
        )
        assert ids == [0, 1]

    def test_search_returns_results(self):
        store = self._make_store()
        store.add_chunks(
            document_id=1,
            contents=["Revenue increased by 15%", "Expenses decreased by 5%"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            metadatas=[
                {"chunk_index": 0, "page_number": 1},
                {"chunk_index": 1, "page_number": 2},
            ],
        )

        loop = _get_event_loop()
        results = loop.run_until_complete(
            store.search(query_embedding=[1.0, 0.0, 0.0], top_k=2)
        )

        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # The first result should be the most similar to [1.0, 0.0, 0.0]
        assert results[0].similarity_score >= results[1].similarity_score
        assert "Revenue" in results[0].content

    def test_search_filters_by_document_id(self):
        store = self._make_store()

        # Add chunks from two different documents
        store.add_chunks(
            document_id=1,
            contents=["Doc 1 content"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[{"chunk_index": 0, "page_number": 1}],
        )
        store.add_chunks(
            document_id=2,
            contents=["Doc 2 content"],
            embeddings=[[0.9, 0.1, 0.0]],
            metadatas=[{"chunk_index": 0, "page_number": 1}],
        )

        loop = _get_event_loop()

        # Search only document 2
        results = loop.run_until_complete(
            store.search(
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
                document_id=2,
            )
        )

        assert len(results) == 1
        assert "Doc 2" in results[0].content

    def test_metadata_sanitisation(self):
        """ChromaDB should handle None and list values in metadata."""
        store = self._make_store()
        # This should not raise — None and lists are sanitised
        ids = store.add_chunks(
            document_id=1,
            contents=["Test content"],
            embeddings=[[0.5] * 3],
            metadatas=[{
                "chunk_index": 0,
                "page_number": None,
                "source_pages": [1, 2, 3],
                "section_title": "Overview",
            }],
        )
        assert ids == [0]
