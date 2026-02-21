-- =============================================================================
-- Database Initialization Script
-- =============================================================================
-- This script runs ONCE when the PostgreSQL container is first created.
-- It enables the pgvector extension, which adds the `vector` data type
-- and similarity search operators (cosine, L2, inner product) to PostgreSQL.
--
-- The pgvector extension transforms PostgreSQL into a vector database,
-- eliminating the need for a separate service like Pinecone or Weaviate.
--
-- If you need to re-run this, delete the pgdata volume:
--   docker compose down -v && docker compose up
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
