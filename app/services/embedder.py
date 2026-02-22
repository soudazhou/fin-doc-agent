# =============================================================================
# Embedding Service — Batch Vector Generation (Provider-Agnostic)
# =============================================================================
#
# Generates vector embeddings using any OpenAI-compatible embedding API.
# Supports OpenAI, Alibaba Cloud (DashScope), and any other provider that
# implements the OpenAI embeddings endpoint.
#
# DESIGN DECISION: OpenAI SDK with configurable base_url.
# The OpenAI SDK is the de facto standard — most providers (Alibaba Cloud,
# DeepSeek, etc.) expose OpenAI-compatible APIs. By making base_url
# configurable, we support all of them with zero code changes.
#
# DESIGN DECISION: Sync-only for now. Celery workers (the primary consumer
# during ingestion) are synchronous. An async version (using AsyncOpenAI)
# can be added if FastAPI endpoints need direct embedding.
#
# DESIGN DECISION: No retry logic in the embedder. Retries are handled at
# the Celery task level (max_retries=3, exponential backoff). This keeps
# the embedder simple and avoids nested retry logic.
#
# TOKEN LIMITS:
# - Each text: max 8,191 tokens
# - We batch at 100 texts per API call (configurable via settings)
# - 100 chunks × 512 tokens = 51,200 tokens per call (well within limits)
# =============================================================================

from __future__ import annotations

import logging
from collections.abc import Sequence

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding Client — Lazy Singleton
# ---------------------------------------------------------------------------
# DESIGN DECISION: Module-level client instance.
# The OpenAI client manages its own HTTP connection pool and is thread-safe.
# Creating one per call would waste connection setup time.
# Lazy initialization avoids import-time failures when API key isn't set.
#
# DESIGN DECISION: API key resolution order:
#   1. OPENAI_API_KEY (explicit embedding key)
#   2. LLM_API_KEY (shared key — e.g. one DashScope key for LLM + embeddings)
# This lets users run both LLM and embeddings off a single Alibaba Cloud key.
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazily initialize and cache the embedding client."""
    global _client
    if _client is None:
        resolved_key = settings.openai_api_key or settings.llm_api_key
        if not resolved_key:
            raise ValueError(
                "No API key configured for embeddings. "
                "Set OPENAI_API_KEY or LLM_API_KEY in .env"
            )

        client_kwargs: dict = {"api_key": resolved_key}
        if settings.embedding_base_url:
            client_kwargs["base_url"] = settings.embedding_base_url

        _client = OpenAI(**client_kwargs)

        logger.info(
            "Initialized embedding client (model=%s, base_url=%s)",
            settings.embedding_model,
            settings.embedding_base_url or "https://api.openai.com/v1",
        )
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_batch(
    texts: Sequence[str],
    batch_size: int | None = None,
) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts using OpenAI API.

    Processes texts in sub-batches to respect API token limits.
    Returns embeddings in the SAME ORDER as the input texts.

    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts per API call. Defaults to
            settings.embedding_batch_size (100).

    Returns:
        List of embedding vectors (each is a list of 1536 floats),
        in the same order as the input texts.

    Raises:
        ValueError: If OpenAI API key is not configured.
        openai.APIError: If the OpenAI API call fails.

    Pipeline position: Step 3 of ingestion (parse → chunk → embed → store).
    """
    if not texts:
        return []

    client = _get_client()
    _batch_size = batch_size or settings.embedding_batch_size

    all_embeddings: list[list[float]] = [[] for _ in texts]

    for i in range(0, len(texts), _batch_size):
        batch = list(texts[i : i + _batch_size])
        logger.info(
            "Embedding batch %d–%d of %d texts (model=%s)",
            i + 1,
            min(i + _batch_size, len(texts)),
            len(texts),
            settings.embedding_model,
        )

        create_kwargs: dict = {
            "model": settings.embedding_model,
            "input": batch,
        }
        if settings.embedding_dimensions:
            create_kwargs["dimensions"] = settings.embedding_dimensions

        response = client.embeddings.create(**create_kwargs)

        # DESIGN DECISION: Sort by response.data[j].index to guarantee
        # output order matches input order. The OpenAI API documentation
        # states that items are returned in the same order, but we sort
        # defensively — order mismatches would silently corrupt embeddings.
        for item in sorted(response.data, key=lambda x: x.index):
            all_embeddings[i + item.index] = item.embedding

        logger.debug(
            "Batch complete: %d embeddings, %d prompt tokens",
            len(batch),
            response.usage.prompt_tokens if response.usage else 0,
        )

    logger.info(
        "Generated %d embeddings (model=%s, dimensions=%d)",
        len(texts),
        settings.embedding_model,
        settings.embedding_dimensions,
    )
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """
    Generate an embedding for a single query string.

    Convenience wrapper for single-text embedding. Used by the search
    agent when embedding user questions during retrieval.

    Args:
        text: The query text to embed.

    Returns:
        A single embedding vector (list of 1536 floats).
    """
    result = embed_batch([text], batch_size=1)
    return result[0]
