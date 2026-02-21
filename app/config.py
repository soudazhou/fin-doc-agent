# =============================================================================
# Application Configuration — Pydantic Settings
# =============================================================================
#
# DESIGN DECISION: We use Pydantic V2's `BaseSettings` for configuration.
# This provides:
# 1. Type-safe configuration with validation at startup
# 2. Automatic loading from environment variables
# 3. Support for .env files (via `env_file` in model_config)
# 4. Clear documentation of all required/optional settings
# 5. Sensible defaults for local development
#
# HOW IT WORKS:
# Pydantic Settings loads values in this priority order (highest first):
#   1. Environment variables (e.g., `DATABASE_URL=...`)
#   2. Values from the .env file
#   3. Default values defined below
#
# This means you can override any setting via environment variables without
# changing code — a twelve-factor app best practice.
#
# USAGE:
#   from app.config import settings
#   print(settings.database_url)
# =============================================================================

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings have sensible defaults for local development with Docker Compose.
    In production, override via environment variables or a .env file.
    """

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = "Financial Document Q&A Agent"
    app_version: str = "0.1.0"
    debug: bool = True

    # -------------------------------------------------------------------------
    # Database — PostgreSQL + pgvector
    # -------------------------------------------------------------------------
    # DESIGN DECISION: We use asyncpg (async driver) for FastAPI endpoints
    # and a separate sync connection string for Celery workers.
    # FastAPI is async-native, but Celery workers are synchronous.
    # Using the wrong driver type will cause runtime errors.
    #
    # Format: postgresql+asyncpg://user:pass@host:port/dbname
    # The `+asyncpg` part tells SQLAlchemy to use the async driver.
    # -------------------------------------------------------------------------
    database_url: str = (
        "postgresql+asyncpg://finagent:finagent_dev@localhost:5432/fin_doc_agent"
    )

    # Sync URL for Celery workers (note: psycopg2, not asyncpg)
    # Celery tasks run in synchronous threads, so they need a sync driver.
    database_url_sync: str = (
        "postgresql+psycopg2://finagent:finagent_dev@localhost:5432/fin_doc_agent"
    )

    # -------------------------------------------------------------------------
    # Redis
    # -------------------------------------------------------------------------
    # Redis database numbers isolate different concerns:
    #   db 0 = Celery broker (task queue)
    #   db 1 = Celery result backend (task results)
    #   db 2 = Application cache (future use)
    # -------------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # -------------------------------------------------------------------------
    # API Keys — External Services
    # -------------------------------------------------------------------------
    # These MUST be set via environment variables or .env file.
    # No defaults are provided to prevent accidental use of invalid keys.
    #
    # ANTHROPIC_API_KEY: For Claude API (Analyst agent reasoning)
    # OPENAI_API_KEY: For text-embedding-3-small (embedding generation only)
    # -------------------------------------------------------------------------
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # -------------------------------------------------------------------------
    # Embedding Configuration
    # -------------------------------------------------------------------------
    # DESIGN DECISION: text-embedding-3-small is chosen for this demo because:
    # 1. Good quality at very low cost ($0.02/1M tokens)
    # 2. 1536 dimensions — standard size, good balance of quality vs storage
    # 3. No GPU required (API-based)
    # In production, you might consider text-embedding-3-large (3072 dims)
    # or a self-hosted model like Nomic Embed V2 for cost savings at scale.
    # -------------------------------------------------------------------------
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # -------------------------------------------------------------------------
    # LLM Configuration
    # -------------------------------------------------------------------------
    # claude-sonnet-4-6 is the reasoning model used by the Analyst agent.
    # It offers the best balance of quality and cost for this use case.
    # Temperature 0.1 keeps outputs deterministic for financial analysis
    # (we want consistent, factual answers, not creative ones).
    # -------------------------------------------------------------------------
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # -------------------------------------------------------------------------
    # Chunking Configuration
    # -------------------------------------------------------------------------
    # DESIGN DECISION: Token-based chunking (not character-based) because:
    # 1. Aligns with LLM token limits — no surprises at inference time
    # 2. tiktoken is the same tokenizer used by OpenAI models
    # 3. 512 tokens ≈ 1-2 paragraphs, good for financial document sections
    # 4. 50 token overlap prevents information loss at chunk boundaries
    # -------------------------------------------------------------------------
    chunk_size: int = 512
    chunk_overlap: int = 50

    # -------------------------------------------------------------------------
    # Retrieval Configuration
    # -------------------------------------------------------------------------
    # top_k: Number of most similar chunks returned by vector search.
    # 5 chunks provides enough context without overwhelming the LLM.
    # similarity_threshold: Minimum cosine similarity score to include a chunk.
    # Chunks below this threshold are filtered out as irrelevant.
    # -------------------------------------------------------------------------
    retrieval_top_k: int = 5
    retrieval_similarity_threshold: float = 0.7

    # -------------------------------------------------------------------------
    # Pydantic Settings Configuration
    # -------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        # Load from .env file in the project root
        env_file=".env",
        # Don't fail if .env file doesn't exist (use defaults)
        env_file_encoding="utf-8",
        # Allow extra fields in environment (don't crash on unknown vars)
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Create and cache a Settings instance.

    DESIGN DECISION: We use @lru_cache to ensure Settings is created only once.
    This avoids re-reading environment variables and .env file on every request.
    The settings object is effectively a singleton.

    In tests, you can override this with FastAPI's dependency_overrides:
        app.dependency_overrides[get_settings] = lambda: Settings(debug=False)
    """
    return Settings()


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------
# Import this directly in most cases:
#   from app.config import settings
#
# Use get_settings() as a FastAPI dependency when you need testability:
#   @router.get("/")
#   def index(settings: Settings = Depends(get_settings)):
# ---------------------------------------------------------------------------
settings = Settings()
