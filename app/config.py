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
    embedding_batch_size: int = 100  # Chunks per OpenAI API call

    # -------------------------------------------------------------------------
    # LLM Configuration — Multi-Provider
    # -------------------------------------------------------------------------
    # DESIGN DECISION: Provider-agnostic LLM layer.
    # As of Feb 2026, Chinese LLMs (DeepSeek, Qwen, GLM-5, MiniMax) match
    # frontier Western models at 10-100x lower cost. We support two providers:
    #   - "anthropic": Claude via native Anthropic SDK
    #   - "openai_compatible": Any OpenAI-compatible API (DeepSeek, Qwen,
    #     GLM-5, Kimi, MiniMax, OpenAI itself)
    #
    # Switching providers is a single .env change — no code changes needed.
    # Use cheap models (DeepSeek V3 at $0.14/1M) for dev, Claude for demos.
    #
    # Example configs:
    #   DeepSeek V3:  provider=openai_compatible, base_url=https://api.deepseek.com/v1, model=deepseek-chat
    #   Qwen 3.5:    provider=openai_compatible, base_url=https://dashscope.aliyuncs.com/compatible-mode/v1, model=qwen-plus
    #   GLM-5:       provider=openai_compatible, base_url=https://open.bigmodel.cn/api/paas/v4, model=glm-5
    #   Claude:      provider=anthropic, model=claude-sonnet-4-6
    # -------------------------------------------------------------------------
    llm_provider: str = "anthropic"  # "anthropic" or "openai_compatible"
    llm_base_url: str | None = None  # Only needed for openai_compatible
    llm_api_key: str | None = None   # Overrides provider-specific key if set
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # -------------------------------------------------------------------------
    # Vector Store Configuration — Pluggable Backend
    # -------------------------------------------------------------------------
    # DESIGN DECISION: Pluggable vector store behind a common interface.
    # We implement both pgvector and Chroma, then benchmark them head-to-head
    # on the same dataset (Recall@k, MRR, latency). This lets data drive the
    # recommendation rather than assumptions.
    #
    # Options:
    #   - "pgvector": PostgreSQL extension (default, no extra infra)
    #   - "chroma": ChromaDB (in-process or client/server)
    # -------------------------------------------------------------------------
    vectorstore_type: str = "pgvector"  # "pgvector" or "chroma"
    chroma_url: str | None = None  # Only needed for Chroma in client/server mode

    # -------------------------------------------------------------------------
    # File Upload
    # -------------------------------------------------------------------------
    # Uploaded PDFs are stored on disk before Celery processes them.
    # This directory is relative to the project root.
    # -------------------------------------------------------------------------
    upload_dir: str = "data/uploads"

    # -------------------------------------------------------------------------
    # Chunking Configuration
    # -------------------------------------------------------------------------
    # DESIGN DECISION: Chunk size is CONFIGURABLE, not hardcoded.
    # Most projects pick 512 tokens by intuition. We benchmark multiple sizes
    # (256, 512, 1024) against a golden dataset to find the optimal size
    # empirically. The /benchmark/retrieval endpoint automates this comparison.
    #
    # Token-based chunking (not character-based) because:
    # 1. Aligns with LLM token limits — no surprises at inference time
    # 2. tiktoken is the same tokenizer used by OpenAI models
    # 3. 512 tokens ≈ 1-2 paragraphs, good for financial document sections
    # 4. Overlap prevents information loss at chunk boundaries
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
    # max_search_iterations: Agentic search loop bound — prevents runaway costs.
    # -------------------------------------------------------------------------
    retrieval_top_k: int = 5
    retrieval_similarity_threshold: float = 0.7
    max_search_iterations: int = 3

    # -------------------------------------------------------------------------
    # Evaluation Configuration (Phase 5)
    # -------------------------------------------------------------------------
    # DESIGN DECISION: Golden datasets stored as JSON files on disk (not DB).
    # They are versioned config tracked in git, not runtime data.
    # The eval_judge_model is the LLM used by DeepEval as an "LLM-as-judge"
    # for metrics like faithfulness and answer relevancy.
    # -------------------------------------------------------------------------
    eval_dataset_dir: str = "data/eval/golden_datasets"
    eval_judge_model: str = "gpt-4.1"  # Cheapest capable judge model
    eval_default_threshold: float = 0.7  # Default pass/fail threshold

    # -------------------------------------------------------------------------
    # Authorisation Configuration
    # -------------------------------------------------------------------------
    # DESIGN DECISION: Auth is toggleable. Disabled during local dev for
    # convenience, enabled in production. API key auth (not OAuth/JWT) is
    # appropriate for an API-first service.
    #
    # Rate limiting uses Redis (already in the stack for Celery).
    # -------------------------------------------------------------------------
    auth_enabled: bool = False  # Toggle on for production / demo
    rate_limit_rpm: int = 100   # Default requests-per-minute per API key
    audit_logging_enabled: bool = True  # Log all queries for compliance

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
