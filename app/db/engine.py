# =============================================================================
# Database Engine & Session Management
# =============================================================================
#
# DESIGN DECISION: Async SQLAlchemy Engine
# FastAPI is an async framework, so we use SQLAlchemy's async engine to avoid
# blocking the event loop during database operations. This means:
# - All DB queries use `await` (e.g., `await session.execute(...)`)
# - We use `asyncpg` as the PostgreSQL driver (fastest async driver)
# - Sessions are created per-request via FastAPI's dependency injection
#
# IMPORTANT: Celery workers are SYNCHRONOUS and cannot use this async engine.
# Celery tasks must use a separate sync engine (see workers/tasks.py).
# This is a common gotcha when combining FastAPI + Celery + SQLAlchemy.
#
# SESSION LIFECYCLE:
# 1. FastAPI request arrives
# 2. `get_async_session` dependency creates a new session
# 3. Route handler uses session for DB operations
# 4. Session is automatically closed when the request completes
# 5. On exception, the transaction is rolled back
# =============================================================================

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# ---------------------------------------------------------------------------
# Async Engine
# ---------------------------------------------------------------------------
# The engine manages a connection pool to PostgreSQL.
#
# Key parameters:
# - echo=True (debug mode): Logs all SQL statements to stdout.
#   Incredibly useful during development to see exactly what queries
#   SQLAlchemy generates. Disable in production for performance.
#
# - pool_size=5: Maximum number of persistent connections in the pool.
#   5 is fine for a demo. In production, tune based on expected concurrency.
#
# - max_overflow=10: Additional connections allowed beyond pool_size
#   during traffic spikes. These connections are closed when no longer needed.
# ---------------------------------------------------------------------------
async_engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=5,
    max_overflow=10,
)

# ---------------------------------------------------------------------------
# Session Factory
# ---------------------------------------------------------------------------
# async_sessionmaker creates new AsyncSession instances.
#
# - expire_on_commit=False: Prevents SQLAlchemy from marking all loaded
#   objects as "expired" after commit. Without this, accessing any attribute
#   after commit would trigger a new database query — which fails in async
#   context outside of a session. This is a common async SQLAlchemy gotcha.
# ---------------------------------------------------------------------------
async_session_factory = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session per request.

    Usage in route handlers:
        @router.get("/items")
        async def list_items(session: AsyncSession = Depends(get_async_session)):
            result = await session.execute(select(Item))
            return result.scalars().all()

    The session is automatically closed when the request completes.
    If an exception occurs, the transaction is rolled back.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
