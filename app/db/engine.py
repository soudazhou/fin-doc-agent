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
# 4. Session auto-commits on exit and is closed when the request completes
# 5. On exception, the transaction is rolled back
#
# COMMIT POLICY:
# Two session patterns exist in this codebase:
#
# 1. Dependency-injected (get_async_session via Depends):
#    Auto-commits when the request handler returns. Endpoints MAY call
#    session.commit() mid-handler if they need a generated ID before the
#    handler exits (e.g., evaluate.py commits an EvalRun to get run_id,
#    then passes it to a background task). The auto-commit at exit is a
#    no-op if no further changes were made.
#
# 2. Self-managed (async_session_factory() directly):
#    Used by background tasks (ask.py _persist_metric) and middleware
#    (audit.py) that run outside the request dependency lifecycle.
#    These MUST commit explicitly.
# =============================================================================

from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

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


# ---------------------------------------------------------------------------
# Sync Engine — For Celery Workers (Lazy Initialization)
# ---------------------------------------------------------------------------
# DESIGN DECISION: Celery workers are synchronous (no async/await).
# They CANNOT use the async engine above. Attempting to do so raises:
#   "Cannot use the async engine in a synchronous context."
#
# This sync engine uses psycopg2 (synchronous PostgreSQL driver) and
# connects to the same database. Both engines share the same schema.
#
# WHY TWO ENGINES?
# FastAPI is async-native → needs asyncpg (non-blocking I/O)
# Celery is sync-native → needs psycopg2 (blocking I/O in threads)
# Using the wrong driver type will cause runtime errors in either context.
#
# DESIGN DECISION: Lazy initialization (not module-level like async_engine).
# The sync engine requires psycopg2, which is only needed by Celery workers.
# Lazy init avoids import errors in contexts where psycopg2 isn't needed
# (e.g., running just the FastAPI server during development).
# ---------------------------------------------------------------------------

_sync_engine = None
_sync_session_factory = None


def _get_sync_engine():
    """Lazily create and cache the sync SQLAlchemy engine."""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(
            settings.database_url_sync,
            echo=settings.debug,
            pool_size=5,
            max_overflow=10,
        )
    return _sync_engine


def _get_sync_session_factory():
    """Lazily create and cache the sync session factory."""
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(
            bind=_get_sync_engine(),
            class_=Session,
            expire_on_commit=False,
        )
    return _sync_session_factory


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """
    Context manager that provides a sync database session for Celery workers.

    Usage in Celery tasks:
        with get_sync_session() as session:
            doc = session.get(Document, document_id)
            doc.status = DocumentStatus.COMPLETED
            # Auto-commits on exit, auto-rollbacks on exception

    DESIGN DECISION: This mirrors the async `get_async_session` pattern
    but uses synchronous SQLAlchemy. Both follow the same lifecycle:
    create → yield → commit (or rollback on error) → close.
    """
    factory = _get_sync_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


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
