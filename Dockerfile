# =============================================================================
# Financial Document Q&A Agent — Dockerfile
# =============================================================================
#
# Multi-stage build for the Python application.
#
# DESIGN DECISIONS:
# 1. Python 3.12-slim: Smallest official Python image with full stdlib.
#    Alpine would be smaller but causes issues with compiled dependencies
#    (numpy, scipy) that many ML libraries require.
#
# 2. uv for dependency management: 10-100x faster than pip.
#    Installs are cached via Docker layer caching for fast rebuilds.
#
# 3. Non-root user: Security best practice. The app runs as `appuser`
#    instead of root, limiting damage from potential vulnerabilities.
#
# 4. /app as working directory: Standard convention for Python web apps.
# =============================================================================

FROM python:3.12-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# - gcc, g++: Required to compile some Python packages with C extensions
# - libpq-dev: PostgreSQL client library (required by asyncpg)
# - curl: For container health checks
# These are kept in the final image because Celery workers also need them.
# In a production multi-stage build, you'd separate build and runtime deps.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install uv — fast Python package manager
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ---------------------------------------------------------------------------
# Install Python dependencies (cached layer)
# ---------------------------------------------------------------------------
# OPTIMIZATION: Copy only dependency files first. Docker caches this layer
# so dependencies are only re-installed when pyproject.toml changes,
# not on every code change. This makes rebuilds much faster.
# ---------------------------------------------------------------------------
WORKDIR /app

COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# ---------------------------------------------------------------------------
# Copy application code
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# Security: Run as non-root user
# ---------------------------------------------------------------------------
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# ---------------------------------------------------------------------------
# Default command: Start FastAPI with uvicorn
# ---------------------------------------------------------------------------
# This is overridden by docker-compose.yml for different services
# (app, celery-worker, flower) but provides a sensible default.
# ---------------------------------------------------------------------------
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
