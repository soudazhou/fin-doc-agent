# =============================================================================
# Workers Package — Celery Background Tasks
# =============================================================================
# Handles asynchronous, long-running operations:
#   - celery_app.py: Celery application configuration
#   - tasks.py: Task definitions (PDF ingestion pipeline)
#
# WHY CELERY?
# Document ingestion involves multiple slow steps:
#   1. PDF parsing (CPU-bound, can take seconds for large docs)
#   2. Embedding generation (network-bound, OpenAI API calls)
#   3. Database inserts (I/O-bound, bulk vector inserts)
#
# Running these synchronously would block the API server.
# Celery processes them in the background, returning a task_id immediately
# so the client can poll for status.
# =============================================================================
