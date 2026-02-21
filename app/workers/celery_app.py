# =============================================================================
# Celery Application Configuration
# =============================================================================
#
# Celery is a distributed task queue that processes background jobs.
# In this project, it handles the document ingestion pipeline:
#   PDF Upload → Parse → Chunk → Embed → Store
#
# WHY CELERY OVER FASTAPI BACKGROUNDTASKS?
# ┌─────────────────────┬──────────────────────┬────────────────────────┐
# │ Feature             │ BackgroundTasks       │ Celery                 │
# ├─────────────────────┼──────────────────────┼────────────────────────┤
# │ Reliability         │ Fire-and-forget       │ Persistent, retries    │
# │ Monitoring          │ None                  │ Flower dashboard       │
# │ Scalability         │ Single process        │ Multi-worker, multi-host│
# │ Task status         │ No tracking           │ Full status tracking   │
# │ Crash recovery      │ Task lost             │ Task re-queued         │
# └─────────────────────┴──────────────────────┴────────────────────────┘
#
# ARCHITECTURE:
# ┌──────────┐     ┌───────┐     ┌──────────────┐     ┌───────┐
# │ FastAPI  │────▶│ Redis │────▶│ Celery Worker│────▶│ Redis │
# │ (producer)│    │(broker)│    │ (consumer)    │    │(result)│
# └──────────┘     └───────┘     └──────────────┘     └───────┘
#    db 0 ──────────┘                                    └── db 1
#
# The broker (Redis db 0) queues tasks. Workers consume and execute them.
# Results are stored in Redis db 1 for the API to poll.
# =============================================================================

from celery import Celery

from app.config import settings

# ---------------------------------------------------------------------------
# Create Celery Application
# ---------------------------------------------------------------------------
# The first argument ("app.workers") is the Celery app name.
# It's used for logging and identifying this app in the Flower dashboard.
# ---------------------------------------------------------------------------
celery_app = Celery(
    "app.workers",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# ---------------------------------------------------------------------------
# Celery Configuration
# ---------------------------------------------------------------------------
# These settings are tuned for reliability and debuggability.
# Each setting is documented with WHY it's set to this value.
# ---------------------------------------------------------------------------
celery_app.conf.update(
    # --- Serialization ---
    # Use JSON (not pickle) for task arguments and results.
    # SECURITY: Pickle can execute arbitrary code during deserialization.
    # JSON is safe and human-readable (easier to debug in Redis).
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # --- Reliability ---
    # task_acks_late: Don't acknowledge the task until it completes.
    # If the worker crashes mid-task, the task is re-queued automatically.
    # Without this, crashing loses the task permanently.
    task_acks_late=True,

    # task_reject_on_worker_lost: If a worker is killed (OOM, SIGKILL),
    # reject the task so it's re-queued. Without this, the task is lost.
    task_reject_on_worker_lost=True,

    # worker_prefetch_multiplier: How many tasks a worker pre-fetches.
    # Set to 1 for fair distribution across workers.
    # Higher values improve throughput but reduce fairness.
    # For long-running tasks like PDF processing, 1 is ideal.
    worker_prefetch_multiplier=1,

    # --- Timeouts ---
    # task_soft_time_limit: Send SIGTERM after 5 minutes.
    # The task can catch this and clean up gracefully.
    task_soft_time_limit=300,

    # task_time_limit: Send SIGKILL after 10 minutes (hard kill).
    # Prevents zombie tasks from consuming resources forever.
    task_time_limit=600,

    # --- Results ---
    # result_expires: Delete results from Redis after 1 hour.
    # Prevents Redis from filling up with old results.
    result_expires=3600,

    # --- Task Discovery ---
    # Auto-discover tasks in the app.workers.tasks module.
    # Celery will look for functions decorated with @celery_app.task.
    include=["app.workers.tasks"],
)
