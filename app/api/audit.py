# =============================================================================
# Audit Logging Middleware — Request/Response Lifecycle Logging
# =============================================================================
#
# Records every API request to the audit_logs table for compliance.
# Financial data access must be traceable: who accessed what, when.
#
# DESIGN DECISION: Starlette middleware (not a FastAPI dependency) because:
# 1. Middleware wraps the ENTIRE request lifecycle (captures status code)
# 2. Captures timing across the full request
# 3. Does not require every endpoint to explicitly opt-in
# 4. Audit writes use their own DB session to avoid lifecycle conflicts
#
# DESIGN DECISION: Non-blocking writes. The audit log is persisted
# after the response is generated. Failures are logged but never crash
# the actual request — availability > auditability.
# =============================================================================

from __future__ import annotations

import logging
import time

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import Response

from app.config import settings
from app.db.engine import async_session_factory
from app.db.models import AuditLog

logger = logging.getLogger(__name__)

# Endpoints to skip audit logging (health check, docs)
_SKIP_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all API requests to the audit_logs table.

    Reads api_key info from request.state (set by auth dependency).
    Reads audit_document_id and audit_question from request.state
    (set by individual endpoint handlers for richer audit data).
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if not settings.audit_logging_enabled:
            return await call_next(request)

        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        start_time = time.monotonic()
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Extract API key info (set by get_current_api_key dependency)
        api_key = getattr(request.state, "api_key", None)
        api_key_id = api_key.id if api_key else None
        api_key_name = api_key.name if api_key else None

        # Extract audit context (set by endpoint handlers)
        document_id = getattr(
            request.state, "audit_document_id", None,
        )
        question = getattr(request.state, "audit_question", None)

        # Client IP
        client_ip = (
            request.client.host if request.client else None
        )

        # Derive endpoint name from path
        path_parts = request.url.path.strip("/").split("/")
        endpoint_name = path_parts[0] if path_parts else ""

        # Persist audit log (own session, non-blocking)
        try:
            async with async_session_factory() as session:
                log_entry = AuditLog(
                    api_key_id=api_key_id,
                    api_key_name=api_key_name,
                    endpoint=endpoint_name,
                    method=request.method,
                    path=str(request.url.path),
                    document_id=document_id,
                    question=(
                        question[:500] if question else None
                    ),
                    client_ip=client_ip,
                    status_code=response.status_code,
                    response_time_ms=elapsed_ms,
                )
                session.add(log_entry)
                await session.commit()
        except Exception as e:
            logger.warning("Failed to write audit log: %s", e)

        return response
