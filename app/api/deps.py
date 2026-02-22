# =============================================================================
# Auth Dependencies — FastAPI Dependency Injection for Authentication
# =============================================================================
#
# Provides three FastAPI dependencies that protect API endpoints:
#
# 1. get_current_api_key() — extract & validate Bearer token
# 2. check_scope()         — verify endpoint-level permission
# 3. check_document_access() — verify document-level ACL
#
# DESIGN DECISION: FastAPI dependency (not middleware) for auth.
# Dependencies integrate cleanly with the DI system:
# - Each endpoint opts-in via Depends(get_current_api_key)
# - The resolved ApiKey object is available in route handlers
# - Testable via dependency_overrides
# - When auth_enabled=False, returns None (anonymous access)
#
# DESIGN DECISION: HTTPBearer(auto_error=False) so that when
# auth is disabled, missing headers don't cause errors. The
# dependency handles the logic internally.
# =============================================================================

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.engine import get_async_session
from app.db.models import ApiKey
from app.services.auth import hash_api_key

logger = logging.getLogger(__name__)

# Security scheme for OpenAPI docs (shows "Authorize" button in Swagger UI)
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(
        _bearer_scheme,
    ),
    session: AsyncSession = Depends(get_async_session),
) -> ApiKey | None:
    """
    FastAPI dependency that validates the API key.

    When auth_enabled=False: returns None (anonymous access).
    When auth_enabled=True:
    - Extracts Bearer token from Authorization header
    - SHA-256 hashes and looks up in api_keys table
    - Validates: is_active, not expired
    - Checks rate limit via Redis
    - Updates last_used_at
    - Stores ApiKey on request.state for audit middleware

    Raises:
        HTTPException 401: Missing or invalid API key
        HTTPException 403: Key is inactive or expired
        HTTPException 429: Rate limit exceeded
    """
    if not settings.auth_enabled:
        return None

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide "
            "'Authorization: Bearer <key>' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_key = credentials.credentials
    key_hash = hash_api_key(raw_key)

    stmt = select(ApiKey).where(ApiKey.key_hash == key_hash)
    result = await session.execute(stmt)
    api_key = result.scalar_one_or_none()

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not api_key.is_active:
        raise HTTPException(
            status_code=403,
            detail="API key has been deactivated.",
        )

    if api_key.expires_at and api_key.expires_at < datetime.now(UTC):
        raise HTTPException(
            status_code=403,
            detail="API key has expired.",
        )

    # Rate limit check (imported lazily to avoid circular imports)
    from app.services.rate_limiter import check_rate_limit
    await check_rate_limit(api_key)

    # Update last_used_at
    api_key.last_used_at = datetime.now(UTC)

    # Store on request.state for audit logging middleware
    request.state.api_key = api_key

    return api_key


def check_scope(api_key: ApiKey | None, required_scope: str) -> None:
    """
    Verify the API key has the required scope.

    Raises HTTPException 403 if the scope is missing.
    No-op when auth is disabled (api_key is None).
    No-op when key has null/empty scopes (full access).
    """
    if api_key is None:
        return

    # Null or empty scopes = full access
    if not api_key.scopes:
        return

    if required_scope not in api_key.scopes:
        raise HTTPException(
            status_code=403,
            detail=f"API key does not have '{required_scope}' scope.",
        )


def check_document_access(
    api_key: ApiKey | None,
    document_id: int | None,
) -> None:
    """
    Verify the API key has access to the specified document.

    Raises HTTPException 403 if access is denied.

    No-op when:
    - Auth disabled (api_key is None)
    - Key has all_documents_access=True
    - document_id is None AND key has all_documents_access

    Restricted keys (all_documents_access=False) MUST specify
    a document_id — searching all documents is not allowed.
    """
    if api_key is None:
        return

    if api_key.all_documents_access:
        return

    if document_id is None:
        raise HTTPException(
            status_code=403,
            detail="This API key requires a document_id. "
            "You do not have 'all documents' access.",
        )

    allowed_ids = {ad.document_id for ad in api_key.allowed_documents}
    if document_id not in allowed_ids:
        raise HTTPException(
            status_code=403,
            detail=f"API key does not have access to "
            f"document {document_id}.",
        )
