# =============================================================================
# Admin API — API Key Management
# =============================================================================
#
# CRUD endpoints for managing API keys. All endpoints require the "admin"
# scope, ensuring only privileged keys can create/modify other keys.
#
# DESIGN DECISION: The raw API key is only returned ONCE at creation
# (POST /admin/keys). After that, only the key_prefix is visible.
# This follows industry practice (GitHub tokens, Stripe keys).
#
# DESIGN DECISION: Soft-delete via is_active=False rather than hard
# delete for the PATCH endpoint. DELETE truly removes the row.
# =============================================================================

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import check_scope, get_current_api_key
from app.db.engine import get_async_session
from app.db.models import ApiKey, ApiKeyDocument, AuditLog
from app.models.requests import (
    CreateApiKeyRequest,
    GrantDocumentAccessRequest,
    UpdateApiKeyRequest,
)
from app.models.responses import (
    ApiKeyCreatedResponse,
    ApiKeyListResponse,
    ApiKeyResponse,
    AuditLogListResponse,
    AuditLogResponse,
)
from app.services.auth import generate_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin"])


# ---------------------------------------------------------------------------
# POST /admin/keys — Create API Key
# ---------------------------------------------------------------------------


@router.post(
    "/admin/keys",
    response_model=ApiKeyCreatedResponse,
    status_code=201,
    summary="Create a new API key",
    description=(
        "Generate a new API key with optional scopes, rate limits, and "
        "document access restrictions. The raw key is only returned in "
        "this response — store it securely."
    ),
)
async def create_api_key(
    request: CreateApiKeyRequest,
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> ApiKeyCreatedResponse:
    """Create a new API key and return it (once)."""
    check_scope(api_key, "admin")

    raw_key, key_prefix, key_hash = generate_api_key()

    new_key = ApiKey(
        name=request.name,
        key_prefix=key_prefix,
        key_hash=key_hash,
        scopes=request.scopes,
        rate_limit_rpm=request.rate_limit_rpm,
        all_documents_access=request.all_documents_access,
        expires_at=request.expires_at,
    )
    session.add(new_key)
    await session.flush()  # Get the ID before commit

    # Add document ACL entries if specified
    if request.document_ids and not request.all_documents_access:
        for doc_id in request.document_ids:
            session.add(ApiKeyDocument(
                api_key_id=new_key.id,
                document_id=doc_id,
            ))

    await session.commit()
    await session.refresh(new_key)

    logger.info(
        "API key created: id=%d, name='%s', prefix='%s'",
        new_key.id, new_key.name, new_key.key_prefix,
    )

    return ApiKeyCreatedResponse(
        id=new_key.id,
        name=new_key.name,
        key_prefix=new_key.key_prefix,
        raw_key=raw_key,
        scopes=new_key.scopes,
        rate_limit_rpm=new_key.rate_limit_rpm,
        all_documents_access=new_key.all_documents_access,
        created_at=new_key.created_at,
        expires_at=new_key.expires_at,
    )


# ---------------------------------------------------------------------------
# GET /admin/keys — List API Keys
# ---------------------------------------------------------------------------


@router.get(
    "/admin/keys",
    response_model=ApiKeyListResponse,
    summary="List all API keys",
)
async def list_api_keys(
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> ApiKeyListResponse:
    """List all API keys (never includes raw key or hash)."""
    check_scope(api_key, "admin")

    stmt = select(ApiKey).order_by(ApiKey.created_at.desc())
    result = await session.execute(stmt)
    keys = list(result.scalars().all())

    return ApiKeyListResponse(
        keys=[_to_key_response(k) for k in keys],
        total=len(keys),
    )


# ---------------------------------------------------------------------------
# GET /admin/keys/{key_id} — Get API Key Details
# ---------------------------------------------------------------------------


@router.get(
    "/admin/keys/{key_id}",
    response_model=ApiKeyResponse,
    summary="Get API key details",
)
async def get_api_key(
    key_id: int,
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> ApiKeyResponse:
    """Get details for a specific API key."""
    check_scope(api_key, "admin")

    target = await _get_key_or_404(session, key_id)
    return _to_key_response(target)


# ---------------------------------------------------------------------------
# PATCH /admin/keys/{key_id} — Update API Key
# ---------------------------------------------------------------------------


@router.patch(
    "/admin/keys/{key_id}",
    response_model=ApiKeyResponse,
    summary="Update an API key",
)
async def update_api_key(
    key_id: int,
    request: UpdateApiKeyRequest,
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> ApiKeyResponse:
    """Partial update for an API key (name, scopes, active, etc.)."""
    check_scope(api_key, "admin")

    target = await _get_key_or_404(session, key_id)

    if request.name is not None:
        target.name = request.name
    if request.scopes is not None:
        target.scopes = request.scopes
    if request.rate_limit_rpm is not None:
        target.rate_limit_rpm = request.rate_limit_rpm
    if request.all_documents_access is not None:
        target.all_documents_access = request.all_documents_access
    if request.is_active is not None:
        target.is_active = request.is_active
    if request.expires_at is not None:
        target.expires_at = request.expires_at

    await session.commit()
    await session.refresh(target)

    logger.info("API key updated: id=%d, name='%s'", target.id, target.name)
    return _to_key_response(target)


# ---------------------------------------------------------------------------
# DELETE /admin/keys/{key_id} — Delete API Key
# ---------------------------------------------------------------------------


@router.delete(
    "/admin/keys/{key_id}",
    status_code=204,
    summary="Delete an API key",
)
async def delete_api_key(
    key_id: int,
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Permanently delete an API key and its document ACL entries."""
    check_scope(api_key, "admin")

    target = await _get_key_or_404(session, key_id)
    await session.delete(target)
    await session.commit()

    logger.info(
        "API key deleted: id=%d, name='%s'", key_id, target.name,
    )


# ---------------------------------------------------------------------------
# POST /admin/keys/{key_id}/documents — Grant Document Access
# ---------------------------------------------------------------------------


@router.post(
    "/admin/keys/{key_id}/documents",
    response_model=ApiKeyResponse,
    summary="Grant document access to an API key",
)
async def grant_document_access(
    key_id: int,
    request: GrantDocumentAccessRequest,
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> ApiKeyResponse:
    """Add document IDs to the key's ACL."""
    check_scope(api_key, "admin")

    target = await _get_key_or_404(session, key_id)

    existing_ids = {ad.document_id for ad in target.allowed_documents}
    for doc_id in request.document_ids:
        if doc_id not in existing_ids:
            session.add(ApiKeyDocument(
                api_key_id=key_id,
                document_id=doc_id,
            ))

    await session.commit()
    await session.refresh(target)
    return _to_key_response(target)


# ---------------------------------------------------------------------------
# DELETE /admin/keys/{key_id}/documents/{doc_id} — Revoke Document Access
# ---------------------------------------------------------------------------


@router.delete(
    "/admin/keys/{key_id}/documents/{document_id}",
    response_model=ApiKeyResponse,
    summary="Revoke document access from an API key",
)
async def revoke_document_access(
    key_id: int,
    document_id: int,
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> ApiKeyResponse:
    """Remove a document ID from the key's ACL."""
    check_scope(api_key, "admin")

    target = await _get_key_or_404(session, key_id)

    stmt = select(ApiKeyDocument).where(
        ApiKeyDocument.api_key_id == key_id,
        ApiKeyDocument.document_id == document_id,
    )
    result = await session.execute(stmt)
    acl_entry = result.scalar_one_or_none()

    if acl_entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not in key's ACL.",
        )

    await session.delete(acl_entry)
    await session.commit()
    await session.refresh(target)
    return _to_key_response(target)


# ---------------------------------------------------------------------------
# GET /admin/audit — Audit Log
# ---------------------------------------------------------------------------


@router.get(
    "/admin/audit",
    response_model=AuditLogListResponse,
    summary="View audit logs",
    description="Query the audit trail. Filter by API key or document.",
)
async def get_audit_logs(
    api_key_id: int | None = Query(default=None),
    document_id: int | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> AuditLogListResponse:
    """Query audit logs with optional filters."""
    check_scope(api_key, "admin")

    stmt = select(AuditLog).order_by(AuditLog.created_at.desc())

    if api_key_id is not None:
        stmt = stmt.where(AuditLog.api_key_id == api_key_id)
    if document_id is not None:
        stmt = stmt.where(AuditLog.document_id == document_id)

    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    logs = list(result.scalars().all())

    # Get total count
    count_stmt = select(func.count(AuditLog.id))
    if api_key_id is not None:
        count_stmt = count_stmt.where(
            AuditLog.api_key_id == api_key_id,
        )
    if document_id is not None:
        count_stmt = count_stmt.where(
            AuditLog.document_id == document_id,
        )
    total = (await session.execute(count_stmt)).scalar() or 0

    return AuditLogListResponse(
        logs=[
            AuditLogResponse(
                id=log.id,
                api_key_id=log.api_key_id,
                api_key_name=log.api_key_name,
                endpoint=log.endpoint,
                method=log.method,
                path=log.path,
                document_id=log.document_id,
                question=log.question,
                client_ip=log.client_ip,
                status_code=log.status_code,
                response_time_ms=log.response_time_ms,
                created_at=log.created_at,
            )
            for log in logs
        ],
        total=total,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_key_or_404(
    session: AsyncSession, key_id: int,
) -> ApiKey:
    """Load an ApiKey by ID or raise 404."""
    stmt = select(ApiKey).where(ApiKey.id == key_id)
    result = await session.execute(stmt)
    api_key = result.scalar_one_or_none()
    if api_key is None:
        raise HTTPException(
            status_code=404, detail=f"API key {key_id} not found.",
        )
    return api_key


def _to_key_response(key: ApiKey) -> ApiKeyResponse:
    """Convert an ApiKey ORM model to a response (excluding sensitive data)."""
    return ApiKeyResponse(
        id=key.id,
        name=key.name,
        key_prefix=key.key_prefix,
        scopes=key.scopes,
        rate_limit_rpm=key.rate_limit_rpm,
        all_documents_access=key.all_documents_access,
        is_active=key.is_active,
        document_ids=[ad.document_id for ad in key.allowed_documents],
        created_at=key.created_at,
        expires_at=key.expires_at,
        last_used_at=key.last_used_at,
    )
