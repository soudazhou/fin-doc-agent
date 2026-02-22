# =============================================================================
# Unit Tests — Authorization & Security (Phase 6)
# =============================================================================
#
# Tests auth components without requiring Redis, a running API, or real API keys.
# Uses mocking for external dependencies (Redis, DB sessions).
#
# Test groups:
#   1. Key generation & hashing (pure functions)
#   2. Auth dependency (get_current_api_key)
#   3. Scope checking (check_scope)
#   4. Document ACL (check_document_access)
#   5. Rate limiter (check_rate_limit)
#   6. Request/response model validation
# =============================================================================

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.services.auth import generate_api_key, hash_api_key


def _run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. Key Generation & Hashing
# ---------------------------------------------------------------------------


class TestKeyGeneration:
    """Tests for API key generation and hashing."""

    def test_key_format_has_prefix(self):
        """Generated key starts with 'sk-'."""
        raw_key, prefix, key_hash = generate_api_key()
        assert raw_key.startswith("sk-")

    def test_key_length(self):
        """Generated key is 'sk-' + 64 hex chars = 67 chars total."""
        raw_key, prefix, key_hash = generate_api_key()
        assert len(raw_key) == 67  # "sk-" (3) + 64 hex chars

    def test_prefix_is_first_8_chars(self):
        """Key prefix is the first 8 characters of the raw key."""
        raw_key, prefix, key_hash = generate_api_key()
        assert prefix == raw_key[:8]

    def test_hash_is_64_hex(self):
        """Key hash is a 64-char hex digest (SHA-256)."""
        raw_key, prefix, key_hash = generate_api_key()
        assert len(key_hash) == 64
        # Verify it's valid hex
        int(key_hash, 16)

    def test_keys_are_unique(self):
        """Two generated keys should differ."""
        raw1, _, hash1 = generate_api_key()
        raw2, _, hash2 = generate_api_key()
        assert raw1 != raw2
        assert hash1 != hash2

    def test_hash_is_deterministic(self):
        """Same raw key always produces the same hash."""
        raw_key = "sk-abc123"
        assert hash_api_key(raw_key) == hash_api_key(raw_key)

    def test_different_keys_different_hashes(self):
        """Different raw keys produce different hashes."""
        assert hash_api_key("sk-key1") != hash_api_key("sk-key2")


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes for auth dependency tests
# ---------------------------------------------------------------------------


@dataclass
class FakeApiKey:
    """Lightweight stand-in for the ApiKey ORM model."""

    id: int = 1
    name: str = "test-key"
    key_prefix: str = "sk-test0"
    key_hash: str = ""
    scopes: list[str] | None = None
    rate_limit_rpm: int | None = None
    all_documents_access: bool = True
    is_active: bool = True
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    created_at: datetime = datetime.now(UTC)
    allowed_documents: list = field(default_factory=list)


@dataclass
class FakeAclEntry:
    """Lightweight stand-in for ApiKeyDocument."""

    document_id: int = 1


@dataclass
class FakeCredentials:
    """Stand-in for HTTPAuthorizationCredentials."""

    credentials: str = "sk-testkey"


class FakeRequestState:
    """Writable request.state."""

    pass


class FakeRequest:
    """Minimal Request stand-in."""

    def __init__(self):
        self.state = FakeRequestState()


# ---------------------------------------------------------------------------
# 2. Auth Dependency (get_current_api_key)
# ---------------------------------------------------------------------------


class TestGetCurrentApiKey:
    """Tests for the get_current_api_key dependency."""

    def test_auth_disabled_returns_none(self):
        """When auth_enabled=False, returns None (anonymous access)."""
        from app.api.deps import get_current_api_key

        with patch("app.api.deps.settings") as mock_settings:
            mock_settings.auth_enabled = False
            result = _run(get_current_api_key(
                request=FakeRequest(),
                credentials=None,
                session=AsyncMock(),
            ))
            assert result is None

    def test_missing_credentials_raises_401(self):
        """When auth enabled and no header, raises 401."""
        from app.api.deps import get_current_api_key

        with patch("app.api.deps.settings") as mock_settings:
            mock_settings.auth_enabled = True
            with pytest.raises(HTTPException) as exc_info:
                _run(get_current_api_key(
                    request=FakeRequest(),
                    credentials=None,
                    session=AsyncMock(),
                ))
            assert exc_info.value.status_code == 401

    def test_invalid_key_raises_401(self):
        """When key not found in DB, raises 401."""
        from app.api.deps import get_current_api_key

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with patch("app.api.deps.settings") as mock_settings:
            mock_settings.auth_enabled = True
            with pytest.raises(HTTPException) as exc_info:
                _run(get_current_api_key(
                    request=FakeRequest(),
                    credentials=FakeCredentials(),
                    session=mock_session,
                ))
            assert exc_info.value.status_code == 401

    def test_inactive_key_raises_403(self):
        """When key is inactive, raises 403."""
        from app.api.deps import get_current_api_key

        fake_key = FakeApiKey(is_active=False)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_key
        mock_session.execute.return_value = mock_result

        with patch("app.api.deps.settings") as mock_settings:
            mock_settings.auth_enabled = True
            with pytest.raises(HTTPException) as exc_info:
                _run(get_current_api_key(
                    request=FakeRequest(),
                    credentials=FakeCredentials(),
                    session=mock_session,
                ))
            assert exc_info.value.status_code == 403
            assert "deactivated" in exc_info.value.detail

    def test_expired_key_raises_403(self):
        """When key has expired, raises 403."""
        from app.api.deps import get_current_api_key

        fake_key = FakeApiKey(
            is_active=True,
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_key
        mock_session.execute.return_value = mock_result

        with patch("app.api.deps.settings") as mock_settings:
            mock_settings.auth_enabled = True
            with pytest.raises(HTTPException) as exc_info:
                _run(get_current_api_key(
                    request=FakeRequest(),
                    credentials=FakeCredentials(),
                    session=mock_session,
                ))
            assert exc_info.value.status_code == 403
            assert "expired" in exc_info.value.detail

    def test_valid_key_returns_api_key(self):
        """When key is valid, returns the ApiKey and updates last_used_at."""
        from app.api.deps import get_current_api_key

        fake_key = FakeApiKey(is_active=True, expires_at=None)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_key
        mock_session.execute.return_value = mock_result

        request = FakeRequest()

        with (
            patch("app.api.deps.settings") as mock_settings,
            patch("app.services.rate_limiter.check_rate_limit", new_callable=AsyncMock),
        ):
            mock_settings.auth_enabled = True
            result = _run(get_current_api_key(
                request=request,
                credentials=FakeCredentials(),
                session=mock_session,
            ))

        assert result is fake_key
        assert fake_key.last_used_at is not None
        assert request.state.api_key is fake_key


# ---------------------------------------------------------------------------
# 3. Scope Checking
# ---------------------------------------------------------------------------


class TestCheckScope:
    """Tests for the check_scope function."""

    def test_none_api_key_passes(self):
        """When auth disabled (api_key=None), scope check is a no-op."""
        from app.api.deps import check_scope
        check_scope(None, "admin")  # Should not raise

    def test_null_scopes_means_full_access(self):
        """Key with null scopes has full access to everything."""
        from app.api.deps import check_scope
        key = FakeApiKey(scopes=None)
        check_scope(key, "admin")  # Should not raise

    def test_empty_scopes_means_full_access(self):
        """Key with empty scopes list has full access."""
        from app.api.deps import check_scope
        key = FakeApiKey(scopes=[])
        check_scope(key, "admin")  # Should not raise

    def test_matching_scope_passes(self):
        """Key with the required scope passes."""
        from app.api.deps import check_scope
        key = FakeApiKey(scopes=["ask", "ingest"])
        check_scope(key, "ask")  # Should not raise

    def test_missing_scope_raises_403(self):
        """Key without the required scope raises 403."""
        from app.api.deps import check_scope
        key = FakeApiKey(scopes=["ask"])
        with pytest.raises(HTTPException) as exc_info:
            check_scope(key, "admin")
        assert exc_info.value.status_code == 403
        assert "admin" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 4. Document ACL
# ---------------------------------------------------------------------------


class TestCheckDocumentAccess:
    """Tests for the check_document_access function."""

    def test_none_api_key_passes(self):
        """When auth disabled (api_key=None), ACL check is a no-op."""
        from app.api.deps import check_document_access
        check_document_access(None, 1)  # Should not raise

    def test_all_documents_access_passes(self):
        """Key with all_documents_access=True bypasses ACL."""
        from app.api.deps import check_document_access
        key = FakeApiKey(all_documents_access=True)
        check_document_access(key, 42)  # Should not raise

    def test_allowed_document_passes(self):
        """Key with specific document in ACL passes."""
        from app.api.deps import check_document_access
        key = FakeApiKey(
            all_documents_access=False,
            allowed_documents=[FakeAclEntry(document_id=5)],
        )
        check_document_access(key, 5)  # Should not raise

    def test_disallowed_document_raises_403(self):
        """Key without document access raises 403."""
        from app.api.deps import check_document_access
        key = FakeApiKey(
            all_documents_access=False,
            allowed_documents=[FakeAclEntry(document_id=5)],
        )
        with pytest.raises(HTTPException) as exc_info:
            check_document_access(key, 99)
        assert exc_info.value.status_code == 403

    def test_restricted_key_without_doc_id_raises_403(self):
        """Restricted key querying all docs (doc_id=None) raises 403."""
        from app.api.deps import check_document_access
        key = FakeApiKey(
            all_documents_access=False,
            allowed_documents=[FakeAclEntry(document_id=5)],
        )
        with pytest.raises(HTTPException) as exc_info:
            check_document_access(key, None)
        assert exc_info.value.status_code == 403
        assert "document_id" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 5. Rate Limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """Tests for the Redis-based rate limiter."""

    def test_none_api_key_skips(self):
        """When api_key is None, rate limit is skipped."""
        from app.services.rate_limiter import check_rate_limit
        _run(check_rate_limit(None))  # Should not raise

    def test_under_limit_passes(self):
        """When under the limit, request passes through."""
        from app.services.rate_limiter import check_rate_limit

        fake_key = FakeApiKey(rate_limit_rpm=100)

        mock_pipe = AsyncMock()
        mock_pipe.execute.return_value = [None, 5, None, None]  # zcard=5

        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        with patch(
            "app.services.rate_limiter._get_rate_limit_redis",
            return_value=mock_redis,
        ):
            _run(check_rate_limit(fake_key))  # Should not raise

    def test_over_limit_raises_429(self):
        """When at or over the limit, raises 429."""
        from app.services.rate_limiter import check_rate_limit

        fake_key = FakeApiKey(rate_limit_rpm=10)

        mock_pipe = AsyncMock()
        mock_pipe.execute.return_value = [None, 10, None, None]

        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        with patch(
            "app.services.rate_limiter._get_rate_limit_redis",
            return_value=mock_redis,
        ):
            with pytest.raises(HTTPException) as exc_info:
                _run(check_rate_limit(fake_key))
            assert exc_info.value.status_code == 429
            assert "Rate limit" in exc_info.value.detail

    def test_redis_unavailable_allows_through(self):
        """When Redis is down, request is allowed (graceful degradation)."""
        from app.services.rate_limiter import check_rate_limit

        fake_key = FakeApiKey(rate_limit_rpm=10)

        with patch(
            "app.services.rate_limiter._get_rate_limit_redis",
            side_effect=ConnectionError("Redis down"),
        ):
            _run(check_rate_limit(fake_key))  # Should not raise


# ---------------------------------------------------------------------------
# 6. Request/Response Model Validation
# ---------------------------------------------------------------------------


class TestRequestModels:
    """Tests for Pydantic request model validation."""

    def test_create_key_requires_name(self):
        """CreateApiKeyRequest requires a name."""
        from pydantic import ValidationError

        from app.models.requests import CreateApiKeyRequest

        with pytest.raises(ValidationError):
            CreateApiKeyRequest()

    def test_create_key_valid(self):
        """CreateApiKeyRequest with name is valid."""
        from app.models.requests import CreateApiKeyRequest

        req = CreateApiKeyRequest(name="test-key")
        assert req.name == "test-key"
        assert req.scopes is None
        assert req.all_documents_access is True

    def test_update_key_all_optional(self):
        """UpdateApiKeyRequest accepts no fields (all optional)."""
        from app.models.requests import UpdateApiKeyRequest

        req = UpdateApiKeyRequest()
        assert req.name is None
        assert req.is_active is None

    def test_grant_docs_requires_ids(self):
        """GrantDocumentAccessRequest requires document_ids."""
        from pydantic import ValidationError

        from app.models.requests import GrantDocumentAccessRequest

        with pytest.raises(ValidationError):
            GrantDocumentAccessRequest()


class TestResponseModels:
    """Tests for Pydantic response model serialization."""

    def test_api_key_response_excludes_hash(self):
        """ApiKeyResponse schema does not include key_hash."""
        from app.models.responses import ApiKeyResponse

        schema = ApiKeyResponse.model_json_schema()
        assert "key_hash" not in schema.get("properties", {})

    def test_api_key_created_response_includes_raw_key(self):
        """ApiKeyCreatedResponse includes raw_key field."""
        from app.models.responses import ApiKeyCreatedResponse

        schema = ApiKeyCreatedResponse.model_json_schema()
        assert "raw_key" in schema.get("properties", {})
