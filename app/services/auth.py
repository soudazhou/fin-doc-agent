# =============================================================================
# Auth Service — API Key Generation & Hashing
# =============================================================================
#
# Pure functions for API key management. No FastAPI dependency — this module
# can be used by the auth dependency, admin endpoints, and tests.
#
# DESIGN DECISION: SHA-256 hashing (not bcrypt). API keys are 32-byte
# random tokens (256 bits of entropy). SHA-256 is:
# - Secure for high-entropy secrets (no rainbow table risk)
# - Fast enough for per-request validation
# - Deterministic (same input → same hash, needed for DB lookup)
#
# Bcrypt is designed for low-entropy human passwords where deliberate
# slowness defends against brute-force. For random API keys, SHA-256
# provides equivalent security at a fraction of the latency.
# =============================================================================

from __future__ import annotations

import hashlib
import secrets


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        (raw_key, key_prefix, key_hash):
        - raw_key: Full key to return to the user (only visible once)
        - key_prefix: First 8 chars for identification in logs/admin
        - key_hash: SHA-256 hex digest for storage in the database
    """
    raw_key = f"sk-{secrets.token_hex(32)}"
    key_prefix = raw_key[:8]
    key_hash = hash_api_key(raw_key)
    return raw_key, key_prefix, key_hash


def hash_api_key(raw_key: str) -> str:
    """Hash an API key using SHA-256. Returns 64-char hex digest."""
    return hashlib.sha256(raw_key.encode()).hexdigest()
