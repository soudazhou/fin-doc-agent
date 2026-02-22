# =============================================================================
# Provider Pricing Registry — Cost Estimation for LLM Providers
# =============================================================================
#
# Maps (provider_type, model_name) → per-token costs in USD.
# Used by the /compare endpoint to estimate cost per provider leg
# and by the /metrics endpoint for cost accounting.
#
# DESIGN DECISION: Static dict rather than database or config file.
# 1. Pricing changes rarely — a code update is acceptable
# 2. Zero latency — no I/O on every cost calculation
# 3. Testable without infrastructure
# 4. Versionable in git — pricing history in the commit log
#
# DESIGN DECISION: Costs stored as USD per TOKEN (not per 1M tokens).
# This avoids floating-point division on every call:
#   cost = input_cost_per_token * input_tokens  (no /1_000_000)
#
# DESIGN DECISION: estimate_cost() returns None for unknown models
# rather than 0.0. Unknown cost != zero cost — None makes the unknown
# state explicit and visible in API responses.
#
# Source: Provider pricing pages as of February 2026.
# Update this dict when prices change.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """Per-token costs for a model."""

    input_cost_per_token: float    # USD per input token
    output_cost_per_token: float   # USD per output token
    provider_label: str            # Human-readable provider name


# ---------------------------------------------------------------------------
# Pricing Registry
# ---------------------------------------------------------------------------
# Keys are (provider_type, model_name) tuples.
# provider_type matches the prefix in provider_id strings used by
# create_provider_from_id(): "anthropic" or "openai_compatible".
# ---------------------------------------------------------------------------

PRICING_REGISTRY: dict[tuple[str, str], ModelPricing] = {
    # --- Anthropic ---
    ("anthropic", "claude-sonnet-4-6"): ModelPricing(
        3.00 / 1_000_000, 15.00 / 1_000_000, "Anthropic",
    ),
    ("anthropic", "claude-opus-4-6"): ModelPricing(
        15.00 / 1_000_000, 75.00 / 1_000_000, "Anthropic",
    ),
    ("anthropic", "claude-haiku-4-5"): ModelPricing(
        0.80 / 1_000_000, 4.00 / 1_000_000, "Anthropic",
    ),

    # --- OpenAI ---
    ("openai_compatible", "gpt-4o"): ModelPricing(
        2.50 / 1_000_000, 10.00 / 1_000_000, "OpenAI",
    ),
    ("openai_compatible", "gpt-4o-mini"): ModelPricing(
        0.15 / 1_000_000, 0.60 / 1_000_000, "OpenAI",
    ),

    # --- DeepSeek ---
    ("openai_compatible", "deepseek-chat"): ModelPricing(
        0.14 / 1_000_000, 0.28 / 1_000_000, "DeepSeek",
    ),
    ("openai_compatible", "deepseek-reasoner"): ModelPricing(
        0.55 / 1_000_000, 2.19 / 1_000_000, "DeepSeek",
    ),

    # --- Qwen (Alibaba Cloud) ---
    ("openai_compatible", "qwen-plus"): ModelPricing(
        0.11 / 1_000_000, 0.44 / 1_000_000, "Alibaba Cloud",
    ),
    ("openai_compatible", "qwen-max"): ModelPricing(
        1.60 / 1_000_000, 6.40 / 1_000_000, "Alibaba Cloud",
    ),

    # --- Zhipu AI (GLM) ---
    ("openai_compatible", "glm-5"): ModelPricing(
        1.00 / 1_000_000, 3.20 / 1_000_000, "Zhipu AI",
    ),

    # --- MiniMax ---
    ("openai_compatible", "minimax-m2.5"): ModelPricing(
        0.20 / 1_000_000, 1.00 / 1_000_000, "MiniMax",
    ),

    # --- Moonshot AI (Kimi) ---
    ("openai_compatible", "kimi-k2.5"): ModelPricing(
        0.60 / 1_000_000, 2.50 / 1_000_000, "Moonshot AI",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_cost(
    provider_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """
    Calculate estimated cost in USD for a completion.

    Returns None if the model is not in the registry (unknown pricing).
    Callers should handle None gracefully — unknown cost != zero cost.

    Args:
        provider_type: "anthropic" or "openai_compatible".
        model: Model name as returned by the LLM API.
        input_tokens: Tokens consumed by the prompt.
        output_tokens: Tokens generated in the response.

    Returns:
        Estimated cost in USD, or None if model not in registry.
    """
    pricing = PRICING_REGISTRY.get((provider_type, model))
    if pricing is None:
        return None
    return (
        pricing.input_cost_per_token * input_tokens
        + pricing.output_cost_per_token * output_tokens
    )


def get_pricing(provider_type: str, model: str) -> ModelPricing | None:
    """Look up pricing for a specific provider+model combination."""
    return PRICING_REGISTRY.get((provider_type, model))
