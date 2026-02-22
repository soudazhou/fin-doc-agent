# =============================================================================
# Unit Tests — Benchmarking & Comparison (Phase 4)
# =============================================================================
#
# Tests the pure-logic components without requiring API keys, databases,
# or real LLM providers. Uses mock objects and direct function calls.
# =============================================================================

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.orchestrator import ask
from app.services.llm import LLMResponse, _parse_provider_id
from app.services.pricing import (
    PRICING_REGISTRY,
    ModelPricing,
    estimate_cost,
    get_pricing,
)


def _run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Test: Pricing Registry
# ---------------------------------------------------------------------------


class TestPricingRegistry:
    """Tests for the provider pricing module."""

    def test_registry_has_entries(self):
        assert len(PRICING_REGISTRY) > 0

    def test_known_model_returns_cost(self):
        # Claude Sonnet: $3/1M input, $15/1M output
        # 1M input + 200K output = $3.00 + $3.00 = $6.00
        cost = estimate_cost(
            "anthropic", "claude-sonnet-4-6", 1_000_000, 200_000,
        )
        assert cost is not None
        assert abs(cost - 6.00) < 0.001

    def test_deepseek_cost(self):
        # DeepSeek V3: $0.14/1M input, $0.28/1M output
        # 500K input + 100K output = $0.07 + $0.028 = $0.098
        cost = estimate_cost(
            "openai_compatible", "deepseek-chat", 500_000, 100_000,
        )
        assert cost is not None
        assert abs(cost - 0.098) < 0.001

    def test_unknown_model_returns_none(self):
        cost = estimate_cost("unknown", "nonexistent-model", 100, 100)
        assert cost is None

    def test_zero_tokens_returns_zero(self):
        cost = estimate_cost("anthropic", "claude-sonnet-4-6", 0, 0)
        assert cost == 0.0

    def test_get_pricing_returns_model_pricing(self):
        pricing = get_pricing("anthropic", "claude-sonnet-4-6")
        assert isinstance(pricing, ModelPricing)
        assert pricing.provider_label == "Anthropic"

    def test_get_pricing_unknown_returns_none(self):
        pricing = get_pricing("unknown", "model")
        assert pricing is None


# ---------------------------------------------------------------------------
# Test: Provider ID Parsing
# ---------------------------------------------------------------------------


class TestParseProviderId:
    """Tests for _parse_provider_id() — the pure parsing function."""

    def test_anthropic_simple(self):
        ptype, model, base_url = _parse_provider_id(
            "anthropic/claude-sonnet-4-6",
        )
        assert ptype == "anthropic"
        assert model == "claude-sonnet-4-6"
        assert base_url is None

    def test_openai_compatible_no_url(self):
        ptype, model, base_url = _parse_provider_id(
            "openai_compatible/deepseek-chat",
        )
        assert ptype == "openai_compatible"
        assert model == "deepseek-chat"
        assert base_url is None

    def test_openai_compatible_with_url(self):
        ptype, model, base_url = _parse_provider_id(
            "openai_compatible/deepseek-chat@https://api.deepseek.com/v1",
        )
        assert ptype == "openai_compatible"
        assert model == "deepseek-chat"
        assert base_url == "https://api.deepseek.com/v1"

    def test_invalid_no_slash(self):
        with pytest.raises(ValueError, match="Invalid provider_id"):
            _parse_provider_id("no-slash-here")

    def test_unknown_provider_type(self):
        with pytest.raises(ValueError, match="Unknown provider type"):
            _parse_provider_id("gemini/gemini-pro")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Invalid provider_id"):
            _parse_provider_id("")


# ---------------------------------------------------------------------------
# Test: Provider Factory (create_provider_from_id)
# ---------------------------------------------------------------------------


class TestCreateProviderFromId:
    """Tests for create_provider_from_id() — validates it calls constructors."""

    def test_anthropic_without_key_raises(self):
        from app.services import llm

        with patch.object(
            llm.settings, "llm_api_key", None
        ), patch.object(
            llm.settings, "anthropic_api_key", ""
        ), pytest.raises(ValueError, match="API key"):
            from app.services.llm import create_provider_from_id
            create_provider_from_id("anthropic/claude-sonnet-4-6")

    def test_openai_compatible_without_key_raises(self):
        from app.services import llm

        with patch.object(
            llm.settings, "llm_api_key", None
        ), patch.object(
            llm.settings, "openai_api_key", ""
        ), pytest.raises(ValueError, match="API key"):
            from app.services.llm import create_provider_from_id
            create_provider_from_id("openai_compatible/gpt-4o")

    def test_invalid_format_raises(self):
        from app.services.llm import create_provider_from_id
        with pytest.raises(ValueError):
            create_provider_from_id("bad-format")


# ---------------------------------------------------------------------------
# Test: Winner Computation
# ---------------------------------------------------------------------------


class TestComputeWinner:
    """Tests for the _compute_winner helper in benchmark.py."""

    def test_fastest_selected(self):
        from app.api.benchmark import _compute_winner
        from app.models.responses import ProviderResult

        results = [
            ProviderResult(
                provider_id="a/fast", answer="x", sources=[], model="m",
                latency_ms=100, input_tokens=10, output_tokens=5,
                search_iterations=1, retrieval_count=3,
                estimated_cost_usd=0.01,
            ),
            ProviderResult(
                provider_id="b/slow", answer="y", sources=[], model="m",
                latency_ms=500, input_tokens=10, output_tokens=5,
                search_iterations=1, retrieval_count=3,
                estimated_cost_usd=0.05,
            ),
        ]
        winner = _compute_winner(results)
        assert winner.fastest_provider == "a/fast"
        assert winner.cheapest_provider == "a/fast"

    def test_cheapest_selected(self):
        from app.api.benchmark import _compute_winner
        from app.models.responses import ProviderResult

        results = [
            ProviderResult(
                provider_id="a/expensive", answer="x", sources=[],
                model="m", latency_ms=100, input_tokens=10,
                output_tokens=5, search_iterations=1, retrieval_count=3,
                estimated_cost_usd=1.00,
            ),
            ProviderResult(
                provider_id="b/cheap", answer="y", sources=[],
                model="m", latency_ms=200, input_tokens=10,
                output_tokens=5, search_iterations=1, retrieval_count=3,
                estimated_cost_usd=0.01,
            ),
        ]
        winner = _compute_winner(results)
        assert winner.fastest_provider == "a/expensive"
        assert winner.cheapest_provider == "b/cheap"

    def test_all_errored_returns_empty(self):
        from app.api.benchmark import _compute_winner
        from app.models.responses import ProviderResult

        results = [
            ProviderResult(
                provider_id="a/broken", answer="", sources=[],
                model="error", latency_ms=50, input_tokens=0,
                output_tokens=0, search_iterations=0, retrieval_count=0,
                error="connection failed",
            ),
        ]
        winner = _compute_winner(results)
        assert winner.fastest_provider is None
        assert winner.cheapest_provider is None

    def test_no_cost_data_cheapest_is_none(self):
        from app.api.benchmark import _compute_winner
        from app.models.responses import ProviderResult

        results = [
            ProviderResult(
                provider_id="a/unknown", answer="x", sources=[],
                model="m", latency_ms=100, input_tokens=10,
                output_tokens=5, search_iterations=1, retrieval_count=3,
                estimated_cost_usd=None,
            ),
        ]
        winner = _compute_winner(results)
        assert winner.fastest_provider == "a/unknown"
        assert winner.cheapest_provider is None


# ---------------------------------------------------------------------------
# Test: Orchestrator LLM Injection
# ---------------------------------------------------------------------------


class TestOrchestratorLLMInjection:
    """Verify that ask(llm=...) passes the injected provider through."""

    def test_injected_llm_used_in_search_and_analyse(self):
        """
        When ask(llm=mock) is called, the mock LLM should be used
        by both search_node and analyse_node, NOT the global singleton.
        """
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = LLMResponse(
            content="Mock answer",
            model="mock-model",
            input_tokens=50,
            output_tokens=10,
        )

        # Mock the vector store to return empty results
        # (this avoids needing a real embedding/DB)
        mock_vs = AsyncMock()
        mock_vs.search.return_value = []

        with patch(
            "app.agents.search.get_vector_store", return_value=mock_vs,
        ), patch(
            "app.agents.search.embed_query", return_value=[0.1] * 1536,
        ), patch(
            "app.agents.orchestrator.get_llm_provider"
        ) as mock_singleton:
            result = _run(ask(
                question="What was revenue?",
                llm=mock_llm,
            ))

            # The singleton should NOT have been called
            mock_singleton.assert_not_called()

        # With empty search results, analyst returns "No relevant info"
        # without calling LLM — so answer comes from the empty-chunks path
        assert "No relevant information" in result.get("answer", "")


# ---------------------------------------------------------------------------
# Test: Pydantic Model Validation
# ---------------------------------------------------------------------------


class TestRequestValidation:
    """Tests for Phase 4 request model validation."""

    def test_compare_request_requires_min_2_providers(self):
        from pydantic import ValidationError

        from app.models.requests import CompareRequest

        with pytest.raises(ValidationError):
            CompareRequest(
                question="test question",
                providers=["only-one/provider"],
            )

    def test_compare_request_allows_max_5_providers(self):
        from pydantic import ValidationError

        from app.models.requests import CompareRequest

        with pytest.raises(ValidationError):
            CompareRequest(
                question="test question",
                providers=["a/1", "b/2", "c/3", "d/4", "e/5", "f/6"],
            )

    def test_compare_request_valid(self):
        from app.models.requests import CompareRequest

        req = CompareRequest(
            question="What was revenue?",
            providers=["anthropic/claude-sonnet-4-6", "openai_compatible/gpt-4o"],
        )
        assert len(req.providers) == 2
        assert req.capability is None

    def test_benchmark_retrieval_defaults(self):
        from app.models.requests import BenchmarkRetrievalRequest

        req = BenchmarkRetrievalRequest()
        assert len(req.sample_queries) == 5
        assert req.top_k_values == [3, 5, 10]
        assert req.vector_stores == ["pgvector"]
