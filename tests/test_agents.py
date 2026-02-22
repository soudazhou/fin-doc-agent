# =============================================================================
# Unit Tests — Agents (Phase 3)
# =============================================================================
#
# Tests the agent components without requiring API keys or databases.
# Uses mock LLM providers and mock vector search results.
# =============================================================================

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

from app.agents.analyst import AnalysisResult, _format_context, analyse
from app.agents.orchestrator import _classify_question
from app.agents.search import SearchResult, _simplify_query
from app.services.llm import LLMResponse
from app.services.vectorstore import VectorSearchResult


def _run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Test: Intent Classification
# ---------------------------------------------------------------------------


class TestClassifyQuestion:
    """Tests for rule-based intent classification."""

    def test_qa_is_default(self):
        assert _classify_question("what was total revenue?") == "qa"

    def test_qa_for_specific_question(self):
        assert _classify_question("how much did they earn?") == "qa"

    def test_summarise_detected(self):
        assert _classify_question("summarise the report") == "summarise"

    def test_summary_keyword(self):
        assert _classify_question("give me a summary") == "summarise"

    def test_overview_keyword(self):
        assert _classify_question("provide an overview") == "summarise"

    def test_compare_detected(self):
        assert _classify_question("compare revenue vs expenses") == "compare"

    def test_versus_keyword(self):
        assert _classify_question("aapl versus msft revenue") == "compare"

    def test_difference_between(self):
        result = _classify_question("difference between q1 and q2")
        assert result == "compare"

    def test_extract_detected(self):
        assert _classify_question("extract all revenue figures") == "extract"

    def test_list_all_keyword(self):
        result = _classify_question("list all quarterly metrics")
        assert result == "extract"

    def test_as_json_keyword(self):
        assert _classify_question("give me the data as json") == "extract"

    def test_table_of_keyword(self):
        result = _classify_question("create a table of expenses")
        assert result == "extract"


# ---------------------------------------------------------------------------
# Test: Context Formatting
# ---------------------------------------------------------------------------


class TestFormatContext:
    """Tests for the analyst's context formatting."""

    def test_single_chunk(self):
        chunks = [
            VectorSearchResult(
                chunk_id=1,
                content="Revenue was $4.2B.",
                page_number=12,
                similarity_score=0.95,
            )
        ]
        result = _format_context(chunks)
        assert "[1] (page 12):" in result
        assert "Revenue was $4.2B." in result

    def test_multiple_chunks(self):
        chunks = [
            VectorSearchResult(
                chunk_id=1, content="Chunk A", page_number=1,
                similarity_score=0.9,
            ),
            VectorSearchResult(
                chunk_id=2, content="Chunk B", page_number=None,
                similarity_score=0.8,
            ),
        ]
        result = _format_context(chunks)
        assert "[1] (page 1):" in result
        assert "[2]:" in result  # No page number
        assert "---" in result  # Separator

    def test_empty_chunks(self):
        result = _format_context([])
        assert result == ""


# ---------------------------------------------------------------------------
# Test: Query Simplification
# ---------------------------------------------------------------------------


class TestSimplifyQuery:
    """Tests for the fallback query simplification."""

    def test_removes_question_prefix(self):
        result = _simplify_query("what is the total revenue?")
        assert "what is" not in result
        assert "revenue" in result

    def test_preserves_key_terms(self):
        result = _simplify_query("how much revenue in q3 2024?")
        assert "revenue" in result
        assert "q3 2024" in result

    def test_empty_after_simplification_returns_original(self):
        result = _simplify_query("what is")
        # After removing "what is", nothing left — returns original
        assert result == "what is"


# ---------------------------------------------------------------------------
# Test: Analyst with Mock LLM
# ---------------------------------------------------------------------------


class TestAnalyse:
    """Tests for the analyst agent with mock LLM."""

    def test_empty_chunks_returns_no_info_message(self):
        mock_llm = AsyncMock()
        result = _run(analyse(
            question="What was revenue?",
            chunks=[],
            capability="qa",
            llm=mock_llm,
        ))
        assert isinstance(result, AnalysisResult)
        assert "No relevant information" in result.answer
        assert result.model == "n/a"
        # LLM should NOT be called when there are no chunks
        mock_llm.complete.assert_not_called()

    def test_qa_calls_llm_with_system_prompt(self):
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = LLMResponse(
            content="Revenue was $4.2B [1].",
            model="test-model",
            input_tokens=100,
            output_tokens=20,
        )

        chunks = [
            VectorSearchResult(
                chunk_id=1,
                content="Total revenue: $4.2 billion.",
                page_number=5,
                similarity_score=0.95,
            )
        ]

        result = _run(analyse(
            question="What was revenue?",
            chunks=chunks,
            capability="qa",
            llm=mock_llm,
        ))

        assert result.answer == "Revenue was $4.2B [1]."
        assert result.model == "test-model"
        mock_llm.complete.assert_called_once()

        # Verify system prompt contains QA-specific instructions
        call_kwargs = mock_llm.complete.call_args
        assert "ONLY the provided context" in call_kwargs.kwargs["system"]

    def test_summarise_uses_different_prompt(self):
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = LLMResponse(
            content="Summary of the document.",
            model="test-model",
            input_tokens=100,
            output_tokens=30,
        )

        chunks = [
            VectorSearchResult(
                chunk_id=1, content="Content here.",
                page_number=1, similarity_score=0.9,
            )
        ]

        _run(analyse(
            question="Summarise the report",
            chunks=chunks,
            capability="summarise",
            llm=mock_llm,
        ))

        call_kwargs = mock_llm.complete.call_args
        assert "structured summary" in call_kwargs.kwargs["system"]


# ---------------------------------------------------------------------------
# Test: LLM Provider Factory
# ---------------------------------------------------------------------------


class TestLLMProviderFactory:
    """Tests for the LLM provider factory function."""

    def test_factory_raises_without_api_key(self):
        """Factory should raise ValueError when no API key is set."""
        from app.services import llm

        # Reset the singleton
        original = llm._provider
        llm._provider = None

        try:
            with patch.object(
                llm.settings, "llm_provider", "anthropic"
            ), patch.object(
                llm.settings, "llm_api_key", None
            ), patch.object(
                llm.settings, "anthropic_api_key", ""
            ):
                try:
                    llm.get_llm_provider()
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "API key" in str(e)
        finally:
            # Restore singleton
            llm._provider = original
