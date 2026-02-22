# =============================================================================
# Unit Tests — Evaluation Runner (Phase 5)
# =============================================================================
#
# Tests the eval runner's pure logic without requiring API keys, databases,
# or real LLM providers. Uses mock objects and direct function calls.
# =============================================================================

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.eval_runner import (
    GoldenDataset,
    GoldenTestCase,
    MetricScore,
    TestCaseResult,
    compute_custom_metrics,
    determine_pass_fail,
    load_dataset,
    run_test_case,
)


def _run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Test: Dataset Loading
# ---------------------------------------------------------------------------


class TestLoadDataset:
    """Tests for load_dataset() — golden dataset parsing."""

    def test_default_dataset_loads(self):
        """The default golden dataset should load without errors."""
        dataset = load_dataset("default")
        assert isinstance(dataset, GoldenDataset)
        assert dataset.dataset_name == "default"
        assert len(dataset.test_cases) >= 29
        assert len(dataset.thresholds) > 0

    def test_missing_dataset_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_dataset("nonexistent_dataset_xyz")

    def test_malformed_dataset_raises(self, tmp_path):
        """A dataset missing required fields should raise ValueError."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps({"dataset_name": "bad"}))

        from app.services import eval_runner

        original_dir = eval_runner.settings.eval_dataset_dir
        eval_runner.settings.eval_dataset_dir = str(tmp_path)
        try:
            with pytest.raises(ValueError, match="missing required"):
                load_dataset("bad")
        finally:
            eval_runner.settings.eval_dataset_dir = original_dir

    def test_empty_test_cases_raises(self, tmp_path):
        """A dataset with empty test_cases should raise ValueError."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text(json.dumps({
            "dataset_name": "empty",
            "thresholds": {"faithfulness": 0.7},
            "test_cases": [],
        }))

        from app.services import eval_runner

        original_dir = eval_runner.settings.eval_dataset_dir
        eval_runner.settings.eval_dataset_dir = str(tmp_path)
        try:
            with pytest.raises(ValueError, match="no test cases"):
                load_dataset("empty")
        finally:
            eval_runner.settings.eval_dataset_dir = original_dir


# ---------------------------------------------------------------------------
# Test: Test Case Execution
# ---------------------------------------------------------------------------


class TestRunTestCase:
    """Tests for run_test_case() — calling ask() and collecting output."""

    def test_successful_execution(self):
        """Mock ask() returns a valid result → TestCaseResult populated."""
        mock_search_result = MagicMock()
        mock_search_result.search_trace = [
            MagicMock(
                query="test query",
                num_results=3,
                avg_similarity=0.85,
                evaluation="sufficient",
            ),
        ]

        mock_ask_result = {
            "answer": "Revenue was $85.8 billion.",
            "sources": [
                {
                    "chunk_id": 42,
                    "content": "Total revenue was $85.8 billion.",
                    "page_number": 3,
                    "similarity_score": 0.92,
                },
            ],
            "search_result": mock_search_result,
            "model": "mock-model",
        }

        test_case = GoldenTestCase(
            id="qa_01",
            capability="qa",
            question="What was the revenue?",
            expected_answer="Revenue was $85.8 billion.",
        )

        with patch(
            "app.agents.orchestrator.ask",
            new_callable=AsyncMock,
            return_value=mock_ask_result,
        ):
            result = _run(run_test_case(test_case, document_id=1))

        assert result.test_case_id == "qa_01"
        assert result.actual_answer == "Revenue was $85.8 billion."
        assert len(result.retrieval_context) == 1
        assert result.search_trace is not None
        assert len(result.search_trace) == 1

    def test_ask_failure_returns_error_result(self):
        """If ask() raises, return a TestCaseResult with error."""
        test_case = GoldenTestCase(
            id="qa_fail",
            capability="qa",
            question="What was the revenue?",
            expected_answer="Revenue was $85.8 billion.",
        )

        with patch(
            "app.agents.orchestrator.ask",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM unavailable"),
        ):
            result = _run(run_test_case(test_case, document_id=1))

        assert result.test_case_id == "qa_fail"
        assert "ERROR" in result.actual_answer
        assert result.passed is False


# ---------------------------------------------------------------------------
# Test: Custom Metric Computation
# ---------------------------------------------------------------------------


class TestComputeCustomMetrics:
    """Tests for compute_custom_metrics()."""

    def test_computes_both_metrics(self):
        result = TestCaseResult(
            test_case_id="qa_01",
            question="What was revenue?",
            expected_answer="$85.8 billion",
            actual_answer="Revenue was $85.8 billion.",
            retrieval_context=["Total revenue was $85.8 billion."],
            sources=[{"chunk_id": 42, "content": "..."}],
            search_trace=None,
            metric_results={},
            passed=False,
        )
        test_case = GoldenTestCase(
            id="qa_01",
            capability="qa",
            question="What was revenue?",
            expected_answer="$85.8 billion",
            numerical_values={"revenue": "$85.8 billion"},
            expected_source_chunks=[42],
        )

        scores = compute_custom_metrics(result, test_case)

        assert "numerical_accuracy" in scores
        assert "retrieval_recall_at_k" in scores
        assert scores["numerical_accuracy"].score == 1.0
        assert scores["retrieval_recall_at_k"].score == 1.0

    def test_no_numerical_values(self):
        """Test case without numerical_values should score 1.0."""
        result = TestCaseResult(
            test_case_id="sum_01",
            question="Summarise the risks.",
            expected_answer="Key risks include...",
            actual_answer="The main risks are...",
            retrieval_context=[],
            sources=[],
            search_trace=None,
            metric_results={},
            passed=False,
        )
        test_case = GoldenTestCase(
            id="sum_01",
            capability="summarise",
            question="Summarise the risks.",
            expected_answer="Key risks include...",
            numerical_values={},
            expected_source_chunks=[],
        )

        scores = compute_custom_metrics(result, test_case)
        assert scores["numerical_accuracy"].score == 1.0
        assert scores["retrieval_recall_at_k"].score == 1.0


# ---------------------------------------------------------------------------
# Test: Pass/Fail Determination
# ---------------------------------------------------------------------------


class TestDeterminePassFail:
    """Tests for determine_pass_fail()."""

    def test_all_metrics_pass(self):
        metrics = {
            "faithfulness": MetricScore(score=0.9),
            "answer_relevancy": MetricScore(score=0.85),
        }
        thresholds = {"faithfulness": 0.7, "answer_relevancy": 0.7}
        assert determine_pass_fail(metrics, thresholds) is True

    def test_one_metric_fails(self):
        metrics = {
            "faithfulness": MetricScore(score=0.9),
            "answer_relevancy": MetricScore(score=0.5),
        }
        thresholds = {"faithfulness": 0.7, "answer_relevancy": 0.7}
        assert determine_pass_fail(metrics, thresholds) is False

    def test_uses_default_threshold(self):
        """Metrics not in thresholds dict use the default."""
        metrics = {
            "custom_metric": MetricScore(score=0.6),
        }
        thresholds = {}  # No threshold for custom_metric
        # Default threshold from settings is 0.7, so 0.6 should fail
        assert determine_pass_fail(
            metrics, thresholds, default_threshold=0.7,
        ) is False

    def test_empty_metrics_passes(self):
        """No metrics → passes by default."""
        assert determine_pass_fail({}, {}) is True
