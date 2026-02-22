# =============================================================================
# Unit Tests — Custom Evaluation Metrics (Phase 5)
# =============================================================================
#
# Tests the two custom metrics: numerical_accuracy and retrieval_recall_at_k.
# These are pure Python functions — no API keys, no mocking needed.
# =============================================================================

from app.services.eval_metrics import (
    _extract_core_number,
    _normalize_number,
    numerical_accuracy,
    retrieval_recall_at_k,
)

# ---------------------------------------------------------------------------
# Test: Numerical Accuracy
# ---------------------------------------------------------------------------


class TestNumericalAccuracy:
    """Tests for the numerical_accuracy metric."""

    def test_exact_match(self):
        result = numerical_accuracy(
            actual_output="Total revenue was $85.8 billion in Q3 2024.",
            expected_values={"revenue": "$85.8 billion"},
        )
        assert result.score == 1.0
        assert "MATCH" in result.reason

    def test_partial_match(self):
        result = numerical_accuracy(
            actual_output="Revenue was $85.8 billion with 5% growth.",
            expected_values={
                "revenue": "$85.8 billion",
                "growth": "5%",
                "eps": "$1.40",
            },
        )
        # 2 out of 3 matched
        assert abs(result.score - 2 / 3) < 0.01
        assert "MISS" in result.reason

    def test_no_match(self):
        result = numerical_accuracy(
            actual_output="The company performed well.",
            expected_values={"revenue": "$85.8 billion"},
        )
        assert result.score == 0.0
        assert "MISS" in result.reason

    def test_empty_expected(self):
        result = numerical_accuracy(
            actual_output="Some answer.",
            expected_values={},
        )
        assert result.score == 1.0
        assert "No numerical values" in result.reason

    def test_normalized_match_billion(self):
        """$85.8B and $85.8 billion should match."""
        result = numerical_accuracy(
            actual_output="Revenue was $85.8B for the quarter.",
            expected_values={"revenue": "$85.8 billion"},
        )
        assert result.score == 1.0

    def test_normalized_match_commas(self):
        """1,234,567 and 1234567 should match."""
        result = numerical_accuracy(
            actual_output="There were 1234567 units sold.",
            expected_values={"units": "1,234,567"},
        )
        assert result.score == 1.0

    def test_percentage_match(self):
        result = numerical_accuracy(
            actual_output="Growth was 46.3% year over year.",
            expected_values={"margin": "46.3%"},
        )
        assert result.score == 1.0

    def test_fuzzy_core_number_match(self):
        """If exact normalized match fails, try core number extraction."""
        result = numerical_accuracy(
            actual_output="The margin was approximately 46.3 percent.",
            expected_values={"margin": "46.3%"},
        )
        # Should match via core number "46.3"
        assert result.score == 1.0

    def test_multiple_values_all_match(self):
        result = numerical_accuracy(
            actual_output="Revenue was $85.8 billion, EPS was $1.40, growth 5%.",
            expected_values={
                "revenue": "$85.8 billion",
                "eps": "$1.40",
                "growth": "5%",
            },
        )
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# Test: Retrieval Recall@k
# ---------------------------------------------------------------------------


class TestRetrievalRecallAtK:
    """Tests for the retrieval_recall_at_k metric."""

    def test_full_recall(self):
        result = retrieval_recall_at_k(
            retrieved_chunk_ids=[1, 2, 3, 4, 5],
            expected_chunk_ids=[2, 4],
        )
        assert result.score == 1.0
        assert "2/2" in result.reason

    def test_partial_recall(self):
        result = retrieval_recall_at_k(
            retrieved_chunk_ids=[1, 2, 3],
            expected_chunk_ids=[2, 4, 6],
        )
        assert abs(result.score - 1 / 3) < 0.01
        assert "1/3" in result.reason

    def test_no_recall(self):
        result = retrieval_recall_at_k(
            retrieved_chunk_ids=[10, 20, 30],
            expected_chunk_ids=[1, 2, 3],
        )
        assert result.score == 0.0
        assert "0/3" in result.reason

    def test_empty_expected(self):
        """Empty expected chunks should return perfect score (skip)."""
        result = retrieval_recall_at_k(
            retrieved_chunk_ids=[1, 2, 3],
            expected_chunk_ids=[],
        )
        assert result.score == 1.0
        assert "skipped" in result.reason.lower()

    def test_k_parameter_truncates(self):
        """Only consider top-k retrieved chunks."""
        result = retrieval_recall_at_k(
            retrieved_chunk_ids=[10, 20, 5, 30, 40],
            expected_chunk_ids=[5, 30],
            k=2,
        )
        # Only [10, 20] considered — neither 5 nor 30 in top-2
        assert result.score == 0.0

    def test_k_parameter_includes(self):
        result = retrieval_recall_at_k(
            retrieved_chunk_ids=[5, 30, 10, 20, 40],
            expected_chunk_ids=[5, 30],
            k=2,
        )
        # [5, 30] considered — both found
        assert result.score == 1.0

    def test_empty_retrieved(self):
        result = retrieval_recall_at_k(
            retrieved_chunk_ids=[],
            expected_chunk_ids=[1, 2],
        )
        assert result.score == 0.0
        assert "Missing" in result.reason


# ---------------------------------------------------------------------------
# Test: Helper Functions
# ---------------------------------------------------------------------------


class TestNormalizeNumber:
    """Tests for _normalize_number helper."""

    def test_removes_commas(self):
        assert "1234567" in _normalize_number("1,234,567")

    def test_normalizes_billion(self):
        assert "85.8b" in _normalize_number("$85.8 billion")

    def test_normalizes_million(self):
        assert "42.5m" in _normalize_number("$42.5 million")

    def test_removes_spaces_after_dollar(self):
        assert "$85.8" in _normalize_number("$ 85.8")


class TestExtractCoreNumber:
    """Tests for _extract_core_number helper."""

    def test_dollar_amount(self):
        assert _extract_core_number("$85.8 billion") == "85.8"

    def test_percentage(self):
        assert _extract_core_number("46.3%") == "46.3"

    def test_plain_number(self):
        assert _extract_core_number("1234") == "1234"

    def test_no_number(self):
        assert _extract_core_number("no numbers here") is None
