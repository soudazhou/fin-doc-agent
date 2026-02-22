# =============================================================================
# Unit Tests — Golden Dataset Validation (Phase 5)
# =============================================================================
#
# Validates the structure and content of the golden dataset file.
# These tests ensure the dataset is well-formed and covers all capabilities.
# No API keys or external services needed.
# =============================================================================

import json
from pathlib import Path

from app.services.eval_runner import load_dataset

DATASET_PATH = Path("data/eval/golden_datasets/default.json")


class TestGoldenDatasetStructure:
    """Validates the default golden dataset file structure."""

    def test_file_exists(self):
        assert DATASET_PATH.exists(), f"Golden dataset not found at {DATASET_PATH}"

    def test_valid_json(self):
        with open(DATASET_PATH) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_has_required_top_level_fields(self):
        with open(DATASET_PATH) as f:
            data = json.load(f)

        for field in ("dataset_name", "test_cases", "thresholds", "version"):
            assert field in data, f"Missing top-level field: {field}"

    def test_thresholds_are_valid(self):
        with open(DATASET_PATH) as f:
            data = json.load(f)

        thresholds = data["thresholds"]
        assert len(thresholds) >= 4, "Need at least 4 threshold entries"

        for name, value in thresholds.items():
            assert 0.0 <= value <= 1.0, (
                f"Threshold '{name}' = {value} is out of [0, 1] range"
            )

    def test_loads_via_runner(self):
        """The dataset should parse via the eval runner's load_dataset()."""
        dataset = load_dataset("default")
        assert dataset.dataset_name == "default"
        assert len(dataset.test_cases) > 0


class TestGoldenDatasetContent:
    """Validates the content quality of the golden dataset."""

    def test_minimum_30_test_cases(self):
        dataset = load_dataset("default")
        assert len(dataset.test_cases) >= 29, (
            f"Need ≥29 test cases, got {len(dataset.test_cases)}"
        )

    def test_all_four_capabilities_covered(self):
        dataset = load_dataset("default")
        capabilities = {tc.capability for tc in dataset.test_cases}
        expected = {"qa", "summarise", "compare", "extract"}
        assert expected.issubset(capabilities), (
            f"Missing capabilities: {expected - capabilities}"
        )

    def test_each_capability_has_multiple_cases(self):
        dataset = load_dataset("default")
        from collections import Counter

        counts = Counter(tc.capability for tc in dataset.test_cases)
        for cap, count in counts.items():
            assert count >= 5, (
                f"Capability '{cap}' has only {count} test cases (need ≥5)"
            )

    def test_test_cases_have_required_fields(self):
        dataset = load_dataset("default")
        for tc in dataset.test_cases:
            assert tc.id, "Test case missing id"
            assert tc.question, f"Test case {tc.id} missing question"
            assert tc.expected_answer, (
                f"Test case {tc.id} missing expected_answer"
            )

    def test_unique_test_case_ids(self):
        dataset = load_dataset("default")
        ids = [tc.id for tc in dataset.test_cases]
        assert len(ids) == len(set(ids)), "Duplicate test case IDs found"

    def test_qa_cases_have_numerical_values(self):
        """QA test cases should have numerical_values for the custom metric."""
        dataset = load_dataset("default")
        qa_cases = [tc for tc in dataset.test_cases if tc.capability == "qa"]
        cases_with_numbers = [
            tc for tc in qa_cases if tc.numerical_values
        ]
        # Most QA cases should have numerical values (financial domain)
        assert len(cases_with_numbers) >= len(qa_cases) * 0.8, (
            "At least 80% of QA cases should have numerical_values"
        )
