# =============================================================================
# Evaluation Runner — Orchestrates Golden Dataset Evaluation
# =============================================================================
#
# This is the core Phase 5 module. It orchestrates the full evaluation loop:
#
# 1. Load golden dataset from JSON file
# 2. For each test case:
#    a. Call ask() to get the RAG pipeline's answer
#    b. Run 4 DeepEval metrics (LLM-as-judge)
#    c. Run 2 custom metrics (deterministic)
#    d. Determine pass/fail based on thresholds
# 3. Aggregate results into EvalRun + EvalTestResult rows
# 4. Persist to database
# 5. Compare with previous run for regression detection
#
# DESIGN DECISION: The runner calls ask() directly (not via HTTP).
# This avoids network overhead, enables LLM injection for cross-provider
# eval, and gives access to the full AgentState (search traces, token
# counts) that HTTP responses might not expose.
#
# DESIGN DECISION: DeepEval metrics run via metric.a_measure() (async).
# This keeps the runner async-native and compatible with FastAPI's
# BackgroundTasks. The 4 DeepEval metrics for each test case run
# sequentially to avoid LLM rate limiting.
#
# DESIGN DECISION: Custom metrics run separately from DeepEval.
# They're deterministic Python functions — no LLM needed, no async
# overhead. Keeping them separate makes it clear which metrics cost
# money (DeepEval) and which are free (custom).
# =============================================================================

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures — Golden Dataset
# ---------------------------------------------------------------------------


@dataclass
class GoldenTestCase:
    """A single test case from the golden dataset."""

    id: str
    capability: str
    question: str
    expected_answer: str
    expected_context: list[str] = field(default_factory=list)
    expected_source_chunks: list[int] = field(default_factory=list)
    numerical_values: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class GoldenDataset:
    """Parsed golden dataset with metadata and thresholds."""

    dataset_name: str
    description: str
    version: str
    thresholds: dict[str, float]
    test_cases: list[GoldenTestCase]


# ---------------------------------------------------------------------------
# Data Structures — Evaluation Results
# ---------------------------------------------------------------------------


@dataclass
class MetricScore:
    """Score and reasoning for a single metric on a single test case."""

    score: float
    reason: str | None = None


@dataclass
class TestCaseResult:
    """Result of evaluating a single golden dataset test case."""

    test_case_id: str
    question: str
    expected_answer: str | None
    actual_answer: str
    retrieval_context: list[str]
    sources: list[dict]
    search_trace: list[dict] | None
    metric_results: dict[str, MetricScore]
    passed: bool


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------


def load_dataset(dataset_name: str) -> GoldenDataset:
    """
    Load and validate a golden dataset from JSON file.

    Args:
        dataset_name: Name of the dataset (maps to filename without extension).
            E.g. "default" → data/eval/golden_datasets/default.json

    Returns:
        Parsed GoldenDataset.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If dataset is malformed or missing required fields.
    """
    dataset_dir = Path(settings.eval_dataset_dir)
    filepath = dataset_dir / f"{dataset_name}.json"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Golden dataset '{dataset_name}' not found at {filepath}"
        )

    with open(filepath) as f:
        raw = json.load(f)

    # Validate required top-level fields
    for required in ("dataset_name", "test_cases", "thresholds"):
        if required not in raw:
            raise ValueError(
                f"Golden dataset missing required field: '{required}'"
            )

    if not raw["test_cases"]:
        raise ValueError("Golden dataset has no test cases")

    # Parse test cases
    test_cases = []
    for tc in raw["test_cases"]:
        if "id" not in tc or "question" not in tc:
            raise ValueError(
                f"Test case missing required field 'id' or 'question': {tc}"
            )
        test_cases.append(GoldenTestCase(
            id=tc["id"],
            capability=tc.get("capability", "qa"),
            question=tc["question"],
            expected_answer=tc.get("expected_answer", ""),
            expected_context=tc.get("expected_context", []),
            expected_source_chunks=tc.get("expected_source_chunks", []),
            numerical_values=tc.get("numerical_values", {}),
            tags=tc.get("tags", []),
        ))

    return GoldenDataset(
        dataset_name=raw["dataset_name"],
        description=raw.get("description", ""),
        version=raw.get("version", "1.0"),
        thresholds=raw["thresholds"],
        test_cases=test_cases,
    )


# ---------------------------------------------------------------------------
# Test Case Execution
# ---------------------------------------------------------------------------


async def run_test_case(
    test_case: GoldenTestCase,
    document_id: int,
    llm=None,
) -> TestCaseResult:
    """
    Execute a single test case through the RAG pipeline.

    Calls orchestrator.ask() and captures the full response including
    answer, sources, and search trace.

    Args:
        test_case: Golden dataset test case to evaluate.
        document_id: Document ID to query against.
        llm: Optional LLM provider for cross-provider eval.

    Returns:
        TestCaseResult with the actual answer and retrieval context.
        metric_results will be empty — populated by compute_metrics().
    """
    from app.agents.orchestrator import ask

    try:
        result = await ask(
            question=test_case.question,
            document_id=document_id,
            capability=test_case.capability,
            llm=llm,
        )
    except Exception as e:
        logger.warning(
            "Test case %s failed during ask(): %s", test_case.id, e,
        )
        return TestCaseResult(
            test_case_id=test_case.id,
            question=test_case.question,
            expected_answer=test_case.expected_answer,
            actual_answer=f"ERROR: {e}",
            retrieval_context=[],
            sources=[],
            search_trace=None,
            metric_results={},
            passed=False,
        )

    # Extract retrieval context (chunk content strings)
    retrieval_context = [
        s.get("content", "") for s in result.get("sources", [])
    ]

    # Extract search trace for failure analysis
    search_result = result.get("search_result")
    search_trace = None
    if search_result and hasattr(search_result, "search_trace"):
        search_trace = [
            {
                "query": it.query,
                "num_results": it.num_results,
                "avg_similarity": it.avg_similarity,
                "evaluation": it.evaluation,
            }
            for it in search_result.search_trace
        ]

    return TestCaseResult(
        test_case_id=test_case.id,
        question=test_case.question,
        expected_answer=test_case.expected_answer,
        actual_answer=result.get("answer", ""),
        retrieval_context=retrieval_context,
        sources=result.get("sources", []),
        search_trace=search_trace,
        metric_results={},
        passed=False,  # Will be determined after metrics
    )


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------


async def compute_deepeval_metrics(
    result: TestCaseResult,
    test_case: GoldenTestCase,
    thresholds: dict[str, float],
) -> dict[str, MetricScore]:
    """
    Run the 4 DeepEval LLM-as-judge metrics.

    Uses DeepEval's async a_measure() API to evaluate:
    - faithfulness: Is the answer grounded in context?
    - answer_relevancy: Does the answer address the question?
    - contextual_precision: Are retrieved chunks relevant?
    - contextual_recall: Did we find all necessary info?

    DESIGN DECISION: Each metric runs sequentially to avoid LLM rate limits.
    With 4 metrics per test case and 30+ test cases, parallel execution
    would spike API usage. Sequential is safer for cost control.
    """
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError:
        logger.warning(
            "deepeval not installed — returning empty DeepEval scores"
        )
        return {}

    # Build the DeepEval test case
    deepeval_case = LLMTestCase(
        input=result.question,
        actual_output=result.actual_answer,
        retrieval_context=result.retrieval_context or None,
        expected_output=test_case.expected_answer or None,
        context=test_case.expected_context or None,
    )

    judge_model = settings.eval_judge_model

    metrics_to_run = [
        ("faithfulness", FaithfulnessMetric(
            threshold=thresholds.get("faithfulness", 0.7),
            model=judge_model,
        )),
        ("answer_relevancy", AnswerRelevancyMetric(
            threshold=thresholds.get("answer_relevancy", 0.7),
            model=judge_model,
        )),
        ("contextual_precision", ContextualPrecisionMetric(
            threshold=thresholds.get("contextual_precision", 0.6),
            model=judge_model,
        )),
        ("contextual_recall", ContextualRecallMetric(
            threshold=thresholds.get("contextual_recall", 0.6),
            model=judge_model,
        )),
    ]

    scores: dict[str, MetricScore] = {}

    for name, metric in metrics_to_run:
        try:
            await metric.a_measure(deepeval_case)
            scores[name] = MetricScore(
                score=metric.score if metric.score is not None else 0.0,
                reason=metric.reason,
            )
        except Exception as e:
            logger.warning("DeepEval metric %s failed: %s", name, e)
            scores[name] = MetricScore(score=0.0, reason=f"Error: {e}")

    return scores


def compute_custom_metrics(
    result: TestCaseResult,
    test_case: GoldenTestCase,
) -> dict[str, MetricScore]:
    """
    Run the 2 custom (non-LLM) metrics.

    These are deterministic Python functions — no API calls, no cost.
    """
    from app.services.eval_metrics import (
        numerical_accuracy,
        retrieval_recall_at_k,
    )

    scores: dict[str, MetricScore] = {}

    # Numerical accuracy (finance-specific)
    num_result = numerical_accuracy(
        actual_output=result.actual_answer,
        expected_values=test_case.numerical_values,
    )
    scores["numerical_accuracy"] = MetricScore(
        score=num_result.score,
        reason=num_result.reason,
    )

    # Retrieval recall@k
    retrieved_ids = [
        s.get("chunk_id", 0) for s in result.sources
    ]
    recall_result = retrieval_recall_at_k(
        retrieved_chunk_ids=retrieved_ids,
        expected_chunk_ids=test_case.expected_source_chunks,
    )
    scores["retrieval_recall_at_k"] = MetricScore(
        score=recall_result.score,
        reason=recall_result.reason,
    )

    return scores


def determine_pass_fail(
    metric_results: dict[str, MetricScore],
    thresholds: dict[str, float],
    default_threshold: float | None = None,
) -> bool:
    """
    Determine whether a test case passes based on metric thresholds.

    A test case passes if ALL metrics meet their respective thresholds.
    """
    threshold_default = default_threshold or settings.eval_default_threshold

    for name, score_obj in metric_results.items():
        threshold = thresholds.get(name, threshold_default)
        if score_obj.score < threshold:
            return False
    return True


# ---------------------------------------------------------------------------
# Full Evaluation Run
# ---------------------------------------------------------------------------


async def run_full_eval(
    document_id: int,
    dataset_name: str = "default",
    provider_id: str | None = None,
    run_id: int | None = None,
) -> None:
    """
    Execute a complete evaluation run and persist results.

    This is the main entry point called by BackgroundTasks from the
    POST /evaluate endpoint. It:
    1. Loads the golden dataset
    2. Runs each test case through the RAG pipeline
    3. Computes all 6 metrics per test case
    4. Persists EvalRun + EvalTestResult rows
    5. Updates the EvalRun with aggregate scores and status

    Args:
        document_id: Document to evaluate against.
        dataset_name: Golden dataset name (default: "default").
        provider_id: Optional provider for cross-provider eval.
        run_id: Pre-created EvalRun ID to update with results.
    """
    from sqlalchemy import select

    from app.db.engine import async_session_factory
    from app.db.models import EvalRun, EvalTestResult

    start_time = time.monotonic()

    # Optionally create a provider instance for cross-provider eval
    llm = None
    model_name = None
    if provider_id:
        try:
            from app.services.llm import create_provider_from_id

            llm = create_provider_from_id(provider_id)
        except ValueError as e:
            logger.error("Invalid provider_id %s: %s", provider_id, e)
            await _update_run_status(
                run_id, "failed", error=f"Invalid provider: {e}",
            )
            return

    try:
        dataset = load_dataset(dataset_name)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load dataset %s: %s", dataset_name, e)
        await _update_run_status(
            run_id, "failed", error=f"Dataset error: {e}",
        )
        return

    # Run each test case and compute metrics
    test_results: list[TestCaseResult] = []
    for tc in dataset.test_cases:
        logger.info("Evaluating test case: %s", tc.id)

        result = await run_test_case(tc, document_id, llm=llm)

        # Compute all 6 metrics
        deepeval_scores = await compute_deepeval_metrics(
            result, tc, dataset.thresholds,
        )
        custom_scores = compute_custom_metrics(result, tc)

        # Merge all metric scores
        all_scores = {**deepeval_scores, **custom_scores}
        result.metric_results = all_scores
        result.passed = determine_pass_fail(
            all_scores, dataset.thresholds,
        )

        # Capture model name from first successful test case
        if model_name is None and result.actual_answer:
            # Model name comes from the last ask() call
            pass  # We get this from the DB or response later

        test_results.append(result)

    # Aggregate scores
    all_metric_names = set()
    for tr in test_results:
        all_metric_names.update(tr.metric_results.keys())

    metric_averages: dict[str, float] = {}
    for name in sorted(all_metric_names):
        scores = [
            tr.metric_results[name].score
            for tr in test_results
            if name in tr.metric_results
        ]
        if scores:
            metric_averages[name] = round(sum(scores) / len(scores), 4)

    overall_score = (
        round(sum(metric_averages.values()) / len(metric_averages), 4)
        if metric_averages else 0.0
    )

    passed_count = sum(1 for tr in test_results if tr.passed)
    failed_count = len(test_results) - passed_count
    duration_ms = int((time.monotonic() - start_time) * 1000)

    # Persist results to database
    async with async_session_factory() as session:
        if run_id:
            # Update the pre-created EvalRun
            stmt = select(EvalRun).where(EvalRun.id == run_id)
            db_result = await session.execute(stmt)
            eval_run = db_result.scalar_one_or_none()

            if eval_run:
                eval_run.status = "completed"
                eval_run.metric_scores = metric_averages
                eval_run.overall_score = overall_score
                eval_run.total_test_cases = len(test_results)
                eval_run.passed = passed_count
                eval_run.failed = failed_count
                eval_run.duration_ms = duration_ms
                eval_run.run_config = {
                    "thresholds": dataset.thresholds,
                    "dataset_version": dataset.version,
                }
        else:
            # Create a new EvalRun
            eval_run = EvalRun(
                document_id=document_id,
                eval_dataset=dataset_name,
                provider_id=provider_id,
                status="completed",
                metric_scores=metric_averages,
                overall_score=overall_score,
                total_test_cases=len(test_results),
                passed=passed_count,
                failed=failed_count,
                run_config={
                    "thresholds": dataset.thresholds,
                    "dataset_version": dataset.version,
                },
                duration_ms=duration_ms,
            )
            session.add(eval_run)
            await session.flush()  # Get the ID

        # Persist individual test case results
        for tr in test_results:
            db_test_result = EvalTestResult(
                eval_run_id=eval_run.id,
                test_case_id=tr.test_case_id,
                question=tr.question,
                expected_answer=tr.expected_answer,
                actual_answer=tr.actual_answer,
                retrieval_context=tr.retrieval_context,
                metric_results={
                    name: {"score": ms.score, "reason": ms.reason}
                    for name, ms in tr.metric_results.items()
                },
                passed=tr.passed,
                search_trace=tr.search_trace,
                sources=tr.sources,
            )
            session.add(db_test_result)

        try:
            await session.commit()
            logger.info(
                "Eval run %d completed: %d/%d passed, overall=%.3f",
                eval_run.id, passed_count, len(test_results), overall_score,
            )
        except Exception as e:
            logger.error("Failed to persist eval results: %s", e)
            await session.rollback()
            await _update_run_status(
                run_id, "failed", error=f"Persistence error: {e}",
            )


async def compare_with_previous(
    run_id: int,
    document_id: int,
    dataset_name: str,
) -> dict | None:
    """
    Compare an eval run with the most recent previous run on the same
    document and dataset.

    Returns a regression comparison dict or None if no previous run exists.
    """
    from sqlalchemy import select

    from app.db.engine import async_session_factory
    from app.db.models import EvalRun

    async with async_session_factory() as session:
        # Find the most recent completed run before this one
        stmt = (
            select(EvalRun)
            .where(
                EvalRun.document_id == document_id,
                EvalRun.eval_dataset == dataset_name,
                EvalRun.status == "completed",
                EvalRun.id < run_id,
            )
            .order_by(EvalRun.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        previous = result.scalar_one_or_none()

        if previous is None:
            return None

        # Load current run
        current_stmt = select(EvalRun).where(EvalRun.id == run_id)
        current_result = await session.execute(current_stmt)
        current = current_result.scalar_one_or_none()

        if current is None or current.metric_scores is None:
            return None

        # Compute deltas
        prev_scores = previous.metric_scores or {}
        curr_scores = current.metric_scores or {}

        metric_deltas = {}
        regressed = []
        improved = []

        for name in set(prev_scores.keys()) | set(curr_scores.keys()):
            prev = prev_scores.get(name)
            curr = curr_scores.get(name)
            if prev is not None and curr is not None:
                delta = round(curr - prev, 4)
                metric_deltas[name] = delta
                if delta < -0.01:
                    regressed.append(name)
                elif delta > 0.01:
                    improved.append(name)

        score_delta = None
        if previous.overall_score is not None and current.overall_score is not None:
            score_delta = round(
                current.overall_score - previous.overall_score, 4,
            )

        return {
            "previous_run_id": previous.id,
            "previous_overall_score": previous.overall_score,
            "score_delta": score_delta,
            "metric_deltas": metric_deltas,
            "regressed_metrics": regressed,
            "improved_metrics": improved,
        }


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


async def _update_run_status(
    run_id: int | None,
    status: str,
    error: str | None = None,
) -> None:
    """Update an EvalRun's status (used for error handling)."""
    if run_id is None:
        return

    from sqlalchemy import select

    from app.db.engine import async_session_factory
    from app.db.models import EvalRun

    try:
        async with async_session_factory() as session:
            stmt = select(EvalRun).where(EvalRun.id == run_id)
            result = await session.execute(stmt)
            eval_run = result.scalar_one_or_none()
            if eval_run:
                eval_run.status = status
                eval_run.error = error
                await session.commit()
    except Exception as e:
        logger.error("Failed to update run %d status: %s", run_id, e)
