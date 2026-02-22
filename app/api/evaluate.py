# =============================================================================
# Evaluation API — Golden Dataset Evaluation & Feedback Loops
# =============================================================================
#
# Four endpoints:
#   POST /evaluate                → Start an evaluation run (background)
#   GET  /evaluate/runs/{run_id}  → Get evaluation results + regression
#   GET  /evaluate/history        → Score trends over time
#   GET  /evaluate/failures       → Detailed failure analysis
#
# DESIGN DECISION: POST /evaluate runs in the background via BackgroundTasks.
# A full evaluation involves 30+ test cases × 6 metrics = potentially
# hundreds of LLM calls. Running synchronously would timeout.
# The pattern mirrors POST /ingest (dispatch task → return ID → poll).
#
# DESIGN DECISION: Route ordering matters. /evaluate/history and
# /evaluate/failures are registered BEFORE /evaluate/runs/{run_id}
# to prevent FastAPI from capturing "history" and "failures" as run_id.
#
# DESIGN DECISION: Regression comparison is computed on-demand when
# fetching results, not stored. This avoids stale data if a previous
# run is deleted, and keeps the comparison always fresh.
# =============================================================================

from __future__ import annotations

import logging
from collections import Counter

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_async_session
from app.db.models import EvalRun, EvalTestResult
from app.models.requests import EvaluateRequest
from app.models.responses import (
    EvalFailureDetail,
    EvalFailuresResponse,
    EvalHistoryEntry,
    EvalHistoryResponse,
    EvalMetric,
    EvalTestCaseResult,
    EvaluateResponse,
    EvaluateStartResponse,
)
from app.services.eval_runner import compare_with_previous, load_dataset, run_full_eval

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Evaluation"])


# ---------------------------------------------------------------------------
# POST /evaluate — Start Evaluation Run
# ---------------------------------------------------------------------------


@router.post(
    "/evaluate",
    response_model=EvaluateStartResponse,
    summary="Start RAG evaluation against golden dataset",
    description=(
        "Triggers evaluation of the RAG pipeline against a golden dataset. "
        "The evaluation runs in the background — poll "
        "GET /evaluate/runs/{run_id} for results."
    ),
)
async def start_evaluation(
    request: EvaluateRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
) -> EvaluateStartResponse:
    """
    Start an evaluation run.

    1. Validate that the dataset exists (fail fast with 400)
    2. Create an EvalRun row with status="running"
    3. Schedule the full eval in BackgroundTasks
    4. Return the run_id immediately
    """
    # Validate dataset exists before creating the run
    try:
        load_dataset(request.eval_dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Create the EvalRun row
    eval_run = EvalRun(
        document_id=request.document_id,
        eval_dataset=request.eval_dataset,
        provider_id=request.provider_id,
        status="running",
    )
    session.add(eval_run)
    await session.commit()
    await session.refresh(eval_run)

    run_id = eval_run.id

    # Schedule the background evaluation
    background_tasks.add_task(
        run_full_eval,
        document_id=request.document_id,
        dataset_name=request.eval_dataset,
        provider_id=request.provider_id,
        run_id=run_id,
    )

    logger.info(
        "Evaluation run %d started for document %d with dataset '%s'",
        run_id, request.document_id, request.eval_dataset,
    )

    return EvaluateStartResponse(
        run_id=run_id,
        status="running",
        message=(
            f"Evaluation started. Poll GET /evaluate/runs/{run_id} for results."
        ),
    )


# ---------------------------------------------------------------------------
# GET /evaluate/history — Evaluation Score Trends
# ---------------------------------------------------------------------------
# NOTE: This must be registered BEFORE /evaluate/runs/{run_id}
# to avoid "history" being captured as a run_id path parameter.
# ---------------------------------------------------------------------------


@router.get(
    "/evaluate/history",
    response_model=EvalHistoryResponse,
    summary="Evaluation score history and trends",
    description=(
        "Returns chronological list of past evaluation runs with a trend "
        "indicator. Filter by document_id, dataset, or provider."
    ),
)
async def get_evaluation_history(
    document_id: int | None = Query(default=None),
    eval_dataset: str = Query(default="default"),
    provider_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
) -> EvalHistoryResponse:
    """
    Query evaluation history with optional filters.

    Trend is computed from the most recent runs:
    - "improving": last 3+ runs show upward score trend
    - "declining": last 3+ runs show downward score trend
    - "stable": scores within ±0.02 across recent runs
    - "insufficient_data": fewer than 2 completed runs
    """
    stmt = (
        select(EvalRun)
        .where(
            EvalRun.eval_dataset == eval_dataset,
            EvalRun.status == "completed",
        )
        .order_by(EvalRun.created_at.desc())
        .limit(limit)
    )

    if document_id is not None:
        stmt = stmt.where(EvalRun.document_id == document_id)
    if provider_id is not None:
        stmt = stmt.where(EvalRun.provider_id == provider_id)

    result = await session.execute(stmt)
    runs = list(result.scalars().all())

    entries = [
        EvalHistoryEntry(
            run_id=r.id,
            document_id=r.document_id,
            eval_dataset=r.eval_dataset,
            provider_id=r.provider_id,
            overall_score=r.overall_score or 0.0,
            metric_scores=r.metric_scores or {},
            total_test_cases=r.total_test_cases or 0,
            passed=r.passed or 0,
            failed=r.failed or 0,
            duration_ms=r.duration_ms,
            created_at=r.created_at,
        )
        for r in runs
    ]

    # Compute trend from recent scores (chronological order)
    trend = _compute_trend([e.overall_score for e in reversed(entries)])

    return EvalHistoryResponse(
        entries=entries,
        total_runs=len(entries),
        trend=trend,
    )


# ---------------------------------------------------------------------------
# GET /evaluate/failures — Failure Analysis
# ---------------------------------------------------------------------------


@router.get(
    "/evaluate/failures",
    response_model=EvalFailuresResponse,
    summary="Detailed failure analysis for an evaluation run",
    description=(
        "Returns detailed information about which test cases failed, "
        "why they failed, and the full search trace for debugging."
    ),
)
async def get_evaluation_failures(
    run_id: int = Query(..., description="Evaluation run ID to analyse"),
    session: AsyncSession = Depends(get_async_session),
) -> EvalFailuresResponse:
    """
    Detailed failure analysis for a specific eval run.

    Shows each failed test case with:
    - Per-metric scores and LLM judge reasoning
    - Which specific metrics fell below threshold
    - The full agentic search trace for debugging
    - The retrieved source chunks
    """
    # Load the eval run and its config (for thresholds)
    run_stmt = select(EvalRun).where(EvalRun.id == run_id)
    run_result = await session.execute(run_stmt)
    eval_run = run_result.scalar_one_or_none()

    if eval_run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Load failed test results
    stmt = (
        select(EvalTestResult)
        .where(
            EvalTestResult.eval_run_id == run_id,
            EvalTestResult.passed == False,  # noqa: E712
        )
        .order_by(EvalTestResult.id)
    )
    result = await session.execute(stmt)
    failed_results = list(result.scalars().all())

    # Get thresholds from run config
    thresholds = (eval_run.run_config or {}).get("thresholds", {})

    # Build failure details
    failures = []
    failing_metric_counts: Counter[str] = Counter()

    for tr in failed_results:
        metric_results = tr.metric_results or {}
        metric_scores = {
            name: data.get("score", 0.0) if isinstance(data, dict) else 0.0
            for name, data in metric_results.items()
        }
        metric_reasons = {
            name: data.get("reason") if isinstance(data, dict) else None
            for name, data in metric_results.items()
        }

        # Determine which metrics specifically failed
        failing = []
        for name, score in metric_scores.items():
            threshold = thresholds.get(name, 0.7)
            if score < threshold:
                failing.append(name)
                failing_metric_counts[name] += 1

        failures.append(EvalFailureDetail(
            test_case_id=tr.test_case_id,
            question=tr.question,
            expected_answer=tr.expected_answer,
            actual_answer=tr.actual_answer,
            metric_scores=metric_scores,
            metric_reasons=metric_reasons,
            failing_metrics=failing,
            search_trace=tr.search_trace,
            sources=tr.sources,
        ))

    # Total test cases for failure rate
    total = eval_run.total_test_cases or 1
    failure_rate = len(failed_results) / total

    most_common = (
        failing_metric_counts.most_common(1)[0][0]
        if failing_metric_counts else None
    )

    return EvalFailuresResponse(
        run_id=run_id,
        failures=failures,
        total_failures=len(failures),
        failure_rate=round(failure_rate, 4),
        most_common_failing_metric=most_common,
    )


# ---------------------------------------------------------------------------
# GET /evaluate/runs/{run_id} — Get Evaluation Results
# ---------------------------------------------------------------------------


@router.get(
    "/evaluate/runs/{run_id}",
    response_model=EvaluateResponse,
    summary="Get evaluation run results",
    description=(
        "Returns full evaluation results including per-metric scores, "
        "individual test case results, and regression comparison."
    ),
)
async def get_evaluation_result(
    run_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> EvaluateResponse:
    """
    Get full results for a specific evaluation run.

    If the run is still "running", returns a partial response.
    If "completed", includes regression comparison with previous run.
    """
    stmt = select(EvalRun).where(EvalRun.id == run_id)
    result = await session.execute(stmt)
    eval_run = result.scalar_one_or_none()

    if eval_run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Build metrics list from aggregate scores
    metrics = []
    if eval_run.metric_scores:
        for name, score in eval_run.metric_scores.items():
            metrics.append(EvalMetric(name=name, score=score))

    # Build test case results if completed
    test_case_results = None
    if eval_run.status == "completed" and eval_run.test_results:
        test_case_results = []
        for tr in eval_run.test_results:
            metric_results = tr.metric_results or {}
            scores = {
                n: d.get("score", 0.0) if isinstance(d, dict) else 0.0
                for n, d in metric_results.items()
            }
            reasons = {
                n: d.get("reason") if isinstance(d, dict) else None
                for n, d in metric_results.items()
            }
            test_case_results.append(EvalTestCaseResult(
                test_case_id=tr.test_case_id,
                question=tr.question,
                expected_answer=tr.expected_answer,
                actual_answer=tr.actual_answer,
                passed=tr.passed,
                metric_scores=scores,
                metric_reasons=reasons,
            ))

    # Regression comparison (only for completed runs)
    regression = None
    if eval_run.status == "completed":
        regression_data = await compare_with_previous(
            run_id=run_id,
            document_id=eval_run.document_id,
            dataset_name=eval_run.eval_dataset,
        )
        if regression_data:
            from app.models.responses import RegressionComparison

            regression = RegressionComparison(**regression_data)

    return EvaluateResponse(
        run_id=eval_run.id,
        status=eval_run.status,
        document_id=eval_run.document_id,
        eval_dataset=eval_run.eval_dataset,
        provider_id=eval_run.provider_id,
        metrics=metrics,
        overall_score=eval_run.overall_score or 0.0,
        total_test_cases=eval_run.total_test_cases or 0,
        passed=eval_run.passed or 0,
        failed=eval_run.failed or 0,
        duration_ms=eval_run.duration_ms,
        regression=regression,
        test_case_results=test_case_results,
        error=eval_run.error,
        created_at=eval_run.created_at,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_trend(scores: list[float]) -> str:
    """
    Compute score trend from a chronological list of scores.

    Args:
        scores: Overall scores in chronological order (oldest first).

    Returns:
        "improving", "declining", "stable", or "insufficient_data".
    """
    if len(scores) < 2:
        return "insufficient_data"

    # Use last 5 scores for trend detection
    recent = scores[-5:]

    if len(recent) < 2:
        return "insufficient_data"

    # Simple linear trend: compare first half avg to second half avg
    mid = len(recent) // 2
    first_half = sum(recent[:mid]) / mid
    second_half = sum(recent[mid:]) / (len(recent) - mid)

    delta = second_half - first_half

    if delta > 0.02:
        return "improving"
    elif delta < -0.02:
        return "declining"
    else:
        return "stable"
