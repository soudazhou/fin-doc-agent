# =============================================================================
# Benchmark & Comparison API — A/B Provider Testing & Performance Metrics
# =============================================================================
#
# Three endpoints:
#   POST /compare             — Run same query across N LLM providers in parallel
#   POST /benchmark/retrieval — Measure retrieval latency at different k/store configs
#   GET  /metrics             — Aggregated performance stats by provider
#
# DESIGN DECISION: All three endpoints in one router file.
# They share the same domain (performance measurement) and the same
# DB session dependency. Splitting further would fragment related code
# without benefit at this scale.
#
# DESIGN DECISION: /compare uses asyncio.gather() for parallel execution.
# Providers are independent — total_latency ≈ max(provider_latencies)
# rather than sum(provider_latencies).
#
# DESIGN DECISION: Provider failures are isolated, not propagated.
# If one provider leg raises, the other providers' results still return.
# The failed leg gets error=str(e) and empty answer. A single flaky
# provider doesn't break the whole comparison.
# =============================================================================

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import ask
from app.api.deps import check_document_access, check_scope, get_current_api_key
from app.db.engine import get_async_session
from app.db.models import ApiKey, QueryMetric
from app.models.requests import BenchmarkRetrievalRequest, CompareRequest
from app.models.responses import (
    BenchmarkRetrievalResponse,
    CompareResponse,
    CompareWinner,
    MetricsResponse,
    ProviderMetricSummary,
    ProviderResult,
    RetrievalBenchmarkResult,
    SourceChunk,
)
from app.services.llm import create_provider_from_id
from app.services.pricing import estimate_cost

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Benchmarking"])


# ---------------------------------------------------------------------------
# POST /compare — A/B Provider Comparison
# ---------------------------------------------------------------------------


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare same query across multiple LLM providers",
    description=(
        "Runs the identical question through N providers in parallel using "
        "the same retrieval pipeline. Returns side-by-side answers with "
        "latency, token usage, and estimated cost per provider."
    ),
)
async def compare_providers(
    http_request: Request,
    request: CompareRequest,
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> CompareResponse:
    """
    Parallel multi-provider comparison.

    1. Validate all provider IDs (fail fast with 400)
    2. Build one coroutine per provider
    3. asyncio.gather() — run all in parallel
    4. Compute winner summary
    5. Persist QueryMetric rows
    """
    check_scope(api_key, "benchmark")
    check_document_access(api_key, request.document_id)
    http_request.state.audit_document_id = request.document_id
    http_request.state.audit_question = request.question

    wall_start = time.monotonic()

    # --- Step 1: Build provider instances (fail fast) ---
    providers = {}
    for pid in request.providers:
        try:
            providers[pid] = create_provider_from_id(pid)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider '{pid}': {e}",
            ) from e

    # --- Step 2: Define per-provider coroutine ---
    async def run_one(pid: str) -> ProviderResult:
        provider = providers[pid]
        start = time.monotonic()
        try:
            result = await ask(
                question=request.question,
                document_id=request.document_id,
                capability=request.capability,
                llm=provider,
            )
        except Exception as e:
            logger.warning("Provider %s failed: %s", pid, e)
            elapsed = int((time.monotonic() - start) * 1000)
            return ProviderResult(
                provider_id=pid,
                answer="",
                sources=[],
                model="error",
                latency_ms=elapsed,
                input_tokens=0,
                output_tokens=0,
                search_iterations=0,
                retrieval_count=0,
                error=str(e),
            )

        elapsed = int((time.monotonic() - start) * 1000)

        # Parse provider type for pricing lookup
        provider_type = pid.split("/")[0]
        model = result.get("model", "unknown")
        in_tok = result.get("input_tokens", 0)
        out_tok = result.get("output_tokens", 0)
        cost = estimate_cost(provider_type, model, in_tok, out_tok)

        # Map sources
        sources = [
            SourceChunk(**s) for s in result.get("sources", [])
        ]

        # Count search iterations from the search trace
        search_result = result.get("search_result")
        iterations = (
            len(search_result.search_trace)
            if search_result else 0
        )

        return ProviderResult(
            provider_id=pid,
            answer=result.get("answer", ""),
            sources=sources,
            model=model,
            latency_ms=elapsed,
            input_tokens=in_tok,
            output_tokens=out_tok,
            estimated_cost_usd=cost,
            search_iterations=iterations,
            retrieval_count=result.get("retrieval_count", 0),
        )

    # --- Step 3: Run all providers in parallel ---
    results: list[ProviderResult] = await asyncio.gather(
        *(run_one(pid) for pid in request.providers),
    )

    # --- Step 4: Compute winner ---
    winner = _compute_winner(results)

    # --- Step 5: Persist metrics ---
    truncated_q = request.question[:500]
    for r in results:
        metric = QueryMetric(
            document_id=request.document_id,
            question=truncated_q,
            capability=request.capability,
            provider_id=r.provider_id,
            model=r.model if r.error is None else None,
            total_latency_ms=r.latency_ms,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            estimated_cost_usd=r.estimated_cost_usd,
            search_iterations=r.search_iterations,
            retrieval_count=r.retrieval_count,
            retrieval_scores=[
                s.similarity_score for s in r.sources
            ] if r.sources else None,
            source="compare",
        )
        session.add(metric)

    try:
        await session.commit()
    except Exception as e:
        logger.warning("Failed to persist compare metrics: %s", e)
        await session.rollback()

    total_ms = int((time.monotonic() - wall_start) * 1000)

    return CompareResponse(
        question=request.question,
        document_id=request.document_id,
        results=results,
        winner=winner,
        total_latency_ms=total_ms,
    )


def _compute_winner(results: list[ProviderResult]) -> CompareWinner:
    """
    Determine which provider wins on each dimension.

    Only considers providers that didn't error out.
    """
    valid = [r for r in results if r.error is None]
    if not valid:
        return CompareWinner()

    fastest = min(valid, key=lambda r: r.latency_ms).provider_id

    costed = [r for r in valid if r.estimated_cost_usd is not None]
    cheapest = (
        min(costed, key=lambda r: r.estimated_cost_usd).provider_id  # type: ignore[arg-type]
        if costed else None
    )

    return CompareWinner(
        fastest_provider=fastest,
        cheapest_provider=cheapest,
    )


# ---------------------------------------------------------------------------
# POST /benchmark/retrieval — Retrieval Latency Benchmark
# ---------------------------------------------------------------------------


@router.post(
    "/benchmark/retrieval",
    response_model=BenchmarkRetrievalResponse,
    summary="Benchmark vector retrieval latency",
    description=(
        "Measures search latency and score distributions across different "
        "top_k values and vector store backends. Does not require a golden "
        "dataset — benchmarks existing chunks."
    ),
)
async def benchmark_retrieval(
    request: BenchmarkRetrievalRequest,
    api_key: ApiKey | None = Depends(get_current_api_key),
) -> BenchmarkRetrievalResponse:
    """
    Retrieval latency benchmark.

    For each (vector_store, top_k) combination:
    1. Embed all sample queries once (reuse across top_k values)
    2. Run search() for each query at this top_k
    3. Record latency per query
    4. Compute p50, p95, avg scores

    DESIGN DECISION: Embed queries once and cache — embedding is the
    same regardless of top_k. This makes the latency measurement fair:
    we're measuring vector search, not embedding.
    """
    check_scope(api_key, "benchmark")
    check_document_access(api_key, request.document_id)
    from app.services.embedder import embed_query
    from app.services.vectorstore import get_vector_store

    results: list[RetrievalBenchmarkResult] = []

    for store_type in request.vector_stores:
        # Get the appropriate store instance
        # DESIGN DECISION: Use get_vector_store() which reads from config.
        # If the user requests "chroma" but config says "pgvector", we
        # still attempt it — the factory may support it. If not, it will
        # raise and we return a 400.
        try:
            store = get_vector_store(override_type=store_type)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Vector store '{store_type}' not available: {e}"
                ),
            ) from e

        # Embed all queries once
        embeddings = []
        for q in request.sample_queries:
            emb = await asyncio.to_thread(embed_query, q)
            embeddings.append(emb)

        for top_k in request.top_k_values:
            latencies: list[float] = []
            all_top_scores: list[float] = []

            for emb in embeddings:
                start = time.monotonic()
                search_results = await store.search(
                    query_embedding=emb,
                    top_k=top_k,
                    document_id=request.document_id,
                )
                latency_ms = (time.monotonic() - start) * 1000
                latencies.append(latency_ms)

                scores = [r.similarity_score for r in search_results]
                if scores:
                    all_top_scores.append(scores[0])

            # Compute percentiles
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            p50 = sorted_lat[n // 2] if n > 0 else 0.0
            p95_idx = min(int(n * 0.95), n - 1) if n > 0 else 0
            p95 = sorted_lat[p95_idx] if n > 0 else 0.0
            avg_lat = sum(latencies) / n if n > 0 else 0.0

            avg_top = (
                sum(all_top_scores) / len(all_top_scores)
                if all_top_scores else 0.0
            )

            results.append(RetrievalBenchmarkResult(
                vector_store=store_type,
                top_k=top_k,
                avg_latency_ms=round(avg_lat, 2),
                p50_latency_ms=round(p50, 2),
                p95_latency_ms=round(p95, 2),
                avg_top_score=round(avg_top, 4),
                queries_run=len(request.sample_queries),
            ))

    # Summary
    best = min(results, key=lambda r: r.p50_latency_ms) if results else None
    summary = (
        f"{best.vector_store} top_k={best.top_k} was fastest at "
        f"{best.p50_latency_ms:.1f}ms p50"
        if best else "No results"
    )

    return BenchmarkRetrievalResponse(
        document_id=request.document_id,
        results=results,
        sample_queries=request.sample_queries,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# GET /metrics — Aggregated Performance Dashboard
# ---------------------------------------------------------------------------


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Aggregated performance metrics by provider",
    description=(
        "Returns aggregated latency, cost, and usage stats from the "
        "query_metrics table, grouped by provider."
    ),
)
async def get_metrics(
    hours: int = Query(
        default=24, ge=1, le=720,
        description="Lookback window in hours",
    ),
    api_key: ApiKey | None = Depends(get_current_api_key),
    session: AsyncSession = Depends(get_async_session),
) -> MetricsResponse:
    """
    Aggregate query_metrics rows grouped by provider_id.

    DESIGN DECISION: Aggregation in Python rather than SQL GROUP BY.
    The dataset is small (demo project) and Python aggregation is
    easier to read, test, and extend. At production scale, switch
    to SQL aggregation with time-series indexing.
    """
    check_scope(api_key, "benchmark")
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    stmt = select(QueryMetric).where(QueryMetric.created_at >= cutoff)
    result = await session.execute(stmt)
    rows = result.scalars().all()

    # Group by provider_id
    by_provider: dict[str | None, list[QueryMetric]] = {}
    for row in rows:
        by_provider.setdefault(row.provider_id, []).append(row)

    summaries = []
    for pid, metrics in by_provider.items():
        latencies = sorted([
            m.total_latency_ms for m in metrics
            if m.total_latency_ms is not None
        ])
        costs = [
            m.estimated_cost_usd for m in metrics
            if m.estimated_cost_usd is not None
        ]
        in_tokens = [
            m.input_tokens for m in metrics
            if m.input_tokens is not None
        ]
        out_tokens = [
            m.output_tokens for m in metrics
            if m.output_tokens is not None
        ]
        iterations = [
            m.search_iterations for m in metrics
            if m.search_iterations is not None
        ]

        n = len(latencies)
        summaries.append(ProviderMetricSummary(
            provider_id=pid,
            query_count=len(metrics),
            avg_latency_ms=(
                round(sum(latencies) / n, 1) if n else None
            ),
            p50_latency_ms=(
                latencies[n // 2] if n else None
            ),
            p95_latency_ms=(
                latencies[min(int(n * 0.95), n - 1)]
                if n else None
            ),
            avg_input_tokens=(
                round(sum(in_tokens) / len(in_tokens), 1)
                if in_tokens else None
            ),
            avg_output_tokens=(
                round(sum(out_tokens) / len(out_tokens), 1)
                if out_tokens else None
            ),
            total_estimated_cost_usd=(
                round(sum(costs), 6) if costs else None
            ),
            avg_cost_per_query_usd=(
                round(sum(costs) / len(costs), 6)
                if costs else None
            ),
            avg_search_iterations=(
                round(sum(iterations) / len(iterations), 2)
                if iterations else None
            ),
        ))

    all_latencies = [
        m.total_latency_ms for m in rows
        if m.total_latency_ms is not None
    ]
    all_costs = [
        m.estimated_cost_usd for m in rows
        if m.estimated_cost_usd is not None
    ]

    return MetricsResponse(
        total_queries=len(rows),
        time_range_hours=hours,
        by_provider=summaries,
        overall_avg_latency_ms=(
            round(sum(all_latencies) / len(all_latencies), 1)
            if all_latencies else None
        ),
        overall_total_cost_usd=(
            round(sum(all_costs), 6) if all_costs else None
        ),
    )
