# =============================================================================
# Ask API — Agentic Document Q&A Endpoint
# =============================================================================
#
# Provides the POST /ask endpoint that invokes the LangGraph agent graph
# to answer questions about ingested financial documents.
#
# FLOW:
#   1. Receive question + optional document_id + optional capability
#   2. Invoke the agent graph (classify → search → analyse)
#   3. Return answer with source citations
#   4. (Phase 4) Record timing + metrics as background task
#
# The heavy lifting happens in the agents package:
#   - orchestrator.py manages the graph
#   - search.py handles agentic retrieval with self-evaluation
#   - analyst.py generates capability-specific answers
#
# This endpoint is thin by design — just request validation, error
# handling, and response mapping.
# =============================================================================

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from app.agents.orchestrator import ask
from app.api.deps import check_document_access, check_scope, get_current_api_key
from app.db.engine import async_session_factory
from app.db.models import ApiKey, QueryMetric
from app.models.requests import AskRequest
from app.models.responses import AskResponse, SourceChunk
from app.services.pricing import estimate_cost

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Question Answering"])


# ---------------------------------------------------------------------------
# POST /ask — Ask a question about financial documents
# ---------------------------------------------------------------------------


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about financial documents",
    description=(
        "Submit a question to the agentic search system. The agent will "
        "autonomously search for relevant document chunks, evaluate "
        "retrieval quality, refine queries if needed, and generate an "
        "answer with source citations."
    ),
)
async def ask_endpoint(
    http_request: Request,
    request: AskRequest,
    background_tasks: BackgroundTasks,
    api_key: ApiKey | None = Depends(get_current_api_key),
) -> AskResponse:
    """
    Invoke the LangGraph agent graph to answer a financial document question.

    The pipeline:
    1. Classify intent → determine capability (qa/summarise/compare/extract)
    2. Agentic search → embed, retrieve, evaluate, refine (up to 3 iterations)
    3. Analyse → generate answer with capability-specific system prompt
    4. (Background) Record timing + metrics to query_metrics table

    Error handling:
    - Missing API key → 503 Service Unavailable
    - LLM API errors → 502 Bad Gateway
    - No results → 200 with "no relevant info" message (not an error)
    """
    # Auth: scope + document access check
    check_scope(api_key, "ask")
    check_document_access(api_key, request.document_id)

    # Audit context
    http_request.state.audit_document_id = request.document_id
    http_request.state.audit_question = request.question

    logger.info(
        "Ask request: question='%s', document_id=%s, capability=%s",
        request.question[:80],
        request.document_id,
        request.capability,
    )

    start_time = time.monotonic()

    try:
        result = await ask(
            question=request.question,
            document_id=request.document_id,
            capability=request.capability,
        )
    except ValueError as e:
        # Missing API key or configuration error
        logger.error("Configuration error: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"Service configuration error: {e}",
        ) from e
    except Exception as e:
        # LLM API errors, network issues, etc.
        logger.exception("Agent graph failed: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"LLM service error: {e}",
        ) from e

    total_latency_ms = int((time.monotonic() - start_time) * 1000)

    # Map the agent state to the response model
    sources = [
        SourceChunk(**source) for source in result.get("sources", [])
    ]

    # Schedule metric persistence as a background task
    # DESIGN DECISION: Background task uses its own session (not the
    # request session) to avoid lifecycle issues. FastAPI's
    # BackgroundTasks run after the response is sent but within the
    # same request lifecycle.
    background_tasks.add_task(
        _persist_metric,
        question=request.question,
        document_id=request.document_id,
        capability=result.get("classified_capability"),
        model=result.get("model"),
        total_latency_ms=total_latency_ms,
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
        retrieval_count=result.get("retrieval_count", 0),
        retrieval_scores=[
            s.get("similarity_score")
            for s in result.get("sources", [])
        ],
    )

    return AskResponse(
        answer=result.get("answer", "No answer generated."),
        sources=sources,
        question=request.question,
        document_id=request.document_id,
        model=result.get("model", "unknown"),
        retrieval_count=result.get("retrieval_count", 0),
    )


# ---------------------------------------------------------------------------
# Background Metric Persistence
# ---------------------------------------------------------------------------


async def _persist_metric(
    question: str,
    document_id: int | None,
    capability: str | None,
    model: str | None,
    total_latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    retrieval_count: int,
    retrieval_scores: list | None,
) -> None:
    """
    Persist a QueryMetric row in the background.

    Uses its own DB session to avoid sharing the request session
    (which may already be closed by the time this runs).
    """
    try:
        # Estimate cost using the default provider type from config
        from app.config import settings
        provider_type = settings.llm_provider
        cost = estimate_cost(
            provider_type, model or "", input_tokens, output_tokens,
        )

        async with async_session_factory() as session:
            metric = QueryMetric(
                document_id=document_id,
                question=question,
                capability=capability,
                model=model,
                total_latency_ms=total_latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                retrieval_count=retrieval_count,
                retrieval_scores=retrieval_scores,
                source="ask",
            )
            session.add(metric)
            await session.commit()
    except Exception as e:
        logger.warning("Failed to persist ask metric: %s", e)
