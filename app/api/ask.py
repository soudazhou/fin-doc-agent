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

from fastapi import APIRouter, HTTPException

from app.agents.orchestrator import ask
from app.models.requests import AskRequest
from app.models.responses import AskResponse, SourceChunk

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
async def ask_endpoint(request: AskRequest) -> AskResponse:
    """
    Invoke the LangGraph agent graph to answer a financial document question.

    The pipeline:
    1. Classify intent → determine capability (qa/summarise/compare/extract)
    2. Agentic search → embed, retrieve, evaluate, refine (up to 3 iterations)
    3. Analyse → generate answer with capability-specific system prompt

    Error handling:
    - Missing API key → 503 Service Unavailable
    - LLM API errors → 502 Bad Gateway
    - No results → 200 with "no relevant info" message (not an error)
    """
    logger.info(
        "Ask request: question='%s', document_id=%s, capability=%s",
        request.question[:80],
        request.document_id,
        request.capability,
    )

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

    # Map the agent state to the response model
    sources = [
        SourceChunk(**source) for source in result.get("sources", [])
    ]

    return AskResponse(
        answer=result.get("answer", "No answer generated."),
        sources=sources,
        question=request.question,
        document_id=request.document_id,
        model=result.get("model", "unknown"),
        retrieval_count=result.get("retrieval_count", 0),
    )
