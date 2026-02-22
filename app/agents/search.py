# =============================================================================
# Agentic Search — Autonomous Retrieval with Self-Evaluation
# =============================================================================
#
# The search agent is the core differentiator from naive RAG. Instead of
# a single embed→retrieve→generate pipeline, the agent:
#
# 1. EMBED + RETRIEVE — vector similarity search
# 2. EVALUATE — LLM judges: "Do these chunks answer the question?"
# 3. REFINE — If insufficient, LLM rewrites the query and we re-search
# 4. REPEAT — Up to max_search_iterations (default 3)
#
# This matters because:
# - The user's query may not match embedding space ("What risks?" won't
#   match a chunk titled "Risk Factors" via cosine similarity alone)
# - Initial retrieval may miss critical information
# - Query rewriting (adding synonyms, restructuring) improves recall
#
# DESIGN DECISION: Plain async function (not a LangGraph subgraph).
# The search loop is simple enough that a Python loop is clearer than
# a graph. LangGraph adds value at the orchestrator level (state mgmt,
# routing) but would over-complicate a 3-iteration loop.
#
# DESIGN DECISION: Self-evaluation uses the same LLM as the analyst.
# No separate "judge" model — keeps costs down and config simple.
# =============================================================================

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.services.embedder import embed_query
from app.services.llm import LLMProvider
from app.services.vectorstore import VectorSearchResult, get_vector_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class SearchIteration:
    """Record of a single search iteration for the trace."""

    query: str
    num_results: int
    avg_similarity: float
    evaluation: str  # "sufficient" or reason for refinement


@dataclass
class SearchResult:
    """
    Complete result from the agentic search loop.

    Contains the retrieved chunks plus a trace of all iterations,
    useful for debugging and transparency in responses.
    """

    chunks: list[VectorSearchResult]
    search_trace: list[SearchIteration] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Self-Evaluation Prompt
# ---------------------------------------------------------------------------

_EVALUATION_SYSTEM = """You are a retrieval quality evaluator for a \
financial document Q&A system.

Your job: determine whether the retrieved chunks contain enough \
information to answer the user's question.

Respond with ONLY valid JSON (no markdown, no explanation):
{
  "sufficient": true or false,
  "reason": "Brief explanation of your judgment",
  "refined_query": "A better search query if insufficient, omit if sufficient"
}

Guidelines:
- "sufficient" = the chunks contain the specific data needed to answer
- If the question asks for a number and no chunk contains that number, \
it's insufficient
- If chunks are tangentially related but don't directly answer, \
it's insufficient
- When suggesting a refined query, try: synonyms, more specific terms, \
or different phrasing"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def agentic_search(
    question: str,
    document_id: int | None,
    llm: LLMProvider,
) -> SearchResult:
    """
    Perform agentic search: retrieve, evaluate, refine in a loop.

    This is the core search function that the orchestrator calls.
    It handles its own iteration internally — the caller just gets
    the final result.

    Args:
        question: The user's question to search for.
        document_id: Optional filter to search within a specific document.
        llm: LLM provider for self-evaluation of retrieval quality.

    Returns:
        SearchResult with deduplicated chunks and search trace.
    """
    vector_store = get_vector_store()
    max_iterations = settings.max_search_iterations
    top_k = settings.retrieval_top_k
    threshold = settings.retrieval_similarity_threshold

    # Track all unique chunks across iterations (deduplicate by chunk_id)
    seen_chunk_ids: set[int] = set()
    all_chunks: list[VectorSearchResult] = []
    trace: list[SearchIteration] = []

    current_query = question

    for iteration in range(max_iterations):
        logger.info(
            "Search iteration %d/%d: query='%s'",
            iteration + 1, max_iterations, current_query,
        )

        # --- Step 1: Embed and retrieve ---
        embedding = await asyncio.to_thread(embed_query, current_query)
        results = await vector_store.search(
            query_embedding=embedding,
            top_k=top_k,
            document_id=document_id,
        )

        # Filter by similarity threshold
        filtered = [
            r for r in results if r.similarity_score >= threshold
        ]

        # Collect unique chunks
        new_chunks = []
        for chunk in filtered:
            if chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.chunk_id)
                all_chunks.append(chunk)
                new_chunks.append(chunk)

        avg_sim = (
            sum(r.similarity_score for r in filtered) / len(filtered)
            if filtered
            else 0.0
        )

        logger.info(
            "Iteration %d: %d results (%d new), avg_similarity=%.3f",
            iteration + 1, len(filtered), len(new_chunks), avg_sim,
        )

        # --- Step 2: Fast path — skip evaluation if confident ---
        # If the top result is very similar and we have enough chunks,
        # no need to spend an LLM call on evaluation.
        if (
            filtered
            and filtered[0].similarity_score > 0.9
            and len(all_chunks) >= 3
        ):
            trace.append(SearchIteration(
                query=current_query,
                num_results=len(filtered),
                avg_similarity=round(avg_sim, 4),
                evaluation="sufficient (high similarity fast path)",
            ))
            logger.info("Fast path: high similarity, skipping evaluation")
            break

        # --- Step 3: No results at all — record and break ---
        if not filtered:
            trace.append(SearchIteration(
                query=current_query,
                num_results=0,
                avg_similarity=0.0,
                evaluation="no results above threshold",
            ))
            # If this is the last iteration, break anyway
            if iteration == max_iterations - 1:
                break
            # Try a simplified version of the query for next iteration
            current_query = _simplify_query(question)
            continue

        # --- Step 4: LLM evaluates retrieval quality ---
        # On the last iteration, skip evaluation — we'll use what we have
        if iteration == max_iterations - 1:
            trace.append(SearchIteration(
                query=current_query,
                num_results=len(filtered),
                avg_similarity=round(avg_sim, 4),
                evaluation="final iteration (using best results)",
            ))
            break

        evaluation = await _evaluate_retrieval(
            question=question,
            chunks=filtered,
            llm=llm,
        )

        trace.append(SearchIteration(
            query=current_query,
            num_results=len(filtered),
            avg_similarity=round(avg_sim, 4),
            evaluation=evaluation.get("reason", "unknown"),
        ))

        if evaluation.get("sufficient", False):
            logger.info("Evaluation: sufficient — stopping search")
            break

        # --- Step 5: Refine query for next iteration ---
        refined = evaluation.get("refined_query", "")
        if refined and refined != current_query:
            logger.info(
                "Evaluation: insufficient — refining query to '%s'",
                refined,
            )
            current_query = refined
        else:
            # LLM didn't suggest a refinement — nothing more to try
            logger.info("Evaluation: insufficient but no refinement suggested")
            break

    # Sort all collected chunks by similarity (highest first)
    all_chunks.sort(key=lambda c: c.similarity_score, reverse=True)

    logger.info(
        "Agentic search complete: %d total chunks, %d iterations",
        len(all_chunks), len(trace),
    )

    return SearchResult(chunks=all_chunks, search_trace=trace)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


async def _evaluate_retrieval(
    question: str,
    chunks: list[VectorSearchResult],
    llm: LLMProvider,
) -> dict:
    """
    Ask the LLM whether retrieved chunks are sufficient to answer
    the question.

    Returns a dict with keys: sufficient (bool), reason (str),
    refined_query (str, optional).
    """
    # Format chunks for the evaluation prompt
    chunk_text = _format_chunks_for_eval(chunks)

    user_message = (
        f"Question: {question}\n\n"
        f"Retrieved chunks ({len(chunks)} total):\n\n{chunk_text}"
    )

    try:
        response = await llm.complete(
            messages=[{"role": "user", "content": user_message}],
            system=_EVALUATION_SYSTEM,
            temperature=0.0,  # Deterministic evaluation
            max_tokens=256,   # Short JSON response
        )

        # Parse JSON response
        return json.loads(response.content)

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(
            "Failed to parse evaluation response: %s. "
            "Treating as sufficient.",
            e,
        )
        return {"sufficient": True, "reason": "parse error, proceeding"}

    except Exception as e:
        logger.warning(
            "Evaluation LLM call failed: %s. "
            "Treating as sufficient to avoid blocking.",
            e,
        )
        return {"sufficient": True, "reason": f"LLM error: {e}"}


def _format_chunks_for_eval(chunks: list[VectorSearchResult]) -> str:
    """Format chunks for the self-evaluation prompt."""
    sections = []
    for i, chunk in enumerate(chunks, 1):
        page = f" (page {chunk.page_number})" if chunk.page_number else ""
        sim = f" [similarity: {chunk.similarity_score:.3f}]"
        sections.append(
            f"--- Chunk {i}{page}{sim} ---\n{chunk.content}"
        )
    return "\n\n".join(sections)


def _simplify_query(question: str) -> str:
    """
    Create a simplified version of the query for retry.

    A basic fallback when LLM evaluation isn't available or doesn't
    suggest a refinement. Extracts key nouns/phrases by removing
    common question words.
    """
    # Remove common question prefixes
    removals = [
        "what is", "what are", "what was", "what were",
        "how much", "how many", "how does", "how did",
        "can you", "could you", "please",
        "tell me about", "explain",
    ]
    simplified = question.lower()
    for phrase in removals:
        simplified = simplified.replace(phrase, "")
    return simplified.strip() or question
