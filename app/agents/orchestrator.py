# =============================================================================
# LangGraph Orchestrator — Agent Graph Assembly
# =============================================================================
#
# The orchestrator wires together the classify, search, and analyse steps
# into a LangGraph StateGraph. This provides:
#
# 1. Explicit state management — all data flows through AgentState
# 2. Debuggable execution — each node's input/output is traceable
# 3. Future extensibility — easy to add conditional edges, new nodes
#
# GRAPH TOPOLOGY:
#   START ──▶ classify ──▶ search ──▶ analyse ──▶ END
#
# DESIGN DECISION: Linear graph (no conditional edges).
# Classification is rule-based (free/instant), search always runs,
# analyse always runs. The agentic loop (self-evaluation, query
# refinement) happens INSIDE the search node, not as graph edges.
# This keeps the graph simple and debuggable.
#
# DESIGN DECISION: Plain TypedDict state (not MessagesState).
# We're not building a chatbot — no need for LangGraph's message
# history management. Our state is structured data flowing through
# a pipeline: question → capability → chunks → answer.
#
# DESIGN DECISION: Graph compiled once at module level.
# LangGraph graph compilation is not free. Compiling on import and
# reusing the compiled graph avoids per-request overhead.
#
# DESIGN DECISION: Rule-based classification over LLM.
# Saves an API call, executes instantly, and is accurate enough for
# 4 well-defined capabilities. LLM classification can be swapped in
# later if needed (just change the classify node).
# =============================================================================

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agents.analyst import AnalysisResult, analyse
from app.agents.search import SearchResult, agentic_search
from app.services.llm import LLMProvider, get_llm_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent State Schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """
    State that flows through the LangGraph graph.

    Uses total=False so nodes only need to return the keys they update.
    LangGraph merges partial updates into the full state.
    """

    # --- Input (set by caller) ---
    question: str
    document_id: int | None
    capability: str | None  # User-provided, or None for auto-classification

    # --- LLM injection (Phase 4) ---
    # When set, nodes use this provider instead of the global singleton.
    # This enables the /compare endpoint to run the same query across
    # multiple providers in parallel, each with its own LLM instance.
    # DESIGN DECISION: Provider object in state rather than a string ID.
    # Nodes need the object to call complete(), and re-parsing the string
    # in every node would duplicate factory logic.
    # NOTE: Not JSON-serialisable. Safe as long as no checkpointer is
    # configured on the graph (current: no checkpointer).
    llm_override: LLMProvider | None

    # --- Intermediate (set by nodes) ---
    classified_capability: str  # Always set after classify node
    search_result: SearchResult | None
    analysis_result: AnalysisResult | None

    # --- Output (set by analyse node) ---
    answer: str
    sources: list[dict[str, Any]]
    model: str
    retrieval_count: int
    input_tokens: int   # Phase 4: for cost calculation in /compare
    output_tokens: int  # Phase 4: for cost calculation in /compare


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------
# Each node receives the full state and returns a partial update dict.
# LangGraph merges the returned dict into the state automatically.
# ---------------------------------------------------------------------------


async def classify_node(state: AgentState) -> dict:
    """
    Determine the query capability (qa, summarise, compare, extract).

    If the user explicitly set a capability, use it. Otherwise, apply
    simple keyword heuristics to auto-classify.

    DESIGN DECISION: Rule-based over LLM classification.
    - Zero latency, zero cost
    - Accurate for 4 well-defined capabilities
    - Easy to test and debug
    - LLM classification can replace this later if needed
    """
    # User explicitly chose a capability
    if state.get("capability"):
        capability = state["capability"]
        logger.info("Using user-specified capability: %s", capability)
        return {"classified_capability": capability}

    # Auto-classify based on question text
    question_lower = state["question"].lower()
    capability = _classify_question(question_lower)
    logger.info(
        "Auto-classified capability: %s (question: '%s')",
        capability, state["question"][:80],
    )
    return {"classified_capability": capability}


async def search_node(state: AgentState) -> dict:
    """
    Execute the agentic search loop.

    Embeds the question, searches the vector store, evaluates
    retrieval quality, and refines the query if needed (up to
    max_search_iterations).
    """
    # Use injected provider if present, otherwise fall back to singleton
    llm = state.get("llm_override") or get_llm_provider()

    result = await agentic_search(
        question=state["question"],
        document_id=state.get("document_id"),
        llm=llm,
    )

    logger.info(
        "Search complete: %d chunks, %d iterations",
        len(result.chunks), len(result.search_trace),
    )

    return {"search_result": result}


async def analyse_node(state: AgentState) -> dict:
    """
    Generate the final answer using the analyst agent.

    Takes the retrieved chunks and capability, calls the LLM with
    the appropriate system prompt, and maps the result to output fields.
    """
    # Use injected provider if present, otherwise fall back to singleton
    llm = state.get("llm_override") or get_llm_provider()
    search_result: SearchResult | None = state.get("search_result")
    chunks = search_result.chunks if search_result else []

    result = await analyse(
        question=state["question"],
        chunks=chunks,
        capability=state["classified_capability"],
        llm=llm,
    )

    # Map VectorSearchResult objects to serialisable dicts for the response
    sources = [
        {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "page_number": chunk.page_number,
            "similarity_score": chunk.similarity_score,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    return {
        "analysis_result": result,
        "answer": result.answer,
        "sources": sources,
        "model": result.model,
        "retrieval_count": len(chunks),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------
# Compiled once at module level. The compiled graph is reusable and
# thread-safe for concurrent FastAPI requests.
# ---------------------------------------------------------------------------

_builder = StateGraph(AgentState)
_builder.add_node("classify", classify_node)
_builder.add_node("search", search_node)
_builder.add_node("analyse", analyse_node)

_builder.add_edge(START, "classify")
_builder.add_edge("classify", "search")
_builder.add_edge("search", "analyse")
_builder.add_edge("analyse", END)

graph = _builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def ask(
    question: str,
    document_id: int | None = None,
    capability: str | None = None,
    llm: LLMProvider | None = None,
) -> AgentState:
    """
    Entry point: invoke the agent graph and return the final state.

    This is what the FastAPI endpoint calls. It sets up the initial
    state and runs the compiled LangGraph graph.

    Args:
        question: The user's question.
        document_id: Optional document to search within.
        capability: Optional explicit capability override.
        llm: Optional LLM provider override. When provided, the graph
            nodes use this instead of the global singleton. Used by
            the /compare endpoint to run multiple providers in parallel.

    Returns:
        The final AgentState with answer, sources, model, etc.
    """
    initial_state: AgentState = {
        "question": question,
        "document_id": document_id,
        "capability": capability,
    }
    if llm is not None:
        initial_state["llm_override"] = llm

    logger.info(
        "Invoking agent graph: question='%s', document_id=%s, "
        "capability=%s",
        question[:80], document_id, capability,
    )

    # LangGraph's ainvoke returns the final state
    result = await graph.ainvoke(initial_state)

    logger.info(
        "Agent graph complete: model=%s, sources=%d",
        result.get("model", "n/a"),
        result.get("retrieval_count", 0),
    )

    return result


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _classify_question(question_lower: str) -> str:
    """
    Rule-based intent classification for 4 capabilities.

    Checks for keyword patterns in order of specificity:
    1. Extract — most specific (asks for structured data)
    2. Compare — involves comparison keywords
    3. Summarise — asks for overview/summary
    4. QA — default fallback (most common)
    """
    # Extract: user wants structured data
    extract_keywords = [
        "extract", "list all", "list every", "table of",
        "as json", "as a table", "pull out", "all the",
        "every mention of", "each quarter",
    ]
    if any(kw in question_lower for kw in extract_keywords):
        return "extract"

    # Compare: user wants comparison
    compare_keywords = [
        "compare", "versus", " vs ", " vs.", "difference between",
        "compared to", "relative to", "how does .* differ",
        "better than", "worse than",
    ]
    if any(kw in question_lower for kw in compare_keywords):
        return "compare"

    # Summarise: user wants overview
    summarise_keywords = [
        "summarise", "summarize", "summary", "overview",
        "key points", "key takeaways", "main findings",
        "give me a brief", "what are the highlights",
    ]
    if any(kw in question_lower for kw in summarise_keywords):
        return "summarise"

    # Default: Q&A
    return "qa"
