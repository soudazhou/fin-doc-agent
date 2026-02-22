# =============================================================================
# Analyst Agent — Capability-Specific Answer Generation
# =============================================================================
#
# The analyst takes retrieved chunks and the user's question, then
# generates an answer using the configured LLM with capability-specific
# system prompts.
#
# CAPABILITIES:
#   qa       — Direct Q&A with source citations
#   summarise — Structured document summary
#   compare  — Cross-document comparison
#   extract  — Structured data extraction (JSON)
#
# DESIGN DECISION: Capability-specific system prompts.
# Each capability has a different instruction set that guides the LLM's
# output format and grounding behavior. This is more reliable than a
# single generic prompt because:
# 1. Specific instructions produce more consistent output formats
# 2. Grounding constraints (e.g., "ONLY the provided context") reduce
#    hallucination for factual Q&A
# 3. Structured output instructions (e.g., "return JSON") for extract
#    produce machine-readable results
#
# DESIGN DECISION: Context formatted with numbered references.
# Chunks are presented as [1], [2], etc. so the LLM can cite specific
# sources in its answer. This enables source tracing in the response.
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.llm import LLMProvider
from app.services.vectorstore import VectorSearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Result from the analyst agent."""

    answer: str
    model: str
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# Capability-Specific System Prompts
# ---------------------------------------------------------------------------
# Each prompt follows the same pattern:
# 1. Role definition
# 2. Grounding instruction (use ONLY the provided context)
# 3. Output format guidance
# 4. Citation instruction
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "qa": (
        "You are a financial document analyst. Answer the user's question "
        "using ONLY the provided context from financial documents.\n\n"
        "Rules:\n"
        "- Base your answer exclusively on the provided context\n"
        "- Cite sources using [1], [2], etc. matching the chunk numbers\n"
        "- Include page numbers when available\n"
        "- If the answer is not in the context, explicitly state: "
        "'The provided documents do not contain this information.'\n"
        "- Be precise with financial figures — never round or estimate\n"
        "- Keep your answer concise and directly relevant"
    ),

    "summarise": (
        "You are a financial document analyst. Create a structured summary "
        "of the provided document content.\n\n"
        "Rules:\n"
        "- Organise the summary with clear headings and bullet points\n"
        "- Cover the key topics present in the context\n"
        "- Cite page numbers for major points\n"
        "- Highlight key financial metrics, risks, and notable items\n"
        "- Keep the summary comprehensive but concise\n"
        "- Use ONLY information from the provided context"
    ),

    "compare": (
        "You are a financial document analyst. Compare the requested "
        "aspects across the provided document content.\n\n"
        "Rules:\n"
        "- Present the comparison in a structured format\n"
        "- Use a markdown table where appropriate\n"
        "- Cite sources using [1], [2], etc. for each data point\n"
        "- Highlight key differences and similarities\n"
        "- If data for comparison is missing, note what's unavailable\n"
        "- Use ONLY information from the provided context"
    ),

    "extract": (
        "You are a financial document analyst. Extract the requested "
        "structured data from the provided context.\n\n"
        "Rules:\n"
        "- Return the extracted data as valid JSON\n"
        "- Include a 'data' key with the extracted values\n"
        "- Include a 'sources' key with page references\n"
        "- Be exact with numbers — copy them verbatim from the source\n"
        "- If requested data is not found, include null values with a note\n"
        "- Use ONLY information from the provided context"
    ),
}

# Default prompt for any unrecognised capability
_DEFAULT_SYSTEM = SYSTEM_PROMPTS["qa"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def analyse(
    question: str,
    chunks: list[VectorSearchResult],
    capability: str,
    llm: LLMProvider,
) -> AnalysisResult:
    """
    Generate an answer from retrieved chunks using the appropriate
    capability-specific system prompt.

    Args:
        question: The user's original question.
        chunks: Retrieved document chunks (sorted by relevance).
        capability: One of "qa", "summarise", "compare", "extract".
        llm: LLM provider to use for generation.

    Returns:
        AnalysisResult with the generated answer and usage metrics.
    """
    # Handle empty chunks gracefully
    if not chunks:
        return AnalysisResult(
            answer=(
                "No relevant information was found in the documents. "
                "Please try rephrasing your question or check that the "
                "document has been ingested."
            ),
            model="n/a",
            input_tokens=0,
            output_tokens=0,
        )

    system_prompt = SYSTEM_PROMPTS.get(capability, _DEFAULT_SYSTEM)
    context = _format_context(chunks)

    user_message = (
        f"Question: {question}\n\n"
        f"Context ({len(chunks)} document chunks):\n\n{context}"
    )

    logger.info(
        "Analyst generating answer: capability=%s, chunks=%d",
        capability, len(chunks),
    )

    response = await llm.complete(
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt,
    )

    logger.info(
        "Analyst complete: model=%s, tokens=%d+%d",
        response.model, response.input_tokens, response.output_tokens,
    )

    return AnalysisResult(
        answer=response.content,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _format_context(chunks: list[VectorSearchResult]) -> str:
    """
    Format retrieved chunks as numbered context for the LLM.

    Each chunk is labeled [1], [2], etc. with page number if available.
    This enables the LLM to cite specific sources in its answer.

    Example output:
        [1] (page 12):
        Revenue for Q3 2024 was $4.2 billion, up 15% year-over-year...

        ---

        [2] (page 15):
        Operating expenses decreased by 3% compared to Q2...
    """
    sections = []
    for i, chunk in enumerate(chunks, 1):
        page_label = (
            f" (page {chunk.page_number})" if chunk.page_number else ""
        )
        sections.append(f"[{i}]{page_label}:\n{chunk.content}")
    return "\n\n---\n\n".join(sections)
