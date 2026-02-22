# =============================================================================
# Token-Based Text Chunker — tiktoken
# =============================================================================
#
# Splits parsed documents into token-based chunks with metadata preservation.
# Each chunk is sized to fit within embedding model limits and annotated with
# source page numbers, section titles, and content type flags.
#
# DESIGN DECISION: Token-based chunking (not character-based) because:
# 1. Aligns with LLM/embedding token limits — no surprises at inference time
# 2. tiktoken uses the same BPE tokenizer as OpenAI models
# 3. Character-based splitting can cut in the middle of multi-byte tokens
# 4. Token counts are exact — critical for benchmarking chunk sizes
#
# DESIGN DECISION: Chunk size is a function parameter (not hardcoded).
# Most projects pick 512 tokens by intuition. We support configurable sizes
# (256, 512, 1024) so we can benchmark empirically and let data decide.
# The /benchmark/retrieval endpoint in Phase 5 automates this comparison.
#
# ALGORITHM:
# 1. Concatenate all parsed elements with \n\n separators
# 2. Build a parallel mapping: character position → source element
# 3. Encode full text into tokens using tiktoken (cl100k_base)
# 4. Slide a window of chunk_size tokens with chunk_overlap overlap
# 5. For each window: decode to text, look up metadata from char mapping
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import tiktoken

from app.services.parser import ParsedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ChunkResult:
    """
    A single chunk ready for embedding and storage.

    Contains the text content, positional metadata (page, index),
    token count, and rich metadata for retrieval filtering.
    """

    content: str  # The text content of this chunk
    page_number: int  # Starting page of this chunk (1-indexed)
    chunk_index: int  # 0-indexed position within the document
    token_count: int  # Exact token count (from tiktoken)
    metadata: dict = field(default_factory=dict)
    # metadata keys:
    #   section_title: str | None — current section header
    #   contains_table: bool — whether this chunk includes tabular content
    #   source_pages: list[int] — all pages this chunk spans
    #   element_types: list[str] — types of elements in this chunk


# ---------------------------------------------------------------------------
# Tiktoken Encoder — Cached Singleton
# ---------------------------------------------------------------------------
# DESIGN DECISION: Cache the tiktoken encoder at module level.
# Loading the encoder reads a ~1.7MB BPE file from disk. Caching it
# avoids repeated I/O across multiple chunk_document() calls.
#
# We use cl100k_base because it's the encoding for text-embedding-3-small.
# Using the same tokenizer ensures our token counts match what the
# embedding model actually sees.
# ---------------------------------------------------------------------------

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Lazily initialize and cache the tiktoken encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    parsed_doc: ParsedDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[ChunkResult]:
    """
    Split a parsed document into token-based chunks with metadata.

    Args:
        parsed_doc: The parsed document from the parser.
        chunk_size: Maximum tokens per chunk (default 512).
        chunk_overlap: Token overlap between consecutive chunks (default 50).

    Returns:
        List of ChunkResult in document order.

    DESIGN DECISION: We cut at exact token boundaries to guarantee size
    limits. This is more important than paragraph-aware splitting because:
    1. Embedding models have hard token limits (8191 for OpenAI)
    2. Benchmarking requires exact chunk sizes for fair comparison
    3. The overlap window recovers context lost at boundaries

    PERFORMANCE NOTE: The char-to-element mapping approach is O(n) in
    document size for each chunk. For documents under 100K tokens
    (typical financial PDFs), this completes in milliseconds. For very
    large documents, a precomputed token-to-char offset map would be
    more efficient.

    Pipeline position: Step 2 of ingestion (parse → chunk → embed → store).
    """
    encoder = _get_encoder()

    if not parsed_doc.elements:
        logger.warning("No elements to chunk in '%s'", parsed_doc.filename)
        return []

    # --- Step 1: Build annotated text ---
    # Concatenate elements with paragraph separators, tracking which
    # element each character position came from.
    separator = "\n\n"
    text_parts: list[str] = []
    # Maps each character position in the full text to an element index
    char_to_element_idx: list[int] = []

    for i, element in enumerate(parsed_doc.elements):
        if i > 0:
            # Add separator between elements
            text_parts.append(separator)
            char_to_element_idx.extend([i - 1] * len(separator))

        text_parts.append(element.text)
        char_to_element_idx.extend([i] * len(element.text))

    full_text = "".join(text_parts)

    # --- Step 2: Encode full text into tokens ---
    all_tokens = encoder.encode(full_text)
    total_tokens = len(all_tokens)

    if total_tokens == 0:
        logger.warning("No tokens after encoding '%s'", parsed_doc.filename)
        return []

    logger.info(
        "Chunking '%s': %d tokens total, chunk_size=%d, overlap=%d",
        parsed_doc.filename, total_tokens, chunk_size, chunk_overlap,
    )

    # --- Step 3: Precompute token-to-character offsets ---
    # For each token position, store the character offset where it starts.
    # This lets us map token windows back to character ranges efficiently.
    token_char_offsets = _build_token_offsets(encoder, all_tokens)

    # --- Step 4: Sliding window over tokens ---
    chunks: list[ChunkResult] = []
    step = max(chunk_size - chunk_overlap, 1)

    for chunk_idx, start in enumerate(range(0, total_tokens, step)):
        end = min(start + chunk_size, total_tokens)
        token_window = all_tokens[start:end]

        # Decode tokens back to text
        chunk_text = encoder.decode(token_window).strip()
        if not chunk_text:
            continue

        # --- Step 5: Look up metadata for this chunk ---
        char_start = token_char_offsets[start]
        char_end = (
            token_char_offsets[end] if end < len(token_char_offsets)
            else len(char_to_element_idx)
        )

        # Find which elements appear in this character range
        element_indices = set()
        for ci in range(char_start, min(char_end, len(char_to_element_idx))):
            element_indices.add(char_to_element_idx[ci])

        chunk_elements = [
            parsed_doc.elements[i]
            for i in sorted(element_indices)
            if i < len(parsed_doc.elements)
        ]

        # Extract metadata from the elements in this chunk
        page_number = chunk_elements[0].page_number if chunk_elements else 1
        source_pages = sorted(set(e.page_number for e in chunk_elements if e.page_number > 0))
        section_title = next(
            (e.section_title for e in chunk_elements if e.section_title),
            None,
        )
        contains_table = any(e.element_type == "table" for e in chunk_elements)
        element_types = sorted(set(e.element_type for e in chunk_elements))

        chunks.append(ChunkResult(
            content=chunk_text,
            page_number=page_number,
            chunk_index=chunk_idx,
            token_count=len(token_window),
            metadata={
                "section_title": section_title,
                "contains_table": contains_table,
                "source_pages": source_pages,
                "element_types": element_types,
            },
        ))

        # Stop if we've reached the end of the document
        if end >= total_tokens:
            break

    logger.info(
        "Chunked '%s' into %d chunks (avg %d tokens/chunk)",
        parsed_doc.filename,
        len(chunks),
        total_tokens // max(len(chunks), 1),
    )

    return chunks


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _build_token_offsets(
    encoder: tiktoken.Encoding,
    tokens: list[int],
) -> list[int]:
    """
    Build a mapping from token index to character offset in the original text.

    For each token position i, token_char_offsets[i] is the character index
    where that token starts in the decoded full text. This avoids the O(n^2)
    cost of decoding prefixes inside the chunking loop.

    Returns:
        List of character offsets, one per token, plus a sentinel at the end.
    """
    offsets: list[int] = []
    char_pos = 0

    for token in tokens:
        offsets.append(char_pos)
        # Decode single token to get its character length
        token_text = encoder.decode([token])
        char_pos += len(token_text)

    # Sentinel: offset for "one past the last token"
    offsets.append(char_pos)
    return offsets
