# =============================================================================
# Unit Tests — Chunker Service
# =============================================================================
#
# Tests the token-based chunking logic without external dependencies.
# No API keys, databases, or network calls needed.
# =============================================================================

from app.services.chunker import chunk_document
from app.services.parser import ParsedDocument, ParsedElement


def _make_parsed_doc(
    texts: list[str],
    page_numbers: list[int] | None = None,
    element_types: list[str] | None = None,
) -> ParsedDocument:
    """Helper to build a ParsedDocument from simple text lists."""
    pages = page_numbers or [1] * len(texts)
    types = element_types or ["text"] * len(texts)
    elements = [
        ParsedElement(
            text=text,
            page_number=page,
            element_type=etype,
            section_title=None,
            level=0,
        )
        for text, page, etype in zip(texts, pages, types, strict=True)
    ]
    return ParsedDocument(
        elements=elements,
        page_count=max(pages) if pages else 0,
        filename="test.pdf",
    )


class TestChunkDocument:
    """Tests for chunk_document()."""

    def test_empty_document_returns_no_chunks(self):
        doc = _make_parsed_doc([])
        chunks = chunk_document(doc, chunk_size=64, chunk_overlap=10)
        assert chunks == []

    def test_single_short_element_produces_one_chunk(self):
        doc = _make_parsed_doc(["This is a short sentence."])
        chunks = chunk_document(doc, chunk_size=64, chunk_overlap=10)
        assert len(chunks) == 1
        assert "short sentence" in chunks[0].content
        assert chunks[0].chunk_index == 0
        assert chunks[0].page_number == 1

    def test_chunk_indices_are_sequential(self):
        # Long enough text to produce multiple chunks
        doc = _make_parsed_doc(["word " * 200])
        chunks = chunk_document(doc, chunk_size=32, chunk_overlap=5)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_token_count_respects_chunk_size(self):
        doc = _make_parsed_doc(["Revenue grew by 15% year over year. " * 100])
        chunks = chunk_document(doc, chunk_size=64, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.token_count <= 64

    def test_overlap_produces_more_chunks(self):
        text = "Financial analysis is important. " * 50
        doc = _make_parsed_doc([text])
        chunks_no_overlap = chunk_document(doc, chunk_size=64, chunk_overlap=0)
        chunks_with_overlap = chunk_document(doc, chunk_size=64, chunk_overlap=20)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_page_numbers_propagated(self):
        doc = _make_parsed_doc(
            texts=["Page one content.", "Page two content."],
            page_numbers=[1, 2],
        )
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=0)
        assert len(chunks) >= 1
        # With a large chunk size, both elements fit in one chunk,
        # but the page_number should be from the first element
        assert chunks[0].page_number >= 1

    def test_metadata_contains_section_title(self):
        elements = [
            ParsedElement(
                text="Introduction to the report.",
                page_number=1,
                element_type="heading",
                section_title="Introduction",
                level=1,
            ),
            ParsedElement(
                text="Detailed analysis follows.",
                page_number=1,
                element_type="text",
                section_title="Introduction",
                level=0,
            ),
        ]
        doc = ParsedDocument(elements=elements, page_count=1, filename="test.pdf")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=0)
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("section_title") == "Introduction"

    def test_table_flag_in_metadata(self):
        doc = _make_parsed_doc(
            texts=["Some text.", "| Q1 | $100M |"],
            element_types=["text", "table"],
        )
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=0)
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("contains_table") is True

    def test_whitespace_only_elements_skipped(self):
        doc = _make_parsed_doc(["   ", "\n\n", "Actual content here."])
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=0)
        assert len(chunks) == 1
        assert "Actual content" in chunks[0].content
