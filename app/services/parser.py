# =============================================================================
# PDF Parser — Docling Document Intelligence
# =============================================================================
#
# Parses financial PDFs using IBM's Docling library, extracting text, tables,
# and section headings with page number and structural metadata.
#
# DESIGN DECISION: Docling over PyPDF/pdfplumber/PyMuPDF because:
# 1. Purpose-built for structured documents (financial tables!)
# 2. Understands multi-header tables, spanning cells, hierarchies
# 3. Built-in OCR for scanned PDFs
# 4. Open-source (MIT license) with active IBM Research backing
#
# DESIGN DECISION: We iterate items (not export_to_markdown()) because:
# 1. export_to_markdown() loses page numbers entirely
# 2. We need per-element page_number for chunk metadata
# 3. We need element_type to distinguish tables from text
# 4. Section headers must be tracked for metadata propagation
#
# DESIGN DECISION: We use our own dataclasses (ParsedElement, ParsedDocument)
# rather than passing Docling types downstream. This decouples the chunker
# and embedder from the parsing library — if we switch from Docling to
# another parser, only this module changes.
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.labels import DocItemLabel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ParsedElement:
    """
    A single structural element extracted from a PDF document.

    Each element corresponds to one paragraph, heading, or table from the
    original PDF, annotated with its page number and section context.
    """

    text: str  # The text content (markdown for tables)
    page_number: int  # 1-indexed page where this element appears
    element_type: str  # "text", "table", or "heading"
    section_title: str | None = None  # Current section header (for metadata)
    level: int = 0  # Heading level (0 = body text, 1+ = heading depth)


@dataclass
class ParsedDocument:
    """
    The complete result of parsing a PDF document.

    Contains all extracted elements in reading order, plus document-level
    metadata (page count, filename) needed by downstream pipeline stages.
    """

    elements: list[ParsedElement] = field(default_factory=list)
    page_count: int = 0
    filename: str = ""


# ---------------------------------------------------------------------------
# Docling Converter — Lazy Singleton
# ---------------------------------------------------------------------------
# DESIGN DECISION: Reuse a single DocumentConverter instance.
# Initialization loads ML models into memory (~2-5 seconds on first use).
# Creating a new instance per document would waste time and memory.
# The converter is thread-safe for concurrent Celery workers.
# ---------------------------------------------------------------------------

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    """Lazily initialize and cache the Docling DocumentConverter."""
    global _converter
    if _converter is None:
        logger.info(
            "Initializing Docling DocumentConverter "
            "(first use, may take a few seconds)..."
        )

        # DESIGN DECISION: Enable table structure extraction and OCR.
        # Financial documents are heavily table-based (income statements,
        # balance sheets, cash flow statements). OCR handles scanned pages.
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = True

        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            }
        )
        logger.info("Docling DocumentConverter initialized")
    return _converter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_pdf(file_path: str) -> ParsedDocument:
    """
    Parse a PDF file using Docling, extracting text, tables, and headings
    with page number and section tracking.

    Args:
        file_path: Absolute path to the PDF file on disk.

    Returns:
        ParsedDocument containing all extracted elements in reading order.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If Docling fails to convert the document.

    Pipeline position: Step 1 of ingestion (parse → chunk → embed → store).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info("Parsing PDF: %s", path.name)
    converter = _get_converter()

    try:
        result = converter.convert(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Docling failed to parse '{path.name}': {exc}"
        ) from exc

    elements: list[ParsedElement] = []
    current_section: str | None = None
    page_numbers_seen: set[int] = set()

    # Iterate all document items in reading order.
    # Each item has a `label` attribute (DocItemLabel enum) and optionally
    # `prov` (provenance) with page number and bounding box.
    for item, level in result.document.iterate_items():
        # --- Extract page number from provenance ---
        # item.prov is a list of ProvenanceItem; [0] is the primary location.
        # Some items may lack provenance (rare); default to 0.
        page_no = 0
        if hasattr(item, "prov") and item.prov:
            page_no = item.prov[0].page_no
        page_numbers_seen.add(page_no)

        # --- Get label for type detection ---
        label = getattr(item, "label", None)

        # --- Section headers: track for metadata propagation ---
        if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            text = getattr(item, "text", "").strip()
            if text:
                current_section = text
                elements.append(ParsedElement(
                    text=text,
                    page_number=page_no,
                    element_type="heading",
                    section_title=current_section,
                    level=level,
                ))

        # --- Tables: export as markdown for LLM consumption ---
        elif label == DocItemLabel.TABLE:
            # DESIGN DECISION: Export tables as markdown-formatted text.
            # LLMs understand pipe-delimited markdown tables better than
            # raw CSV or HTML. We use pandas to_markdown() via the
            # Docling export_to_dataframe() method.
            table_md = _table_to_markdown(item, result.document)
            if table_md:
                elements.append(ParsedElement(
                    text=table_md,
                    page_number=page_no,
                    element_type="table",
                    section_title=current_section,
                    level=level,
                ))

        # --- Regular text paragraphs ---
        elif label in (DocItemLabel.TEXT, DocItemLabel.LIST_ITEM,
                       DocItemLabel.CAPTION, DocItemLabel.FOOTNOTE):
            text = getattr(item, "text", "").strip()
            if text:
                elements.append(ParsedElement(
                    text=text,
                    page_number=page_no,
                    element_type="text",
                    section_title=current_section,
                    level=level,
                ))

    # Determine page count from provenance data
    page_count = max(page_numbers_seen) if page_numbers_seen - {0} else 0

    logger.info(
        "Parsed '%s': %d elements (%d headings, %d tables, %d text blocks), %d pages",
        path.name,
        len(elements),
        sum(1 for e in elements if e.element_type == "heading"),
        sum(1 for e in elements if e.element_type == "table"),
        sum(1 for e in elements if e.element_type == "text"),
        page_count,
    )

    return ParsedDocument(
        elements=elements,
        page_count=page_count,
        filename=path.name,
    )


def _table_to_markdown(table_item: object, document: object) -> str:
    """
    Convert a Docling TableItem to a markdown-formatted string.

    Attempts to use export_to_dataframe() → pandas to_markdown().
    Falls back to a simple text representation if that fails.
    """
    try:
        if hasattr(table_item, "export_to_dataframe"):
            df = table_item.export_to_dataframe()
            return df.to_markdown(index=False)
    except Exception as exc:
        logger.warning("Table export to DataFrame failed: %s", exc)

    # Fallback: use text representation if available
    text = getattr(table_item, "text", "")
    return text.strip() if text else ""
