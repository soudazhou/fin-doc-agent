# =============================================================================
# API Request Models — Pydantic V2 Schemas
# =============================================================================
#
# These models define the shape of data coming INTO the API.
# FastAPI uses them for:
# 1. Request body validation (automatic 422 errors for invalid data)
# 2. OpenAPI documentation generation (visible at /docs)
# 3. Type hints for IDE autocompletion in route handlers
#
# DESIGN DECISION: Pydantic V2 over V1
# V2 is a complete Rust-powered rewrite with 5-50x faster validation.
# Key V2 features used here:
# - `model_config = ConfigDict(...)` replaces the inner `class Config:`
# - Field constraints via `Field(...)` with clearer error messages
# - Native JSON Schema generation for OpenAPI docs
# =============================================================================

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """
    Request body for POST /ask — Ask a question about a financial document.

    The orchestrator auto-detects the capability from the question, or you
    can specify it explicitly via the `capability` field.

    Example:
        {
            "question": "What was Apple's total revenue in Q3 2024?",
            "document_id": 1,
            "capability": "qa"
        }
    """

    # The natural language question to answer
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to ask about the financial document",
        examples=["What was the total revenue in Q3 2024?"],
    )

    # Optional: restrict search to a specific document
    # If None, searches across ALL ingested documents
    document_id: int | None = Field(
        default=None,
        description="Restrict search to a specific document ID. If omitted, searches all documents.",
        examples=[1],
    )

    # Optional: explicitly select which agent capability to use.
    # If None, the orchestrator auto-classifies the intent from the question.
    # DESIGN DECISION: Auto-detection is the default because users shouldn't
    # need to know our internal routing. Explicit override is available for
    # testing and for clients that want deterministic routing.
    capability: Literal["qa", "summarise", "compare", "extract"] | None = Field(
        default=None,
        description=(
            "Agent capability to use. If omitted, auto-detected from the question. "
            "Options: 'qa' (answer questions), 'summarise' (structured summary), "
            "'compare' (cross-document comparison), 'extract' (structured data extraction)."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "What was the total revenue in Q3 2024?",
                    "document_id": 1,
                    "capability": "qa",
                },
                {
                    "question": "Summarise the risk factors section",
                    "document_id": 1,
                    "capability": "summarise",
                },
                {
                    "question": "Extract all quarterly revenue figures as a table",
                    "document_id": 1,
                    "capability": "extract",
                },
            ]
        }
    )


class EvaluateRequest(BaseModel):
    """
    Request body for POST /evaluate — Run RAG evaluation suite.

    Triggers evaluation of the RAG pipeline against a golden dataset
    of question/answer pairs. Returns metrics like faithfulness,
    answer relevancy, context precision, and context recall.

    Example:
        {
            "document_id": 1,
            "eval_dataset": "default"
        }
    """

    # Which document to evaluate against
    document_id: int = Field(
        ...,
        description="The document ID to evaluate the RAG pipeline against",
    )

    # Which evaluation dataset to use (extensible for different doc types)
    eval_dataset: str = Field(
        default="default",
        description="Name of the evaluation dataset to use",
    )
