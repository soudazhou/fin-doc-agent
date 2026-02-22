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


# ---------------------------------------------------------------------------
# Phase 4: Benchmarking & Comparison Requests
# ---------------------------------------------------------------------------


class CompareRequest(BaseModel):
    """
    Request body for POST /compare — run same query across multiple providers.

    DESIGN DECISION: Provider IDs are strings (not full config objects).
    API keys are read from server-side env only — never from request bodies.
    The provider_id format is self-describing: "type/model@base_url".
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to run across all providers",
    )
    document_id: int | None = Field(
        default=None,
        description="Restrict search to a specific document. Null = all.",
    )
    capability: Literal["qa", "summarise", "compare", "extract"] | None = Field(
        default=None,
        description="Agent capability override. Auto-detected if omitted.",
    )
    providers: list[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description=(
            "Provider IDs to compare. Format: 'type/model' or "
            "'type/model@base_url'. Min 2, max 5."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "What was total revenue in Q3 2024?",
                    "document_id": 1,
                    "providers": [
                        "anthropic/claude-sonnet-4-6",
                        "openai_compatible/deepseek-chat@https://api.deepseek.com/v1",
                    ],
                }
            ]
        }
    )


class BenchmarkRetrievalRequest(BaseModel):
    """
    Request body for POST /benchmark/retrieval — measure search latency.

    Benchmarks retrieval LATENCY across existing chunks, not Recall@k
    (which requires a golden dataset — deferred to Phase 5).

    DESIGN DECISION: Does not re-ingest at different chunk sizes.
    Use POST /ingest?chunk_size=256 separately, then benchmark.
    """

    document_id: int | None = Field(
        default=None,
        description="Document to benchmark against. Null = all.",
    )
    sample_queries: list[str] = Field(
        default=[
            "total revenue",
            "operating expenses",
            "risk factors",
            "earnings per share",
            "cash flow from operations",
        ],
        min_length=1,
        max_length=20,
        description="Queries to benchmark. Defaults to 5 financial templates.",
    )
    top_k_values: list[int] = Field(
        default=[3, 5, 10],
        description="Values of k to test in retrieval.",
    )
    vector_stores: list[Literal["pgvector", "chroma"]] = Field(
        default=["pgvector"],
        description="Vector stores to benchmark.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": 1,
                    "sample_queries": ["total revenue", "risk factors"],
                    "top_k_values": [5, 10],
                    "vector_stores": ["pgvector"],
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# Phase 5: Evaluation Requests
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    """
    Request body for POST /evaluate — Run RAG evaluation suite.

    Triggers evaluation of the RAG pipeline against a golden dataset
    of question/answer pairs. Returns metrics like faithfulness,
    answer relevancy, context precision, and context recall.

    Example:
        {
            "document_id": 1,
            "eval_dataset": "default",
            "provider_id": "anthropic/claude-sonnet-4-6"
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

    # Optional: specify a provider for cross-provider evaluation
    # Uses Phase 4's create_provider_from_id() format
    provider_id: str | None = Field(
        default=None,
        description=(
            "Provider to evaluate (e.g., 'anthropic/claude-sonnet-4-6'). "
            "If omitted, uses the default configured provider."
        ),
    )
