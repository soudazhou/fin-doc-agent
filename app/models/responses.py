# =============================================================================
# API Response Models — Pydantic V2 Schemas
# =============================================================================
#
# These models define the shape of data coming OUT of the API.
# They serve as the contract between backend and frontend:
# 1. Ensure consistent response structure across all endpoints
# 2. Automatically serialized to JSON by FastAPI
# 3. Generate OpenAPI response schemas (visible at /docs)
# 4. Prevent accidental exposure of internal fields (e.g., raw embeddings)
#
# DESIGN DECISION: Separate response models from DB models
# The database stores 1536-dimensional embedding vectors per chunk.
# We never want to send those over the wire — they're huge and useless
# to the client. Response models let us control exactly what's exposed.
# =============================================================================

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Response for GET /health — confirms the API is running."""

    status: str = "ok"
    version: str
    service: str


class DocumentResponse(BaseModel):
    """
    Response model for document metadata.
    Returned after ingestion and in document listings.
    """

    id: int
    filename: str
    file_size: int
    page_count: int | None
    status: str
    error_message: str | None = None
    celery_task_id: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class IngestResponse(BaseModel):
    """
    Response for POST /ingest — confirms document upload and async processing.

    The document is NOT immediately available for querying. The client should
    poll GET /ingest/{task_id} to check when processing is complete.
    """

    document_id: int = Field(description="ID of the created document record")
    task_id: str = Field(description="Celery task ID for tracking ingestion progress")
    status: str = Field(
        default="processing",
        description="Current status: 'processing' means ingestion is in progress",
    )
    message: str = Field(
        default="Document uploaded. Ingestion in progress.",
        description="Human-readable status message",
    )


class IngestStatusResponse(BaseModel):
    """
    Response for GET /ingest/{task_id} — ingestion pipeline status.

    Clients poll this endpoint to know when a document is ready for querying.
    """

    task_id: str
    status: str = Field(description="Task status: PENDING, STARTED, SUCCESS, FAILURE")
    document: DocumentResponse | None = Field(
        default=None,
        description="Document details (available when status is SUCCESS)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (available when status is FAILURE)",
    )


class SourceChunk(BaseModel):
    """
    A source chunk returned as part of a Q&A answer.

    Provides provenance for the answer — the user can verify which parts
    of the original document were used to generate the response.
    This is critical for financial applications where traceability matters.
    """

    chunk_id: int = Field(description="Database ID of the chunk")
    content: str = Field(description="The text content of the chunk")
    page_number: int | None = Field(description="Page number in the original PDF")
    similarity_score: float = Field(
        description="Cosine similarity score (0-1, higher = more relevant)"
    )
    metadata: dict | None = Field(
        default=None,
        description="Additional metadata (section title, table indicator, etc.)",
    )


class AskResponse(BaseModel):
    """
    Response for POST /ask — the answer to a financial document question.

    Includes the answer, source citations, and metadata about the
    retrieval and generation process. Source citations are critical
    for financial applications — users need to verify answers against
    original documents.
    """

    answer: str = Field(description="The generated answer to the question")
    sources: list[SourceChunk] = Field(
        description="Source chunks used to generate the answer, ranked by relevance"
    )
    question: str = Field(description="The original question (echoed back)")
    document_id: int | None = Field(
        description="Document ID that was searched (null if all documents)"
    )
    model: str = Field(description="LLM model used for generation")
    retrieval_count: int = Field(
        description="Number of chunks retrieved from vector search"
    )


# ---------------------------------------------------------------------------
# Phase 4: Benchmarking & Comparison Responses
# ---------------------------------------------------------------------------


class ProviderResult(BaseModel):
    """Result from a single provider leg of POST /compare."""

    provider_id: str = Field(
        description="Provider identifier (e.g., 'anthropic/claude-sonnet-4-6')",
    )
    answer: str = Field(description="The generated answer")
    sources: list[SourceChunk] = Field(description="Retrieved source chunks")
    model: str = Field(description="Actual model name returned by the API")
    latency_ms: int = Field(description="End-to-end latency in milliseconds")
    input_tokens: int = Field(description="Input tokens consumed")
    output_tokens: int = Field(description="Output tokens generated")
    estimated_cost_usd: float | None = Field(
        default=None,
        description="Estimated cost in USD. Null if model not in pricing registry.",
    )
    search_iterations: int = Field(
        description="Number of agentic search iterations",
    )
    retrieval_count: int = Field(description="Number of chunks retrieved")
    error: str | None = Field(
        default=None,
        description="Error message if this provider leg failed.",
    )


class CompareWinner(BaseModel):
    """Summary of which provider 'won' on each dimension."""

    fastest_provider: str | None = Field(
        default=None, description="Provider with lowest latency_ms",
    )
    cheapest_provider: str | None = Field(
        default=None,
        description="Provider with lowest estimated_cost_usd",
    )


class CompareResponse(BaseModel):
    """Response for POST /compare — side-by-side provider results."""

    question: str
    document_id: int | None
    results: list[ProviderResult]
    winner: CompareWinner
    total_latency_ms: int = Field(
        description="Wall-clock time for all providers (parallel execution)",
    )


class RetrievalBenchmarkResult(BaseModel):
    """Result for a single (vector_store, top_k) combination."""

    vector_store: str
    top_k: int
    avg_latency_ms: float = Field(description="Average search latency (ms)")
    p50_latency_ms: float = Field(description="Median search latency (ms)")
    p95_latency_ms: float = Field(
        description="95th percentile search latency (ms)",
    )
    avg_top_score: float = Field(
        description="Average similarity score of the top result",
    )
    queries_run: int = Field(description="Number of sample queries executed")


class BenchmarkRetrievalResponse(BaseModel):
    """Response for POST /benchmark/retrieval."""

    document_id: int | None
    results: list[RetrievalBenchmarkResult]
    sample_queries: list[str]
    summary: str = Field(
        description="Human-readable summary of results",
    )


class ProviderMetricSummary(BaseModel):
    """Aggregated metrics for a single provider in GET /metrics."""

    provider_id: str | None = Field(
        description="Provider identifier. Null for /ask calls.",
    )
    query_count: int
    avg_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    total_estimated_cost_usd: float | None = None
    avg_cost_per_query_usd: float | None = None
    avg_search_iterations: float | None = None


class MetricsResponse(BaseModel):
    """Response for GET /metrics — aggregated performance stats."""

    total_queries: int
    time_range_hours: int = Field(
        description="Lookback window for the aggregation",
    )
    by_provider: list[ProviderMetricSummary]
    overall_avg_latency_ms: float | None = None
    overall_total_cost_usd: float | None = None


# ---------------------------------------------------------------------------
# Phase 5: Evaluation Responses
# ---------------------------------------------------------------------------


class EvalMetric(BaseModel):
    """
    A single evaluation metric result.

    Each metric measures a different aspect of RAG quality:
    - faithfulness: Is the answer grounded in the retrieved context?
    - answer_relevancy: Does the answer address the question?
    - context_precision: Are the retrieved chunks relevant?
    - context_recall: Did we retrieve all necessary information?
    """

    name: str = Field(description="Metric name (e.g., 'faithfulness')")
    score: float = Field(
        description="Metric score (0.0-1.0, higher is better)",
        ge=0.0,
        le=1.0,
    )
    reason: str | None = Field(
        default=None,
        description="Explanation of the score (from DeepEval)",
    )


class EvaluateResponse(BaseModel):
    """
    Response for POST /evaluate — RAG evaluation results.

    Returns per-metric scores and an overall assessment.
    These metrics help identify weaknesses in the RAG pipeline:
    - Low faithfulness → LLM is hallucinating, needs better prompting
    - Low context precision → Retrieval returning irrelevant chunks
    - Low context recall → Chunking or embedding issues
    """

    document_id: int
    eval_dataset: str
    metrics: list[EvalMetric] = Field(description="Individual metric scores")
    overall_score: float = Field(
        description="Average score across all metrics (0.0-1.0)"
    )
    total_test_cases: int = Field(description="Number of Q&A pairs evaluated")
    passed: int = Field(description="Number of test cases that passed thresholds")
    failed: int = Field(description="Number of test cases that failed thresholds")
