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


class EvaluateStartResponse(BaseModel):
    """
    Response for POST /evaluate — confirms evaluation has started.

    The evaluation runs in the background (potentially 30+ LLM calls).
    Poll GET /evaluate/runs/{run_id} for results.
    """

    run_id: int = Field(description="ID of the created evaluation run")
    status: str = Field(
        default="running",
        description="Run status: 'running' means evaluation is in progress",
    )
    message: str = Field(
        default="Evaluation started. Poll GET /evaluate/runs/{run_id} for results.",
    )


class EvalTestCaseResult(BaseModel):
    """Per-test-case result detail within an evaluation run."""

    test_case_id: str = Field(description="ID from the golden dataset")
    question: str
    expected_answer: str | None = None
    actual_answer: str
    passed: bool
    metric_scores: dict[str, float] = Field(
        description="Metric name → score (0.0-1.0)",
    )
    metric_reasons: dict[str, str | None] = Field(
        default_factory=dict,
        description="Metric name → LLM judge reason (if applicable)",
    )


class RegressionComparison(BaseModel):
    """
    Comparison between current eval run and the most recent previous run.

    Positive score_delta means improvement; negative means regression.
    """

    previous_run_id: int | None = Field(
        default=None,
        description="ID of the previous run compared against",
    )
    previous_overall_score: float | None = None
    score_delta: float | None = Field(
        default=None,
        description="Overall score change (positive = improvement)",
    )
    metric_deltas: dict[str, float] | None = Field(
        default=None,
        description="Per-metric score changes",
    )
    regressed_metrics: list[str] = Field(
        default_factory=list,
        description="Metrics that got worse (delta < -0.01)",
    )
    improved_metrics: list[str] = Field(
        default_factory=list,
        description="Metrics that improved (delta > 0.01)",
    )


class EvaluateResponse(BaseModel):
    """
    Response for GET /evaluate/runs/{run_id} — full evaluation results.

    Returns per-metric scores, individual test case results, and
    regression comparison with the previous run.
    These metrics help identify weaknesses in the RAG pipeline:
    - Low faithfulness → LLM is hallucinating, needs better prompting
    - Low context precision → Retrieval returning irrelevant chunks
    - Low context recall → Chunking or embedding issues
    """

    run_id: int = Field(description="Evaluation run ID")
    status: str = Field(description="Run status: running, completed, failed")
    document_id: int
    eval_dataset: str
    provider_id: str | None = None
    metrics: list[EvalMetric] = Field(
        default_factory=list,
        description="Aggregate metric scores",
    )
    overall_score: float = Field(
        default=0.0,
        description="Average score across all metrics (0.0-1.0)",
    )
    total_test_cases: int = Field(
        default=0, description="Number of Q&A pairs evaluated",
    )
    passed: int = Field(
        default=0, description="Number of test cases that passed thresholds",
    )
    failed: int = Field(
        default=0, description="Number of test cases that failed thresholds",
    )
    duration_ms: int | None = Field(
        default=None, description="Total evaluation duration in milliseconds",
    )
    regression: RegressionComparison | None = Field(
        default=None,
        description="Comparison with previous run (null if first run)",
    )
    test_case_results: list[EvalTestCaseResult] | None = Field(
        default=None,
        description="Per-test-case details (null while running)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if status is 'failed'",
    )
    created_at: datetime | None = None


class EvalHistoryEntry(BaseModel):
    """A single entry in the evaluation history timeline."""

    run_id: int
    document_id: int
    eval_dataset: str
    provider_id: str | None = None
    overall_score: float
    metric_scores: dict[str, float] = Field(
        description="Metric name → average score",
    )
    total_test_cases: int
    passed: int
    failed: int
    duration_ms: int | None = None
    created_at: datetime


class EvalHistoryResponse(BaseModel):
    """
    Response for GET /evaluate/history — evaluation score trends over time.

    Shows chronological list of eval runs with a trend indicator.
    """

    entries: list[EvalHistoryEntry]
    total_runs: int
    trend: str = Field(
        description=(
            "Score trend: 'improving', 'declining', 'stable', "
            "or 'insufficient_data' (need 2+ runs)"
        ),
    )


class EvalFailureDetail(BaseModel):
    """Detailed failure information for a single test case."""

    test_case_id: str
    question: str
    expected_answer: str | None = None
    actual_answer: str
    metric_scores: dict[str, float]
    metric_reasons: dict[str, str | None] = Field(default_factory=dict)
    failing_metrics: list[str] = Field(
        description="Metrics that fell below their threshold",
    )
    search_trace: list[dict] | None = Field(
        default=None,
        description="Full agentic search trace for debugging",
    )
    sources: list[dict] | None = Field(
        default=None,
        description="Retrieved source chunks",
    )


class EvalFailuresResponse(BaseModel):
    """
    Response for GET /evaluate/failures — detailed failure analysis.

    Used for debugging: shows which test cases failed, why, and the
    full search trace so you can identify retrieval or generation issues.
    """

    run_id: int
    failures: list[EvalFailureDetail]
    total_failures: int
    failure_rate: float = Field(
        description="Fraction of test cases that failed (0.0-1.0)",
    )
    most_common_failing_metric: str | None = Field(
        default=None,
        description="Metric that fails most often across test cases",
    )


# ---------------------------------------------------------------------------
# Phase 6: Authorization Response Models
# ---------------------------------------------------------------------------


class ApiKeyResponse(BaseModel):
    """
    Response for API key details.

    Never includes raw key or hash — only the prefix for identification.
    """

    id: int
    name: str
    key_prefix: str
    scopes: list[str] | None = None
    rate_limit_rpm: int | None = None
    all_documents_access: bool
    is_active: bool
    document_ids: list[int] = Field(default_factory=list)
    created_at: datetime
    expires_at: datetime | None = None
    last_used_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class ApiKeyCreatedResponse(BaseModel):
    """
    Response for POST /admin/keys — returned once at key creation.

    WARNING: The raw_key is only returned in this response.
    It is never stored or retrievable after creation.
    """

    id: int
    name: str
    key_prefix: str
    raw_key: str = Field(
        description=(
            "The full API key. Store it securely "
            "— it will NOT be shown again."
        ),
    )
    scopes: list[str] | None = None
    rate_limit_rpm: int | None = None
    all_documents_access: bool
    created_at: datetime
    expires_at: datetime | None = None


class ApiKeyListResponse(BaseModel):
    """Response for GET /admin/keys — list of all API keys."""

    keys: list[ApiKeyResponse]
    total: int


class AuditLogResponse(BaseModel):
    """Response for a single audit log entry."""

    id: int
    api_key_id: int | None = None
    api_key_name: str | None = None
    endpoint: str
    method: str
    path: str
    document_id: int | None = None
    question: str | None = None
    client_ip: str | None = None
    status_code: int | None = None
    response_time_ms: int | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AuditLogListResponse(BaseModel):
    """Response for GET /admin/audit — list of audit log entries."""

    logs: list[AuditLogResponse]
    total: int
