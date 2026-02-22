#!/usr/bin/env bash
# =============================================================================
# Financial Document Q&A Agent — End-to-End Demo
# =============================================================================
#
# Walks through every major feature of the system:
#   1. Health check
#   2. Document ingestion (PDF upload + async processing)
#   3. Agentic Q&A (all 4 capabilities)
#   4. Provider comparison (A/B testing)
#   5. Retrieval benchmark
#   6. Metrics dashboard
#   7. Evaluation suite
#
# PREREQUISITES:
#   - docker compose up  (all services running)
#   - .env configured with API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
#   - curl and jq installed
#
# USAGE:
#   ./scripts/demo.sh                           # Uses defaults
#   ./scripts/demo.sh path/to/report.pdf        # Ingest a specific PDF
#   DOCUMENT_ID=1 ./scripts/demo.sh             # Skip ingestion, use existing doc
#   BASE_URL=http://remote:8000 ./scripts/demo.sh
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL="${BASE_URL:-http://localhost:8000}"
PDF_FILE="${1:-}"
DOCUMENT_ID="${DOCUMENT_ID:-}"
POLL_INTERVAL=3     # seconds between status polls
POLL_TIMEOUT=300    # max seconds to wait for async tasks

# ---------------------------------------------------------------------------
# Colours and helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }
step()    { echo -e "\n${BOLD}▶ $1${NC}"; }

# Pretty-print JSON (fallback to raw if jq missing)
ppjson() {
    if command -v jq &>/dev/null; then
        jq '.'
    else
        cat
    fi
}

# ---------------------------------------------------------------------------
# Prerequisite check
# ---------------------------------------------------------------------------

check_deps() {
    local missing=0
    for cmd in curl; do
        if ! command -v "$cmd" &>/dev/null; then
            error "'$cmd' is required but not installed."
            missing=1
        fi
    done
    if ! command -v jq &>/dev/null; then
        warn "'jq' not found — JSON output will not be pretty-printed."
    fi
    if [ "$missing" -eq 1 ]; then
        exit 1
    fi
}

# Poll an async task until it reaches a terminal state
# Usage: poll_task <url> <status_field> <success_value>
poll_task() {
    local url="$1"
    local status_field="$2"
    local success_value="$3"
    local elapsed=0

    while [ "$elapsed" -lt "$POLL_TIMEOUT" ]; do
        local response
        response=$(curl -s "$url")
        local status
        status=$(echo "$response" | jq -r ".$status_field" 2>/dev/null || echo "unknown")

        if [ "$status" = "$success_value" ]; then
            echo "$response"
            return 0
        elif [ "$status" = "FAILURE" ] || [ "$status" = "failed" ]; then
            error "Task failed:"
            echo "$response" | ppjson
            return 1
        fi

        info "Status: $status — waiting ${POLL_INTERVAL}s... (${elapsed}s elapsed)"
        sleep "$POLL_INTERVAL"
        elapsed=$((elapsed + POLL_INTERVAL))
    done

    error "Timed out after ${POLL_TIMEOUT}s"
    return 1
}

# =============================================================================
# DEMO START
# =============================================================================

check_deps

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║     Financial Document Q&A Agent — Live Demo            ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  Base URL:    $BASE_URL"
echo "  PDF file:    ${PDF_FILE:-<none — will use existing document>}"
echo "  Document ID: ${DOCUMENT_ID:-<will be assigned after ingestion>}"
echo ""

# =============================================================================
# 1. HEALTH CHECK
# =============================================================================

header "1. Health Check"

step "GET /health"
HEALTH=$(curl -s "$BASE_URL/health")
echo "$HEALTH" | ppjson

STATUS=$(echo "$HEALTH" | jq -r '.status' 2>/dev/null || echo "unknown")
if [ "$STATUS" = "healthy" ]; then
    info "API is healthy"
else
    error "API returned status: $STATUS"
    error "Is 'docker compose up' running?"
    exit 1
fi

# =============================================================================
# 2. DOCUMENT INGESTION
# =============================================================================

header "2. Document Ingestion"

if [ -n "$DOCUMENT_ID" ]; then
    info "Using existing document ID: $DOCUMENT_ID (skipping ingestion)"

elif [ -n "$PDF_FILE" ]; then
    if [ ! -f "$PDF_FILE" ]; then
        error "File not found: $PDF_FILE"
        exit 1
    fi

    step "POST /ingest — Uploading $(basename "$PDF_FILE")"
    INGEST_RESPONSE=$(curl -s -X POST "$BASE_URL/ingest" \
        -F "file=@${PDF_FILE}")
    echo "$INGEST_RESPONSE" | ppjson

    TASK_ID=$(echo "$INGEST_RESPONSE" | jq -r '.task_id' 2>/dev/null)
    DOCUMENT_ID=$(echo "$INGEST_RESPONSE" | jq -r '.document_id' 2>/dev/null)
    info "Document ID: $DOCUMENT_ID, Task ID: $TASK_ID"

    step "Polling GET /ingest/$TASK_ID until complete..."
    if RESULT=$(poll_task "$BASE_URL/ingest/$TASK_ID" "status" "SUCCESS"); then
        info "Ingestion complete!"
        echo "$RESULT" | ppjson
    else
        warn "Ingestion failed — continuing demo with document_id=$DOCUMENT_ID"
    fi
else
    warn "No PDF file provided and no DOCUMENT_ID set."
    echo ""
    echo "  To ingest a document:"
    echo "    ./scripts/demo.sh path/to/earnings_report.pdf"
    echo ""
    echo "  To use an existing document:"
    echo "    DOCUMENT_ID=1 ./scripts/demo.sh"
    echo ""
    echo "  Skipping ingestion and Q&A sections..."
fi

# =============================================================================
# 3. AGENTIC QUESTION ANSWERING
# =============================================================================

header "3. Agentic Question Answering (4 Capabilities)"

if [ -z "$DOCUMENT_ID" ]; then
    warn "No document ID — skipping Q&A. Provide a PDF or set DOCUMENT_ID."
else

    # --- Q&A ---
    step "Q&A: \"What was the total revenue?\""
    curl -s -X POST "$BASE_URL/ask" \
        -H "Content-Type: application/json" \
        -d "{
            \"question\": \"What was the total revenue?\",
            \"document_id\": $DOCUMENT_ID,
            \"capability\": \"qa\"
        }" | ppjson

    # --- Summarise ---
    step "Summarise: \"Summarise the risk factors section\""
    curl -s -X POST "$BASE_URL/ask" \
        -H "Content-Type: application/json" \
        -d "{
            \"question\": \"Summarise the risk factors section\",
            \"document_id\": $DOCUMENT_ID,
            \"capability\": \"summarise\"
        }" | ppjson

    # --- Compare ---
    step "Compare: \"Compare revenue growth between product segments\""
    curl -s -X POST "$BASE_URL/ask" \
        -H "Content-Type: application/json" \
        -d "{
            \"question\": \"Compare revenue growth between product segments\",
            \"document_id\": $DOCUMENT_ID,
            \"capability\": \"compare\"
        }" | ppjson

    # --- Extract ---
    step "Extract: \"Extract all quarterly revenue figures as JSON\""
    curl -s -X POST "$BASE_URL/ask" \
        -H "Content-Type: application/json" \
        -d "{
            \"question\": \"Extract all quarterly revenue figures as JSON\",
            \"document_id\": $DOCUMENT_ID,
            \"capability\": \"extract\"
        }" | ppjson

fi

# =============================================================================
# 4. PROVIDER COMPARISON (A/B Testing)
# =============================================================================

header "4. Provider Comparison (A/B Testing)"

if [ -z "$DOCUMENT_ID" ]; then
    warn "No document ID — skipping comparison."
else
    step "POST /compare — Claude vs DeepSeek on the same question"
    echo -e "${YELLOW}NOTE: This requires both ANTHROPIC_API_KEY and DEEPSEEK_API_KEY in .env${NC}"
    echo -e "${YELLOW}      If a provider fails, the other still returns results.${NC}"
    echo ""

    curl -s -X POST "$BASE_URL/compare" \
        -H "Content-Type: application/json" \
        -d "{
            \"question\": \"What was the total revenue?\",
            \"document_id\": $DOCUMENT_ID,
            \"providers\": [
                \"anthropic/claude-sonnet-4-6\",
                \"openai_compatible/deepseek-chat@https://api.deepseek.com/v1\"
            ]
        }" | ppjson
fi

# =============================================================================
# 5. RETRIEVAL BENCHMARK
# =============================================================================

header "5. Retrieval Benchmark"

if [ -z "$DOCUMENT_ID" ]; then
    warn "No document ID — skipping benchmark."
else
    step "POST /benchmark/retrieval — Testing top_k=5 and top_k=10"
    curl -s -X POST "$BASE_URL/benchmark/retrieval" \
        -H "Content-Type: application/json" \
        -d "{
            \"document_id\": $DOCUMENT_ID,
            \"sample_queries\": [
                \"total revenue\",
                \"operating expenses\",
                \"risk factors\",
                \"earnings per share\",
                \"cash flow from operations\"
            ],
            \"top_k_values\": [5, 10],
            \"vector_stores\": [\"chroma\"]
        }" | ppjson
fi

# =============================================================================
# 6. METRICS DASHBOARD
# =============================================================================

header "6. Metrics Dashboard"

step "GET /metrics — Aggregated performance stats (last 24 hours)"
curl -s "$BASE_URL/metrics?hours=24" | ppjson

# =============================================================================
# 7. EVALUATION SUITE
# =============================================================================

header "7. Evaluation Suite"

if [ -z "$DOCUMENT_ID" ]; then
    warn "No document ID — skipping evaluation."
else
    step "POST /evaluate — Running golden dataset evaluation"
    EVAL_RESPONSE=$(curl -s -X POST "$BASE_URL/evaluate" \
        -H "Content-Type: application/json" \
        -d "{
            \"document_id\": $DOCUMENT_ID,
            \"eval_dataset\": \"default\"
        }")
    echo "$EVAL_RESPONSE" | ppjson

    RUN_ID=$(echo "$EVAL_RESPONSE" | jq -r '.run_id' 2>/dev/null)

    if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ]; then
        step "Polling GET /evaluate/runs/$RUN_ID until complete..."
        if EVAL_RESULT=$(poll_task "$BASE_URL/evaluate/runs/$RUN_ID" "status" "completed"); then
            info "Evaluation complete!"
            echo "$EVAL_RESULT" | ppjson

            step "GET /evaluate/failures — Failure analysis"
            curl -s "$BASE_URL/evaluate/failures?run_id=$RUN_ID" | ppjson

            step "GET /evaluate/history — Score trends"
            curl -s "$BASE_URL/evaluate/history?document_id=$DOCUMENT_ID" | ppjson
        else
            warn "Evaluation did not complete in time."
        fi
    else
        warn "Could not start evaluation (missing run_id in response)."
    fi
fi

# =============================================================================
# DONE
# =============================================================================

header "Demo Complete"

echo "  Endpoints demonstrated:"
echo "    GET  /health              — System health"
echo "    POST /ingest              — Document upload"
echo "    GET  /ingest/{task_id}    — Ingestion status polling"
echo "    POST /ask                 — Agentic Q&A (4 capabilities)"
echo "    POST /compare             — A/B provider comparison"
echo "    POST /benchmark/retrieval — Retrieval latency benchmark"
echo "    GET  /metrics             — Performance dashboard"
echo "    POST /evaluate            — Golden dataset evaluation"
echo "    GET  /evaluate/runs/{id}  — Evaluation results"
echo "    GET  /evaluate/failures   — Failure analysis"
echo "    GET  /evaluate/history    — Score trends"
echo ""
echo "  Swagger UI:  $BASE_URL/docs"
echo "  Flower:      http://localhost:5555"
echo ""
