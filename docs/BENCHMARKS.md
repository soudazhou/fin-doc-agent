# Benchmarks

This document describes the benchmarking methodology, metrics, and links to individual benchmark runs. Each run is stored as a separate file in [`benchmark-runs/`](benchmark-runs/) so results are versioned and comparable over time.

## How to Run Benchmarks

```bash
# 1. Start all services
docker compose up --build

# 2. Ingest a document
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/samples/apple_q3_2024_earnings.pdf"

# 3. Run retrieval benchmark
curl -X POST http://localhost:8000/benchmark/retrieval \
  -H "Content-Type: application/json" \
  -d '{"document_id": 1}'

# 4. Run provider comparison (use your DashScope endpoint)
DASHSCOPE="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What was the total revenue?\",
       \"providers\": [
         \"openai_compatible/qwen3.5-plus@${DASHSCOPE}\",
         \"openai_compatible/qwen-plus@${DASHSCOPE}\",
         \"openai_compatible/qwen-turbo@${DASHSCOPE}\",
         \"openai_compatible/qwq-plus@${DASHSCOPE}\"
       ]}"

# 5. View aggregated metrics
curl http://localhost:8000/metrics

# 6. Run evaluation suite (requires aligned golden dataset)
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"document_id": 1}'
```

After running benchmarks, create a new results file:

```bash
# Copy the template and fill in your results
cp docs/benchmark-runs/2026-02-22-dashscope-singapore-qwen.md \
   docs/benchmark-runs/YYYY-MM-DD-description.md
```

## Metric Definitions

### Retrieval Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Recall@k** | Fraction of relevant chunks found in the top-k results | > 0.85 at k=5 |
| **MRR** (Mean Reciprocal Rank) | Average of 1/rank for the first relevant result | > 0.8 |
| **p50 Latency** | Median search latency in milliseconds | < 50ms |
| **p95 Latency** | 95th percentile search latency | < 200ms |
| **Avg Top Score** | Mean cosine similarity of the top-1 result | > 0.85 |

### Evaluation Metrics (Golden Dataset)

The evaluation suite runs 29 test cases across 4 capabilities against 6 metrics:

| Metric | What It Measures | Threshold |
|--------|-----------------|-----------|
| **Faithfulness** | Is the answer grounded in the retrieved context? | 0.7 |
| **Answer Relevancy** | Does the answer address the question? | 0.7 |
| **Contextual Precision** | Are the retrieved chunks relevant to the query? | 0.6 |
| **Contextual Recall** | Did retrieval find all necessary information? | 0.6 |
| **Numerical Accuracy** | Are financial figures (revenue, EPS, margins) correct? | 0.8 |
| **Retrieval Recall@k** | Did the correct source chunks appear in top-k? | 0.7 |

## Benchmark Dimensions

### 1. Chunk Size Comparison

Ingest the same document at different chunk sizes and compare retrieval quality:

| Chunk Size | Expected Behaviour |
|------------|-------------------|
| 256 tokens | More precise chunks, may lose context across sections |
| 512 tokens | Balanced — default setting, ~1-2 paragraphs |
| 1024 tokens | More context per chunk, may dilute relevance scores |

```bash
# Ingest at different chunk sizes
curl -X POST http://localhost:8000/ingest -F "file=@report.pdf" -F "chunk_size=256"
curl -X POST http://localhost:8000/ingest -F "file=@report.pdf" -F "chunk_size=512"
curl -X POST http://localhost:8000/ingest -F "file=@report.pdf" -F "chunk_size=1024"
```

### 2. Vector Store Comparison

Both stores are behind a pluggable interface. Benchmark on the same dataset:

| Store | Strengths | Trade-offs |
|-------|----------|------------|
| **pgvector** | No extra infra, HNSW indexing, ACID transactions | Requires PostgreSQL, slower cold start |
| **Chroma** | Simple API, fast prototyping, in-process mode | Separate service in production, no ACID |

### 3. Provider Comparison

The `/compare` endpoint runs the same query across providers in parallel:

| Provider | Model | Input Cost | Output Cost | Best For |
|----------|-------|-----------|-------------|----------|
| Anthropic | Claude Sonnet 4.6 | $3.00/1M | $15.00/1M | Best quality, demos |
| Anthropic | Claude Haiku 4.5 | $0.80/1M | $4.00/1M | Fast, cost-effective |
| DeepSeek | V3 | $0.14/1M | $0.28/1M | Development (20x cheaper) |
| DeepSeek | R1 | $0.55/1M | $2.19/1M | Complex reasoning |
| Alibaba | Qwen 3.5-Plus | ~$0.40/1M | — | Latest flagship |
| Alibaba | Qwen-Plus | ~$0.40/1M | ~$1.20/1M | Best value |
| Alibaba | Qwen-Turbo | ~$0.10/1M | ~$0.30/1M | Fastest, cheapest |
| Alibaba | QwQ-Plus | ~$0.40/1M | ~$1.20/1M | Reasoning (like R1) |
| Zhipu AI | GLM-5 | $1.00/1M | $3.20/1M | MIT licensed, self-hostable |
| MiniMax | M2.5 | $0.20/1M | $1.00/1M | 3rd globally on SWE-bench |
| Moonshot AI | Kimi K2.5 | $0.60/1M | $2.50/1M | Long-context specialist |

*Pricing as of February 2026. Source: provider pricing pages.*

## Benchmark Runs

Individual benchmark results are stored in [`benchmark-runs/`](benchmark-runs/). Each run captures the full context: date, operator, document, models, infrastructure, and results.

| Date | Description | Models | Key Finding | Link |
|------|------------|--------|-------------|------|
| 2026-02-22 | DashScope Singapore — Qwen model comparison | qwen3.5-plus, qwen-plus, qwen-turbo, qwq-plus | qwen-plus best overall (7.8s avg, $0.0004/query); pgvector sub-3ms p50 | [Results](benchmark-runs/2026-02-22-dashscope-singapore-qwen.md) |

### How to Add a New Run

1. Run the benchmarks using the commands above
2. Create a new file: `docs/benchmark-runs/YYYY-MM-DD-description.md`
3. Use an existing run as a template (copy the structure)
4. Add a row to the table above linking to the new file
5. Commit and open a PR
