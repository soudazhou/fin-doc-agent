# Benchmarks

This document describes the benchmarking methodology, metrics, and how to reproduce results. Run benchmarks after ingesting documents to populate results.

## How to Run Benchmarks

```bash
# 1. Start all services
docker compose up

# 2. Ingest a document (or use the demo script)
./scripts/demo.sh path/to/earnings_report.pdf

# 3. Run retrieval benchmark
curl -X POST http://localhost:8000/benchmark/retrieval \
  -H "Content-Type: application/json" \
  -d '{"document_id": 1,
       "sample_queries": ["total revenue", "operating expenses", "risk factors",
                          "earnings per share", "cash flow from operations"],
       "top_k_values": [5, 10, 20],
       "vector_stores": ["pgvector", "chroma"]}'

# 4. Run provider comparison
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total revenue?", "document_id": 1,
       "providers": ["anthropic/claude-sonnet-4-6",
                     "openai_compatible/deepseek-chat@https://api.deepseek.com/v1"]}'

# 5. View aggregated metrics
curl http://localhost:8000/metrics?hours=24

# 6. Run evaluation suite
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"document_id": 1, "eval_dataset": "default"}'
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
| Alibaba | Qwen 3.5-Plus | $0.11/1M | $0.44/1M | Cheapest flagship |
| Zhipu AI | GLM-5 | $1.00/1M | $3.20/1M | MIT licensed, self-hostable |
| MiniMax | M2.5 | $0.20/1M | $1.00/1M | 3rd globally on SWE-bench |
| Moonshot AI | Kimi K2.5 | $0.60/1M | $2.50/1M | Long-context specialist |

*Pricing as of February 2026. Source: provider pricing pages.*

## Results

*Results will be populated after running benchmarks with real documents. Use the demo script or API endpoints above to generate data.*

### Retrieval Benchmark Results

```
Run: POST /benchmark/retrieval
```

| Vector Store | top_k | Avg Latency (ms) | p50 (ms) | p95 (ms) | Avg Top Score |
|-------------|-------|------------------|----------|----------|---------------|
| *pending* | — | — | — | — | — |

### Provider Comparison Results

```
Run: POST /compare
```

| Provider | Latency (ms) | Input Tokens | Output Tokens | Cost (USD) |
|----------|-------------|-------------|---------------|------------|
| *pending* | — | — | — | — |

### Evaluation Results

```
Run: POST /evaluate
```

| Metric | Score | Threshold | Pass? |
|--------|-------|-----------|-------|
| *pending* | — | — | — |
