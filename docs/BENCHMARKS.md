# Benchmarks

Benchmark results from end-to-end testing on 22 Feb 2026. All tests run against the Apple Q3 2024 earnings report (7 pages, 6 chunks at 512 tokens) using Alibaba Cloud Model Studio (DashScope Singapore).

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

## Results

All results collected on 22 Feb 2026 using DashScope Singapore (`dashscope-intl.aliyuncs.com`).

### Retrieval Benchmark Results

**Configuration:** pgvector (PostgreSQL 17 + pgvector extension), text-embedding-v4 (1536 dimensions), 6 chunks at 512 tokens.

**Test queries:** "total revenue", "operating expenses", "risk factors", "earnings per share", "cash flow from operations"

| Vector Store | top_k | Avg Latency (ms) | p50 (ms) | p95 (ms) | Avg Top Score |
|-------------|-------|------------------|----------|----------|---------------|
| pgvector | 3 | 18.3 | 3.2 | 78.3 | 0.33 |
| pgvector | 5 | 3.0 | 2.9 | 3.3 | 0.33 |
| pgvector | 10 | 2.9 | 2.9 | 3.0 | 0.33 |

**Observations:**

- pgvector search latency is sub-3ms at p50 — excellent for real-time Q&A
- The first query (top_k=3) shows a higher average due to cold-start connection setup; subsequent queries are consistent
- Low avg top score (0.33) reflects that the benchmark uses generic keyword queries ("total revenue"), not full natural-language questions. The agentic search pipeline (which embeds actual user questions) achieves 0.52–0.83 similarity scores in practice

### Provider Comparison Results

**Test:** 3 questions run across 4 Qwen model tiers via DashScope Singapore. All models received identical retrieved context and system prompts.

**Questions tested:**

1. "What was Apple's total revenue in Q3 2024 and how did it compare to Q3 2023?"
2. "What was Apple's gross margin in Q3 2024 and what drove the improvement?"
3. "How much was Apple Services revenue and what percentage of total revenue did it represent?"

#### Per-Query Results

**Q1: Total revenue comparison**

| Model | Latency | Tokens (in/out) | Answer Quality |
|-------|---------|-----------------|----------------|
| qwen3.5-plus | 18,576ms | 3,071 / 930 | Detailed with comparison table and analysis |
| qwen-plus | 9,349ms | 2,962 / 320 | Concise, correct, well-cited |
| qwen-turbo | 20,472ms | 2,966 / 1,234 | Verbose with markdown tables |
| qwq-plus | 42,111ms | 2,962 / 1,135 | Deep reasoning with step-by-step analysis |

**Q2: Gross margin**

| Model | Latency | Tokens (in/out) | Answer Quality |
|-------|---------|-----------------|----------------|
| qwen3.5-plus | 5,381ms | 3,083 / 100 | Brief, correct |
| qwen-plus | 6,968ms | 2,974 / 137 | Good detail with drivers explained |
| qwen-turbo | 4,731ms | 2,978 / 79 | Minimal but correct |
| qwq-plus | 23,908ms | 2,974 / 459 | Extensive reasoning chain |

**Q3: Services revenue**

| Model | Latency | Tokens (in/out) | Answer Quality |
|-------|---------|-----------------|----------------|
| qwen3.5-plus | 6,082ms | 1,304 / 111 | Correct, calculated percentage |
| qwen-plus | 6,773ms | 1,259 / 165 | Detailed with YoY comparison |
| qwen-turbo | 3,089ms | 1,263 / 30 | One-line answer |
| qwq-plus | 24,916ms | 1,259 / 472 | Full reasoning trace |

#### Aggregated Metrics (24 queries total)

| Model | Queries | Avg Latency | p50 | p95 | Avg Tokens (in/out) | Avg Cost/Query |
|-------|---------|-------------|-----|-----|---------------------|----------------|
| **qwen-plus** | 4 | 7,808ms | 8,143ms | 9,349ms | 2,541 / 202 | $0.000368 |
| **qwen-turbo** | 4 | 8,096ms | 4,731ms | 20,472ms | 2,545 / 341 | n/a |
| **qwen3.5-plus** | 3 | 10,013ms | 6,082ms | 18,576ms | 2,486 / 380 | n/a |
| **qwq-plus** | 3 | 30,312ms | 24,916ms | 42,111ms | 2,398 / 689 | n/a |

#### Key Findings

1. **Best overall: qwen-plus** — Consistently fast (7.8s avg), cheapest tracked cost ($0.0004/query), good answer quality with proper citations. Best balance of speed, cost, and quality.

2. **Fastest per-query: qwen-turbo** — 3.1s on simple questions (Q3), but highly variable (3–20s). Answers are minimal — often one sentence without elaboration. Best for latency-sensitive applications where brevity is acceptable.

3. **Most thorough: qwq-plus** — Reasoning model produces step-by-step analysis chains. 3–4x slower than qwen-plus (30s avg) and generates 3x more output tokens. Best for complex analytical questions where reasoning transparency matters.

4. **Latest generation: qwen3.5-plus** — Similar latency to qwen-plus but more variable. Answers are well-structured. Pricing not yet tracked in our registry.

5. **All models answered correctly** — Every model correctly identified $85.8B revenue, 46.3% gross margin, and $24.2B Services revenue. The differences are in answer depth and formatting, not accuracy.

#### Recommendation

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Production Q&A | qwen-plus | Best speed/cost/quality balance |
| Quick lookups | qwen-turbo | Fastest for simple factual queries |
| Deep analysis | qwq-plus | Full reasoning chain, best for "explain why" |
| Demo / showcase | qwen3.5-plus or Claude Sonnet 4.6 | Most polished output formatting |

### Evaluation Results

*Evaluation with DeepEval LLM-as-judge metrics requires running the eval suite against ingested documents. Run `POST /evaluate` after ingestion to populate these results.*

| Metric | Score | Threshold | Pass? |
|--------|-------|-----------|-------|
| Faithfulness | — | 0.7 | — |
| Answer Relevancy | — | 0.7 | — |
| Contextual Precision | — | 0.6 | — |
| Contextual Recall | — | 0.6 | — |
| Numerical Accuracy | — | 0.8 | — |
| Retrieval Recall@k | — | 0.7 | — |

## Notes

- **Embedding model:** Alibaba Cloud `text-embedding-v4` (1536 dimensions) via DashScope Singapore
- **Similarity threshold:** `RETRIEVAL_SIMILARITY_THRESHOLD=0.45` (lowered from default 0.7 for DashScope embeddings; see README for details)
- **Infrastructure:** Docker Desktop on macOS (Apple Silicon), all services in containers, Docling PDF parsing on CPU
- **Third-party models (GLM, DeepSeek, MiniMax):** Available on DashScope Mainland China only, not on the international endpoint. To compare these providers, use their direct APIs with separate API keys
