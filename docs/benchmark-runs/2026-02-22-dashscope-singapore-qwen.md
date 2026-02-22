# Benchmark Run: DashScope Singapore — Qwen Model Comparison

**Date:** 22 February 2026
**Operator:** @soudazhou
**Document:** Apple Q3 2024 Earnings Report (synthetic, 7 pages, 6 chunks at 512 tokens)
**Embedding:** Alibaba Cloud `text-embedding-v4` (1536 dimensions) via DashScope Singapore
**Vector Store:** pgvector (PostgreSQL 17 + pgvector extension, HNSW index)
**Infrastructure:** Docker Desktop on macOS (Apple Silicon), all services in containers, Docling PDF parsing on CPU

## Purpose

First end-to-end benchmark after completing all 7 implementation phases. Goals:

1. Validate retrieval performance with DashScope embeddings
2. Compare 4 Qwen model tiers on the same financial questions
3. Establish baseline metrics for future comparison

## Retrieval Benchmark

**Endpoint:** `POST /benchmark/retrieval`
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

## Provider Comparison

**Endpoint:** `POST /compare`
**Models tested:** qwen3.5-plus, qwen-plus, qwen-turbo, qwq-plus (all via DashScope Singapore)
**Method:** 3 financial questions, each run across all 4 models with identical retrieved context and system prompts

### Questions

1. "What was Apple's total revenue in Q3 2024 and how did it compare to Q3 2023?"
2. "What was Apple's gross margin in Q3 2024 and what drove the improvement?"
3. "How much was Apple Services revenue and what percentage of total revenue did it represent?"

### Per-Query Results

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

### Aggregated Metrics (24 queries total)

| Model | Queries | Avg Latency | p50 | p95 | Avg Tokens (in/out) | Avg Cost/Query |
|-------|---------|-------------|-----|-----|---------------------|----------------|
| **qwen-plus** | 4 | 7,808ms | 8,143ms | 9,349ms | 2,541 / 202 | $0.000368 |
| **qwen-turbo** | 4 | 8,096ms | 4,731ms | 20,472ms | 2,545 / 341 | n/a |
| **qwen3.5-plus** | 3 | 10,013ms | 6,082ms | 18,576ms | 2,486 / 380 | n/a |
| **qwq-plus** | 3 | 30,312ms | 24,916ms | 42,111ms | 2,398 / 689 | n/a |

## Key Findings

1. **Best overall: qwen-plus** — Consistently fast (7.8s avg), cheapest tracked cost ($0.0004/query), good answer quality with proper citations. Best balance of speed, cost, and quality.

2. **Fastest per-query: qwen-turbo** — 3.1s on simple questions (Q3), but highly variable (3–20s). Answers are minimal — often one sentence without elaboration. Best for latency-sensitive applications where brevity is acceptable.

3. **Most thorough: qwq-plus** — Reasoning model produces step-by-step analysis chains. 3–4x slower than qwen-plus (30s avg) and generates 3x more output tokens. Best for complex analytical questions where reasoning transparency matters.

4. **Latest generation: qwen3.5-plus** — Similar latency to qwen-plus but more variable. Answers are well-structured. Pricing not yet tracked in our registry.

5. **All models answered correctly** — Every model correctly identified $85.8B revenue, 46.3% gross margin, and $24.2B Services revenue. The differences are in answer depth and formatting, not accuracy.

## Recommendations

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Production Q&A | qwen-plus | Best speed/cost/quality balance |
| Quick lookups | qwen-turbo | Fastest for simple factual queries |
| Deep analysis | qwq-plus | Full reasoning chain, best for "explain why" |
| Demo / showcase | qwen3.5-plus or Claude Sonnet 4.6 | Most polished output formatting |

## Issues Discovered

| Issue | Impact | Resolution |
|-------|--------|------------|
| `text-embedding-v3` defaults to 1024 dims | Ingestion fails with pgvector VECTOR(1536) column | Switched to `text-embedding-v4`; now pass `dimensions` param to API |
| Similarity threshold 0.7 too aggressive | Empty results on some queries (DashScope scores ~0.52) | Lowered `RETRIEVAL_SIMILARITY_THRESHOLD` to 0.45 in `.env` |

## Notes

- **Third-party models (GLM, DeepSeek, MiniMax, Kimi):** Available on DashScope Mainland China only (`dashscope.aliyuncs.com`), not on the international endpoint (`dashscope-intl.aliyuncs.com`). To compare these providers, use their direct APIs with separate API keys.
- **Evaluation suite:** Not run in this benchmark — requires DeepEval LLM-as-judge which adds cost. Planned for a future run.
