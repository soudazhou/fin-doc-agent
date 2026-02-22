# Code Review Report

Date: 2026-02-22
Reviewer: Codex agent
Scope: `/workspace/fin-doc-agent`

## Executive Summary

The project is thoughtfully structured and documented, with strong modular boundaries across API, agent orchestration, services, and DB models. I found **two high-impact implementation risks** and **two medium-impact quality issues** that should be prioritized.

## Findings

### 1) High — Internal exception details are returned to clients from `/ask`

`ask_endpoint` catches broad exceptions and includes `str(e)` directly in the public HTTP 502 response body. This may leak internal topology, provider details, stack-adjacent information, or transient backend errors to external consumers.

- File: `app/api/ask.py`
- Current behavior: `detail=f"LLM service error: {e}"`
- Recommendation: return a generic error message to clients and log the exception server-side (already done with `logger.exception`).

### 2) High — Unbounded question content is persisted in query metrics

`_persist_metric` writes the full incoming question to `QueryMetric.question`. In finance/compliance contexts, this can capture sensitive customer prompts, identifiers, or confidential text and keep it indefinitely.

- File: `app/api/ask.py`
- Current behavior: raw `question` is persisted as-is.
- Recommendation: add configurable truncation and/or redaction before persistence, and document retention policy.

### 3) Medium — Test/dev environment drift is easy to hit without `uv sync --group dev`

The repo expects Python 3.12+ and several optional dev dependencies, but a default environment may fail test collection with import errors (`pydantic_settings`, `langgraph`, `tiktoken`, `chromadb`) and Python-version mismatch (`datetime.UTC` in 3.10).

- File: `pyproject.toml`
- Recommendation: add a short "Quick Start Validation" section in README with explicit environment setup (`uv sync --group dev`) and required Python version check.

### 4) Medium — Duplicate commit patterns in request-scoped DB sessions

Multiple endpoints call `await session.commit()` explicitly while `get_async_session` also commits when dependency scope exits. This is not incorrect, but creates inconsistent transaction semantics and can surprise maintainers.

- Files: `app/db/engine.py`, `app/api/admin.py`, `app/api/evaluate.py`, `app/api/ask.py`
- Recommendation: define and document one policy:
  - either explicit commit in endpoints/services, or
  - centralized commit-on-exit with explicit flush where needed.

## Positive Notes

- Excellent in-file architectural comments and rationale throughout core modules.
- Good separation between request models, response models, auth dependencies, and service logic.
- Practical ACL + scope model with audit logging hooks.
- Clear benchmark/eval scaffolding for iterative quality improvements.
