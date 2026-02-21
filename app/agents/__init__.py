# =============================================================================
# Agents Package — LangGraph Multi-Agent Orchestration
# =============================================================================
# Implements a graph-based multi-agent system with agentic search:
#   - orchestrator.py: LangGraph graph — classifies intent, routes to
#     capability-specific agents, manages search iteration loop
#   - retriever.py: Searches vector store for relevant chunks, supports
#     pluggable backends (pgvector, Chroma)
#   - analyst.py: LLM reasoning agent — synthesizes answers with citations,
#     self-evaluates sufficiency, requests re-retrieval if needed
#
# Capabilities: qa, summarise, compare, extract
# Search loop: plan → retrieve → evaluate → refine (max N iterations)
# =============================================================================
