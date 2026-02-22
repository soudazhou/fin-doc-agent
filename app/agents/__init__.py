# =============================================================================
# Agents Package — LangGraph Multi-Agent Orchestration
# =============================================================================
# Implements a graph-based multi-agent system with agentic search:
#
#   orchestrator.py — LangGraph StateGraph: classify → search → analyse
#   search.py       — Agentic search loop: embed → retrieve → evaluate → refine
#   analyst.py      — LLM answer generation with capability-specific prompts
#
# Capabilities: qa, summarise, compare, extract
# Search loop: embed → retrieve → evaluate → refine (max N iterations)
# =============================================================================
