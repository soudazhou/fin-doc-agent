# =============================================================================
# Agents Package — LangGraph Multi-Agent Orchestration
# =============================================================================
# Implements a graph-based multi-agent system for financial document Q&A:
#   - orchestrator.py: LangGraph graph that coordinates agent flow
#   - retriever.py: Searches pgvector for relevant document chunks
#   - analyst.py: Uses Claude to synthesize answers from retrieved context
# =============================================================================
