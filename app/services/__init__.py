# =============================================================================
# Services Package — Business Logic
# =============================================================================
# Contains the core business logic, separated from API handlers:
#   - parser.py: PDF parsing with Docling (table-aware extraction)
#   - chunker.py: Token-based text chunking (configurable size: 256/512/1024)
#   - embedder.py: OpenAI embedding generation (batch processing)
#   - evaluator.py: RAG evaluation with DeepEval (faithfulness, relevancy,
#     context precision/recall) + regression tracking
#   - vectorstore.py: Pluggable vector store protocol (pgvector, Chroma)
#   - llm.py: Multi-provider LLM abstraction (Anthropic, OpenAI-compatible)
# =============================================================================
