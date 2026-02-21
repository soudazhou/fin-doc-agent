# =============================================================================
# Models Package — Pydantic V2 Schemas
# =============================================================================
# Defines request/response schemas for the API.
# These are SEPARATE from the database models (app/db/models.py).
#
# DESIGN DECISION: Separating API schemas from DB models is a best practice:
# 1. API schemas define what clients see (public contract)
# 2. DB models define how data is stored (internal concern)
# 3. They can evolve independently (e.g., add API fields without DB migration)
# 4. Prevents accidentally exposing internal fields (e.g., embedding vectors)
# =============================================================================
