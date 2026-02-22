# =============================================================================
# Multi-Provider LLM Abstraction — Pluggable AI Backend
# =============================================================================
#
# Provides a common interface for LLM completions, with concrete
# implementations for Anthropic (Claude) and OpenAI-compatible APIs
# (DeepSeek, Qwen, GLM-5, MiniMax, Kimi, OpenAI).
#
# DESIGN DECISION: Protocol (structural typing) over ABC.
# Matches the VectorStore pattern in vectorstore.py — consistent across
# the project. Any class with the right `complete()` method works.
#
# DESIGN DECISION: Native SDKs over LangChain wrappers.
# LangChain's ChatAnthropic/ChatOpenAI add layers of abstraction we don't
# need. Using anthropic and openai SDKs directly gives us:
# 1. Fewer dependencies and moving parts
# 2. Direct control over request parameters
# 3. Easier debugging (no wrapper translation)
# 4. Access to provider-specific features if needed
#
# DESIGN DECISION: Async only.
# This service is called from FastAPI route handlers (async context).
# Celery workers don't need LLM calls in Phase 3.
#
# ARCHITECTURE:
#   LLMProvider (Protocol)
#   ├── AnthropicProvider       — Claude via native Anthropic SDK
#   │   └── complete()          — system prompt as top-level kwarg
#   ├── OpenAICompatibleProvider — Any OpenAI-compatible API
#   │   └── complete()          — system prompt as message role
#   ├── get_llm_provider()      — Singleton factory, reads from config
#   └── create_provider_from_id() — Non-singleton factory for /compare
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """
    Standardised response from any LLM provider.

    Normalises the different response formats (Anthropic vs OpenAI)
    into a single structure that downstream code can consume.
    """

    content: str           # The generated text
    model: str             # Model identifier (e.g., "claude-sonnet-4-6")
    input_tokens: int      # Tokens consumed by the prompt
    output_tokens: int     # Tokens generated in the response


# ---------------------------------------------------------------------------
# Protocol Definition
# ---------------------------------------------------------------------------


class LLMProvider(Protocol):
    """
    Protocol defining the LLM provider interface.

    Both Anthropic and OpenAI-compatible implementations must provide
    the `complete()` method. Checked statically by mypy.
    """

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: Conversation messages as dicts with "role" and "content".
                Roles: "user", "assistant" (no "system" — use the system param).
            system: System prompt for the LLM. Handled differently per provider:
                - Anthropic: top-level `system=` kwarg
                - OpenAI: prepended as {"role": "system", ...} message
            temperature: Override sampling temperature (default from config).
            max_tokens: Override max output tokens (default from config).

        Returns:
            LLMResponse with generated text and usage metrics.
        """
        ...


# ---------------------------------------------------------------------------
# Implementation 1: Anthropic (Claude)
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """
    Anthropic Claude provider using the native SDK.

    DESIGN DECISION: Uses AsyncAnthropic for non-blocking calls in FastAPI.
    The Anthropic SDK handles its own connection pooling and retries.

    KEY API DIFFERENCE: Anthropic takes system prompts as a top-level
    `system=` kwarg, NOT as a message with role "system". This is the
    opposite of OpenAI's pattern and a common source of bugs.

    DESIGN DECISION (Phase 4): Constructor accepts optional kwargs for
    api_key and model, enabling create_provider_from_id() to build
    fresh instances for /compare without touching the global singleton.
    No-arg construction still reads from settings (backward-compatible).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        from anthropic import AsyncAnthropic

        resolved_key = api_key or settings.llm_api_key or settings.anthropic_api_key
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key configured. Set LLM_API_KEY or "
                "ANTHROPIC_API_KEY in .env"
            )

        self._client = AsyncAnthropic(api_key=resolved_key)
        self._model = model or settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens

        logger.info(
            "Initialized AnthropicProvider (model=%s)", self._model
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion using Claude."""
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature or self._temperature,
        }

        # Anthropic: system prompt is a top-level kwarg, not a message
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)

        # Extract text from the first content block
        content = ""
        for block in response.content:
            if block.type == "text":
                content = block.text
                break

        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


# ---------------------------------------------------------------------------
# Implementation 2: OpenAI-Compatible (DeepSeek, Qwen, GLM-5, etc.)
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider:
    """
    OpenAI-compatible provider for any API that follows the OpenAI spec.

    DESIGN DECISION: Most Chinese LLMs (DeepSeek, Qwen, GLM-5, MiniMax,
    Kimi) expose OpenAI-compatible APIs. By using the OpenAI SDK with a
    custom base_url, we support all of them with a single implementation.

    Switching providers is a config change:
        LLM_PROVIDER=openai_compatible
        LLM_BASE_URL=https://api.deepseek.com/v1
        LLM_API_KEY=your-key
        LLM_MODEL=deepseek-chat

    DESIGN DECISION (Phase 4): Constructor accepts optional kwargs for
    api_key, model, and base_url for the same reason as AnthropicProvider.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        resolved_key = api_key or settings.llm_api_key or settings.openai_api_key
        if not resolved_key:
            raise ValueError(
                "No API key configured for OpenAI-compatible provider. "
                "Set LLM_API_KEY in .env"
            )

        client_kwargs: dict = {"api_key": resolved_key}
        resolved_base_url = base_url or settings.llm_base_url
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self._client = AsyncOpenAI(**client_kwargs)
        self._model = model or settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens

        logger.info(
            "Initialized OpenAICompatibleProvider (model=%s, base_url=%s)",
            self._model,
            resolved_base_url or "https://api.openai.com/v1",
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion using an OpenAI-compatible API."""
        # OpenAI: system prompt goes as the first message
        all_messages: list[dict[str, str]] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=all_messages,
            max_tokens=max_tokens or self._max_tokens,
            temperature=temperature or self._temperature,
        )

        content = response.choices[0].message.content or ""

        # Token counts: OpenAI uses different field names than Anthropic
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            content=content,
            model=response.model or self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

# Lazy singleton — avoid re-creating client on every request
_provider: AnthropicProvider | OpenAICompatibleProvider | None = None


def get_llm_provider() -> AnthropicProvider | OpenAICompatibleProvider:
    """
    Factory that returns the configured LLM provider.

    Reads `llm_provider` from settings:
    - "anthropic" → AnthropicProvider (Claude)
    - "openai_compatible" → OpenAICompatibleProvider (DeepSeek, Qwen, etc.)

    DESIGN DECISION: Lazy singleton. The SDK clients manage their own
    connection pools and are thread-safe. Creating one per request would
    waste connection setup time.
    """
    global _provider
    if _provider is None:
        if settings.llm_provider == "openai_compatible":
            _provider = OpenAICompatibleProvider()
        else:
            _provider = AnthropicProvider()
    return _provider


# ---------------------------------------------------------------------------
# Non-Singleton Factory — For /compare (Phase 4)
# ---------------------------------------------------------------------------
# The /compare endpoint needs to run the same query across N different
# providers in parallel. Each provider needs its own client instance
# (different API keys, base URLs, models). This factory creates fresh
# instances without touching the global singleton.
#
# DESIGN DECISION: Separate from get_llm_provider() because the
# singleton factory serves normal /ask traffic (one provider, reused).
# This factory serves /compare traffic (multiple providers, ephemeral).
# Mixing them would require resetting the singleton on every comparison.
# ---------------------------------------------------------------------------


_KNOWN_PROVIDER_TYPES = {"anthropic", "openai_compatible"}


def _parse_provider_id(
    provider_id: str,
) -> tuple[str, str, str | None]:
    """
    Parse a provider_id string into (provider_type, model, base_url).

    Formats supported:
        "anthropic/claude-sonnet-4-6"
            → ("anthropic", "claude-sonnet-4-6", None)
        "openai_compatible/deepseek-chat"
            → ("openai_compatible", "deepseek-chat", None)
        "openai_compatible/deepseek-chat@https://api.deepseek.com/v1"
            → ("openai_compatible", "deepseek-chat", "https://api.deepseek.com/v1")

    Raises:
        ValueError: If the format is unrecognisable or provider type unknown.
    """
    if "/" not in provider_id:
        raise ValueError(
            f"Invalid provider_id '{provider_id}'. "
            "Expected format: 'provider_type/model' or "
            "'provider_type/model@base_url'"
        )

    provider_type, rest = provider_id.split("/", 1)

    base_url: str | None = None
    if "@" in rest:
        model, base_url = rest.split("@", 1)
    else:
        model = rest

    if provider_type not in _KNOWN_PROVIDER_TYPES:
        raise ValueError(
            f"Unknown provider type '{provider_type}'. "
            f"Supported types: {sorted(_KNOWN_PROVIDER_TYPES)}"
        )

    return provider_type, model, base_url


def create_provider_from_id(
    provider_id: str,
    api_key: str | None = None,
) -> AnthropicProvider | OpenAICompatibleProvider:
    """
    Create a fresh, non-singleton LLM provider from a provider ID string.

    Used by the /compare endpoint to instantiate multiple providers
    for parallel execution. Each call returns an independent instance.

    Args:
        provider_id: Provider string (see _parse_provider_id for format).
        api_key: Optional API key override. If None, reads from env.

    Returns:
        A new provider instance (AnthropicProvider or OpenAICompatibleProvider).

    Raises:
        ValueError: If provider_id is invalid or API key is missing.
    """
    provider_type, model, base_url = _parse_provider_id(provider_id)

    if provider_type == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)

    # openai_compatible
    return OpenAICompatibleProvider(
        api_key=api_key, model=model, base_url=base_url,
    )
