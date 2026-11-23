"""LLM and forecasting configuration helpers driven by environment variables."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping
from typing import Literal

from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

_OTHER_LLM_KEYS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")

# Free-tier requests per minute seen in Google AI Studio (2026-11-23).
# The throttle keeps each model a little under its limit.
_FREE_TIER_RPM = (
    ("flash-lite", 15),
    ("gemma", 30),
    ("flash", 5),
)
_DEFAULT_GROUNDED_MODELS = "gemini/gemini-2.5-flash,gemini/gemini-2.5-flash-lite"


def free_tier_interval(model: str) -> float:
    """Seconds to wait between two calls to one Gemini model on the free tier."""
    name = model.lower()
    for key, rpm in _FREE_TIER_RPM:
        if key in name:
            return 60.0 / rpm * 1.15
    return 60.0 / 5 * 1.15


class ThrottledLlm(GeneralLlm):
    """GeneralLlm that spaces calls per model (shared by all instances)."""

    _locks: dict[str, asyncio.Lock] = {}
    _last_call: dict[str, float] = {}

    def __init__(self, *args, min_interval: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.min_interval = (
            free_tier_interval(self.model) if min_interval is None else min_interval
        )

    async def _mockable_direct_call_to_model(self, prompt):  # type: ignore[override]
        lock = self._locks.setdefault(self.model, asyncio.Lock())
        async with lock:
            wait = self._last_call.get(self.model, 0.0) + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call[self.model] = time.monotonic()
        return await super()._mockable_direct_call_to_model(prompt)


class FallbackResearchLlm(ThrottledLlm):
    """Research with Google Search grounding; fall back to the next model on error.

    Free-tier grounding is available on Gemini 2.5 models only and their daily
    limit is small, so the last fallback is the plain (not grounded) model.
    """

    def __init__(self, grounded_models: list[str], fallback: GeneralLlm) -> None:
        super().__init__(model=fallback.model, temperature=0.1)
        self.chain: list[GeneralLlm] = [
            ThrottledLlm(
                model=m,
                temperature=0.1,
                timeout=180,
                allowed_tries=1,
                tools=[{"googleSearch": {}}],
            )
            for m in grounded_models
        ]
        self.fallback = fallback

    async def invoke(self, prompt, system_prompt=None) -> str:  # type: ignore[override]
        for llm in self.chain:
            try:
                return await llm.invoke(prompt, system_prompt)
            except Exception as e:  # quota or tool errors: try the next model
                logger.warning("Grounded research with %s failed: %s", llm.model, str(e)[:200])
        return await self.fallback.invoke(prompt, system_prompt)


def _get_env(env: Mapping[str, str], key: str) -> str | None:
    val = env.get(key)
    if val is None:
        return None
    stripped = val.strip()
    if not stripped or stripped == "REPLACE_ME":
        return None
    return stripped


def _has_asknews(env: Mapping[str, str]) -> bool:
    has_pair = bool(
        _get_env(env, "ASKNEWS_CLIENT_ID") and _get_env(env, "ASKNEWS_SECRET")
    )
    return has_pair or bool(_get_env(env, "ASKNEWS_API_KEY"))


def active_provider(
    env: Mapping[str, str] | None = None,
) -> Literal["openrouter", "gemini", "default"]:
    """Return which LLM provider configuration build_llms() will select."""
    resolved = os.environ if env is None else env
    if _get_env(resolved, "OPENROUTER_API_KEY"):
        return "openrouter"
    if _get_env(resolved, "GEMINI_API_KEY"):
        return "gemini"
    return "default"


def is_gemini_only(env: Mapping[str, str] | None = None) -> bool:
    """Return True when GEMINI_API_KEY is set and no other LLM key is set."""
    resolved = os.environ if env is None else env
    return bool(_get_env(resolved, "GEMINI_API_KEY")) and not any(
        _get_env(resolved, key) for key in _OTHER_LLM_KEYS
    )


def build_llms(env: Mapping[str, str] | None = None) -> dict | None:
    """Build the ForecastBot llms dictionary from env vars, or None for defaults."""
    resolved = os.environ if env is None else env
    provider = active_provider(resolved)

    if provider == "openrouter":
        default_model = (
            _get_env(resolved, "BOT_FORECAST_MODEL") or "openrouter/openai/gpt-4o"
        )
        cheap_model = (
            _get_env(resolved, "BOT_CHEAP_MODEL") or "openrouter/openai/gpt-4o-mini"
        )
        researcher = (
            "asknews/news-summaries"
            if _has_asknews(resolved)
            else _get_env(resolved, "BOT_RESEARCH_MODEL")
            or "openrouter/openai/gpt-4o-search-preview"
        )
        return {
            "default": GeneralLlm(
                model=default_model,
                temperature=0.3,
                timeout=120,
                allowed_tries=3,
            ),
            "summarizer": cheap_model,
            "parser": cheap_model,
            "researcher": researcher,
        }

    if provider == "gemini":
        default_model = (
            _get_env(resolved, "BOT_GEMINI_MODEL")
            or "gemini/gemini-flash-lite-latest"
        )
        cheap_model = (
            _get_env(resolved, "BOT_GEMINI_CHEAP_MODEL")
            or "gemini/gemini-flash-lite-latest"
        )
        default_llm = ThrottledLlm(
            model=default_model, temperature=0.3, timeout=120, allowed_tries=4
        )
        cheap_llm = ThrottledLlm(
            model=cheap_model, temperature=0.3, timeout=120, allowed_tries=4
        )
        researcher: str | GeneralLlm
        search = (_get_env(resolved, "BOT_GEMINI_SEARCH") or "true").lower()
        if _has_asknews(resolved):
            researcher = "asknews/news-summaries"
        elif search in ("1", "true"):
            grounded = (
                _get_env(resolved, "BOT_GEMINI_GROUNDED_MODELS")
                or _DEFAULT_GROUNDED_MODELS
            )
            researcher = FallbackResearchLlm(
                [m.strip() for m in grounded.split(",") if m.strip()],
                fallback=ThrottledLlm(
                    model=default_model, temperature=0.1, timeout=120, allowed_tries=3
                ),
            )
        else:
            researcher = default_llm
        return {
            "default": default_llm,
            "summarizer": cheap_llm,
            "parser": cheap_llm,
            "researcher": researcher,
        }

    return None


def predictions_per_report(env: Mapping[str, str] | None = None) -> int:
    """Return predictions per research report from BOT_PREDICTIONS or provider default."""
    resolved = os.environ if env is None else env
    raw = _get_env(resolved, "BOT_PREDICTIONS")
    if raw is not None:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = None
        if parsed is not None and 1 <= parsed <= 10:
            return parsed
    return 3 if is_gemini_only(resolved) else 5
