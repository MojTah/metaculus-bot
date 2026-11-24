"""LLM and forecasting configuration helpers driven by environment variables."""

from __future__ import annotations

import asyncio
import logging
import os
import re
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
_DEFAULT_BACKUP_MODELS = "gemini/gemini-2.5-flash-lite,gemini/gemini-2.5-flash"


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


class FallbackChainLlm(ThrottledLlm):
    """Throttled model that tries backup models when the main one fails.

    Free-tier Gemini often answers 503 "high demand" or 429 (daily limit);
    a backup model keeps the question from failing.
    """

    def __init__(self, *args, backups: list[str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backups: list[GeneralLlm] = [
            ThrottledLlm(
                model=m,
                temperature=kwargs.get("temperature", 0.3),
                timeout=kwargs.get("timeout", 120),
                allowed_tries=2,
            )
            for m in (backups or [])
        ]

    async def invoke(self, prompt, system_prompt=None) -> str:  # type: ignore[override]
        try:
            return await super().invoke(prompt, system_prompt)
        except Exception as first_error:
            last_error: Exception = first_error
            logger.warning("%s failed: %s", self.model, str(first_error)[:200])
            for llm in self.backups:
                try:
                    return await llm.invoke(prompt, system_prompt)
                except Exception as e:
                    last_error = e
                    logger.warning("Backup %s failed: %s", llm.model, str(e)[:200])
            raise last_error


class RoundRobinLlm(ThrottledLlm):
    """Send each call to the next model in a list (a cheap multi-model ensemble).

    With N predictions per question and N models, every model gives one
    prediction and forecasting-tools takes the median. Each model has its own
    free-tier quota, so this also spreads the daily load.
    """

    def __init__(self, members: list[GeneralLlm]) -> None:
        if not members:
            raise ValueError("RoundRobinLlm needs at least one model")
        super().__init__(model=members[0].model, temperature=0.3)
        self.members = members
        self._next = 0

    async def invoke(self, prompt, system_prompt=None) -> str:  # type: ignore[override]
        llm = self.members[self._next % len(self.members)]
        self._next += 1
        return await llm.invoke(prompt, system_prompt)


def _split_models(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


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
        backups = _split_models(
            _get_env(resolved, "BOT_GEMINI_BACKUP_MODELS") or _DEFAULT_BACKUP_MODELS
        )
        default_llm: GeneralLlm = FallbackChainLlm(
            model=default_model,
            temperature=0.3,
            timeout=120,
            allowed_tries=3,
            backups=backups,
        )
        ensemble = _split_models(_get_env(resolved, "BOT_GEMINI_ENSEMBLE") or "")
        if ensemble:
            default_llm = RoundRobinLlm(
                [
                    FallbackChainLlm(
                        model=m,
                        temperature=0.3,
                        timeout=180,
                        allowed_tries=3,
                        backups=backups,
                        min_interval=15.0 if "gemma" in m.lower() else None,
                    )
                    for m in ensemble
                ]
            )
        cheap_llm = FallbackChainLlm(
            model=cheap_model,
            temperature=0.3,
            timeout=120,
            allowed_tries=3,
            backups=backups,
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
                _split_models(grounded),
                fallback=FallbackChainLlm(
                    model=default_model,
                    temperature=0.1,
                    timeout=120,
                    allowed_tries=3,
                    backups=backups,
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


_PROBABILITY_RE = re.compile(r"probability\W{0,5}\s*(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE)


def parse_final_probability(text: str) -> float | None:
    """Return the last "Probability: ZZ%" in text as a decimal, or None."""
    matches = _PROBABILITY_RE.findall(text or "")
    if not matches:
        return None
    value = float(matches[-1])
    if not 0 <= value <= 100:
        return None
    return value / 100.0
