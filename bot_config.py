"""LLM and forecasting configuration helpers driven by environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from forecasting_tools import GeneralLlm

_OTHER_LLM_KEYS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


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
            _get_env(resolved, "BOT_GEMINI_MODEL") or "gemini/gemini-flash-latest"
        )
        cheap_model = (
            _get_env(resolved, "BOT_GEMINI_CHEAP_MODEL")
            or "gemini/gemini-flash-lite-latest"
        )
        researcher: str | GeneralLlm
        if _has_asknews(resolved):
            researcher = "asknews/news-summaries"
        elif (_get_env(resolved, "BOT_GEMINI_SEARCH") or "").lower() in ("1", "true"):
            # Gemini with Google Search grounding (litellm passes the tool on).
            researcher = GeneralLlm(
                model=default_model,
                temperature=0.1,
                timeout=180,
                allowed_tries=3,
                tools=[{"googleSearch": {}}],
            )
        else:
            researcher = default_model
        return {
            "default": GeneralLlm(
                model=default_model,
                temperature=0.3,
                timeout=120,
                allowed_tries=4,
            ),
            "summarizer": cheap_model,
            "parser": cheap_model,
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
