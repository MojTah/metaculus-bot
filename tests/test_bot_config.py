from forecasting_tools import GeneralLlm

from bot_config import (
    FallbackResearchLlm,
    ThrottledLlm,
    build_llms,
    free_tier_interval,
    is_gemini_only,
    predictions_per_report,
)


def name(x):
    return GeneralLlm.to_model_name(x)


def test_openrouter_defaults():
    env = {"OPENROUTER_API_KEY": "sk-or-test"}
    llms = build_llms(env)
    assert llms is not None
    assert name(llms["default"]) == "openrouter/openai/gpt-4o"
    assert name(llms["summarizer"]) == "openrouter/openai/gpt-4o-mini"
    assert name(llms["parser"]) == "openrouter/openai/gpt-4o-mini"
    assert name(llms["researcher"]) == "openrouter/openai/gpt-4o-search-preview"
    assert is_gemini_only(env) is False
    assert predictions_per_report(env) == 5


def test_openrouter_model_overrides():
    env = {
        "OPENROUTER_API_KEY": "sk-or-test",
        "BOT_FORECAST_MODEL": "openrouter/anthropic/claude-sonnet-4",
        "BOT_CHEAP_MODEL": "openrouter/openai/gpt-4.1-mini",
        "BOT_RESEARCH_MODEL": "openrouter/perplexity/sonar",
    }
    llms = build_llms(env)
    assert llms is not None
    assert name(llms["default"]) == "openrouter/anthropic/claude-sonnet-4"
    assert name(llms["summarizer"]) == "openrouter/openai/gpt-4.1-mini"
    assert name(llms["parser"]) == "openrouter/openai/gpt-4.1-mini"
    assert name(llms["researcher"]) == "openrouter/perplexity/sonar"


def test_gemini_only_defaults_and_predictions():
    env = {"GEMINI_API_KEY": "gemini-test-key"}
    llms = build_llms(env)
    assert llms is not None
    assert name(llms["default"]) == "gemini/gemini-flash-lite-latest"
    assert name(llms["summarizer"]) == "gemini/gemini-flash-lite-latest"
    assert name(llms["parser"]) == "gemini/gemini-flash-lite-latest"
    for role in ("default", "summarizer", "parser"):
        assert isinstance(llms[role], ThrottledLlm)
    researcher = llms["researcher"]
    assert isinstance(researcher, FallbackResearchLlm)
    assert [m.model for m in researcher.chain] == [
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash-lite",
    ]
    assert researcher.chain[0].litellm_kwargs["tools"] == [{"googleSearch": {}}]
    assert researcher.fallback.model == "gemini/gemini-flash-lite-latest"
    assert is_gemini_only(env) is True
    assert predictions_per_report(env) == 3


def test_gemini_model_overrides():
    env = {
        "GEMINI_API_KEY": "gemini-test-key",
        "BOT_GEMINI_MODEL": "gemini/gemini-2.5-pro",
        "BOT_GEMINI_CHEAP_MODEL": "gemini/gemini-2.5-flash-lite",
        "BOT_GEMINI_SEARCH": "false",
    }
    llms = build_llms(env)
    assert llms is not None
    assert name(llms["default"]) == "gemini/gemini-2.5-pro"
    assert name(llms["researcher"]) == "gemini/gemini-2.5-pro"
    assert name(llms["summarizer"]) == "gemini/gemini-2.5-flash-lite"
    assert name(llms["parser"]) == "gemini/gemini-2.5-flash-lite"


def test_gemini_grounded_models_override():
    llms = build_llms(
        {"GEMINI_API_KEY": "k", "BOT_GEMINI_GROUNDED_MODELS": "gemini/a, gemini/b ,"}
    )
    assert llms is not None
    assert [m.model for m in llms["researcher"].chain] == ["gemini/a", "gemini/b"]


def test_free_tier_interval():
    assert 4 < free_tier_interval("gemini/gemini-flash-lite-latest") < 5
    assert 13 < free_tier_interval("gemini/gemini-2.5-flash") < 14
    assert free_tier_interval("gemini/gemma-4-31b-it") < 3


def test_asknews_researcher():
    llms_pair = build_llms(
        {
            "OPENROUTER_API_KEY": "sk-or-test",
            "ASKNEWS_CLIENT_ID": "client-id",
            "ASKNEWS_SECRET": "client-secret",
        }
    )
    assert llms_pair is not None
    assert name(llms_pair["researcher"]) == "asknews/news-summaries"

    env_key_gemini = {
        "GEMINI_API_KEY": "gemini-test-key",
        "ASKNEWS_API_KEY": "asknews-api-key",
    }
    llms_gemini = build_llms(env_key_gemini)
    assert llms_gemini is not None
    assert name(llms_gemini["researcher"]) == "asknews/news-summaries"
    assert is_gemini_only(env_key_gemini) is True

    llms_partial = build_llms(
        {"OPENROUTER_API_KEY": "sk-or-test", "ASKNEWS_CLIENT_ID": "client-id"}
    )
    assert llms_partial is not None
    assert name(llms_partial["researcher"]) == "openrouter/openai/gpt-4o-search-preview"


def test_replace_me_and_empty_count_as_unset():
    assert build_llms({"OPENROUTER_API_KEY": "REPLACE_ME"}) is None
    assert build_llms({"OPENROUTER_API_KEY": "REPLACE_ME", "GEMINI_API_KEY": ""}) is None

    env_fallback = {"OPENROUTER_API_KEY": "REPLACE_ME", "GEMINI_API_KEY": "k"}
    llms = build_llms(env_fallback)
    assert llms is not None
    assert name(llms["default"]) == "gemini/gemini-flash-lite-latest"
    assert is_gemini_only(env_fallback) is True

    llms_or = build_llms(
        {
            "OPENROUTER_API_KEY": "sk-or-test",
            "BOT_FORECAST_MODEL": "REPLACE_ME",
            "BOT_CHEAP_MODEL": "",
            "ASKNEWS_API_KEY": "REPLACE_ME",
        }
    )
    assert llms_or is not None
    assert name(llms_or["default"]) == "openrouter/openai/gpt-4o"
    assert name(llms_or["summarizer"]) == "openrouter/openai/gpt-4o-mini"
    assert name(llms_or["researcher"]) == "openrouter/openai/gpt-4o-search-preview"


def test_no_keys_returns_none():
    assert build_llms({}) is None
    assert is_gemini_only({}) is False
    assert predictions_per_report({}) == 5


def test_openrouter_takes_precedence_over_gemini():
    env = {"OPENROUTER_API_KEY": "sk-or-test", "GEMINI_API_KEY": "k"}
    llms = build_llms(env)
    assert llms is not None
    assert name(llms["default"]) == "openrouter/openai/gpt-4o"
    assert is_gemini_only(env) is False
    assert predictions_per_report(env) == 5


def test_gemini_with_other_llm_key_is_not_gemini_only():
    env = {"GEMINI_API_KEY": "k", "OPENAI_API_KEY": "sk-openai-test"}
    assert is_gemini_only(env) is False
    assert predictions_per_report(env) == 5


def test_predictions_per_report_valid_and_invalid():
    assert predictions_per_report({"BOT_PREDICTIONS": "1"}) == 1
    assert predictions_per_report({"BOT_PREDICTIONS": "7"}) == 7
    assert predictions_per_report({"BOT_PREDICTIONS": "10"}) == 10
    assert predictions_per_report({"GEMINI_API_KEY": "k", "BOT_PREDICTIONS": "8"}) == 8
    for invalid in ("0", "-1", "11", "abc", "3.5", "", "REPLACE_ME"):
        assert predictions_per_report({"BOT_PREDICTIONS": invalid}) == 5
        assert (
            predictions_per_report({"GEMINI_API_KEY": "k", "BOT_PREDICTIONS": invalid})
            == 3
        )


def test_fallback_research_uses_next_model_on_error():
    import asyncio

    class Boom(ThrottledLlm):
        async def invoke(self, prompt, system_prompt=None):
            raise RuntimeError("quota")

    class Ok(ThrottledLlm):
        async def invoke(self, prompt, system_prompt=None):
            return "plain research"

    r = FallbackResearchLlm(["gemini/x"], fallback=Ok(model="gemini/y"))
    r.chain = [Boom(model="gemini/x")]
    assert asyncio.run(r.invoke("q")) == "plain research"


def test_throttle_spaces_calls(monkeypatch):
    import asyncio

    from forecasting_tools import GeneralLlm as _G

    calls = []

    async def fake_call(self, prompt):
        calls.append(__import__("time").monotonic())
        return "ok"

    monkeypatch.setattr(_G, "_mockable_direct_call_to_model", fake_call)
    llm = ThrottledLlm(model="gemini/throttle-test", min_interval=0.2)

    async def run():
        await asyncio.gather(*(llm._mockable_direct_call_to_model("p") for _ in range(3)))

    asyncio.run(run())
    assert len(calls) == 3
    assert calls[1] - calls[0] >= 0.19 and calls[2] - calls[1] >= 0.19
