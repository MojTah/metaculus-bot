from forecasting_tools import GeneralLlm

from bot_config import build_llms, is_gemini_only, predictions_per_report


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
    assert name(llms["default"]) == "gemini/gemini-flash-latest"
    assert name(llms["summarizer"]) == "gemini/gemini-flash-lite-latest"
    assert name(llms["parser"]) == "gemini/gemini-flash-lite-latest"
    assert name(llms["researcher"]) == "gemini/gemini-flash-latest"
    assert is_gemini_only(env) is True
    assert predictions_per_report(env) == 3


def test_gemini_model_overrides():
    env = {
        "GEMINI_API_KEY": "gemini-test-key",
        "BOT_GEMINI_MODEL": "gemini/gemini-2.5-pro",
        "BOT_GEMINI_CHEAP_MODEL": "gemini/gemini-2.5-flash-lite",
    }
    llms = build_llms(env)
    assert llms is not None
    assert name(llms["default"]) == "gemini/gemini-2.5-pro"
    assert name(llms["researcher"]) == "gemini/gemini-2.5-pro"
    assert name(llms["summarizer"]) == "gemini/gemini-2.5-flash-lite"
    assert name(llms["parser"]) == "gemini/gemini-2.5-flash-lite"


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
    assert name(llms["default"]) == "gemini/gemini-flash-latest"
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


def test_gemini_search_grounding_toggle():
    llms = build_llms({"GEMINI_API_KEY": "k", "BOT_GEMINI_SEARCH": "true"})
    assert llms is not None
    researcher = llms["researcher"]
    assert isinstance(researcher, GeneralLlm)
    assert name(researcher) == "gemini/gemini-flash-latest"
    assert researcher.litellm_kwargs["tools"] == [{"googleSearch": {}}]
    off = build_llms({"GEMINI_API_KEY": "k", "BOT_GEMINI_SEARCH": "0"})
    assert off is not None and off["researcher"] == "gemini/gemini-flash-latest"
