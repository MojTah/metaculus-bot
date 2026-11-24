"""Check which Gemini models (and Google Search grounding) answer on this key.

Prints one line per check: model, grounded or not, HTTP status, short reason.
Uses 1 tiny request per check.
"""

import os
import sys

import requests

KEY = os.environ.get("GEMINI_API_KEY", "")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
MODELS = sys.argv[1:] or [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemini-3.8-flash",
]


def check(model: str, grounded: bool) -> str:
    body = {"contents": [{"parts": [{"text": "Reply with the word OK and today's top news headline."}]}]}
    if grounded:
        body["tools"] = [{"google_search": {}}]
    try:
        r = requests.post(URL.format(m=model), params={"key": KEY}, json=body, timeout=60)
    except Exception as e:  # network error
        return f"ERR {type(e).__name__}"
    if r.ok:
        meta = (r.json().get("candidates") or [{}])[0].get("groundingMetadata")
        return f"200 grounded_meta={'yes' if meta else 'no'}"
    try:
        msg = r.json().get("error", {}).get("message", "")[:140]
    except Exception:
        msg = r.text[:140]
    return f"{r.status_code} {msg}"


for m in MODELS:
    for g in (False, True):
        if g and m.startswith("gemma"):
            continue
        print(f"PROBE {m:28s} grounded={g!s:5s} -> {check(m, g)}", flush=True)

# Free news source used by news_research.py
try:
    from news_research import fetch_google_news

    items = fetch_google_news("Federal Reserve interest rates")
    print(f"PROBE google-news-rss -> {len(items)} items; first: {items[0]['date'] if items else '-'} | {items[0]['title'][:80] if items else '-'}")
except Exception as e:
    print(f"PROBE google-news-rss -> ERR {type(e).__name__}: {str(e)[:120]}")
