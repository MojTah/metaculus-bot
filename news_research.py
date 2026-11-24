"""Free web research from Google News RSS (no API key, no sign-up).

Google retired the free Gemini 2.5 models (the only free tier with Google
Search grounding) for new users on 2026-11-24, so the bot needs another free
news source. Flow per question:
1. the LLM writes up to 3 short search queries;
2. Google News RSS returns recent headlines with source, date and snippet;
3. the LLM writes the research rundown from those headlines.
If the news fetch fails, the LLM writes the rundown without headlines.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import requests
from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

_RSS_URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
_TAG_RE = re.compile(r"<[^>]+>")


def fetch_google_news(query: str, max_items: int = 8, timeout: float = 20.0) -> list[dict]:
    """Return recent news items for a query: title, source, date, snippet."""
    url = _RSS_URL.format(q=quote_plus(query))
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (forecasting bot)"})
    r.raise_for_status()
    return parse_rss(r.content, max_items)


def parse_rss(content: bytes | str, max_items: int = 8) -> list[dict]:
    root = ET.fromstring(content)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        snippet = _TAG_RE.sub(" ", html.unescape(item.findtext("description") or ""))
        items.append(
            {
                "title": title,
                "source": (item.findtext("source") or "").strip(),
                "date": (item.findtext("pubDate") or "").strip(),
                "snippet": " ".join(snippet.split())[:300],
            }
        )
        if len(items) >= max_items:
            break
    return items


def parse_queries(text: str, limit: int = 3) -> list[str]:
    """Read one query per line; drop numbering, bullets and quotes."""
    out = []
    for line in (text or "").splitlines():
        q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        if 2 <= len(q) <= 120 and not q.lower().startswith(("here", "queries", "search")):
            out.append(q)
        if len(out) >= limit:
            break
    return out


def format_items(items: list[dict]) -> str:
    return "\n".join(
        f"- [{it['date']}] {it['title']} ({it['source']}): {it['snippet']}" for it in items
    )


class NewsResearchLlm(GeneralLlm):
    """Researcher that adds Google News headlines before the LLM writes research."""

    def __init__(self, llm: GeneralLlm, max_items: int = 15) -> None:
        super().__init__(model=llm.model, temperature=0.1)
        self.llm = llm
        self.max_items = max_items

    async def _gather_news(self, research_prompt: str) -> list[dict]:
        query_text = await self.llm.invoke(
            "Write up to 3 short Google News search queries (2-6 words each, one per "
            "line, no numbering, no other text) that find the latest news needed to "
            "forecast this question:\n\n" + research_prompt[:3000]
        )
        queries = parse_queries(query_text)
        seen: set[str] = set()
        items: list[dict] = []
        for q in queries:
            try:
                found = await asyncio.to_thread(fetch_google_news, q)
            except Exception as e:
                logger.warning("News search failed for %r: %s", q, str(e)[:200])
                continue
            for it in found:
                key = it["title"].lower()
                if key not in seen:
                    seen.add(key)
                    items.append(it)
        logger.info("News research: %d queries, %d items", len(queries), len(items))
        return items[: self.max_items]

    async def invoke(self, prompt, system_prompt=None) -> str:  # type: ignore[override]
        try:
            items = await self._gather_news(prompt)
        except Exception as e:
            logger.warning("News gathering failed: %s", str(e)[:200])
            items = []
        if not items:
            return await self.llm.invoke(prompt, system_prompt)
        return await self.llm.invoke(
            prompt
            + "\n\nRecent news headlines (Google News, newest information first where "
            "possible). Use them, cite dates, and say when they do not answer the "
            "question:\n"
            + format_items(items),
            system_prompt,
        )
