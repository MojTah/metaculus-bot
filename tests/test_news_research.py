import asyncio

from forecasting_tools import GeneralLlm

from news_research import NewsResearchLlm, format_items, parse_queries, parse_rss

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Fed holds rates steady - Reuters</title><pubDate>Mon, 23 Nov 2026 10:00:00 GMT</pubDate>
<description>&lt;a href="x"&gt;Fed holds rates&lt;/a&gt; &amp;nbsp; officials say</description><source url="r">Reuters</source></item>
<item><title></title></item>
<item><title>Second</title><pubDate>Sun, 22 Nov 2026</pubDate><description>b</description></item>
</channel></rss>"""


def test_parse_rss():
    items = parse_rss(RSS)
    assert len(items) == 2
    assert items[0]["source"] == "Reuters"
    assert "<a" not in items[0]["snippet"] and "Fed holds rates" in items[0]["snippet"]
    assert "Reuters" in format_items(items)
    assert len(parse_rss(RSS, max_items=1)) == 1


def test_parse_queries():
    text = 'Here are the queries:\n1. "Fed rate decision December"\n- ECB rates\n* x\nfourth one\nfifth'
    assert parse_queries(text) == ["Fed rate decision December", "ECB rates", "fourth one"]


class FakeLlm(GeneralLlm):
    def __init__(self):
        super().__init__(model="gemini/fake")
        self.prompts = []

    async def invoke(self, prompt, system_prompt=None):
        self.prompts.append(prompt)
        if prompt.startswith("Write up to 3"):
            return "fed rates\nfed rates"
        return "research"


def test_news_research_adds_headlines(monkeypatch):
    import news_research

    monkeypatch.setattr(news_research, "fetch_google_news", lambda q: parse_rss(RSS))
    llm = FakeLlm()
    out = asyncio.run(NewsResearchLlm(llm).invoke("Question: will the Fed cut?"))
    assert out == "research"
    assert "Fed holds rates steady" in llm.prompts[-1]
    # duplicate headlines from the repeated query are dropped
    assert llm.prompts[-1].count("Fed holds rates steady - Reuters") == 1


def test_news_research_falls_back_without_news(monkeypatch):
    import news_research

    def boom(q):
        raise RuntimeError("blocked")

    monkeypatch.setattr(news_research, "fetch_google_news", boom)
    llm = FakeLlm()
    assert asyncio.run(NewsResearchLlm(llm).invoke("Q")) == "research"
    assert llm.prompts[-1] == "Q"
