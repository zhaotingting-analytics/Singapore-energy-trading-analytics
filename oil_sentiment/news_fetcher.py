"""
Fetch oil-related news from RSS feeds and NewsAPI.

Priority order:
  1. NewsAPI  (requires NEWS_API_KEY in .env)
  2. Public RSS feeds (Reuters, BBC, NYT)
  3. Mock data  (fallback for testing / offline use)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

import config

log = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    title: str
    summary: str
    url: str
    published_at: datetime
    source: str
    full_text: str = field(default="")

    @property
    def combined_text(self) -> str:
        """Title + summary used as the primary sentiment input."""
        return f"{self.title}. {self.summary}"


# ---------------------------------------------------------------------------
# RSS fetcher
# ---------------------------------------------------------------------------

def _fetch_rss(feed_url: str, keywords: List[str]) -> List[NewsArticle]:
    try:
        import feedparser
    except ImportError:
        log.warning("feedparser not installed; skipping RSS source %s", feed_url)
        return []

    try:
        feed = feedparser.parse(feed_url)
    except Exception as exc:
        log.warning("RSS fetch failed for %s: %s", feed_url, exc)
        return []

    articles: List[NewsArticle] = []
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)

    cutoff = datetime.now(timezone.utc) - timedelta(days=config.NEWS_LOOKBACK_DAYS)

    for entry in feed.entries:
        title = getattr(entry, "title", "")
        summary = getattr(entry, "summary", "")
        combined = f"{title} {summary}"

        if not pattern.search(combined):
            continue

        # Parse published date
        published_at: Optional[datetime] = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        else:
            published_at = datetime.now(timezone.utc)

        if published_at < cutoff:
            continue

        articles.append(NewsArticle(
            title=title,
            summary=_strip_html(summary),
            url=getattr(entry, "link", ""),
            published_at=published_at,
            source=feed.feed.get("title", feed_url),
        ))

    return articles


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


# ---------------------------------------------------------------------------
# NewsAPI fetcher
# ---------------------------------------------------------------------------

def _fetch_newsapi(api_key: str, keywords: List[str]) -> List[NewsArticle]:
    if not api_key:
        return []

    articles: List[NewsArticle] = []
    from_date = (datetime.now(timezone.utc) - timedelta(days=config.NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    query = " OR ".join(f'"{k}"' for k in keywords[:5])  # NewsAPI free tier limits query length

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 100,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("NewsAPI request failed: %s", exc)
        return []

    for item in data.get("articles", []):
        try:
            published_at = datetime.fromisoformat(
                item["publishedAt"].replace("Z", "+00:00")
            )
        except (KeyError, ValueError):
            published_at = datetime.now(timezone.utc)

        articles.append(NewsArticle(
            title=item.get("title", "") or "",
            summary=item.get("description", "") or "",
            url=item.get("url", ""),
            published_at=published_at,
            source=item.get("source", {}).get("name", "NewsAPI"),
            full_text=item.get("content", "") or "",
        ))

    return articles


# ---------------------------------------------------------------------------
# Mock data for offline testing
# ---------------------------------------------------------------------------

def _mock_articles() -> List[NewsArticle]:
    now = datetime.now(timezone.utc)
    samples = [
        ("OPEC+ agrees to surprise production cut, sending oil prices surging",
         "OPEC+ members agreed to cut output by 1 million barrels per day, far exceeding expectations."),
        ("Oil prices tumble on fears of global recession",
         "Crude oil fell more than 3% as data showed slowing manufacturing across major economies."),
        ("Brent crude stabilises near $85 as traders weigh supply risks",
         "Oil markets steadied after volatile swings driven by geopolitical uncertainty in the Middle East."),
        ("US crude inventories rise unexpectedly, putting pressure on WTI",
         "EIA data showed a build of 4.2 million barrels, dampening the recent rally in oil prices."),
        ("Saudi Arabia extends voluntary oil output cut by another month",
         "The kingdom reaffirmed its commitment to market stability by prolonging the voluntary cut."),
        ("IEA raises global oil demand forecast on strong Chinese recovery",
         "The agency revised up its 2024 demand outlook by 200,000 bpd citing robust jet fuel consumption."),
        ("Oil drops as interest rate hike fears dampen energy demand outlook",
         "Crude benchmarks declined as Federal Reserve officials signalled further monetary tightening."),
        ("Russia halts diesel exports, tightening refined products market",
         "Moscow's temporary ban on diesel exports caused immediate price spikes in European markets."),
        ("Hedge funds increase long positions in oil ahead of OPEC meeting",
         "Speculative positioning in crude reached its highest level since March, analysts noted."),
        ("Nigeria oil output disrupted by pipeline sabotage",
         "Force majeure was declared at a key terminal after armed attacks on pipeline infrastructure."),
    ]
    articles = []
    for i, (title, summary) in enumerate(samples):
        articles.append(NewsArticle(
            title=title,
            summary=summary,
            url=f"https://example.com/oil-news-{i}",
            published_at=now - timedelta(days=i * 3),
            source="mock",
        ))
    return articles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news(
    keywords: Optional[List[str]] = None,
    use_mock: bool = False,
) -> List[NewsArticle]:
    """Return de-duplicated oil news articles sorted newest-first.

    Args:
        keywords: override config.NEWS_KEYWORDS
        use_mock: skip network calls and return mock data
    """
    if keywords is None:
        keywords = config.NEWS_KEYWORDS

    if use_mock:
        log.info("Using mock news data")
        return _mock_articles()

    articles: List[NewsArticle] = []

    # 1. NewsAPI
    newsapi_articles = _fetch_newsapi(config.NEWS_API_KEY, keywords)
    if newsapi_articles:
        log.info("NewsAPI returned %d articles", len(newsapi_articles))
        articles.extend(newsapi_articles)

    # 2. RSS feeds
    for feed_url in config.NEWS_SOURCES:
        rss_articles = _fetch_rss(feed_url, keywords)
        log.info("RSS %s returned %d articles", feed_url, len(rss_articles))
        articles.extend(rss_articles)

    # Fall back to mock if nothing came back
    if not articles:
        log.warning("No articles fetched from network sources; using mock data")
        return _mock_articles()

    # De-duplicate by URL
    seen: set[str] = set()
    unique = []
    for a in articles:
        if a.url not in seen:
            seen.add(a.url)
            unique.append(a)

    unique.sort(key=lambda a: a.published_at, reverse=True)
    log.info("Total unique articles: %d", len(unique))
    return unique
