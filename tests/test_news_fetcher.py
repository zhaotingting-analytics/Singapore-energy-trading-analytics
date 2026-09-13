"""Unit tests for the news fetcher module."""

import pytest
from datetime import datetime, timezone
from oil_sentiment.news_fetcher import fetch_news, NewsArticle, _strip_html


class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_passthrough_plain(self):
        assert _strip_html("plain text") == "plain text"

    def test_empty(self):
        assert _strip_html("") == ""


class TestMockFetch:
    def test_returns_list(self):
        articles = fetch_news(use_mock=True)
        assert isinstance(articles, list)
        assert len(articles) > 0

    def test_all_are_news_articles(self):
        articles = fetch_news(use_mock=True)
        for a in articles:
            assert isinstance(a, NewsArticle)

    def test_sorted_newest_first(self):
        articles = fetch_news(use_mock=True)
        dates = [a.published_at for a in articles]
        assert dates == sorted(dates, reverse=True)

    def test_no_duplicate_urls(self):
        articles = fetch_news(use_mock=True)
        urls = [a.url for a in articles]
        assert len(urls) == len(set(urls))

    def test_combined_text_nonempty(self):
        articles = fetch_news(use_mock=True)
        for a in articles:
            assert len(a.combined_text) > 0

    def test_published_at_is_timezone_aware(self):
        articles = fetch_news(use_mock=True)
        for a in articles:
            assert a.published_at.tzinfo is not None
