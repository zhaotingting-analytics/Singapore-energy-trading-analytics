"""Unit tests for the sentiment analyzer (VADER + TextBlob only — no heavy downloads)."""

import pytest
from oil_sentiment.news_fetcher import NewsArticle
from oil_sentiment.sentiment_analyzer import SentimentAnalyzer, _label_from_score
from datetime import datetime, timezone


def _article(title: str, summary: str = "") -> NewsArticle:
    return NewsArticle(
        title=title,
        summary=summary,
        url=f"https://example.com/{hash(title)}",
        published_at=datetime.now(timezone.utc),
        source="test",
    )


class TestLabelFromScore:
    def test_positive(self):
        assert _label_from_score(0.1) == "positive"

    def test_negative(self):
        assert _label_from_score(-0.1) == "negative"

    def test_neutral(self):
        assert _label_from_score(0.0) == "neutral"

    def test_boundary_positive(self):
        assert _label_from_score(0.05) == "positive"

    def test_boundary_negative(self):
        assert _label_from_score(-0.05) == "negative"

    def test_just_below_positive(self):
        assert _label_from_score(0.04) == "neutral"


class TestVaderTextBlobOnly:
    """Use only lightweight models so CI doesn't download FinBERT."""

    @pytest.fixture
    def analyzer(self):
        return SentimentAnalyzer(models=["vader", "textblob"])

    def test_positive_headline(self, analyzer):
        article = _article("Oil prices surge strongly, excellent gains boost market confidence")
        result = analyzer.analyze(article)
        assert result.composite_score > 0, "Bullish headline should score positive"
        assert result.label == "positive"

    def test_negative_headline(self, analyzer):
        article = _article("Oil prices crash as recession fears devastate energy demand")
        result = analyzer.analyze(article)
        assert result.composite_score < 0, "Bearish headline should score negative"
        assert result.label == "negative"

    def test_neutral_headline(self, analyzer):
        article = _article("Oil prices unchanged as traders await OPEC decision")
        result = analyzer.analyze(article)
        assert result.label in {"neutral", "positive", "negative"}  # just no crash

    def test_score_bounds(self, analyzer):
        article = _article("OPEC+ agrees to extend voluntary cuts by 1 million bpd")
        result = analyzer.analyze(article)
        assert -1.0 <= result.composite_score <= 1.0

    def test_vader_score_populated(self, analyzer):
        article = _article("Brent crude oil rises on supply disruption news")
        result = analyzer.analyze(article)
        assert result.vader_score is not None

    def test_textblob_score_populated(self, analyzer):
        article = _article("US crude inventories declined more than expected this week")
        result = analyzer.analyze(article)
        assert result.textblob_score is not None

    def test_batch_analyze(self, analyzer):
        articles = [
            _article("Oil surges after Saudi Arabia announces surprise output cut"),
            _article("WTI crude tumbles on weak Chinese manufacturing data"),
            _article("Crude oil holds steady ahead of OPEC meeting"),
        ]
        results = analyzer.analyze_batch(articles)
        assert len(results) == 3
        for r in results:
            assert -1.0 <= r.composite_score <= 1.0

    def test_article_url_preserved(self, analyzer):
        article = _article("Oil market update", "Supply and demand in balance")
        result = analyzer.analyze(article)
        assert result.article_url == article.url

    def test_empty_text_does_not_crash(self, analyzer):
        article = _article("", "")
        result = analyzer.analyze(article)
        assert result.composite_score is not None
