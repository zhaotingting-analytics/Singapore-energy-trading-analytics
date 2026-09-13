"""
Multi-model sentiment analyzer for financial news.

Three analysers are supported:
  - VADER    — rule-based, fast, no model download, handles punctuation/caps well.
  - TextBlob — simple lexicon-based baseline.
  - FinBERT  — ProsusAI/finbert, fine-tuned on financial corpora (optional; ~400 MB).

Results are merged into a composite [-1, +1] score using configurable weights.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from oil_sentiment.news_fetcher import NewsArticle

log = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    article_url: str
    title: str
    vader_score: Optional[float] = None       # compound: -1 to +1
    textblob_score: Optional[float] = None    # polarity: -1 to +1
    finbert_score: Optional[float] = None     # mapped to -1 to +1
    finbert_label: Optional[str] = None       # "positive" / "negative" / "neutral"
    composite_score: float = 0.0              # weighted ensemble
    label: str = "neutral"                    # "positive" / "negative" / "neutral"
    extra: Dict = field(default_factory=dict)


def _label_from_score(score: float, pos_thresh=0.05, neg_thresh=-0.05) -> str:
    if score >= pos_thresh:
        return "positive"
    if score <= neg_thresh:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Individual analysers
# ---------------------------------------------------------------------------

class VaderAnalyzer:
    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        return self._analyzer.polarity_scores(text)["compound"]


class TextBlobAnalyzer:
    def score(self, text: str) -> float:
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity


class FinBERTAnalyzer:
    """Lazy-loads ProsusAI/finbert on first use.

    Maps the three-class softmax to a scalar in [-1, +1]:
      positive probability - negative probability
    """

    _pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            log.info("Loading FinBERT model (first run may take a while)…")
            self._pipeline = pipeline(
                "text-classification",
                model=config.FINBERT_MODEL,
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )
            log.info("FinBERT loaded.")

    def score(self, text: str):
        self._load()
        results = self._pipeline(text[:512])[0]  # list of {label, score}
        scores = {r["label"].lower(): r["score"] for r in results}
        composite = scores.get("positive", 0) - scores.get("negative", 0)
        label = max(scores, key=scores.get)
        return composite, label


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """Run multiple models and return a weighted composite score."""

    def __init__(self, models: Optional[List[str]] = None):
        self.models = models or config.SENTIMENT_MODELS
        self._vader: Optional[VaderAnalyzer] = None
        self._textblob: Optional[TextBlobAnalyzer] = None
        self._finbert: Optional[FinBERTAnalyzer] = None

        if "vader" in self.models:
            try:
                self._vader = VaderAnalyzer()
            except ImportError:
                log.warning("vaderSentiment not installed; skipping.")

        if "textblob" in self.models:
            try:
                self._textblob = TextBlobAnalyzer()
            except ImportError:
                log.warning("textblob not installed; skipping.")

        if "finbert" in self.models:
            # Instantiate lazily — only loads weights on first .score() call
            try:
                self._finbert = FinBERTAnalyzer()
            except ImportError:
                log.warning("transformers not installed; skipping FinBERT.")

    def analyze(self, article: NewsArticle) -> SentimentResult:
        text = article.combined_text
        result = SentimentResult(article_url=article.url, title=article.title)

        weighted_sum = 0.0
        total_weight = 0.0

        if self._vader:
            result.vader_score = self._vader.score(text)
            w = config.SENTIMENT_WEIGHTS.get("vader", 1.0)
            weighted_sum += result.vader_score * w
            total_weight += w

        if self._textblob:
            result.textblob_score = self._textblob.score(text)
            w = config.SENTIMENT_WEIGHTS.get("textblob", 1.0)
            weighted_sum += result.textblob_score * w
            total_weight += w

        if self._finbert:
            try:
                composite, label = self._finbert.score(text)
                result.finbert_score = composite
                result.finbert_label = label
                w = config.SENTIMENT_WEIGHTS.get("finbert", 1.0)
                weighted_sum += composite * w
                total_weight += w
            except Exception as exc:
                log.warning("FinBERT inference failed for '%s': %s", article.title[:50], exc)

        result.composite_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        result.label = _label_from_score(result.composite_score)
        return result

    def analyze_batch(self, articles: List[NewsArticle]) -> List[SentimentResult]:
        results = []
        for article in articles:
            try:
                results.append(self.analyze(article))
            except Exception as exc:
                log.error("Sentiment analysis failed for '%s': %s", article.title[:50], exc)
        return results
