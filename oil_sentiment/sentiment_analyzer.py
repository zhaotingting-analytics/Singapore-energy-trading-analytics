"""
Multi-model sentiment analyzer for financial news.

Four analysers are supported:
  - VADER      — rule-based, fast, no model download.
  - TextBlob   — simple lexicon baseline.
  - FinBERT    — ProsusAI/finbert transformer (optional; ~400 MB).
  - neural_lstm— custom BiLSTM + Attention (trained on-the-fly or loaded
                 from output/neural_model/).

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
    vader_score: Optional[float] = None        # compound: -1 to +1
    textblob_score: Optional[float] = None     # polarity: -1 to +1
    finbert_score: Optional[float] = None      # mapped to -1 to +1
    finbert_label: Optional[str] = None
    neural_score: Optional[float] = None       # BiLSTM: -1 to +1
    neural_label: Optional[str] = None
    composite_score: float = 0.0
    label: str = "neutral"
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
    """Lazy-loads ProsusAI/finbert on first use."""

    _pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            log.info("Loading FinBERT model…")
            self._pipeline = pipeline(
                "text-classification",
                model=config.FINBERT_MODEL,
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )

    def score(self, text: str):
        self._load()
        results = self._pipeline(text[:512])[0]
        scores = {r["label"].lower(): r["score"] for r in results}
        composite = scores.get("positive", 0) - scores.get("negative", 0)
        return composite, max(scores, key=scores.get)


class NeuralLSTMAnalyzer:
    """
    Wraps NeuralSentimentAnalyzer for use inside the ensemble.

    On first use it either loads saved weights from output/neural_model/
    or trains from scratch on the articles passed to SentimentAnalyzer.
    Pass `articles` to SentimentAnalyzer.__init__ so training happens
    before batch inference begins.
    """

    def __init__(self, articles=None, model_dir: Optional[str] = None):
        from oil_sentiment.neural_model import NeuralSentimentAnalyzer
        self._nn = NeuralSentimentAnalyzer(model_dir=model_dir)

        if not self._nn._saved_model_exists():
            log.info("No saved neural model found — training from scratch…")
            self._nn.fit(articles)
        else:
            log.info("Neural model loaded from %s", self._nn.model_dir)

    def score(self, text: str):
        return self._nn.score(text)  # (composite_score, label)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """Run multiple models and return a weighted composite score.

    Args:
        models:   list of model names to activate.  Defaults to
                  config.SENTIMENT_MODELS.
        articles: list of NewsArticle objects — forwarded to
                  NeuralLSTMAnalyzer for bootstrap training when
                  "neural_lstm" is in `models` and no saved model exists.
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        articles=None,
    ):
        self.models = models or config.SENTIMENT_MODELS
        self._vader:   Optional[VaderAnalyzer]      = None
        self._textblob: Optional[TextBlobAnalyzer]  = None
        self._finbert: Optional[FinBERTAnalyzer]    = None
        self._neural:  Optional[NeuralLSTMAnalyzer] = None

        if "vader" in self.models:
            try:
                self._vader = VaderAnalyzer()
            except ImportError:
                log.warning("vaderSentiment not installed; skipping VADER.")

        if "textblob" in self.models:
            try:
                self._textblob = TextBlobAnalyzer()
            except ImportError:
                log.warning("textblob not installed; skipping TextBlob.")

        if "finbert" in self.models:
            try:
                self._finbert = FinBERTAnalyzer()
            except ImportError:
                log.warning("transformers not installed; skipping FinBERT.")

        if "neural_lstm" in self.models:
            try:
                self._neural = NeuralLSTMAnalyzer(articles=articles)
            except Exception as exc:
                log.warning("NeuralLSTM init failed: %s", exc)

    # ------------------------------------------------------------------
    def analyze(self, article: NewsArticle) -> SentimentResult:
        text   = article.combined_text
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
                log.warning("FinBERT failed for '%s': %s", article.title[:50], exc)

        if self._neural:
            try:
                composite, label = self._neural.score(text)
                result.neural_score = composite
                result.neural_label = label
                w = config.SENTIMENT_WEIGHTS.get("neural_lstm", 1.0)
                weighted_sum += composite * w
                total_weight += w
            except Exception as exc:
                log.warning("NeuralLSTM failed for '%s': %s", article.title[:50], exc)

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
