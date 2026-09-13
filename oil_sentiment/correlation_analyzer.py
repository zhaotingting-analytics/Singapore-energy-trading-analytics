"""
Correlate daily news sentiment with oil price movements.

Produces:
  - merged DataFrame (date / sentiment / price)
  - Pearson + Spearman correlations
  - Rolling 7-day correlation
  - Lead/lag correlation at ±5 days
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

import config
from oil_sentiment.news_fetcher import NewsArticle
from oil_sentiment.sentiment_analyzer import SentimentResult

log = logging.getLogger(__name__)


def build_daily_sentiment(
    articles: List[NewsArticle],
    results: List[SentimentResult],
) -> pd.DataFrame:
    """Aggregate per-article scores to a daily mean composite score.

    Returns DataFrame with columns: date, composite_score, article_count,
    positive_count, negative_count, neutral_count.
    """
    url_to_date = {a.url: a.published_at.date() for a in articles}
    rows = []
    for r in results:
        date = url_to_date.get(r.article_url)
        if date is None:
            continue
        rows.append({
            "date": pd.Timestamp(date),
            "composite_score": r.composite_score,
            "label": r.label,
        })

    if not rows:
        return pd.DataFrame(columns=["date", "composite_score", "article_count",
                                     "positive_count", "negative_count", "neutral_count"])

    df = pd.DataFrame(rows)
    daily = (
        df.groupby("date")
        .agg(
            composite_score=("composite_score", "mean"),
            article_count=("composite_score", "count"),
            positive_count=("label", lambda s: (s == "positive").sum()),
            negative_count=("label", lambda s: (s == "negative").sum()),
            neutral_count=("label", lambda s: (s == "neutral").sum()),
        )
        .reset_index()
        .sort_values("date")
    )
    return daily


def merge_sentiment_price(
    daily_sentiment: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str = "WTI",
) -> pd.DataFrame:
    """Left-join sentiment onto price data for the given ticker."""
    price_df = prices[prices["ticker"] == ticker][["date", "close", "pct_change", "direction"]].copy()
    price_df["date"] = pd.to_datetime(price_df["date"])

    merged = pd.merge(price_df, daily_sentiment, on="date", how="left")
    merged["composite_score"] = merged["composite_score"].ffill()
    merged = merged.dropna(subset=["pct_change", "composite_score"])
    return merged.sort_values("date").reset_index(drop=True)


def compute_correlations(merged: pd.DataFrame) -> Dict:
    """Pearson & Spearman between sentiment and same-day / next-day price change."""
    if len(merged) < 5:
        return {}

    x = merged["composite_score"].values
    y_same = merged["pct_change"].values
    y_next = merged["pct_change"].shift(-1).dropna().values
    x_next = x[:len(y_next)]

    def _corr_pair(a, b):
        if len(a) < 3:
            return {}
        pearson_r, pearson_p = stats.pearsonr(a, b)
        spearman_r, spearman_p = stats.spearmanr(a, b)
        return {
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "n": len(a),
        }

    return {
        "same_day": _corr_pair(x, y_same),
        "next_day": _corr_pair(x_next, y_next),
    }


def compute_rolling_correlation(
    merged: pd.DataFrame,
    window: int = None,
) -> pd.DataFrame:
    """Return a DataFrame with date + rolling Pearson correlation."""
    if window is None:
        window = config.ROLLING_WINDOW_DAYS

    df = merged[["date", "composite_score", "pct_change"]].copy()
    df["rolling_corr"] = (
        df["composite_score"]
        .rolling(window, min_periods=3)
        .corr(df["pct_change"])
    )
    return df[["date", "rolling_corr"]].dropna()


def compute_lag_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of sentiment at lag k with price change at lag 0.

    Positive lag = sentiment leads price (predictive).
    Negative lag = price leads sentiment (reactive).
    """
    rows = []
    x_base = merged["composite_score"].values
    y_base = merged["pct_change"].values
    n = len(x_base)

    for lag in config.CORRELATION_LAG_DAYS:
        if lag >= 0:
            x = x_base[: n - lag] if lag > 0 else x_base
            y = y_base[lag:] if lag > 0 else y_base
        else:
            shift = -lag
            x = x_base[shift:]
            y = y_base[: n - shift]

        if len(x) < 3:
            continue
        try:
            r, p = stats.pearsonr(x, y)
            rows.append({"lag_days": lag, "pearson_r": round(r, 4), "p_value": round(p, 4)})
        except Exception:
            pass

    return pd.DataFrame(rows)
