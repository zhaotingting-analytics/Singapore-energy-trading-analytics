"""
End-to-end pipeline: fetch news → analyse sentiment → correlate with prices → report.

Usage (CLI):
    python -m oil_sentiment.pipeline --ticker WTI --mock --output output/report.html
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

import config
from oil_sentiment.news_fetcher import NewsArticle, fetch_news
from oil_sentiment.sentiment_analyzer import SentimentAnalyzer, SentimentResult
from oil_sentiment.price_data import fetch_oil_prices
from oil_sentiment.correlation_analyzer import (
    build_daily_sentiment,
    compute_correlations,
    compute_lag_correlations,
    compute_rolling_correlation,
    merge_sentiment_price,
)

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    articles: List[NewsArticle]
    sentiment_results: List[SentimentResult]
    daily_sentiment: pd.DataFrame
    prices: pd.DataFrame
    merged: pd.DataFrame
    correlations: dict
    rolling_corr: pd.DataFrame
    lag_df: pd.DataFrame
    report_path: Optional[str] = None


def run_pipeline(
    ticker: str = "WTI",
    models: Optional[List[str]] = None,
    use_mock: bool = False,
    output_path: Optional[str] = None,
) -> PipelineResult:
    """Execute the full sentiment analysis pipeline.

    Args:
        ticker: "WTI" or "Brent"
        models: list of sentiment models to use; None → config.SENTIMENT_MODELS
        use_mock: use offline synthetic data (no network / API keys needed)
        output_path: if provided, write HTML report here

    Returns:
        PipelineResult with all intermediate and final data
    """
    if models is None:
        models = config.SENTIMENT_MODELS

    log.info("=== Step 1/4: Fetching news ===")
    articles = fetch_news(use_mock=use_mock)
    log.info("Fetched %d articles", len(articles))

    log.info("=== Step 2/4: Analysing sentiment ===")
    analyzer = SentimentAnalyzer(models=models)
    sentiment_results = analyzer.analyze_batch(articles)
    log.info("Sentiment analysed for %d articles", len(sentiment_results))

    log.info("=== Step 3/4: Fetching oil prices ===")
    prices = fetch_oil_prices(use_mock=use_mock)

    log.info("=== Step 4/4: Correlating & building report ===")
    daily_sentiment = build_daily_sentiment(articles, sentiment_results)
    merged = merge_sentiment_price(daily_sentiment, prices, ticker=ticker)
    correlations = compute_correlations(merged)
    rolling_corr = compute_rolling_correlation(merged)
    lag_df = compute_lag_correlations(merged)

    report_path: Optional[str] = None
    if output_path:
        from oil_sentiment.visualizer import build_html_report
        build_html_report(
            merged=merged,
            daily_sentiment=daily_sentiment,
            rolling_corr=rolling_corr,
            lag_df=lag_df,
            correlations=correlations,
            ticker=ticker,
            output_path=output_path,
        )
        report_path = output_path

    _print_summary(correlations, daily_sentiment, ticker)

    return PipelineResult(
        articles=articles,
        sentiment_results=sentiment_results,
        daily_sentiment=daily_sentiment,
        prices=prices,
        merged=merged,
        correlations=correlations,
        rolling_corr=rolling_corr,
        lag_df=lag_df,
        report_path=report_path,
    )


def _print_summary(correlations: dict, daily_sentiment: pd.DataFrame, ticker: str):
    avg = daily_sentiment["composite_score"].mean() if not daily_sentiment.empty else 0
    label = "BULLISH" if avg > 0.05 else "BEARISH" if avg < -0.05 else "NEUTRAL"

    print("\n" + "=" * 55)
    print(f"  Oil Sentiment Analysis Summary — {ticker}")
    print("=" * 55)
    print(f"  Average sentiment score : {avg:+.4f}  ({label})")

    same = correlations.get("same_day", {})
    nxt  = correlations.get("next_day", {})
    if same:
        sig = "***" if same.get("pearson_p", 1) < 0.01 else (
              "**" if same.get("pearson_p", 1) < 0.05 else
              "*"  if same.get("pearson_p", 1) < 0.10 else "ns")
        print(f"  Same-day  Pearson r     : {same.get('pearson_r'):+.4f}  {sig}")
    if nxt:
        sig = "***" if nxt.get("pearson_p", 1) < 0.01 else (
              "**" if nxt.get("pearson_p", 1) < 0.05 else
              "*"  if nxt.get("pearson_p", 1) < 0.10 else "ns")
        print(f"  Next-day  Pearson r     : {nxt.get('pearson_r'):+.4f}  {sig}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Oil price news sentiment analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker", default="WTI", choices=["WTI", "Brent"],
                   help="Oil price benchmark to analyse")
    p.add_argument("--models", nargs="+", default=None,
                   choices=["vader", "textblob", "finbert"],
                   help="Sentiment models to use (default: all three)")
    p.add_argument("--mock", action="store_true",
                   help="Use synthetic offline data (no API keys needed)")
    p.add_argument("--output", default=os.path.join(config.OUTPUT_DIR, config.REPORT_FILENAME),
                   help="Path to write the HTML dashboard")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_pipeline(
        ticker=args.ticker,
        models=args.models,
        use_mock=args.mock,
        output_path=args.output,
    )

    if result.report_path:
        print(f"Dashboard saved → {result.report_path}")


if __name__ == "__main__":
    main()
