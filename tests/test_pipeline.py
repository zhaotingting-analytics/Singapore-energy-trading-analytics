"""Integration tests for the full pipeline using mock / offline data."""

import pytest
import pandas as pd
from oil_sentiment.pipeline import run_pipeline


@pytest.fixture(scope="module")
def pipeline_result():
    """Run pipeline once in mock mode and reuse across tests."""
    return run_pipeline(
        ticker="WTI",
        models=["vader", "textblob"],  # skip FinBERT to keep CI fast
        use_mock=True,
        output_path=None,
    )


class TestPipelineOutputShapes:
    def test_articles_nonempty(self, pipeline_result):
        assert len(pipeline_result.articles) > 0

    def test_sentiment_results_match_articles(self, pipeline_result):
        assert len(pipeline_result.sentiment_results) == len(pipeline_result.articles)

    def test_daily_sentiment_is_dataframe(self, pipeline_result):
        assert isinstance(pipeline_result.daily_sentiment, pd.DataFrame)
        assert "date" in pipeline_result.daily_sentiment.columns
        assert "composite_score" in pipeline_result.daily_sentiment.columns

    def test_prices_has_wti(self, pipeline_result):
        tickers = pipeline_result.prices["ticker"].unique().tolist()
        assert "WTI" in tickers

    def test_merged_has_required_columns(self, pipeline_result):
        required = {"date", "close", "pct_change", "composite_score"}
        assert required.issubset(pipeline_result.merged.columns)

    def test_merged_nonempty(self, pipeline_result):
        assert len(pipeline_result.merged) > 0

    def test_correlations_dict(self, pipeline_result):
        assert isinstance(pipeline_result.correlations, dict)
        for key in ("same_day", "next_day"):
            if key in pipeline_result.correlations:
                assert "pearson_r" in pipeline_result.correlations[key]

    def test_rolling_corr_has_date(self, pipeline_result):
        assert "date" in pipeline_result.rolling_corr.columns
        assert "rolling_corr" in pipeline_result.rolling_corr.columns

    def test_lag_df_has_lag_column(self, pipeline_result):
        assert "lag_days" in pipeline_result.lag_df.columns
        assert "pearson_r" in pipeline_result.lag_df.columns

    def test_composite_scores_in_bounds(self, pipeline_result):
        scores = pipeline_result.daily_sentiment["composite_score"]
        assert scores.between(-1.0, 1.0).all(), "All scores must be in [-1, +1]"


class TestPipelineLabels:
    def test_labels_are_valid(self, pipeline_result):
        valid = {"positive", "negative", "neutral"}
        for r in pipeline_result.sentiment_results:
            assert r.label in valid

    def test_label_consistent_with_score(self, pipeline_result):
        for r in pipeline_result.sentiment_results:
            if r.composite_score >= 0.05:
                assert r.label == "positive"
            elif r.composite_score <= -0.05:
                assert r.label == "negative"
            else:
                assert r.label == "neutral"
