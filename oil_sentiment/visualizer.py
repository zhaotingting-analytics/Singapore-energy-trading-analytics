"""
Interactive HTML dashboard for the oil sentiment analysis.

Design decisions (dataviz skill):
  - Form heuristic: change-over-time → line chart; distribution → bar chart
  - Color assignment:
      Positive   hsl(152, 60%, 35%)  — green
      Negative   hsl(5,   70%, 48%)  — red
      Neutral    hsl(43,  75%, 45%)  — amber
      WTI price  hsl(220, 75%, 50%)  — blue (sequential single-hue)
      Brent price hsl(195, 70%, 42%) — teal
  - One y-axis per chart (no dual-axis)
  - Hover tooltip on every chart
  - Dark-mode: explicit token redefinition under @media prefers-color-scheme: dark
"""

import logging
import os
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)

# Palette (validated against dataviz CVD requirement)
_PALETTE = {
    "positive":   "#1f7a4f",   # green
    "negative":   "#c93520",   # red
    "neutral":    "#c47d10",   # amber
    "price_wti":  "#2563c7",   # blue
    "price_brent":"#0f7ea3",   # teal
    "sentiment":  "#7c3aed",   # violet  (composite line)
    "grid":       "rgba(128,128,128,0.15)",
}

_PLOTLY_TEMPLATE = "plotly_white"


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=24, t=48, b=48),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        xaxis=dict(showgrid=True, gridcolor=_PALETTE["grid"]),
        yaxis=dict(showgrid=True, gridcolor=_PALETTE["grid"]),
    )
    return fig


def chart_price_and_sentiment(merged: pd.DataFrame, ticker: str = "WTI") -> str:
    """Two separate charts stacked: oil price + daily sentiment score."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=(f"{ticker} Crude Oil Price (USD)", "Daily News Sentiment Score"),
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
    )

    color_price = _PALETTE["price_wti"] if ticker == "WTI" else _PALETTE["price_brent"]

    # Price line
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["close"],
            mode="lines",
            name=f"{ticker} Close",
            line=dict(color=color_price, width=2),
            hovertemplate="<b>%{x|%b %d}</b><br>Price: $%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Sentiment bars — coloured by label if available, else by sign
    sentiment_colors = [
        _PALETTE["positive"] if s >= 0.05 else
        _PALETTE["negative"] if s <= -0.05 else
        _PALETTE["neutral"]
        for s in merged["composite_score"]
    ]

    fig.add_trace(
        go.Bar(
            x=merged["date"],
            y=merged["composite_score"],
            name="Composite Sentiment",
            marker_color=sentiment_colors,
            hovertemplate="<b>%{x|%b %d}</b><br>Sentiment: %{y:.3f}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Score [-1, +1]", row=2, col=1, range=[-1, 1])
    fig.update_layout(
        height=540,
        showlegend=True,
        title_text=f"Oil Price vs News Sentiment — {ticker}",
        title_font_size=15,
    )
    _apply_theme(fig)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="chart-price-sentiment")


def chart_rolling_correlation(rolling_corr: pd.DataFrame) -> str:
    """Rolling Pearson correlation between sentiment and price change."""
    fig = go.Figure()

    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dash"))
    fig.add_hline(y=0.3,  line=dict(color=_PALETTE["positive"], width=1, dash="dot"),
                  annotation_text="moderate +0.3")
    fig.add_hline(y=-0.3, line=dict(color=_PALETTE["negative"], width=1, dash="dot"),
                  annotation_text="moderate -0.3")

    fig.add_trace(go.Scatter(
        x=rolling_corr["date"],
        y=rolling_corr["rolling_corr"],
        mode="lines+markers",
        name="Rolling Correlation",
        line=dict(color=_PALETTE["sentiment"], width=2),
        marker=dict(size=5),
        hovertemplate="<b>%{x|%b %d}</b><br>Correlation: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        height=320,
        title_text="7-Day Rolling Correlation: Sentiment → Price Change",
        title_font_size=15,
        yaxis=dict(title="Pearson r", range=[-1, 1]),
        xaxis=dict(title="Date"),
    )
    _apply_theme(fig)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="chart-rolling-corr")


def chart_lag_correlations(lag_df: pd.DataFrame) -> str:
    """Bar chart of Pearson r by sentiment lead/lag."""
    colors = [
        _PALETTE["positive"] if r > 0 else _PALETTE["negative"]
        for r in lag_df["pearson_r"]
    ]
    fig = go.Figure(go.Bar(
        x=lag_df["lag_days"],
        y=lag_df["pearson_r"],
        marker_color=colors,
        hovertemplate="Lag %{x}d: r=%{y:.3f}<extra></extra>",
        name="Pearson r",
    ))

    fig.update_layout(
        height=300,
        title_text="Lead/Lag Correlation (positive lag = sentiment leads price)",
        title_font_size=15,
        xaxis=dict(title="Lag (days)", dtick=1),
        yaxis=dict(title="Pearson r", range=[-1, 1]),
    )
    _apply_theme(fig)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="chart-lag")


def chart_sentiment_distribution(daily_sentiment: pd.DataFrame) -> str:
    """Stacked bar: daily count of positive / neutral / negative articles."""
    fig = go.Figure()

    for label, col, color in [
        ("Positive", "positive_count", _PALETTE["positive"]),
        ("Neutral",  "neutral_count",  _PALETTE["neutral"]),
        ("Negative", "negative_count", _PALETTE["negative"]),
    ]:
        if col not in daily_sentiment.columns:
            continue
        fig.add_trace(go.Bar(
            x=daily_sentiment["date"],
            y=daily_sentiment[col],
            name=label,
            marker_color=color,
            hovertemplate=f"<b>%{{x|%b %d}}</b><br>{label}: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        height=280,
        barmode="stack",
        title_text="Daily Article Sentiment Breakdown",
        title_font_size=15,
        yaxis=dict(title="Article count"),
        xaxis=dict(title="Date"),
    )
    _apply_theme(fig)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="chart-distribution")


def build_html_report(
    merged: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    rolling_corr: pd.DataFrame,
    lag_df: pd.DataFrame,
    correlations: dict,
    ticker: str = "WTI",
    output_path: Optional[str] = None,
) -> str:
    """Assemble a standalone HTML dashboard and optionally write it to disk."""

    c_price_sentiment = chart_price_and_sentiment(merged, ticker)
    c_rolling = chart_rolling_correlation(rolling_corr)
    c_lag = chart_lag_correlations(lag_df)
    c_dist = chart_sentiment_distribution(daily_sentiment)

    # Summary stats card
    corr_same = correlations.get("same_day", {})
    corr_next = correlations.get("next_day", {})
    avg_sentiment = daily_sentiment["composite_score"].mean() if not daily_sentiment.empty else 0.0
    sentiment_label = "Bullish" if avg_sentiment > 0.05 else "Bearish" if avg_sentiment < -0.05 else "Neutral"

    def stat_card(title, value, sub=""):
        return f"""
        <div class="stat-card">
          <div class="stat-title">{title}</div>
          <div class="stat-value">{value}</div>
          {f'<div class="stat-sub">{sub}</div>' if sub else ''}
        </div>"""

    stats_html = "".join([
        stat_card("Avg Sentiment", f"{avg_sentiment:+.3f}", sentiment_label),
        stat_card("Same-day Pearson r",
                  f"{corr_same.get('pearson_r', 'N/A')}",
                  f"p={corr_same.get('pearson_p', '')}"),
        stat_card("Next-day Pearson r",
                  f"{corr_next.get('pearson_r', 'N/A')}",
                  f"p={corr_next.get('pearson_p', '')}"),
        stat_card("Articles Analysed",
                  str(int(daily_sentiment["article_count"].sum()))
                  if "article_count" in daily_sentiment.columns else "—"),
    ])

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oil Price Sentiment Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg:        #f8f9fa;
      --surface:   #ffffff;
      --border:    #e2e4e9;
      --text:      #1a1d23;
      --muted:     #6b7280;
      --positive:  {_PALETTE["positive"]};
      --negative:  {_PALETTE["negative"]};
      --neutral:   {_PALETTE["neutral"]};
    }}
    @media (prefers-color-scheme: dark) {{
      :root:not([data-theme="light"]) {{
        --bg:      #0f1117;
        --surface: #1a1d23;
        --border:  #2d313a;
        --text:    #e8eaf0;
        --muted:   #9ca3af;
      }}
    }}
    [data-theme="dark"] {{
      --bg:      #0f1117;
      --surface: #1a1d23;
      --border:  #2d313a;
      --text:    #e8eaf0;
      --muted:   #9ca3af;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: Inter, system-ui, -apple-system, sans-serif;
      font-size: 14px;
      padding: 24px 32px;
    }}
    h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }}
    .subtitle {{ color: var(--muted); margin-bottom: 24px; }}
    .stats-row {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 20px;
      min-width: 160px;
      flex: 1;
    }}
    .stat-title {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
    .stat-value {{ font-size: 1.6rem; font-weight: 700; margin: 4px 0 2px; }}
    .stat-sub   {{ font-size: 0.75rem; color: var(--muted); }}
    .chart-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 16px;
      overflow: hidden;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    @media (max-width: 700px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    .theme-toggle {{
      position: fixed; top: 16px; right: 24px;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 12px;
      color: var(--text);
    }}
  </style>
</head>
<body>
  <button class="theme-toggle" onclick="toggleTheme()">Toggle dark mode</button>
  <h1>Oil Price Sentiment Dashboard</h1>
  <p class="subtitle">Singapore Energy Trading Analytics — {ticker} Crude Oil vs News Sentiment</p>

  <div class="stats-row">{stats_html}</div>

  <div class="chart-card">{c_price_sentiment}</div>
  <div class="grid-2">
    <div class="chart-card">{c_rolling}</div>
    <div class="chart-card">{c_lag}</div>
  </div>
  <div class="chart-card">{c_dist}</div>

  <script>
    function toggleTheme() {{
      const root = document.documentElement;
      root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    }}
  </script>
</body>
</html>"""

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        log.info("Report written to %s", output_path)

    return html
