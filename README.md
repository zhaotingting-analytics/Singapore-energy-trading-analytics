# Singapore Energy Trading Analytics

Analyse how oil-related news sentiment correlates with crude oil price movements.

## Architecture

```
oil_sentiment/
├── news_fetcher.py        ← RSS feeds + NewsAPI + mock data fallback
├── sentiment_analyzer.py  ← VADER · TextBlob · FinBERT ensemble
├── price_data.py          ← Yahoo Finance (WTI / Brent) + mock prices
├── correlation_analyzer.py← same-day / next-day / rolling / lag correlations
├── visualizer.py          ← interactive Plotly HTML dashboard
└── pipeline.py            ← end-to-end orchestration + CLI
```

## Quick start (offline / no API key)

```bash
pip install -r requirements.txt
python -m oil_sentiment.pipeline --mock --output output/report.html
open output/report.html
```

## With real news data

1. Copy `.env.example` → `.env` and add your [NewsAPI](https://newsapi.org) key.
2. Run:
```bash
python -m oil_sentiment.pipeline --ticker WTI --output output/report.html
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `WTI` | `WTI` or `Brent` |
| `--models` | all three | `vader` `textblob` `finbert` |
| `--mock` | off | Use synthetic offline data |
| `--output` | `output/oil_sentiment_report.html` | Dashboard output path |

## Sentiment models

| Model | Type | Notes |
|-------|------|-------|
| VADER | Rule-based | Fast; handles caps, punctuation, intensifiers |
| TextBlob | Lexicon | Simple polarity baseline |
| FinBERT | Transformer | ProsusAI/finbert; fine-tuned on financial corpora (~400 MB) |

The composite score is a weighted average: FinBERT 50 % · VADER 35 % · TextBlob 15 %.

## Running tests

```bash
pytest tests/ -v
```

Tests use `--mock` mode — no API keys or model downloads required.
