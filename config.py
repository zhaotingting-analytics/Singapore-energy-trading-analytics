"""
Central configuration for the oil sentiment analysis pipeline.
Copy .env.example to .env and fill in your API keys.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- News Sources ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_SOURCES = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
]
NEWS_KEYWORDS = [
    "crude oil", "oil price", "OPEC", "petroleum", "Brent crude",
    "WTI oil", "oil market", "energy prices", "oil supply", "oil demand",
    "oil production", "oil refinery", "oil reserves"
]
NEWS_LOOKBACK_DAYS = 30

# --- Oil Price Tickers ---
OIL_TICKERS = {
    "WTI": "CL=F",    # West Texas Intermediate
    "Brent": "BZ=F",  # Brent Crude
}

# --- Sentiment Models ---
SENTIMENT_MODELS = ["vader", "textblob", "finbert"]
FINBERT_MODEL = "ProsusAI/finbert"

# Weights for ensemble score
SENTIMENT_WEIGHTS = {
    "vader": 0.35,
    "textblob": 0.15,
    "finbert": 0.50,
}

# --- Analysis ---
ROLLING_WINDOW_DAYS = 7
CORRELATION_LAG_DAYS = list(range(-5, 6))  # -5 to +5 day lag

# --- Output ---
OUTPUT_DIR = "output"
REPORT_FILENAME = "oil_sentiment_report.html"
