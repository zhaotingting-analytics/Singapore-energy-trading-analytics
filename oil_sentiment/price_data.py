"""
Fetch historical oil price data from Yahoo Finance.

Tickers used:
  CL=F  — WTI Crude Oil Futures (front month)
  BZ=F  — Brent Crude Oil Futures (front month)

Returned as a tidy DataFrame with columns:
  date, ticker, close, pct_change, direction
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import numpy as np

import config

log = logging.getLogger(__name__)


def fetch_oil_prices(
    lookback_days: Optional[int] = None,
    use_mock: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV data for configured oil tickers.

    Args:
        lookback_days: how far back to fetch (defaults to config.NEWS_LOOKBACK_DAYS + 10)
        use_mock: return synthetic data without network calls

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume,
        pct_change, direction
    """
    if lookback_days is None:
        lookback_days = config.NEWS_LOOKBACK_DAYS + 10

    if use_mock:
        return _mock_prices(lookback_days)

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)

    dfs = []
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed; falling back to mock prices.")
        return _mock_prices(lookback_days)

    for name, ticker in config.OIL_TICKERS.items():
        try:
            raw = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                log.warning("No data for %s (%s)", name, ticker)
                continue

            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index).normalize()
            df.index.name = "date"
            df = df.reset_index()
            df["ticker"] = name
            dfs.append(df)
        except Exception as exc:
            log.warning("yfinance download failed for %s: %s", ticker, exc)

    if not dfs:
        log.warning("All price fetches failed; using mock data.")
        return _mock_prices(lookback_days)

    result = pd.concat(dfs, ignore_index=True)
    result = _add_derived_columns(result)
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pct_change"] = df.groupby("ticker")["close"].pct_change() * 100
    df["direction"] = df["pct_change"].apply(
        lambda x: "up" if x > 0 else ("down" if x < 0 else "flat")
    )
    return df


def _mock_prices(lookback_days: int) -> pd.DataFrame:
    """Synthetic oil price walk for offline development."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(
        end=datetime.now().date(),
        periods=lookback_days,
        freq="B",  # business days only
    )
    rows = []
    for name, base in [("WTI", 80.0), ("Brent", 84.0)]:
        prices = [base]
        for _ in range(len(dates) - 1):
            prices.append(prices[-1] * (1 + rng.normal(0, 0.012)))

        for date, price in zip(dates, prices):
            rows.append({
                "date": date,
                "ticker": name,
                "open": price * (1 + rng.normal(0, 0.003)),
                "high": price * (1 + abs(rng.normal(0, 0.006))),
                "low": price * (1 - abs(rng.normal(0, 0.006))),
                "close": price,
                "volume": int(rng.integers(100_000, 500_000)),
            })

    df = pd.DataFrame(rows)
    df = _add_derived_columns(df)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
