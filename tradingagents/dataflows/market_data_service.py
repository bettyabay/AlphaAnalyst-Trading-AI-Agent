"""
Utility helpers to read OHLCV data from Supabase tables.
All UI features should pull from the ingested market_data tables instead of
external market data APIs.

Simple way to test every feature:

1. Add debug prints (temporarily) inside `fetch_ohlcv` and `get_latest_price`:

   - e.g. `print("fetch_ohlcv", symbol, interval, start_value, end_value)`
   - and `print("get_latest_price", symbol, intervals)`

2. Run `streamlit run app.py` and use each feature in the UI:

   - When you click a button / open a view, you should see these debug lines
     in the terminal. That confirms the feature is going to Supabase, not
     directly to an external API.

3. Optionally break external APIs:

   - Temporarily remove/invalidate `POLYGON_API_KEY` and block internet.
   - Ingestion will fail (as expected), but all read-only features will still
     work as long as the tables already have data, because they never call
     yfinance/Polygon directly—only Supabase via `fetch_ohlcv` /
     `get_latest_price`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from tradingagents.database.config import get_supabase

TABLE_CONFIG: Dict[str, Dict[str, object]] = {
    "1d": {"table": "market_data", "time_field": "date", "is_date": True},
    "day": {"table": "market_data", "time_field": "date", "is_date": True},
    "daily": {"table": "market_data", "time_field": "date", "is_date": True},
    "5min": {"table": "market_data_5min", "time_field": "timestamp", "is_date": False},
    "1min": {"table": "market_data_stocks_1min", "time_field": "timestamp", "is_date": False},
}

DEFAULT_PERIOD_LOOKUP = {
    "1mo": 31,
    "3mo": 93,
    "6mo": 186,
    "9mo": 279,
    "1y": 365,
    "2y": 730,
    "3y": 1095,
    "5y": 1825,
    "10y": 3650,
    "ytd": 365,
    "max": 1825,
}

MAX_SUPABASE_LIMIT = 1000  # Supabase REST API hard limit per request


def period_to_days(period: Optional[str], default: int = 180) -> int:
    """Convert a yfinance-style period string into a number of days."""
    if not period:
        return default
    period = period.strip().lower()
    if period in DEFAULT_PERIOD_LOOKUP:
        return DEFAULT_PERIOD_LOOKUP[period]
    if period.endswith("d"):
        try:
            return int(period[:-1])
        except ValueError:
            return default
    if period.endswith("mo"):
        try:
            months = int(period[:-2])
            return max(1, months * 31)
        except ValueError:
            return default
    if period.endswith("y"):
        try:
            years = int(period[:-1])
            return max(1, years * 365)
        except ValueError:
            return default
    return default


def fetch_ohlcv(
    symbol: str,
    interval: str = "1d",
    lookback_days: int = 365,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = MAX_SUPABASE_LIMIT,
    asset_class: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol from Supabase.

    Args:
        symbol: Stock ticker.
        interval: One of '1d', '5min', '1min'.
        lookback_days: Historical window when start isn't provided.
        start/end: Explicit datetime bounds (UTC).
        limit: Supabase limit (<=1000).
        asset_class: Optional asset class to route to specific tables (Commodities, Indices, Currencies, Stocks).
    """
    symbol = (symbol or "").upper()
    if not symbol:
        return pd.DataFrame()

    config = TABLE_CONFIG.get(interval)
    if not config:
        raise ValueError(f"Unsupported interval '{interval}'.")

    supabase = get_supabase()
    if not supabase:
        return pd.DataFrame()
    
    # Debug logging (can be disabled in production)
    import os
    if os.getenv("DEBUG_DB_QUERIES", "false").lower() == "true":
        print(f"🔍 [DB] fetch_ohlcv({symbol}, interval={interval}, lookback_days={lookback_days}, asset_class={asset_class})")

    table = config["table"]
    
    # Dynamic table routing for 1min data
    if interval == "1min":
        # 1. Prioritize explicit asset_class routing
        if asset_class == "Commodities":
            table = "market_data_commodities_1min"
        elif asset_class == "Indices":
            table = "market_data_indices_1min"
        elif asset_class == "Currencies":
            table = "market_data_currencies_1min"
        elif asset_class == "Stocks":
            table = "market_data_stocks_1min"  # Default stock table
        # 2. Fallback to symbol pattern matching
        elif "*" in symbol:
            table = "market_data_commodities_1min"
        elif symbol.startswith("^"):
            table = "market_data_indices_1min"
        elif "/" in symbol:
            table = "market_data_currencies_1min"
            
    time_field = config["time_field"]
    is_date = bool(config["is_date"])

    # Determine time bounds
    if start is None:
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    if end is None:
        end = datetime.now(timezone.utc)

    start_value = start.date().isoformat() if is_date else start.isoformat()
    end_value = end.date().isoformat() if is_date else end.isoformat()

    fields = f"{time_field},open,high,low,close,volume"

    try:
        query = (
            supabase.table(table)
            .select(fields)
            .eq("symbol", symbol)
            .gte(time_field, start_value)
            .lte(time_field, end_value)
            .order(time_field, desc=True)
            .limit(min(limit, MAX_SUPABASE_LIMIT))
        )
        resp = query.execute()
        data = resp.data if hasattr(resp, "data") else resp
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df[time_field] = pd.to_datetime(df[time_field])
        df = df.sort_values(time_field)

        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df.rename(columns=rename_map, inplace=True)
        for col in rename_map.values():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.set_index(time_field, inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date" if is_date else "Timestamp"
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        
        # Debug logging
        import os
        if os.getenv("DEBUG_DB_QUERIES", "false").lower() == "true":
            print(f"✅ [DB] Retrieved {len(df)} records from {table} for {symbol}")
        
        return df
    except Exception as exc:
        print(f"Error fetching OHLCV for {symbol} ({interval}): {exc}")
        return pd.DataFrame()


def fetch_latest_bar(symbol: str, interval: str = "1d", asset_class: Optional[str] = None) -> Optional[Dict[str, object]]:
    """Return the most recent bar for a symbol."""
    symbol = (symbol or "").upper()
    if not symbol:
        return None

    config = TABLE_CONFIG.get(interval)
    if not config:
        return None

    supabase = get_supabase()
    if not supabase:
        return None

    time_field = config["time_field"]
    table = config["table"]

    # Dynamic table routing for 1min data
    if interval == "1min":
        # 1. Prioritize explicit asset_class routing
        if asset_class == "Commodities":
            table = "market_data_commodities_1min"
        elif asset_class == "Indices":
            table = "market_data_indices_1min"
        elif asset_class == "Currencies":
            table = "market_data_currencies_1min"
        elif asset_class == "Stocks":
            table = "market_data_stocks_1min"
        # 2. Fallback
        elif "*" in symbol:
            table = "market_data_commodities_1min"
        elif symbol.startswith("^"):
            table = "market_data_indices_1min"
        elif "/" in symbol:
            table = "market_data_currencies_1min"

    fields = f"{time_field},open,high,low,close,volume,source"

    try:
        resp = (
            supabase.table(table)
            .select(fields)
            .eq("symbol", symbol)
            .order(time_field, desc=True)
            .limit(1)
            .execute()
        )
        data = resp.data if hasattr(resp, "data") else resp
        if not data:
            return None

        row = data[0]
        row[time_field] = pd.to_datetime(row[time_field])
        return row
    except Exception as exc:
        print(f"Error fetching latest bar for {symbol}: {exc}")
        return None


def get_latest_price(
    symbol: str,
    preferred_intervals: Optional[Sequence[str]] = None,
    asset_class: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    """
    Return the latest available price across preferred intervals.

    The returned dict contains:
        - price: float
        - interval: interval the price came from
        - timestamp: pandas.Timestamp
        - source: database table/source label
    """
    intervals = preferred_intervals or ("1min", "5min", "1d")
    for interval in intervals:
        bar = fetch_latest_bar(symbol, interval=interval, asset_class=asset_class)
        if not bar:
            continue
        close_value = bar.get("close")
        if close_value is None:
            continue
        try:
            price = float(close_value)
        except (TypeError, ValueError):
            continue
        return {
            "price": price,
            "interval": interval,
            "timestamp": bar.get("timestamp")
            or bar.get("date")
            or bar.get(TABLE_CONFIG[interval]["time_field"]),
            "source": bar.get("source", TABLE_CONFIG[interval]["table"]),
        }
    return None


