"""
Indicator Confluence & Filtering utilities.

This module provides:
    - generate_technical_features(market_df)
    - snapshot_indicators_for_signals(signals_df, market_df_with_indicators)
    - get_distribution_stats(enriched_signals_df, indicator_col, pnl_col)
    - apply_filters(signals_df, rules)
    - IndicatorCache: Session-based caching for indicator snapshots

All time-based calculations are designed to work with GMT+4 (Asia/Dubai)
timestamps. Callers are responsible for passing in DataFrames where
timestamps are already normalized to GMT+4, which is the standard across
the app.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
import json
from datetime import datetime

import numpy as np
import pandas as pd


GMT4_TZ = "Asia/Dubai"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a DatetimeIndex (tz-aware if possible)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # Try common timestamp column names
    for col in ("timestamp", "Timestamp", "date", "Date"):
        if col in df.columns:
            df = df.set_index(col)
            break

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .ewm(span=span, adjust=False, min_periods=span)
        .mean()
    )


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    close = pd.to_numeric(series, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _stochastic_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    high_roll = pd.to_numeric(high, errors="coerce").rolling(
        window=k_period, min_periods=k_period
    )
    low_roll = pd.to_numeric(low, errors="coerce").rolling(
        window=k_period, min_periods=k_period
    )
    lowest_low = low_roll.min()
    highest_high = high_roll.max()
    k = (pd.to_numeric(close, errors="coerce") - lowest_low) / (
        highest_high - lowest_low
    ) * 100
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def _bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    price = pd.to_numeric(close, errors="coerce")
    ma = price.rolling(window=period, min_periods=period).mean()
    std = price.rolling(window=period, min_periods=period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / ma.replace(0, np.nan)
    return upper, lower, width


def generate_technical_features(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a broad set of technical indicators on OHLCV market data.

    Required columns in market_df (case-sensitive): 'Open', 'High', 'Low', 'Close', 'Volume'.

    Indicators (minimum set):
        - Momentum: RSI(14), Stochastic Oscillator %K / %D
        - Trend: SMA(50), SMA(200), EMA(20), MACD (12, 26, 9)
        - Volatility: Bollinger Bands (upper, lower, width)
        - Interaction: Distance from Price to SMA(200) in percentage

    Look-ahead safety:
        Indicators are shifted by 1 bar so that the value aligned with
        a given candle only uses information up to the PREVIOUS candle's
        close. This guarantees that when a trade is opened exactly at
        the start of a new bar, we never use that bar's close.
    """
    if market_df is None or market_df.empty:
        return market_df

    df = market_df.copy()
    df = _ensure_datetime_index(df).sort_index()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Momentum
    df["rsi_14_raw"] = _rsi(close, period=14)
    stoch_k_raw, stoch_d_raw = _stochastic_oscillator(high, low, close, k_period=14, d_period=3)
    df["stoch_k_raw"] = stoch_k_raw
    df["stoch_d_raw"] = stoch_d_raw

    # Trend
    close_num = pd.to_numeric(close, errors="coerce")
    df["sma_50_raw"] = close_num.rolling(window=50, min_periods=50).mean()
    df["sma_200_raw"] = close_num.rolling(window=200, min_periods=200).mean()
    df["ema_20_raw"] = _ema(close_num, span=20)

    ema_12 = _ema(close_num, span=12)
    ema_26 = _ema(close_num, span=26)
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - macd_signal

    df["macd_raw"] = macd_line
    df["macd_signal_raw"] = macd_signal
    df["macd_hist_raw"] = macd_hist

    # Volatility - Bollinger Bands
    bb_upper_raw, bb_lower_raw, bb_width_raw = _bollinger_bands(close_num, period=20, num_std=2.0)
    df["bb_upper_raw"] = bb_upper_raw
    df["bb_lower_raw"] = bb_lower_raw
    df["bb_width_raw"] = bb_width_raw

    # Interaction: Distance from price to SMA 200 (percentage)
    df["sma_200_distance_pct_raw"] = (
        (close_num - df["sma_200_raw"]) / df["sma_200_raw"].replace(0, np.nan) * 100
    )

    # Look-ahead safety: shift all indicator columns by 1 bar
    indicator_cols = [c for c in df.columns if c.endswith("_raw")]
    for col in indicator_cols:
        safe_col = col.replace("_raw", "")
        df[safe_col] = df[col].shift(1)

    # Keep original OHLCV plus safe indicator columns
    keep_cols = ["Open", "High", "Low", "Close", "Volume"] + [
        c.replace("_raw", "") for c in indicator_cols
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


def snapshot_indicators_for_signals(
    signals_df: pd.DataFrame,
    market_df_with_indicators: pd.DataFrame,
    entry_time_col: str = "entry_datetime",
) -> pd.DataFrame:
    """
    Map indicator values to each signal at the moment the trade was opened.

    This uses pd.merge_asof with direction='backward' so that for a given
    entry time, we take the latest completed candle at or BEFORE that time.
    Combined with the 1-bar indicator shift in generate_technical_features,
    this guarantees no look-ahead to the current candle's close.
    """
    if signals_df is None or signals_df.empty:
        return signals_df
    if market_df_with_indicators is None or market_df_with_indicators.empty:
        return signals_df

    sig = signals_df.copy()
    mkt = market_df_with_indicators.copy()

    # Ensure datetime types and GMT+4 awareness where possible
    if entry_time_col not in sig.columns:
        raise ValueError(f"signals_df is missing required column: {entry_time_col}")

    sig[entry_time_col] = pd.to_datetime(sig[entry_time_col])
    sig = sig.sort_values(entry_time_col)

    mkt = _ensure_datetime_index(mkt).sort_index()

    # merge_asof requires both sides to be sorted and with comparable dtypes
    enriched = pd.merge_asof(
        sig,
        mkt,
        left_on=entry_time_col,
        right_index=True,
        direction="backward",
    )
    return enriched


@dataclass
class DistributionBin:
    bin_label: str
    bin_left: float
    bin_right: float
    count_total: int
    count_wins: int
    count_losses: int

    @property
    def win_rate(self) -> Optional[float]:
        return (self.count_wins / self.count_total * 100) if self.count_total > 0 else None


def get_distribution_stats(
    enriched_signals_df: pd.DataFrame,
    indicator_col: str,
    pnl_col: str = "pips_made",
    bins: Optional[Sequence[float]] = None,
) -> List[DistributionBin]:
    """
    Compute win-rate distribution for a given indicator.

    - Separates trades into wins (pnl_col > 0) and losses (<= 0)
    - Bins indicator values and computes win rate per bin
    - For oscillators on 0-100 scale (e.g. RSI), default bins are
      [0,20,40,60,80,100,120] if not provided.
    """
    df = enriched_signals_df.copy()
    if df.empty or indicator_col not in df.columns:
        return []

    if pnl_col not in df.columns:
        # Fallback to generic profit_loss if pips_made is not present
        if "profit_loss" in df.columns:
            pnl_col = "profit_loss"
        else:
            raise ValueError(f"Could not find PnL column: {pnl_col}")

    df = df[[indicator_col, pnl_col]].dropna()
    if df.empty:
        return []

    values = pd.to_numeric(df[indicator_col], errors="coerce")
    df = df[values.notna()].copy()
    df[indicator_col] = values[values.notna()]
    if df.empty:
        return []

    # Default bins
    if bins is None:
        col_lower = df[indicator_col].min()
        col_upper = df[indicator_col].max()
        # Heuristic: if indicator looks like oscillator (bounded ~0-100), use fixed bins
        if col_lower >= -5 and col_upper <= 105:
            bins = [0, 20, 40, 60, 70, 80, 100, 120]
        else:
            # Price-like: use 8 equal-width bins
            bins = np.linspace(col_lower, col_upper, num=9).tolist()

    df["__bin"] = pd.cut(df[indicator_col], bins=bins, include_lowest=True)

    results: List[DistributionBin] = []
    grouped = df.groupby("__bin")

    for interval, group in grouped:
        if group.empty or not isinstance(interval, pd.Interval):
            continue

        total = len(group)
        wins = int((group[pnl_col] > 0).sum())
        losses = total - wins

        bin_label = f"{interval.left:.2f}–{interval.right:.2f}"
        results.append(
            DistributionBin(
                bin_label=bin_label,
                bin_left=float(interval.left),
                bin_right=float(interval.right),
                count_total=int(total),
                count_wins=wins,
                count_losses=losses,
            )
        )

    return results


def apply_filters(signals_df: pd.DataFrame, rules: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """
    Apply a list of simple indicator-based rules to filter trades.

    Each rule is a dict with:
        - indicator: column name in signals_df
        - operator: one of '<', '>'
        - value: numeric threshold

    Returns:
        Filtered DataFrame with only trades that survive all rules.
    """
    if signals_df is None or signals_df.empty or not rules:
        return signals_df

    filtered = signals_df.copy()

    for rule in rules:
        indicator = rule.get("indicator")
        operator = rule.get("operator")
        value = rule.get("value")

        if indicator not in filtered.columns:
            # Skip silently if indicator missing
            continue

        if value is None:
            continue

        if operator == "<":
            filtered = filtered[pd.to_numeric(filtered[indicator], errors="coerce") < float(value)]
        elif operator == ">":
            filtered = filtered[pd.to_numeric(filtered[indicator], errors="coerce") > float(value)]

    return filtered


class IndicatorCache:
    """
    Session-based cache for indicator snapshots to avoid re-fetching OHLCV
    for the same symbol/time range.
    
    Usage:
        cache = IndicatorCache()
        cache_key = cache.get_cache_key(symbol, start, end)
        if cache_key in cache:
            market_with_indicators = cache[cache_key]
        else:
            market_df = fetch_ohlcv(...)
            market_with_indicators = generate_technical_features(market_df)
            cache[cache_key] = market_with_indicators
    """
    
    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_cache_key(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
        """Generate a cache key from symbol and time range."""
        # Normalize timestamps to strings for consistent keys
        start_str = pd.to_datetime(start).strftime("%Y%m%d_%H%M%S")
        end_str = pd.to_datetime(end).strftime("%Y%m%d_%H%M%S")
        return f"{symbol}_{start_str}_{end_str}"
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached DataFrame."""
        return self._cache.get(key)
    
    def set(self, key: str, df: pd.DataFrame) -> None:
        """Store DataFrame in cache."""
        self._cache[key] = df.copy()
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache


def export_enriched_to_json(
    enriched_df: pd.DataFrame,
    filename: Optional[str] = None,
    include_indicators: bool = True,
) -> str:
    """
    Export enriched trades DataFrame to JSON format for offline analysis.
    
    Args:
        enriched_df: DataFrame with trades and indicator columns
        filename: Optional filename (if None, generates timestamp-based name)
        include_indicators: Whether to include indicator columns (default: True)
    
    Returns:
        JSON string representation of the data
    """
    if enriched_df is None or enriched_df.empty:
        return json.dumps({"error": "No data to export"}, indent=2)
    
    df = enriched_df.copy()
    
    # Convert datetime columns to ISO format strings
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    for col in datetime_cols:
        df[col] = df[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
    
    # Convert index to column if it's a datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if df.index.name:
            df.rename(columns={df.index.name: "index"}, inplace=True)
    
    # Select columns to export
    if not include_indicators:
        # Exclude indicator columns
        indicator_cols = [
            "rsi_14", "stoch_k", "stoch_d", "sma_50", "sma_200", "ema_20",
            "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", 
            "bb_width", "sma_200_distance_pct"
        ]
        cols_to_keep = [c for c in df.columns if c not in indicator_cols]
        df = df[cols_to_keep]
    
    # Replace NaN/None with null for JSON compatibility
    df = df.replace({np.nan: None, pd.NA: None})
    
    # Convert to records format (list of dicts)
    records = df.to_dict(orient='records')
    
    # Build export metadata
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "total_records": len(records),
        "columns": list(df.columns),
        "include_indicators": include_indicators,
        "data": records
    }
    
    return json.dumps(export_data, indent=2, default=str)

