"""
Feature engineering helper that pulls stored Supabase data, computes
multi-timeframe technical features, and optionally builds an LLM prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from groq import Groq

from tradingagents.dataflows.market_data_service import fetch_ohlcv
from tradingagents.database.config import get_supabase


UTC = timezone.utc


def _calc_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= period:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else None


def _calc_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df.empty or len(df) < period + 1:
        return None
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close.shift()
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return float(atr.iloc[-1]) if not atr.empty else None


def _pct_change(series: pd.Series, lookback: int) -> Optional[float]:
    series = pd.to_numeric(series, errors="coerce")
    if len(series) <= lookback:
        return None
    earlier = series.iloc[-lookback - 1]
    latest = series.iloc[-1]
    if earlier == 0 or pd.isna(earlier) or pd.isna(latest):
        return None
    return float(((latest - earlier) / earlier) * 100)


def _volatility(series: pd.Series, window: int = 20) -> Optional[float]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < window:
        return None
    returns = series.pct_change().dropna().tail(window)
    if returns.empty:
        return None
    return float(returns.std() * np.sqrt(window) * 100)


def _volume_ratio(volume: pd.Series, recent: int = 20, prior: int = 20) -> Optional[float]:
    volume = pd.to_numeric(volume, errors="coerce").dropna()
    if len(volume) < recent + prior:
        return None
    recent_avg = volume.tail(recent).mean()
    prior_avg = volume.tail(recent + prior).head(prior).mean()
    if prior_avg == 0 or pd.isna(prior_avg):
        return None
    return float(recent_avg / prior_avg)


def _format_ohlcv_table(df: pd.DataFrame, label: str = "Timestamp") -> str:
    if df.empty:
        return "No data"

    table = df.copy().reset_index()
    if table.empty:
        return "No data"

    idx_col = table.columns[0]
    table[idx_col] = pd.to_datetime(table[idx_col]).dt.strftime("%Y-%m-%d %H:%M")
    if idx_col != label:
        table.rename(columns={idx_col: label}, inplace=True)

    cols = [label, "Open", "High", "Low", "Close", "Volume"]
    for col in cols:
        if col not in table.columns:
            table[col] = np.nan
    try:
        return table[cols].to_markdown(index=False)
    except Exception:
        return table[cols].to_string(index=False)


def _format_window(index: pd.Index) -> str:
    if index.empty:
        return "N/A"
    start_ts = pd.to_datetime(index[0])
    end_ts = pd.to_datetime(index[-1])
    same_day = start_ts.date() == end_ts.date()
    if same_day:
        date_label = start_ts.strftime("%Y-%m-%d")
        return f"{start_ts.strftime('%H:%M')}-{end_ts.strftime('%H:%M')} ({date_label})"
    start = start_ts.strftime("%Y-%m-%d %H:%M")
    end = end_ts.strftime("%Y-%m-%d %H:%M")
    return f"{start} → {end}"


def _has_expected_frequency(index: pd.Index, minutes: int) -> bool:
    if len(index) < 2:
        return True
    expected = pd.Timedelta(minutes=minutes)
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return True
    return bool((diffs == expected).all())


@dataclass
class FeatureResult:
    symbol: str
    feature_blocks: Dict[str, Dict[str, object]]
    samples: Dict[str, str]
    context: str
    prompt: str
    llm_response: Optional[str]
    provider: str


class FeatureLab:
    """Utility to compute stored features and optionally run an LLM prompt."""

    def __init__(self):
        import os

        self.groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or ""
        self.client = None
        if self.groq_key.startswith("gsk_"):
            try:
                self.client = Groq(api_key=self.groq_key)
            except Exception:
                self.client = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, symbol: str, user_prompt: str) -> FeatureResult:
        symbol = (symbol or "").upper()
        if not symbol:
            raise ValueError("Symbol is required")

        blocks = self._build_feature_blocks(symbol)
        samples = self._build_samples(symbol)
        context = self._build_context(symbol, blocks, samples)
        prompt = self._build_prompt(symbol, context, user_prompt)
        response = self._call_llm(prompt) if self.client else None

        provider = "groq" if self.client else "prompt-only"
        return FeatureResult(
            symbol=symbol,
            feature_blocks=blocks,
            samples=samples,
            context=context,
            prompt=prompt,
            llm_response=response,
            provider=provider,
        )

    def _check_data_availability(self, symbol: str, interval: str) -> Dict[str, object]:
        """Check what data exists in the database for diagnostic purposes."""
        supabase = get_supabase()
        if not supabase:
            return {"error": "Supabase not configured"}
        
        table_map = {"1min": "market_data_1min", "5min": "market_data_5min"}
        table = table_map.get(interval)
        if not table:
            return {"error": f"Unknown interval: {interval}"}
        
        try:
            # Get total count
            count_resp = supabase.table(table).select("timestamp", count="exact").eq("symbol", symbol).execute()
            total_count = count_resp.count if hasattr(count_resp, 'count') else None
            
            # Get earliest and latest timestamps
            earliest_resp = supabase.table(table).select("timestamp").eq("symbol", symbol).order("timestamp", desc=False).limit(1).execute()
            latest_resp = supabase.table(table).select("timestamp").eq("symbol", symbol).order("timestamp", desc=True).limit(1).execute()
            
            earliest = earliest_resp.data[0]["timestamp"] if earliest_resp.data else None
            latest = latest_resp.data[0]["timestamp"] if latest_resp.data else None
            
            return {
                "symbol": symbol,
                "interval": interval,
                "table": table,
                "total_count": total_count,
                "earliest": earliest,
                "latest": latest,
            }
        except Exception as e:
            return {"error": str(e)}

    def run_quantum_screen(self, symbol: str, command_ts: Optional[str] = None) -> Dict[str, object]:
        symbol = (symbol or "").upper()
        if not symbol:
            raise ValueError("Symbol is required")

        m1 = self._fetch_df(symbol, "1min", lookback_days=3)
        if m1.empty:
            # Check what data actually exists
            m1_diag = self._check_data_availability(symbol, "1min")
            m1_extended = self._fetch_df(symbol, "1min", lookback_days=30)
            
            if m1_extended.empty:
                diag_msg = ""
                if "error" not in m1_diag:
                    diag_msg = (
                        f" Diagnostic: Found {m1_diag.get('total_count', 0)} total records in database. "
                        f"Earliest: {m1_diag.get('earliest', 'N/A')}, Latest: {m1_diag.get('latest', 'N/A')}."
                    )
                raise ValueError(
                    f"No 1-minute data available for {symbol} in the last 30 days.{diag_msg} "
                    "Run ingestion first. Check that data exists in market_data_1min table."
                )
            else:
                diag_msg = ""
                if "error" not in m1_diag:
                    diag_msg = f" Latest data in DB: {m1_diag.get('latest', 'N/A')}."
                raise ValueError(
                    f"No 1-minute data available for {symbol} in the last 3 days. "
                    f"Found {len(m1_extended)} records in the last 30 days, but need recent data.{diag_msg} "
                    "The data may be too old. Consider ingesting more recent data."
                )

        m5 = self._fetch_df(symbol, "5min", lookback_days=7)
        if m5.empty:
            # Check what data actually exists
            m5_diag = self._check_data_availability(symbol, "5min")
            m5_extended = self._fetch_df(symbol, "5min", lookback_days=30)
            
            if m5_extended.empty:
                diag_msg = ""
                if "error" not in m5_diag:
                    diag_msg = (
                        f" Diagnostic: Found {m5_diag.get('total_count', 0)} total records in database. "
                        f"Earliest: {m5_diag.get('earliest', 'N/A')}, Latest: {m5_diag.get('latest', 'N/A')}."
                    )
                raise ValueError(
                    f"No 5-minute data available for {symbol} in the last 30 days.{diag_msg} "
                    "Run ingestion first. Check that data exists in market_data_5min table."
                )
            else:
                diag_msg = ""
                if "error" not in m5_diag:
                    diag_msg = f" Latest data in DB: {m5_diag.get('latest', 'N/A')}."
                raise ValueError(
                    f"No 5-minute data available for {symbol} in the last 7 days. "
                    f"Found {len(m5_extended)} records in the last 30 days, but need recent data.{diag_msg} "
                    "The data may be too old. Consider ingesting more recent data."
                )

        if len(m1) < 20:
            raise ValueError("Need at least 20 consecutive 1-minute bars (start ingesting from Aug 1 to current time).")
        if len(m5) < 6:
            raise ValueError("Need at least 6 consecutive 5-minute bars (ingest data from Oct 1, 2023 onward).")

        # Fetch daily data for trend and context
        daily = self._fetch_df(symbol, "1d", lookback_days=60)
        if daily.empty:
            # Daily data is optional but recommended - warn but continue
            print(f"Warning: No daily data available for {symbol}. Daily context will be missing.")

        timestamp_label = command_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics = self._compute_quantum_metrics(m1, m5, daily)
        extraction = self._build_quantum_extraction(symbol, m1, m5, daily, metrics)
        prompt = self._build_quantum_prompt(symbol, metrics, extraction, timestamp_label)
        summary = (
            f"{symbol} | Vol: {metrics['volume_score']:.2f} | VWAP: {metrics['vwap_score']:.2f} | "
            f"Mom: {metrics['momentum_score']:.2f} | Cat: {metrics['catalyst_score']:.2f} | "
            f"Composite: {metrics['composite_score']:.2f} | {metrics['verdict']}"
        )

        return {
            "metrics": metrics,
            "prompt": prompt,
            "summary": summary,
            "timestamp": timestamp_label,
            "extraction": extraction,
        }

    # ------------------------------------------------------------------ #
    # Feature calculations
    # ------------------------------------------------------------------ #
    def _fetch_df(
        self,
        symbol: str,
        interval: str,
        lookback_days: int,
    ) -> pd.DataFrame:
        df = fetch_ohlcv(symbol, interval=interval, lookback_days=lookback_days)
        return df.copy()

    def _build_feature_blocks(self, symbol: str) -> Dict[str, Dict[str, object]]:
        daily = self._fetch_df(symbol, "1d", lookback_days=500)
        m5 = self._fetch_df(symbol, "5min", lookback_days=14)
        m1 = self._fetch_df(symbol, "1min", lookback_days=3)

        blocks = {
            "daily": self._summarize_daily(daily),
            "m5": self._summarize_intraday(m5, label="5-minute"),
            "m1": self._summarize_intraday(m1, label="1-minute"),
        }
        return blocks

    def _summarize_daily(self, df: pd.DataFrame) -> Dict[str, object]:
        if df.empty:
            return {"status": "missing"}

        close = df["Close"]
        block = {
            "records": int(len(df)),
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
            "last_close": float(close.iloc[-1]),
            "return_5d_pct": _pct_change(close, 5),
            "return_20d_pct": _pct_change(close, 20),
            "return_60d_pct": _pct_change(close, 60),
            "sma20": float(close.tail(20).mean()) if len(close) >= 20 else None,
            "sma50": float(close.tail(50).mean()) if len(close) >= 50 else None,
            "sma200": float(close.tail(200).mean()) if len(close) >= 200 else None,
            "atr14": _calc_atr(df, 14),
            "volatility20_pct": _volatility(close, 20),
            "volume_ratio": _volume_ratio(df["Volume"]),
        }
        return block

    def _summarize_intraday(self, df: pd.DataFrame, label: str) -> Dict[str, object]:
        if df.empty:
            return {"status": "missing", "label": label}

        close = df["Close"]
        block = {
            "label": label,
            "records": int(len(df)),
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
            "last_close": float(close.iloc[-1]),
            "rsi14": _calc_rsi(close),
            "return_pct": _pct_change(close, min(50, len(close) - 2)) if len(close) > 2 else None,
            "volume_ratio": _volume_ratio(df["Volume"], recent=50, prior=50),
        }
        return block

    def _build_samples(self, symbol: str) -> Dict[str, str]:
        def format_sample(df: pd.DataFrame, rows: int = 5, label: str = "Date") -> str:
            if df.empty:
                return "No data"
            tail = df.tail(rows).reset_index()
            tail[label] = tail[label].dt.strftime("%Y-%m-%d %H:%M")
            cols = [label, "Open", "High", "Low", "Close", "Volume"]
            try:
                return tail[cols].to_markdown(index=False)
            except Exception:
                return tail[cols].to_string(index=False)

        daily = self._fetch_df(symbol, "1d", lookback_days=30)
        m5 = self._fetch_df(symbol, "5min", lookback_days=3)
        m1 = self._fetch_df(symbol, "1min", lookback_days=1)

        samples = {
            "daily": format_sample(daily, label="Date"),
            "m5": format_sample(m5, label="Timestamp"),
            "m1": format_sample(m1, label="Timestamp"),
        }
        return samples

    # ------------------------------------------------------------------ #
    # Prompt building
    # ------------------------------------------------------------------ #
    def _build_context(
        self,
        symbol: str,
        blocks: Dict[str, Dict[str, object]],
        samples: Dict[str, str],
    ) -> str:
        daily = blocks.get("daily", {})
        m5 = blocks.get("m5", {})
        m1 = blocks.get("m1", {})

        def fmt(value: Optional[float], suffix: str = "") -> str:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "N/A"
            return f"{value:.2f}{suffix}"

        lines = [
            f"FEATURE SNAPSHOT FOR {symbol}",
            "",
            "DAILY (D1):",
            f"- Coverage: {daily.get('start', 'N/A')} → {daily.get('end', 'N/A')} ({daily.get('records', 0)} bars)",
            f"- Last Close: {fmt(daily.get('last_close'), '')}",
            f"- Δ5d: {fmt(daily.get('return_5d_pct'), '%')} | Δ20d: {fmt(daily.get('return_20d_pct'), '%')} | Δ60d: {fmt(daily.get('return_60d_pct'), '%')}",
            f"- SMA20/SMA50/SMA200: {fmt(daily.get('sma20'))}, {fmt(daily.get('sma50'))}, {fmt(daily.get('sma200'))}",
            f"- ATR14: {fmt(daily.get('atr14'))} | Volatility20: {fmt(daily.get('volatility20_pct'), '%')}",
            f"- Volume Ratio (20 vs prior 20): {fmt(daily.get('volume_ratio'))}",
            "",
            "TACTICAL (M5):",
            f"- Coverage: {m5.get('start', 'N/A')} → {m5.get('end', 'N/A')} ({m5.get('records', 0)} bars)",
            f"- Last Close: {fmt(m5.get('last_close'))} | RSI14: {fmt(m5.get('rsi14'))}",
            f"- Volume Ratio: {fmt(m5.get('volume_ratio'))}",
            "",
            "EXECUTION (M1):",
            f"- Coverage: {m1.get('start', 'N/A')} → {m1.get('end', 'N/A')} ({m1.get('records', 0)} bars)",
            f"- Last Close: {fmt(m1.get('last_close'))} | RSI14: {fmt(m1.get('rsi14'))}",
            "",
            "RECENT D1:",
            samples.get("daily", "N/A"),
            "",
            "RECENT M5:",
            samples.get("m5", "N/A"),
            "",
            "RECENT M1:",
            samples.get("m1", "N/A"),
        ]
        return "\n".join(lines)

    def _build_prompt(self, symbol: str, context: str, user_prompt: str) -> str:
        template = f"""
You are an equities analyst. Use ONLY the stored features provided below.
Respond in markdown with sections: **Observations**, **Signals**, **Risks**, **Actions**.

{context}

USER INSTRUCTIONS:
{user_prompt.strip()}
"""
        return template.strip()

    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.client:
            return None
        try:
            chat = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a disciplined trading assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return chat.choices[0].message.content if chat.choices else None
        except Exception as exc:
            return f"LLM call failed: {exc}"

    # ------------------------------------------------------------------ #
    # QUANTUMTRADER calculations
    # ------------------------------------------------------------------ #
    def _compute_quantum_metrics(self, m1: pd.DataFrame, m5: pd.DataFrame, daily: pd.DataFrame = None) -> Dict[str, float]:
        close = pd.to_numeric(m1["Close"], errors="coerce")
        volume = pd.to_numeric(m1["Volume"], errors="coerce")

        current_volume = float(volume.iloc[-1])
        volume_ma20 = float(volume.tail(20).mean()) if len(volume) >= 20 else None
        volume_ratio = (
            float(current_volume / volume_ma20) if volume_ma20 and volume_ma20 != 0 else None
        )
        volume_score = self._score_volume(volume_ratio)

        vwap_value, vwap_distance = self._compute_vwap(close, m1, volume)
        vwap_score = self._score_vwap(vwap_distance)

        momentum_inputs = self._compute_momentum(close)
        price_5min_ago = momentum_inputs[0]
        roc_5min = momentum_inputs[1]
        momentum_score = self._score_momentum(roc_5min)

        catalyst_score = self._score_catalyst(assumed_sector_return=0.006)

        composite_score = round(
            0.3 * volume_score + 0.3 * vwap_score + 0.3 * momentum_score + 0.1 * catalyst_score,
            2,
        )
        verdict = "PASS" if composite_score >= 6 else "FAIL"

        atr_5min = _calc_atr(m5, 14) if not m5.empty else None
        rsi_1min = _calc_rsi(close)
        window120 = close.tail(120)
        high_120 = float(window120.max()) if not window120.empty else None
        low_120 = float(window120.min()) if not window120.empty else None
        ema15_current, ema15_previous = self._compute_ema15_pair(m5)

        # Compute daily metrics if available
        daily_metrics = self._compute_daily_metrics(daily, float(close.iloc[-1])) if daily is not None and not daily.empty else {}

        metrics = {
            "current_volume": round(current_volume, 2),
            "volume_ma20": round(volume_ma20, 2) if volume_ma20 is not None else None,
            "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
            "volume_score": round(volume_score, 2),
            "current_price": round(float(close.iloc[-1]), 4),
            "vwap": round(vwap_value, 4) if vwap_value is not None else None,
            "vwap_distance_pct": round(vwap_distance, 2) if vwap_distance is not None else None,
            "vwap_score": round(vwap_score, 2),
            "price_5min_ago": round(price_5min_ago, 4) if price_5min_ago is not None else None,
            "roc_5min_pct": round(roc_5min, 2) if roc_5min is not None else None,
            "momentum_score": round(momentum_score, 2),
            "catalyst_score": round(catalyst_score, 2),
            "composite_score": composite_score,
            "verdict": verdict,
            "atr_5min": round(atr_5min, 4) if atr_5min is not None else None,
            "rsi_1min": round(rsi_1min, 2) if rsi_1min is not None else None,
            "high_120": round(high_120, 4) if high_120 is not None else None,
            "low_120": round(low_120, 4) if low_120 is not None else None,
            "ema15_current": round(ema15_current, 4) if ema15_current is not None else None,
            "ema15_previous": round(ema15_previous, 4) if ema15_previous is not None else None,
            **daily_metrics,  # Merge daily metrics into main metrics dict
        }
        return metrics

    def _compute_daily_metrics(self, daily: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Compute daily timeframe metrics for trend context."""
        if daily.empty:
            return {}
        
        daily_close = pd.to_numeric(daily["Close"], errors="coerce")
        daily_volume = pd.to_numeric(daily["Volume"], errors="coerce")
        
        # Daily SMAs
        sma20 = float(daily_close.tail(20).mean()) if len(daily_close) >= 20 else None
        sma50 = float(daily_close.tail(50).mean()) if len(daily_close) >= 50 else None
        sma200 = float(daily_close.tail(200).mean()) if len(daily_close) >= 200 else None
        
        # Daily returns
        return_5d = _pct_change(daily_close, 5)
        return_20d = _pct_change(daily_close, 20)
        
        # Daily ATR
        atr14_daily = _calc_atr(daily, 14)
        
        # Daily RSI
        rsi14_daily = _calc_rsi(daily_close)
        
        # Daily high/low (last 20 days)
        daily_high_20d = float(daily_close.tail(20).max()) if len(daily_close) >= 20 else None
        daily_low_20d = float(daily_close.tail(20).min()) if len(daily_close) >= 20 else None
        
        # Daily volume ratio
        volume_ratio_daily = _volume_ratio(daily_volume)
        
        # Trend determination (price vs SMAs)
        trend = "NEUTRAL"
        if sma20 and sma50:
            if current_price > sma20 > sma50:
                trend = "BULLISH"
            elif current_price < sma20 < sma50:
                trend = "BEARISH"
        
        # Distance from daily SMAs
        distance_sma20 = ((current_price - sma20) / sma20 * 100) if sma20 and sma20 != 0 else None
        distance_sma50 = ((current_price - sma50) / sma50 * 100) if sma50 and sma50 != 0 else None
        
        return {
            "daily_sma20": round(sma20, 4) if sma20 is not None else None,
            "daily_sma50": round(sma50, 4) if sma50 is not None else None,
            "daily_sma200": round(sma200, 4) if sma200 is not None else None,
            "daily_return_5d_pct": round(return_5d, 2) if return_5d is not None else None,
            "daily_return_20d_pct": round(return_20d, 2) if return_20d is not None else None,
            "daily_atr14": round(atr14_daily, 4) if atr14_daily is not None else None,
            "daily_rsi14": round(rsi14_daily, 2) if rsi14_daily is not None else None,
            "daily_high_20d": round(daily_high_20d, 4) if daily_high_20d is not None else None,
            "daily_low_20d": round(daily_low_20d, 4) if daily_low_20d is not None else None,
            "daily_volume_ratio": round(volume_ratio_daily, 2) if volume_ratio_daily is not None else None,
            "daily_trend": trend,
            "daily_distance_sma20_pct": round(distance_sma20, 2) if distance_sma20 is not None else None,
            "daily_distance_sma50_pct": round(distance_sma50, 2) if distance_sma50 is not None else None,
        }

    def _build_quantum_extraction(
        self,
        symbol: str,
        m1: pd.DataFrame,
        m5: pd.DataFrame,
        daily: pd.DataFrame,
        metrics: Dict[str, float],
    ) -> Dict[str, object]:
        m1_recent = m1.tail(20)
        m5_recent = m5.tail(6)
        daily_recent = daily.tail(10) if daily is not None and not daily.empty else pd.DataFrame()

        m1_window = _format_window(m1_recent.index) if not m1_recent.empty else "N/A"
        m5_window = _format_window(m5_recent.index) if not m5_recent.empty else "N/A"
        daily_window = _format_window(daily_recent.index) if not daily_recent.empty else "N/A"

        alignment_ok = False
        if not m1_recent.empty and not m5_recent.empty:
            alignment_ok = (
                abs(
                    (m1_recent.index[-1].to_pydatetime() - m5_recent.index[-1].to_pydatetime()).total_seconds()
                )
                <= 300
            )

        sequence_ok = _has_expected_frequency(m1_recent.index, 1) and _has_expected_frequency(
            m5_recent.index, 5
        )
        freshness_sec = (
            datetime.now(UTC) - m1.index[-1].to_pydatetime()
            if not m1.empty
            else float("inf")
        ).total_seconds()
        freshness_ok = freshness_sec <= 60
        volume_ok = bool(m1_recent["Volume"].dropna().ge(0).all()) if not m1_recent.empty else True

        return {
            "symbol": symbol,
            "m1_table": _format_ohlcv_table(m1_recent),
            "m1_window": m1_window,
            "m5_table": _format_ohlcv_table(m5_recent),
            "m5_window": m5_window,
            "daily_table": _format_ohlcv_table(daily_recent) if not daily_recent.empty else "No daily data available",
            "daily_window": daily_window,
            "validation": {
                "timestamp_alignment": alignment_ok,
                "sequence_ok": sequence_ok,
                "freshness_sec": round(freshness_sec, 1) if freshness_sec != float("inf") else None,
                "freshness_ok": freshness_ok,
                "volume_ok": volume_ok,
            },
            "metrics": {
                "vwap": metrics.get("vwap"),
                "atr_5min": metrics.get("atr_5min"),
                "ema15_current": metrics.get("ema15_current"),
                "ema15_previous": metrics.get("ema15_previous"),
                "rsi_1min": metrics.get("rsi_1min"),
                "high_120": metrics.get("high_120"),
                "low_120": metrics.get("low_120"),
                "volume_ma20": metrics.get("volume_ma20"),
                "daily_sma20": metrics.get("daily_sma20"),
                "daily_sma50": metrics.get("daily_sma50"),
                "daily_sma200": metrics.get("daily_sma200"),
                "daily_return_5d_pct": metrics.get("daily_return_5d_pct"),
                "daily_return_20d_pct": metrics.get("daily_return_20d_pct"),
                "daily_atr14": metrics.get("daily_atr14"),
                "daily_rsi14": metrics.get("daily_rsi14"),
                "daily_trend": metrics.get("daily_trend"),
                "daily_high_20d": metrics.get("daily_high_20d"),
                "daily_low_20d": metrics.get("daily_low_20d"),
            },
        }

    def _compute_vwap(
        self,
        close: pd.Series,
        m1: pd.DataFrame,
        volume: pd.Series,
    ) -> Tuple[Optional[float], Optional[float]]:
        high = pd.to_numeric(m1["High"], errors="coerce")
        low = pd.to_numeric(m1["Low"], errors="coerce")
        typical_price = (high + low + close) / 3
        rolling_vwap = (typical_price * volume).cumsum() / volume.replace(0, np.nan).cumsum()

        if rolling_vwap.empty or pd.isna(rolling_vwap.iloc[-1]):
            return None, None

        vwap_value = float(rolling_vwap.iloc[-1])
        current_price = float(close.iloc[-1])
        if vwap_value == 0:
            return vwap_value, None
        distance = ((current_price - vwap_value) / vwap_value) * 100
        return vwap_value, float(distance)

    def _compute_momentum(self, close: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        if len(close) < 6:
            return None, None
        price_5min_ago = float(close.iloc[-6])
        latest_price = float(close.iloc[-1])
        if price_5min_ago == 0:
            return price_5min_ago, None
        roc = ((latest_price - price_5min_ago) / price_5min_ago) * 100
        return price_5min_ago, float(roc)

    def _compute_ema15_pair(self, m5: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        if m5.empty:
            return None, None
        try:
            close15 = (
                pd.to_numeric(m5["Close"], errors="coerce")
                .resample("15T")
                .last()
                .dropna()
            )
        except Exception:
            return None, None
        if close15.empty:
            return None, None
        ema_series = close15.ewm(span=20, adjust=False).mean().dropna()
        if ema_series.empty:
            return None, None
        current = float(ema_series.iloc[-1])
        previous = float(ema_series.iloc[-2]) if len(ema_series) >= 2 else None
        return current, previous

    def _score_volume(self, ratio: Optional[float]) -> float:
        if ratio is None or np.isnan(ratio):
            return 0.0
        return float(np.clip(ratio * 4.0, 0.0, 10.0))

    def _score_vwap(self, distance_pct: Optional[float]) -> float:
        if distance_pct is None or np.isnan(distance_pct):
            return 0.0
        return float(np.clip(10.0 - abs(distance_pct), 0.0, 10.0))

    def _score_momentum(self, roc: Optional[float]) -> float:
        if roc is None or np.isnan(roc):
            return 0.0
        return float(np.clip(roc + 5.0, 0.0, 10.0))

    def _score_catalyst(self, assumed_sector_return: float) -> float:
        # Simple heuristic: base 5, add 5 * sector_return (sector_return given as decimal, e.g., 0.006 for 0.6%)
        return float(np.clip(5.0 + (assumed_sector_return * 500.0), 0.0, 10.0))

    def _build_quantum_prompt(
        self,
        symbol: str,
        metrics: Dict[str, float],
        extraction: Dict[str, object],
        timestamp_label: str,
    ) -> str:
        def fmt(value, units: str = "", precision: int = 2):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "N/A"
            if isinstance(value, (int, float)):
                return f"{value:,.{precision}f}{units}"
            return str(value)

        def flag(ok: bool, ok_text: str = "OK", warn_text: str = "CHECK") -> str:
            return f"{'✅' if ok else '⚠️'} {'OK' if ok else warn_text}"

        validation = extraction.get("validation", {})
        extraction_metrics = extraction.get("metrics", {})
        freshness_sec = validation.get("freshness_sec")
        freshness_line = (
            f"{freshness_sec:.1f}s ({'OK' if validation.get('freshness_ok') else 'CHECK'})"
            if freshness_sec is not None
            else "N/A"
        )

        prompt = f"""QUANTUMTRADER v0.1 - DATA EXTRACTION TEST
Command: EXTRACT_RAW_DATA [{symbol}] [{timestamp_label}]

Required Data Output:
• Daily OHLCV: Last 10 periods ({extraction.get('daily_window', 'N/A')})
{extraction.get('daily_table', 'No daily data available')}

• 5-minute OHLCV: Last 6 periods ({extraction.get('m5_window', 'N/A')})
{extraction.get('m5_table', 'No data')}

• 1-minute OHLCV: Last 20 periods ({extraction.get('m1_window', 'N/A')})
{extraction.get('m1_table', 'No data')}

Daily Context:
• Daily Trend: {extraction_metrics.get('daily_trend', 'N/A')}
• Daily SMAs: SMA20={fmt(extraction_metrics.get('daily_sma20'))} | SMA50={fmt(extraction_metrics.get('daily_sma50'))} | SMA200={fmt(extraction_metrics.get('daily_sma200'))}
• Daily Returns: 5d={fmt(extraction_metrics.get('daily_return_5d_pct'), units='%', precision=2)} | 20d={fmt(extraction_metrics.get('daily_return_20d_pct'), units='%', precision=2)}
• Daily RSI(14): {fmt(extraction_metrics.get('daily_rsi14'))}
• Daily ATR(14): {fmt(extraction_metrics.get('daily_atr14'))}
• Daily Range (20d): High {fmt(extraction_metrics.get('daily_high_20d'))} | Low {fmt(extraction_metrics.get('daily_low_20d'))}
• Price vs Daily SMA20: {fmt(extraction_metrics.get('daily_distance_sma20_pct'), units='%', precision=2) if extraction_metrics.get('daily_distance_sma20_pct') is not None else 'N/A'}

Intraday Metrics:
• VWAP: {fmt(extraction_metrics.get('vwap'))}
• ATR(14)[5min]: {fmt(extraction_metrics.get('atr_5min'))}
• EMA(20)[15min]: Current {fmt(extraction_metrics.get('ema15_current'))} | Previous {fmt(extraction_metrics.get('ema15_previous'))}
• RSI(14)[1min]: {fmt(extraction_metrics.get('rsi_1min'))}
• 120-minute high/low: High {fmt(extraction_metrics.get('high_120'))} | Low {fmt(extraction_metrics.get('low_120'))}
• Volume MA(20)[1min]: {fmt(extraction_metrics.get('volume_ma20'), precision=0)}

Validation Checks:
• Timestamp alignment across timeframes: {flag(validation.get('timestamp_alignment', False))}
• No missing periods (1m & 5m): {flag(validation.get('sequence_ok', False))}
• Data freshness < 1 minute lag: {freshness_line}
• Volume data integrity (non-negative): {flag(validation.get('volume_ok', False))}

QUANTUMTRADER v0.1 - CALCULATION ENGINE TEST
Command: RUN_QUANT_SCREEN [{symbol}] [{timestamp_label}]

Phase 1 Scoring Calculations Required:
1. VOLUME_SCORE:
   • Current 1min volume: {fmt(metrics['current_volume'], precision=0)}
   • Volume MA(20): {fmt(metrics['volume_ma20'], precision=0)}
   • Volume Ratio: {fmt(metrics['volume_ratio'])}
   • Volume Score: {fmt(metrics['volume_score'])}

2. VWAP_SCORE:
   • Current Price: {fmt(metrics['current_price'])}
   • VWAP: {fmt(metrics['vwap'])}
   • VWAP Distance %: {fmt(metrics['vwap_distance_pct'], units='%', precision=2)}
   • VWAP Score: {fmt(metrics['vwap_score'])}

3. MOMENTUM_SCORE:
   • Price 5min ago: {fmt(metrics['price_5min_ago'])}
   • ROC_5min: {fmt(metrics['roc_5min_pct'], units='%', precision=2)}
   • Momentum Score: {fmt(metrics['momentum_score'])}

4. CATALYST_SCORE (simulated):
   • Assume: No scheduled news, sector_return = +0.6%
   • Catalyst Score: {fmt(metrics['catalyst_score'])}

5. COMPOSITE_SCORE:
   • Weighted sum: {fmt(metrics['composite_score'])}
   • PASS/FAIL: {metrics['verdict']}

Expected Output Format:
{symbol} | Vol: {fmt(metrics['volume_score'])} | VWAP: {fmt(metrics['vwap_score'])} | Mom: {fmt(metrics['momentum_score'])} | Cat: {fmt(metrics['catalyst_score'])} | Composite: {fmt(metrics['composite_score'])} | {metrics['verdict']}

DATA POLICY:
• Responses must rely strictly on the Supabase-ingested OHLCV tables reflected above.
• Do NOT fetch, assume, or hallucinate data from any external source.

QUANTUMTRADER v0.1 - TRADE DECISION ENGINE
Command: EVALUATE_TRADE_DECISION [{symbol}] [{timestamp_label}]

CONDITIONS FOR "TRADE YES":

1. Composite_Score ≥ 6.5
   • Current Composite Score: {fmt(metrics['composite_score'])}
   • Threshold: 6.5
   • Status: {'✅ PASS' if metrics['composite_score'] >= 6.5 else '❌ FAIL'}

2. All Phase 1 gates passed ✓
   • Volume spike, trend, RSI, and liquidity checks required
   • Run Phase 1 screening separately to verify

3. R:R ratio achievable ≥ 1:2
   • Minimum Risk:Reward ratio must be 1:2 or better
   • Calculate based on ATR-based stop loss and targets

4. Position size calculable within $2,000 exposure limit
   • Maximum position exposure: $2,000 per trade
   • Position size based on risk per share and exposure limit

5. No conflicting daily trend (avoid counter-trend)
   • 5-minute trend must align with higher timeframes
   • Avoid trading against daily trend

DIRECTION DECISION:
- If 5-min trend = UP and aligned with higher timeframes → BUY
- If 5-min trend = DOWN and aligned with higher timeframes → SELL
- If conflicting → "NO TRADE - Trend conflict"

RISK MANAGEMENT OVERLAY:
- Max daily loss: $400 (4% of $10k)
- Max concurrent trades: 3
- Auto-close all positions at 4:00 PM ET

DECISION OUTPUT FORMAT:
{symbol} | Decision: [TRADE YES / NO TRADE] | Direction: [BUY/SELL/CONFLICT] | Reason: [Explanation]
"""
        return prompt.strip()


