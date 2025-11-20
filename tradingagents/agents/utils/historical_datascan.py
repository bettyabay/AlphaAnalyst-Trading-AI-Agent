import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from groq import Groq

from tradingagents.database.config import get_supabase
from tradingagents.dataflows.market_data_service import TABLE_CONFIG


class HistoricalDataScanEngine:
    """Utility to run the DataScan Analyst prompt with stored Supabase data."""

    DAILY_LOOKBACK_DAYS = 1825  # 5 years
    M5_LOOKBACK_DAYS = 730      # 2 years
    M1_LOOKBACK_DAYS = 60       # 2 months (approx. 60 calendar days)

    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY", "") or os.getenv("GROK_API_KEY", "")
        self.client = None
        if self.groq_key and self.groq_key.startswith("gsk_"):
            try:
                self.client = Groq(api_key=self.groq_key)
            except Exception:
                self.client = None

    def run_analysis(self, symbol: str) -> Dict[str, Any]:
        """Run the DataScan prompt end-to-end."""
        symbol = (symbol or "").upper()
        if not symbol:
            return {"error": "Symbol is required for DataScan analysis."}

        timeframes = self._collect_timeframe_data(symbol)
        if timeframes.get("error"):
            return timeframes

        prompt = self._build_prompt(symbol, timeframes)

        result: Dict[str, Any] = {
            "prompt": prompt,
            "timeframe_stats": timeframes["stats"],
            "samples": timeframes["samples"],
        }

        if not self.client:
            result["message"] = (
                "GROQ_API_KEY not configured or invalid. Copy the prompt above into your "
                "LLM of choice to run the DataScan analysis manually."
            )
            return result

        try:
            chat = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0.15,
                messages=[
                    {
                        "role": "system",
                        "content": self._datascan_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            response = chat.choices[0].message.content if chat.choices else ""
            result["report"] = response
            return result
        except Exception as exc:
            result["error"] = f"Groq request failed: {exc}"
            return result

    def _collect_timeframe_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch daily, 5-min, and 1-min data slices from Supabase."""
        supabase = get_supabase()
        if not supabase:
            return {"error": "Supabase not configured. Please set SUPABASE_URL and SUPABASE_KEY."}

        now = datetime.now(timezone.utc)
        daily_df = self._fetch_interval_df(
            supabase,
            symbol,
            interval="1d",
            start=now - timedelta(days=self.DAILY_LOOKBACK_DAYS),
            end=now,
            max_rows=2200,
        )
        m5_df = self._fetch_interval_df(
            supabase,
            symbol,
            interval="5min",
            start=now - timedelta(days=self.M5_LOOKBACK_DAYS),
            end=now,
            max_rows=60000,
        )
        m1_df = self._fetch_interval_df(
            supabase,
            symbol,
            interval="1min",
            start=now - timedelta(days=self.M1_LOOKBACK_DAYS),
            end=now,
            max_rows=40000,
        )

        if daily_df.empty:
            return {"error": f"No daily data found for {symbol}. Please ingest historical data first."}

        stats = {
            "daily": self._build_stats(daily_df, "1d"),
            "m5": self._build_stats(m5_df, "5min"),
            "m1": self._build_stats(m1_df, "1min"),
        }

        samples = {
            "daily": self._format_sample_table(daily_df, "Date"),
            "m5": self._format_sample_table(m5_df, "Timestamp"),
            "m1": self._format_sample_table(m1_df, "Timestamp"),
        }

        return {
            "daily_df": daily_df,
            "m5_df": m5_df,
            "m1_df": m1_df,
            "stats": stats,
            "samples": samples,
        }

    def _fetch_interval_df(
        self,
        supabase,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        config = TABLE_CONFIG.get(interval)
        if not config:
            return pd.DataFrame()

        table = config["table"]
        time_field = config["time_field"]
        is_date = bool(config["is_date"])

        start_value = start.date().isoformat() if is_date else start.isoformat()
        end_value = end.date().isoformat() if is_date else end.isoformat()

        fields = f"{time_field},open,high,low,close,volume"
        page_size = 1000
        offset = 0
        rows: list[Dict[str, Any]] = []

        while True:
            query = (
                supabase.table(table)
                .select(fields)
                .eq("symbol", symbol)
                .gte(time_field, start_value)
                .lte(time_field, end_value)
                .order(time_field, desc=True)
                .range(offset, offset + page_size - 1)
            )
            resp = query.execute()
            data = resp.data if hasattr(resp, "data") else resp
            if not data:
                break

            rows.extend(data)
            if len(data) < page_size:
                break

            offset += page_size
            if max_rows and len(rows) >= max_rows:
                break

        if not rows:
            return pd.DataFrame()

        if max_rows and len(rows) > max_rows:
            rows = rows[:max_rows]

        df = pd.DataFrame(rows)
        df[time_field] = pd.to_datetime(df[time_field])
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        df = df.sort_values(time_field)
        df.set_index(time_field, inplace=True)
        df.index.name = "Date" if is_date else "Timestamp"
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        return df

    def _build_stats(self, df: pd.DataFrame, interval: str) -> Dict[str, Any]:
        if df.empty:
            return {"records": 0, "coverage": "No data"}

        first_idx = df.index[0]
        last_idx = df.index[-1]
        stats = {
            "records": int(len(df)),
            "start": first_idx.isoformat(),
            "end": last_idx.isoformat(),
        }

        close = pd.to_numeric(df["Close"], errors="coerce")

        if interval == "1d":
            sma50 = self._calc_sma(close, 50)
            sma200 = self._calc_sma(close, 200)
            last_price = close.iloc[-1]
            trend = "Ranging"
            if pd.notna(sma50) and pd.notna(sma200):
                if sma50 > sma200:
                    trend = "Bullish"
                elif sma50 < sma200:
                    trend = "Bearish"
            stats.update(
                {
                    "trend": trend,
                    "sma50": self._round_float(sma50),
                    "sma200": self._round_float(sma200),
                    "last_close": self._round_float(last_price),
                    "support_levels": self._key_levels(close, mode="support"),
                    "resistance_levels": self._key_levels(close, mode="resistance"),
                    "volume_note": self._volume_note(df["Volume"]),
                }
            )
        else:
            rsi = self._calc_rsi(close)
            stats.update(
                {
                    "last_close": self._round_float(close.iloc[-1]),
                    "rsi14": self._round_float(rsi),
                }
            )

        return stats

    def _key_levels(self, close: pd.Series, mode: str) -> Tuple[float, float, float]:
        if close.empty:
            return (0.0, 0.0, 0.0)
        quantiles = (0.1, 0.2, 0.3) if mode == "support" else (0.7, 0.8, 0.9)
        levels = []
        for q in quantiles:
            try:
                level = float(close.quantile(q))
                levels.append(round(level))
            except Exception:
                levels.append(0.0)
        return tuple(levels)

    def _volume_note(self, volume: pd.Series) -> str:
        volume = pd.to_numeric(volume, errors="coerce").dropna()
        if len(volume) < 40:
            return "Limited volume history"

        recent = volume.tail(20).mean()
        prior = volume.tail(40).head(20).mean()
        if recent > prior * 1.15:
            return "Volume rising on up-moves"
        if recent < prior * 0.85:
            return "Volume fading vs prior month"
        return "Volume steady"

    def _calc_sma(self, series: pd.Series, period: int) -> Optional[float]:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) < period:
            return None
        return float(series.tail(period).mean())

    def _calc_rsi(self, series: pd.Series, period: int = 14) -> Optional[float]:
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

    def _format_sample_table(self, df: pd.DataFrame, index_name: str, rows: int = 10) -> str:
        if df.empty:
            return "No data available"
        tail = df.tail(rows).reset_index()
        tail[index_name] = tail[index_name].dt.strftime("%Y-%m-%d %H:%M")
        cols = [index_name, "Open", "High", "Low", "Close", "Volume"]
        tail = tail[cols]
        try:
            return tail.to_markdown(index=False)
        except Exception:
            # pandas needs the optional `tabulate` dependency for markdown export.
            # Fallback to a plain-text table if tabulate isn't installed.
            return tail.to_string(index=False)

    def _round_float(self, value: Optional[float]) -> Optional[float]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return round(float(value), 2)

    def _datascan_system_prompt(self) -> str:
        return (
            'You are the "DataScan Analyst," an AI quantitative analysis engine. '
            "Perform rigorous multi-timeframe technical analysis on the supplied historical data "
            "to identify the most statistically significant patterns and high-probability trading signals. "
            "Follow the provided instructions exactly and output in the required markdown template."
        )

    def _build_prompt(self, symbol: str, timeframe_data: Dict[str, Any]) -> str:
        stats = timeframe_data["stats"]
        samples = timeframe_data["samples"]

        daily_stats = stats.get("daily", {})
        m5_stats = stats.get("m5", {})
        m1_stats = stats.get("m1", {})

        def fmt_levels(levels):
            if not levels:
                return "N/A"
            return ", ".join(f"${int(level)}" for level in levels)

        prompt_sections = [
            f"HISTORICAL DATA PACKAGE FOR {symbol}",
            "",
            "DAILY (D1) SUMMARY - 5 YEARS",
            f"- Records: {daily_stats.get('records', 'N/A')}",
            f"- Coverage: {daily_stats.get('start', 'N/A')} → {daily_stats.get('end', 'N/A')}",
            f"- Trend via SMA: {daily_stats.get('trend', 'Unknown')} (SMA50={daily_stats.get('sma50')}, SMA200={daily_stats.get('sma200')})",
            f"- Support Levels: {fmt_levels(daily_stats.get('support_levels'))}",
            f"- Resistance Levels: {fmt_levels(daily_stats.get('resistance_levels'))}",
            f"- Volume Note: {daily_stats.get('volume_note', 'N/A')}",
            "",
            "TACTICAL (M5) SUMMARY - 2 YEARS",
            f"- Records: {m5_stats.get('records', 'N/A')}",
            f"- Coverage: {m5_stats.get('start', 'N/A')} → {m5_stats.get('end', 'N/A')}",
            f"- Latest Close: {m5_stats.get('last_close', 'N/A')}",
            f"- RSI(14): {m5_stats.get('rsi14', 'N/A')}",
            "",
            "EXECUTION (M1) SUMMARY - 2 MONTHS",
            f"- Records: {m1_stats.get('records', 'N/A')}",
            f"- Coverage: {m1_stats.get('start', 'N/A')} → {m1_stats.get('end', 'N/A')}",
            f"- Latest Close: {m1_stats.get('last_close', 'N/A')}",
            f"- RSI(14): {m1_stats.get('rsi14', 'N/A')}",
            "",
            "RECENT D1 CANDLES:",
            samples.get("daily", "N/A"),
            "",
            "RECENT M5 CANDLES:",
            samples.get("m5", "N/A"),
            "",
            "RECENT M1 CANDLES:",
            samples.get("m1", "N/A"),
        ]

        prompt_sections.append("")
        prompt_sections.append(
            "Please run the HISTORICAL DATA ANALYST protocol using the context above. "
            "Return the final report in the required markdown structure."
        )

        return "\n".join(prompt_sections)

