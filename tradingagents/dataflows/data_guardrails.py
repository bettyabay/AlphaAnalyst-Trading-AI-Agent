"""
Data coverage guardrails to ensure required historical ranges exist before analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

from tradingagents.config.watchlist import get_watchlist_symbols
from tradingagents.database.config import get_supabase
from tradingagents.dataflows.market_data_service import TABLE_CONFIG


EAT_TZ = ZoneInfo("Africa/Nairobi")
UTC = timezone.utc


@dataclass
class CoverageRequirement:
    """Represents a required coverage window for a given interval."""

    interval: str
    label: str
    start: datetime
    end: datetime
    chunk_days: int

    def as_dict(self) -> Dict[str, str]:
        return {
            "interval": self.interval,
            "label": self.label,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "chunk_days": self.chunk_days,
        }


@dataclass
class CoverageRecord:
    """Holds coverage status for a single symbol/interval."""

    symbol: str
    interval: str
    requirement: CoverageRequirement
    db_start: Optional[datetime]
    db_end: Optional[datetime]
    row_count: Optional[int]
    head_gap_minutes: float
    tail_gap_minutes: float

    def status(self) -> str:
        """
        Determine coverage status, accounting for weekends and expected gaps.
        Status is only "partial" if there are significant gaps during trading days.
        Small gaps (< 30 days) are acceptable regardless of weekdays.
        """
        if self.db_start is None or self.db_end is None:
            return "missing"
        
        # Calculate total required period for percentage calculation
        total_required_minutes = max(1.0, (self.requirement.end - self.requirement.start).total_seconds() / 60.0)
        
        # Check if gaps are significant (not just weekends or small acceptable gaps)
        head_significant = False
        tail_significant = False
        
        # Acceptable gap thresholds: 30 days OR 5% of total period, whichever is larger
        acceptable_gap_minutes = max(43200, total_required_minutes * 0.05)  # 30 days = 43200 minutes, or 5% of total
        
        if self.head_gap_minutes > 0:
            # Head gap is acceptable if it's small (< 30 days or < 5% of total period)
            if self.head_gap_minutes <= acceptable_gap_minutes:
                head_significant = False
            else:
                # Larger gap - check if it contains weekdays
                gap_start = self.requirement.start
                gap_end = self.db_start
                gap_duration_days = self.head_gap_minutes / 1440
                sample_count = min(10, max(1, int(gap_duration_days)))
                if sample_count > 0:
                    for i in range(sample_count):
                        # Distribute samples evenly across the gap
                        if sample_count > 1:
                            progress = i / (sample_count - 1)
                        else:
                            progress = 0.0
                        sample_date = gap_start + timedelta(days=gap_duration_days * progress)
                        if sample_date.weekday() < 5:  # Monday-Friday (0-4)
                            head_significant = True
                            break
                # If gap is very large (> 1 week), consider it significant even if all weekends
                if self.head_gap_minutes > 10080:
                    head_significant = True
        
        if self.tail_gap_minutes > 0:
            # Tail gap is acceptable if it's small (< 30 days or < 5% of total period)
            if self.tail_gap_minutes <= acceptable_gap_minutes:
                tail_significant = False
            else:
                # Larger gap - check if it contains weekdays
                gap_start = self.db_end
                gap_end = self.requirement.end
                gap_duration_days = self.tail_gap_minutes / 1440
                sample_count = min(10, max(1, int(gap_duration_days)))
                if sample_count > 0:
                    for i in range(sample_count):
                        # Distribute samples evenly across the gap
                        if sample_count > 1:
                            progress = i / (sample_count - 1)
                        else:
                            progress = 0.0
                        sample_date = gap_start + timedelta(days=gap_duration_days * progress)
                        if sample_date.weekday() < 5:  # Monday-Friday (0-4)
                            tail_significant = True
                            break
                # If gap is very large (> 1 week), consider it significant even if all weekends
                if self.tail_gap_minutes > 10080:
                    tail_significant = True
        
        # Only mark as "partial" if there are significant gaps
        if not head_significant and not tail_significant:
            return "ready"
        
        return "partial"

    def needs_backfill(self) -> bool:
        return self.status() != "ready"

    def to_dict(self) -> Dict[str, object]:
        req = self.requirement
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "label": req.label,
            "required_start": req.start.isoformat(),
            "required_end": req.end.isoformat(),
            "db_start": self.db_start.isoformat() if self.db_start else "N/A (no data found)",
            "db_end": self.db_end.isoformat() if self.db_end else "N/A (no data found)",
            "rows": self.row_count if self.row_count is not None else 0,
            "head_gap_min": round(self.head_gap_minutes, 1),
            "tail_gap_min": round(self.tail_gap_minutes, 1),
            "status": self.status(),
        }


logger = logging.getLogger(__name__)


class DataCoverageService:
    """Ensures the database holds the minimum required historical windows."""

    def __init__(self, reference_dt: Optional[datetime] = None, asset_class: Optional[str] = None):
        self.supabase = get_supabase()
        self.reference_dt = reference_dt
        self.asset_class = asset_class  # "Commodities", "Currencies", "Stocks", "Indices"

    # --------------------------------------------------------------------- #
    # Requirement helpers
    # --------------------------------------------------------------------- #
    def _now_eat(self) -> datetime:
        if self.reference_dt:
            if self.reference_dt.tzinfo:
                return self.reference_dt.astimezone(EAT_TZ)
            return self.reference_dt.replace(tzinfo=UTC).astimezone(EAT_TZ)
        return datetime.now(EAT_TZ)

    def _minutes_gap(self, start: datetime, end: datetime) -> float:
        return max(0.0, (end - start).total_seconds() / 60.0)

    def _is_weekend(self, dt: datetime) -> bool:
        """Check if a datetime falls on a weekend (Saturday or Sunday)."""
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        return weekday >= 5  # Saturday (5) or Sunday (6)

    def _count_trading_minutes(self, start: datetime, end: datetime) -> float:
        """
        Count expected trading minutes between two datetimes, excluding weekends.
        Assumes 24/5 trading (weekdays only) for forex/commodities, or market hours for stocks.
        For simplicity, we count all weekday minutes and exclude weekend minutes.
        """
        if end <= start:
            return 0.0
        
        total_minutes = self._minutes_gap(start, end)
        
        # Count weekend minutes to subtract
        weekend_minutes = 0.0
        current = start
        while current < end:
            if self._is_weekend(current):
                # Count minutes in this weekend day
                day_end = min(
                    current.replace(hour=23, minute=59, second=59, microsecond=999999),
                    end
                )
                weekend_minutes += self._minutes_gap(current, day_end)
            # Move to next day
            next_day = current + timedelta(days=1)
            next_day = next_day.replace(hour=0, minute=0, second=0, microsecond=0)
            if next_day >= end:
                break
            current = next_day
        
        return max(0.0, total_minutes - weekend_minutes)

    def _is_gap_significant(self, gap_minutes: float, start: datetime, end: datetime) -> bool:
        """
        Determine if a gap is significant (not just weekends).
        A gap is significant if it's more than 2 days (2880 minutes) of trading time,
        or if it spans non-weekend periods.
        """
        if gap_minutes <= 0:
            return False
        
        # If gap is less than 2 days, it's likely just weekends or minor gaps
        if gap_minutes < 2880:  # 2 days = 2880 minutes
            return False
        
        # Check if the gap period contains non-weekend days
        # Sample a few dates in the gap to see if any are weekdays
        gap_start = start
        gap_end = end
        if gap_end <= gap_start:
            return False
        
        # Sample up to 10 dates across the gap
        gap_duration = (gap_end - gap_start).total_seconds() / 86400  # days
        sample_count = min(10, int(gap_duration))
        
        if sample_count == 0:
            return gap_minutes > 1440  # More than 1 day
        
        for i in range(sample_count):
            sample_date = gap_start + timedelta(days=(gap_duration * i / max(1, sample_count - 1)))
            if not self._is_weekend(sample_date):
                # Found a weekday in the gap, so it's significant
                return True
        
        # All sampled dates are weekends, but gap is large - still consider significant
        return gap_minutes > 10080  # More than 1 week

    def _requirements(self) -> Dict[str, CoverageRequirement]:
        """
        Build requirements for 1-minute data only, with asset-class-specific start dates.
        End date is set based on Polygon API limits:
            - Indices: 1 year back (Polygon free plan limit)
            - Stocks/Commodities/Currencies: 2 years (Polygon API limit)
        For indices, start date is calculated as 1 year back from reference date.
            - Indices: (now - 1 year) → now (1 year back from reference date)
            - Stocks: 2024-01-01 → 2026-01-01 (2 years)
            - Commodities: 2024-01-10 → 2026-01-10 (2 years)
            - Currencies: 2024-01-10 → 2026-01-10 (2 years)
        """
        now_utc = datetime.now(UTC)
        
        # Determine start date based on asset class
        asset_class = (self.asset_class or "").strip()
        if asset_class == "Indices":
            # Indices: 1 year back from now (Polygon free plan limit)
            start_date = (now_utc - timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            start_str = start_date.strftime("%b %d, %Y")
            end_str = end_date.strftime("%b %d, %Y")
            label = f"1-min bars ({start_str} → {end_str}, 1 year)"
        elif asset_class == "Stocks":
            start_date = datetime(2024, 1, 1, tzinfo=UTC)
            end_date = datetime(2026, 1, 1, tzinfo=UTC)
            label = "1-min bars (Jan 1, 2024 → Jan 1, 2026, 2 years)"
        elif asset_class in ["Commodities", "Currencies"]:
            start_date = datetime(2024, 1, 10, tzinfo=UTC)
            end_date = datetime(2026, 1, 10, tzinfo=UTC)
            label = "1-min bars (Jan 10, 2024 → Jan 10, 2026, 2 years)"
        else:
            # Default fallback
            start_date = datetime(2024, 1, 1, tzinfo=UTC)
            end_date = datetime(2026, 1, 1, tzinfo=UTC)
            label = "1-min bars (Jan 1, 2024 → Jan 1, 2026, 2 years)"

        requirements = {
            "1min": CoverageRequirement(
                interval="1min",
                label=label,
                start=start_date,
                end=end_date,
                chunk_days=3,
            ),
        }

        # Guard against negative ranges
        for req in requirements.values():
            if req.end < req.start:
                req.end = req.start
        return requirements

    # --------------------------------------------------------------------- #
    # Database helpers
    # --------------------------------------------------------------------- #
    def _table_fields(self, interval: str) -> Tuple[str, str, bool]:
        config = TABLE_CONFIG.get(interval) or TABLE_CONFIG.get(
            "1d" if interval == "daily" else interval
        )
        if not config:
            raise ValueError(f"No table config for interval '{interval}'")
        return config["table"], config["time_field"], bool(config["is_date"])

    def _parse_ts(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            ts = pd.to_datetime(value)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            else:
                ts = ts.tz_convert(UTC) if hasattr(ts, "tz_convert") else ts.astimezone(UTC)
            return ts.to_pydatetime()
        except Exception:
            return None

    def _get_bounds(self, symbol: str, interval: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        if not self.supabase:
            return None, None

        # Get base table and fields
        try:
            table, time_field, _ = self._table_fields(interval)
        except ValueError:
            logger.warning(f"No table config for interval '{interval}'")
            return None, None

        # For 1min data, determine which tables to check based on asset class and symbol format
        tables_to_try = []
        if interval == "1min":
            symbol_upper = symbol.upper()
            # 1. Prioritize explicit asset_class routing
            if self.asset_class == "Commodities":
                tables_to_try = ["market_data_commodities_1min"]
            elif self.asset_class == "Indices":
                tables_to_try = ["market_data_indices_1min"]
            elif self.asset_class == "Currencies":
                tables_to_try = ["market_data_currencies_1min"]
            elif self.asset_class == "Stocks":
                tables_to_try = ["market_data_stocks_1min"]
            else:
                # 2. Fallback to symbol pattern matching - try multiple tables
                # Check for commodities first (XAU, GOLD) even if it has C: prefix
                if "XAU" in symbol_upper or "GOLD" in symbol_upper or "*" in symbol_upper:
                    tables_to_try.append("market_data_commodities_1min")
                # Check for currencies if it has C: prefix or / separator
                if symbol_upper.startswith("C:") or "/" in symbol_upper:
                    tables_to_try.append("market_data_currencies_1min")
                # Check for indices if it has I: prefix or ^ prefix
                if symbol_upper.startswith("I:") or symbol_upper.startswith("^"):
                    tables_to_try.append("market_data_indices_1min")
                # Default to stocks if no match
                if not tables_to_try:
                    tables_to_try.append("market_data_stocks_1min")
        else:
            # For non-1min intervals, use the table from config
            tables_to_try = [table]

        # Try multiple symbol formats (e.g., "C:XAUUSD" might be stored as "XAUUSD" or vice versa)
        symbols_to_try = [symbol]
        symbol_upper = symbol.upper()
        
        # Add alternative formats
        if symbol_upper.startswith("C:"):
            # For "C:XAUUSD", try both "C:XAUUSD" and "XAUUSD"
            base_symbol = symbol_upper[2:]  # Remove "C:" prefix
            symbols_to_try.extend([base_symbol, symbol_upper])
        elif symbol_upper.startswith("I:"):
            # For "I:SPX", try both "I:SPX", "^SPX", and "SPX"
            base_symbol = symbol_upper[2:]
            symbols_to_try.extend([f"^{base_symbol}", base_symbol, symbol_upper])
        elif "XAU" in symbol_upper or "GOLD" in symbol_upper:
            # For "XAUUSD", try both "XAUUSD" and "C:XAUUSD"
            symbols_to_try.extend([f"C:{symbol_upper}", symbol_upper])
        elif symbol_upper.startswith("^"):
            # For "^SPX", try both "^SPX", "I:SPX", and "SPX"
            base_symbol = symbol_upper[1:]
            symbols_to_try.extend([f"I:{base_symbol}", base_symbol, symbol_upper])
        
        # Remove duplicates while preserving order
        symbols_to_try = list(dict.fromkeys(symbols_to_try))
        
        # Try each table and symbol format combination
        for try_table in tables_to_try:
            for try_symbol in symbols_to_try:
                try:
                    earliest_resp = (
                        self.supabase.table(try_table)
                        .select(time_field)
                        .eq("symbol", try_symbol)
                        .order(time_field, desc=False)
                        .limit(1)
                        .execute()
                    )
                    latest_resp = (
                        self.supabase.table(try_table)
                        .select(time_field)
                        .eq("symbol", try_symbol)
                        .order(time_field, desc=True)
                        .limit(1)
                        .execute()
                    )
                    
                    # Check if we got results
                    earliest_data = getattr(earliest_resp, "data", None)
                    latest_data = getattr(latest_resp, "data", None)
                    
                    if earliest_data and len(earliest_data) > 0 and latest_data and len(latest_data) > 0:
                        earliest_val = earliest_data[0].get(time_field)
                        latest_val = latest_data[0].get(time_field)
                        db_start = self._parse_ts(earliest_val)
                        db_end = self._parse_ts(latest_val)
                        if db_start and db_end:
                            logger.debug(
                                f"Found bounds for {symbol} (tried as '{try_symbol}') in {try_table}: {db_start} to {db_end}"
                            )
                            return db_start, db_end
                except Exception as exc:
                    # Continue to next table/symbol combination
                    logger.debug(
                        f"Failed to query bounds for {symbol} (tried as '{try_symbol}') in {try_table}: {exc}"
                    )
                    continue
        
        # No results found in any format
        logger.warning(
            f"Coverage guardrail: No data found for symbol '{symbol}' in table '{table}' "
            f"(tried formats: {symbols_to_try}, interval: {interval})"
        )
        return None, None

    def _count_rows(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> Optional[int]:
        if not self.supabase:
            return None

        try:
            table, time_field, is_date = self._table_fields(interval)
        except ValueError:
            logger.warning(f"No table config for interval '{interval}'")
            return None

        # For 1min data, determine which tables to check (same logic as _get_bounds)
        tables_to_try = []
        if interval == "1min":
            symbol_upper = symbol.upper()
            # 1. Prioritize explicit asset_class routing
            if self.asset_class == "Commodities":
                tables_to_try = ["market_data_commodities_1min"]
            elif self.asset_class == "Indices":
                tables_to_try = ["market_data_indices_1min"]
            elif self.asset_class == "Currencies":
                tables_to_try = ["market_data_currencies_1min"]
            elif self.asset_class == "Stocks":
                tables_to_try = ["market_data_stocks_1min"]
            else:
                # 2. Fallback to symbol pattern matching - try multiple tables
                if "XAU" in symbol_upper or "GOLD" in symbol_upper or "*" in symbol_upper:
                    tables_to_try.append("market_data_commodities_1min")
                if symbol_upper.startswith("C:") or "/" in symbol_upper:
                    tables_to_try.append("market_data_currencies_1min")
                if symbol_upper.startswith("I:") or symbol_upper.startswith("^"):
                    tables_to_try.append("market_data_indices_1min")
                if not tables_to_try:
                    tables_to_try.append("market_data_stocks_1min")
        else:
            tables_to_try = [table]

        start_value = start.date().isoformat() if is_date else start.isoformat()
        end_value = end.date().isoformat() if is_date else end.isoformat()

        # Try multiple symbol formats (same as _get_bounds)
        symbols_to_try = [symbol]
        symbol_upper = symbol.upper()
        if symbol_upper.startswith("C:"):
            symbols_to_try.extend([symbol_upper[2:], symbol_upper])
        elif symbol_upper.startswith("I:"):
            base_symbol = symbol_upper[2:]
            symbols_to_try.extend([f"^{base_symbol}", base_symbol, symbol_upper])
        elif "XAU" in symbol_upper or "GOLD" in symbol_upper:
            symbols_to_try.extend([f"C:{symbol_upper}", symbol_upper])
        elif symbol_upper.startswith("^"):
            base_symbol = symbol_upper[1:]
            symbols_to_try.extend([f"I:{base_symbol}", base_symbol, symbol_upper])
        symbols_to_try = list(dict.fromkeys(symbols_to_try))

        # Try each table and symbol format combination
        for try_table in tables_to_try:
            for try_symbol in symbols_to_try:
                try:
                    resp = (
                        self.supabase.table(try_table)
                        .select(time_field, count="exact")
                        .eq("symbol", try_symbol)
                        .gte(time_field, start_value)
                        .lte(time_field, end_value)
                        .limit(1)
                        .execute()
                    )
                    count = getattr(resp, "count", None)
                    if count is not None and count > 0:
                        logger.debug(f"Found {count} rows for {symbol} (tried as '{try_symbol}') in {try_table}")
                        return count
                except Exception as exc:
                    logger.debug(f"Failed to count rows for {symbol} (tried as '{try_symbol}') in {try_table}: {exc}")
                    continue
        
        # No rows found in any table/format combination
        return 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def build_symbol_report(self, symbol: str) -> List[CoverageRecord]:
        symbol = (symbol or "").upper()
        if not symbol:
            return []
        reqs = self._requirements()
        records: List[CoverageRecord] = []
        for interval_key, requirement in reqs.items():
            table_interval = interval_key if interval_key != "1d" else "1d"
            db_start, db_end = self._get_bounds(symbol, table_interval)

            head_gap = 0.0
            tail_gap = 0.0
            if db_start:
                if db_start > requirement.start:
                    head_gap = self._minutes_gap(requirement.start, db_start)
            else:
                head_gap = self._minutes_gap(requirement.start, requirement.end)

            if db_end:
                if db_end < requirement.end:
                    tail_gap = self._minutes_gap(db_end, requirement.end)
            else:
                tail_gap = self._minutes_gap(requirement.start, requirement.end)

            row_count = self._count_rows(symbol, table_interval, requirement.start, requirement.end)
            records.append(
                CoverageRecord(
                    symbol=symbol,
                    interval=table_interval,
                    requirement=requirement,
                    db_start=db_start,
                    db_end=db_end,
                    row_count=row_count,
                    head_gap_minutes=head_gap,
                    tail_gap_minutes=tail_gap,
                )
            )
        return records

    def build_watchlist_report(self, symbols: Optional[Sequence[str]] = None) -> List[CoverageRecord]:
        symbols = symbols or get_watchlist_symbols()
        report: List[CoverageRecord] = []
        for symbol in symbols:
            report.extend(self.build_symbol_report(symbol))
        return report

    # ------------------------------------------------------------------ #
    # Backfill helpers
    # ------------------------------------------------------------------ #
    def _create_backfill_tasks(self, record: CoverageRecord) -> List[Dict[str, object]]:
        tasks: List[Dict[str, object]] = []
        req = record.requirement
        tolerance = timedelta(minutes=5)

        # Head gap (missing start of window)
        if record.db_start is None or record.db_start - req.start > tolerance:
            head_end = min(req.end, (record.db_start - timedelta(minutes=1)) if record.db_start else req.end)
            if head_end > req.start:
                tasks.append(
                    {
                        "symbol": record.symbol,
                        "interval": record.interval,
                        "start": req.start,
                        "end": head_end,
                        "chunk_days": req.chunk_days,
                        "resume_from_latest": False,
                        "reason": "head_gap",
                    }
                )

        # Tail gap (missing latest data)
        if record.db_end is None or req.end - record.db_end > tolerance:
            tail_start = max(req.start, (record.db_end + timedelta(minutes=1)) if record.db_end else req.start)
            if req.end > tail_start:
                tasks.append(
                    {
                        "symbol": record.symbol,
                        "interval": record.interval,
                        "start": tail_start,
                        "end": req.end,
                        "chunk_days": req.chunk_days,
                        "resume_from_latest": True,
                        "reason": "tail_gap",
                    }
                )

        return tasks

    def backfill_missing(
        self,
        pipeline,
        records: Optional[Sequence[CoverageRecord]] = None,
    ) -> List[Dict[str, object]]:
        """
        Attempt to backfill any gaps using the provided DataIngestionPipeline.
        For 1min data, uses ingest_from_polygon_api which properly handles symbol conversion.
        Returns execution logs for UI display.
        """
        records = records or []
        logs: List[Dict[str, object]] = []
        
        # Import here to avoid circular imports
        from tradingagents.dataflows.universal_ingestion import ingest_from_polygon_api
        from tradingagents.dataflows.ingestion_pipeline import convert_instrument_to_polygon_symbol
        from ingest_indices_polygon import ingest_indices_from_polygon
        
        for record in records:
            for task in self._create_backfill_tasks(record):
                start_dt = task["start"].replace(tzinfo=None)
                end_dt = task["end"].replace(tzinfo=None)
                if end_dt <= start_dt:
                    continue

                try:
                    symbol = task["symbol"]
                    interval = task["interval"]
                    
                    # For 1min data, convert symbol to Polygon format and use GMT+4 timezone
                    if interval == "1min" and self.asset_class:
                        # Special handling for indices: use dedicated indices ingestion, not the generic pipeline.
                        if self.asset_class == "Indices":
                            # Convert display symbol (e.g. "^SPX") to Polygon index symbol (e.g. "I:SPX")
                            api_symbol = convert_instrument_to_polygon_symbol(self.asset_class, symbol)
                            print(f"🔄 Indices backfill: symbol '{symbol}' → Polygon '{api_symbol}' (1min)")

                            # Call the indices ingestion helper which:
                            # - Maps I:SPX → SPY internally for minute data
                            # - Stores data into market_data_indices_1min with db_symbol below
                            result = ingest_indices_from_polygon(
                                api_symbol=api_symbol,
                                interval="1min",
                                years=1,
                                db_symbol=symbol,
                            )
                            success = bool(result.get("success")) if isinstance(result, dict) else False
                        else:
                            # Non-index 1min data: use the generic pipeline with proper symbol conversion
                            # This ensures currencies get "C:" prefix (USDJPY → C:USDJPY), etc.
                            polygon_symbol = convert_instrument_to_polygon_symbol(self.asset_class, symbol)
                            print(f"🔄 Converting symbol '{symbol}' to Polygon format '{polygon_symbol}' for asset class '{self.asset_class}'")
                            
                            if not pipeline:
                                logs.append({
                                    "symbol": symbol,
                                    "interval": interval,
                                    "reason": task["reason"],
                                    "start": start_dt.isoformat(),
                                    "end": end_dt.isoformat(),
                                    "success": False,
                                    "message": "Pipeline not provided",
                                })
                                continue
                            
                            # Use pipeline with converted Polygon symbol and GMT+4 timezone conversion
                            # The pipeline will convert UTC timestamps from Polygon to GMT+4 (Asia/Dubai) before storing
                            # Pass asset_class so pipeline can also do symbol conversion as a safeguard
                            success = pipeline.ingest_historical_data(
                                symbol=polygon_symbol,  # Use converted Polygon symbol format (e.g., "C:USDJPY")
                                interval="daily" if interval == "1d" else interval,
                                start_date=start_dt,
                                end_date=end_dt,
                                chunk_days=task["chunk_days"],
                                resume_from_latest=False,  # Don't use auto-resume for backfill - use explicit dates
                                target_timezone="Asia/Dubai",  # Convert UTC to GMT+4 (Asia/Dubai) before storing
                                asset_class=self.asset_class,  # Pass asset_class for symbol conversion safeguard
                            )
                    else:
                        # For non-1min data, use pipeline as before
                        if not pipeline:
                            logs.append({
                                "symbol": symbol,
                                "interval": interval,
                                "reason": task["reason"],
                                "start": start_dt.isoformat(),
                                "end": end_dt.isoformat(),
                                "success": False,
                                "message": "Pipeline not provided",
                            })
                            continue
                        
                        success = pipeline.ingest_historical_data(
                            symbol=symbol,
                            interval="daily" if interval == "1d" else interval,
                            start_date=start_dt,
                            end_date=end_dt,
                            chunk_days=task["chunk_days"],
                            resume_from_latest=task["resume_from_latest"],
                        )
                    
                    logs.append(
                        {
                            "symbol": symbol,
                            "interval": interval,
                            "reason": task["reason"],
                            "start": start_dt.isoformat(),
                            "end": end_dt.isoformat(),
                            "success": success,
                        }
                    )
                except ValueError as e:
                    # Catch 401 errors from polygon_integration
                    error_msg = str(e)
                    logs.append(
                        {
                            "symbol": task["symbol"],
                            "interval": task["interval"],
                            "reason": task["reason"],
                            "start": start_dt.isoformat(),
                            "end": end_dt.isoformat(),
                            "success": False,
                            "message": error_msg,
                        }
                    )
                except Exception as e:
                    # Catch other errors
                    error_msg = str(e)
                    logs.append(
                        {
                            "symbol": task["symbol"],
                            "interval": task["interval"],
                            "reason": task["reason"],
                            "start": start_dt.isoformat(),
                            "end": end_dt.isoformat(),
                            "success": False,
                            "message": error_msg,
                        }
                    )
        return logs


