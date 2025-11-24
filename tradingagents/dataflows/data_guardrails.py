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
        if self.db_start is None or self.db_end is None:
            return "missing"
        if self.head_gap_minutes <= 0 and self.tail_gap_minutes <= 0:
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
            "db_start": self.db_start.isoformat() if self.db_start else None,
            "db_end": self.db_end.isoformat() if self.db_end else None,
            "rows": self.row_count,
            "head_gap_min": round(self.head_gap_minutes, 1),
            "tail_gap_min": round(self.tail_gap_minutes, 1),
            "status": self.status(),
        }


logger = logging.getLogger(__name__)


class DataCoverageService:
    """Ensures the database holds the minimum required historical windows."""

    def __init__(self, reference_dt: Optional[datetime] = None):
        self.supabase = get_supabase()
        self.reference_dt = reference_dt

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

    def _requirements(self) -> Dict[str, CoverageRequirement]:
        """
        Build requirements for each interval based on the provided specification:
            - 1 minute: Aug 1 2025 → now minus 15 minutes (EAT reference)
            - 5 minute: Oct 1 2023 → now minus 15 minutes
            - Daily: Oct 1 2023 → previous EAT day close
        """
        now_eat = self._now_eat()
        last_15_cutoff = (now_eat - timedelta(minutes=15)).astimezone(UTC)
        prev_day_close = (
            (now_eat - timedelta(days=1))
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .astimezone(UTC)
        )

        requirements = {
            "1min": CoverageRequirement(
                interval="1min",
                label="1-min bars (Aug 1 2025 → now-15m)",
                start=datetime(2025, 8, 1, tzinfo=UTC),
                end=last_15_cutoff,
                chunk_days=3,
            ),
            "5min": CoverageRequirement(
                interval="5min",
                label="5-min bars (Oct 1 2023 → now-15m)",
                start=datetime(2023, 10, 1, tzinfo=UTC),
                end=last_15_cutoff,
                chunk_days=30,
            ),
            "1d": CoverageRequirement(
                interval="1d",
                label="Daily bars (Oct 1 2023 → previous day)",
                start=datetime(2023, 10, 1, tzinfo=UTC),
                end=prev_day_close,
                chunk_days=365,
            ),
        }

        # Guard against negative ranges (e.g., if prev_day_close < start)
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

        try:
            table, time_field, _ = self._table_fields(interval)
            earliest = (
                self.supabase.table(table)
                .select(time_field)
                .eq("symbol", symbol)
                .order(time_field, desc=False)
                .limit(1)
                .execute()
            )
            latest = (
                self.supabase.table(table)
                .select(time_field)
                .eq("symbol", symbol)
                .order(time_field, desc=True)
                .limit(1)
                .execute()
            )
        except Exception as exc:  # pragma: no cover - network safeguard
            logger.warning(
                "Coverage guardrail failed to query bounds for %s (%s): %s",
                symbol,
                interval,
                exc,
            )
            return None, None

        earliest_val = (
            earliest.data[0].get(time_field) if getattr(earliest, "data", None) else None
        )
        latest_val = (
            latest.data[0].get(time_field) if getattr(latest, "data", None) else None
        )
        return self._parse_ts(earliest_val), self._parse_ts(latest_val)

    def _count_rows(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> Optional[int]:
        if not self.supabase:
            return None

        table, time_field, is_date = self._table_fields(interval)
        start_value = start.date().isoformat() if is_date else start.isoformat()
        end_value = end.date().isoformat() if is_date else end.isoformat()

        try:
            resp = (
                self.supabase.table(table)
                .select(time_field, count="exact")
                .eq("symbol", symbol)
                .gte(time_field, start_value)
                .lte(time_field, end_value)
                .limit(1)
                .execute()
            )
            return getattr(resp, "count", None)
        except Exception as exc:  # pragma: no cover - network safeguard
            logger.warning(
                "Coverage guardrail failed to count rows for %s (%s): %s",
                symbol,
                interval,
                exc,
            )
            return None

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
        Returns execution logs for UI display.
        """
        if not pipeline:
            return [{"status": "error", "message": "Pipeline not provided"}]

        records = records or []
        logs: List[Dict[str, object]] = []
        for record in records:
            for task in self._create_backfill_tasks(record):
                start_dt = task["start"].replace(tzinfo=None)
                end_dt = task["end"].replace(tzinfo=None)
                if end_dt <= start_dt:
                    continue

                success = pipeline.ingest_historical_data(
                    symbol=task["symbol"],
                    interval="daily" if task["interval"] == "1d" else task["interval"],
                    start_date=start_dt,
                    end_date=end_dt,
                    chunk_days=task["chunk_days"],
                    resume_from_latest=task["resume_from_latest"],
                )
                logs.append(
                    {
                        "symbol": task["symbol"],
                        "interval": task["interval"],
                        "reason": task["reason"],
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat(),
                        "success": success,
                    }
                )
        return logs


