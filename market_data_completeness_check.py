"""
Market Data Completeness Checker

This script analyzes 1-minute OHLCV market data completeness for instruments
stored in Supabase/PostgreSQL tables.

Asset Coverage Rules:
- Commodities, Currencies, Stocks: Last 2 years of 1-minute data required
- Indices: Last 1 year of 1-minute data required

Features:
- Enumerates all instruments from database tables
- Generates expected timestamp sequences based on trading hours
- Compares expected vs actual data
- Detects gaps, missing minutes, and coverage issues
- Computes completeness metrics
- Returns structured results for UI display
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import pytz
from tradingagents.database.config import get_supabase


# Asset class to table mapping
ASSET_CLASS_TABLES = {
    "Commodities": "market_data_commodities_1min",
    "Currencies": "market_data_currencies_1min",
    "Stocks": "market_data_stocks_1min",
    "Indices": "market_data_indices_1min"
}

# Required coverage periods (in days)
REQUIRED_COVERAGE_DAYS = {
    "Commodities": 730,  # 2 years
    "Currencies": 730,    # 2 years
    "Stocks": 730,        # 2 years
    "Indices": 365        # 1 year
}


class TradingHoursCalendar:
    """
    Trading hours calendar for different asset classes.
    Designed to be pluggable - can be extended with market calendar libraries later.
    """
    
    @staticmethod
    def is_trading_minute(timestamp: datetime, asset_class: str) -> bool:
        """
        Determine if a given UTC timestamp is within trading hours for the asset class.
        
        Args:
            timestamp: UTC datetime
            asset_class: One of "Commodities", "Currencies", "Stocks", "Indices"
        
        Returns:
            True if the minute should have trading data, False otherwise
        """
        # Convert to UTC if not already
        if timestamp.tzinfo is None:
            timestamp = pytz.UTC.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(pytz.UTC)
        
        # Get day of week (0=Monday, 6=Sunday)
        weekday = timestamp.weekday()
        hour = timestamp.hour
        minute = timestamp.minute
        
        # 24/7 markets: Commodities and Currencies trade from Sunday 22:00 UTC to Friday 22:00 UTC
        if asset_class in ["Commodities", "Currencies"]:
            # Saturday: No trading
            if weekday == 5:  # Saturday
                return False
            
            # Sunday: Trading starts at 22:00 UTC
            if weekday == 6:  # Sunday
                return hour >= 22
            
            # Monday-Thursday: Full trading (24 hours)
            if weekday < 4:  # Monday (0) to Thursday (3)
                return True
            
            # Friday: Trading ends at 22:00 UTC
            if weekday == 4:  # Friday
                return hour < 22
            
            return False
        
        # US Market Hours (for Stocks and Indices)
        # Regular trading: 9:30 AM - 4:00 PM ET
        # Extended hours: 4:00 AM - 9:30 AM ET (pre-market) and 4:00 PM - 8:00 PM ET (after-hours)
        # ET is UTC-5 (EST) or UTC-4 (EDT) - using UTC-4 (EDT) as default for simplicity
        # 4:00 AM ET = 8:00 UTC, 8:00 PM ET = 0:00 UTC next day
        # So trading hours: 8:00 UTC to 0:00 UTC next day (wrapping)
        # Regular hours: 13:30 UTC (9:30 AM ET) to 20:00 UTC (4:00 PM ET)
        
        if asset_class in ["Stocks", "Indices"]:
            # US Eastern Time market hours (using EDT = UTC-4)
            # Extended hours: 4:00 AM ET - 9:30 AM ET = 8:00 UTC - 13:30 UTC
            # Regular hours: 9:30 AM ET - 4:00 PM ET = 13:30 UTC - 20:00 UTC
            # Extended hours: 4:00 PM ET - 8:00 PM ET = 20:00 UTC - 0:00 UTC next day
            # Total: 8:00 UTC to 0:00 UTC next day (16 hours)
            # Note: Hour 0 (midnight) is the end of previous day's session
            
            if hour >= 8:  # 8:00 UTC to 23:59 UTC
                return True
            elif hour == 0:  # 0:00 UTC (end of previous day's session)
                return True
            # Gap: 1:00 UTC to 7:59 UTC (7 hours)
            return False
        
        return False
    
    @staticmethod
    def generate_expected_timestamps(
        start_dt: datetime,
        end_dt: datetime,
        asset_class: str
    ) -> Set[datetime]:
        """
        Generate set of expected 1-minute timestamps for the given date range.
        
        Args:
            start_dt: Start datetime (UTC)
            end_dt: End datetime (UTC)
            asset_class: Asset class to determine trading hours
        
        Returns:
            Set of datetime objects representing expected trading minutes
        """
        expected = set()
        current = start_dt.replace(second=0, microsecond=0)
        
        while current <= end_dt:
            if TradingHoursCalendar.is_trading_minute(current, asset_class):
                expected.add(current)
            current += timedelta(minutes=1)
        
        return expected


def enumerate_instruments(asset_class: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Enumerate all instruments from market data tables.
    
    Args:
        asset_class: Optional filter for specific asset class
    
    Returns:
        Dictionary mapping asset_class -> list of symbols
    """
    sb = get_supabase()
    if not sb:
        return {}
    
    instruments = {}
    
    classes_to_check = [asset_class] if asset_class else ASSET_CLASS_TABLES.keys()
    
    for ac in classes_to_check:
        if ac not in ASSET_CLASS_TABLES:
            continue
        
        table_name = ASSET_CLASS_TABLES[ac]
        try:
            # Get distinct symbols from table
            result = sb.table(table_name).select("symbol").execute()
            
            if result.data:
                symbols = list(set([row["symbol"] for row in result.data]))
                symbols.sort()
                instruments[ac] = symbols
            else:
                instruments[ac] = []
        except Exception as e:
            print(f"Error enumerating instruments from {table_name}: {e}")
            instruments[ac] = []
    
    return instruments


def get_actual_timestamps(
    symbol: str,
    asset_class: str,
    start_dt: datetime,
    end_dt: datetime
) -> Set[datetime]:
    """
    Fetch actual timestamps from database for a symbol.
    
    Args:
        symbol: Symbol to check
        asset_class: Asset class
        start_dt: Start datetime (UTC)
        end_dt: End datetime (UTC)
    
    Returns:
        Set of datetime objects representing actual timestamps in database
    """
    sb = get_supabase()
    if not sb:
        return set()
    
    table_name = ASSET_CLASS_TABLES.get(asset_class)
    if not table_name:
        return set()
    
    try:
        # Query timestamps for the symbol in the date range
        start_str = start_dt.isoformat()
        end_str = end_dt.isoformat()
        
        result = sb.table(table_name)\
            .select("timestamp")\
            .eq("symbol", symbol)\
            .gte("timestamp", start_str)\
            .lte("timestamp", end_str)\
            .order("timestamp", desc=False)\
            .execute()
        
        if not result.data:
            return set()
        
        # Convert to datetime objects (UTC)
        timestamps = set()
        for row in result.data:
            ts_str = row["timestamp"]
            # Parse ISO format timestamp
            if isinstance(ts_str, str):
                # Handle timezone-aware strings
                if ts_str.endswith('Z'):
                    ts_str = ts_str[:-1] + '+00:00'
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            else:
                dt = ts_str
            
            # Ensure UTC timezone
            if dt.tzinfo is None:
                dt = pytz.UTC.localize(dt)
            else:
                dt = dt.astimezone(pytz.UTC)
            
            # Round to minute (remove seconds/microseconds)
            dt = dt.replace(second=0, microsecond=0)
            timestamps.add(dt)
        
        return timestamps
    
    except Exception as e:
        print(f"Error fetching timestamps for {symbol} from {table_name}: {e}")
        return set()


def detect_gaps(actual_timestamps: Set[datetime], expected_timestamps: Set[datetime]) -> List[Tuple[datetime, datetime]]:
    """
    Detect gaps in the data (missing consecutive minutes).
    
    Args:
        actual_timestamps: Set of actual timestamps
        expected_timestamps: Set of expected timestamps
    
    Returns:
        List of (gap_start, gap_end) tuples
    """
    gaps = []
    
    # Find missing timestamps
    missing = expected_timestamps - actual_timestamps
    
    if not missing:
        return gaps
    
    # Sort missing timestamps
    missing_sorted = sorted(missing)
    
    # Group consecutive missing timestamps into gaps
    if missing_sorted:
        gap_start = missing_sorted[0]
        gap_end = gap_start
        
        for i in range(1, len(missing_sorted)):
            if missing_sorted[i] == gap_end + timedelta(minutes=1):
                # Consecutive, extend gap
                gap_end = missing_sorted[i]
            else:
                # Gap ended, save it and start new gap
                gaps.append((gap_start, gap_end))
                gap_start = missing_sorted[i]
                gap_end = gap_start
        
        # Add final gap
        gaps.append((gap_start, gap_end))
    
    return gaps


def identify_missing_weeks(gaps: List[Tuple[datetime, datetime]], asset_class: str) -> List[Dict]:
    """
    Identify missing trading weeks from gaps and provide ingestion suggestions.
    
    Args:
        gaps: List of (gap_start, gap_end) tuples
        asset_class: Asset class to determine trading week boundaries
    
    Returns:
        List of dictionaries with week information and ingestion suggestions
    """
    missing_weeks = []
    
    if asset_class not in ["Commodities", "Currencies"]:
        return missing_weeks
    
    for gap_start, gap_end in gaps:
        gap_duration_minutes = int((gap_end - gap_start).total_seconds() / 60) + 1
        
        # Check if this looks like a full trading week (approximately 7200 minutes)
        # Allow some tolerance: 7000-7500 minutes
        if 7000 <= gap_duration_minutes <= 7500:
            # This is likely a full trading week
            # For commodities/currencies: Sunday 22:00 UTC to Friday 21:59 UTC
            # Ingestion should start from Sunday 22:00 and go to Friday 22:00 (inclusive)
            ingestion_start = gap_start  # Sunday 22:00
            ingestion_end = gap_end + timedelta(minutes=1)  # Friday 22:00 (add 1 minute to include 21:59)
            
            missing_weeks.append({
                "week_start": gap_start.isoformat(),
                "week_end": gap_end.isoformat(),
                "ingestion_start": ingestion_start.isoformat(),
                "ingestion_end": ingestion_end.isoformat(),
                "duration_minutes": gap_duration_minutes,
                "is_full_week": True,
                "week_label": f"{gap_start.strftime('%Y-%m-%d')} to {gap_end.strftime('%Y-%m-%d')}"
            })
        elif gap_duration_minutes > 100:  # Significant gaps (more than 100 minutes)
            # Partial gap - still suggest ingestion
            missing_weeks.append({
                "week_start": gap_start.isoformat(),
                "week_end": gap_end.isoformat(),
                "ingestion_start": gap_start.isoformat(),
                "ingestion_end": gap_end.isoformat(),
                "duration_minutes": gap_duration_minutes,
                "is_full_week": False,
                "week_label": f"{gap_start.strftime('%Y-%m-%d %H:%M')} to {gap_end.strftime('%Y-%m-%d %H:%M')}"
            })
    
    return missing_weeks


def normalize_gold_symbol(symbol: str) -> str:
    """
    Normalize Gold symbol to standard format C:XAUUSD for database consistency.
    
    Args:
        symbol: Symbol string (can be ^XAUUSD, C:XAUUSD, XAUUSD, GOLD, etc.)
    
    Returns:
        Normalized symbol: C:XAUUSD for Gold-related symbols, otherwise returns as-is
    """
    symbol_upper = symbol.upper().strip()
    
    # Normalize all Gold symbol variants to C:XAUUSD
    if "XAU" in symbol_upper or symbol_upper == "GOLD" or symbol_upper in ["^XAUUSD", "XAUUSD", "C:XAUUSD"]:
        return "C:XAUUSD"
    
    return symbol


def get_latest_ingested_timestamp(symbol: str, asset_class: str) -> Optional[datetime]:
    """
    Get the latest ingested timestamp for a symbol from the database.
    
    Args:
        symbol: Symbol to check
        asset_class: Asset class
    
    Returns:
        Latest timestamp as datetime (UTC) or None if no data found
    """
    sb = get_supabase()
    if not sb:
        return None
    
    table_name = ASSET_CLASS_TABLES.get(asset_class)
    if not table_name:
        return None
    
    try:
        # Normalize symbol first (especially for Gold)
        normalized_symbol = normalize_gold_symbol(symbol) if asset_class == "Commodities" else symbol
        
        # Try multiple symbol formats
        symbols_to_try = [normalized_symbol]
        if normalized_symbol == "C:XAUUSD":
            symbols_to_try.extend(["^XAUUSD", "GOLD", "XAUUSD"])
        elif normalized_symbol == "^SPX":
            symbols_to_try.extend(["I:SPX", "S&P 500", "SPX", "SPY"])
        
        for try_sym in symbols_to_try:
            try:
                result = sb.table(table_name)\
                    .select("timestamp")\
                    .eq("symbol", try_sym)\
                    .order("timestamp", desc=True)\
                    .limit(1)\
                    .execute()
                
                if result.data and len(result.data) > 0:
                    latest_ts_str = result.data[0].get("timestamp")
                    if latest_ts_str:
                        # Parse ISO format timestamp
                        if isinstance(latest_ts_str, str):
                            if latest_ts_str.endswith('Z'):
                                latest_ts_str = latest_ts_str[:-1] + '+00:00'
                            dt = datetime.fromisoformat(latest_ts_str.replace('Z', '+00:00'))
                        else:
                            dt = latest_ts_str
                        
                        # Ensure UTC timezone
                        if dt.tzinfo is None:
                            dt = pytz.UTC.localize(dt)
                        else:
                            dt = dt.astimezone(pytz.UTC)
                        
                        # Round to minute
                        dt = dt.replace(second=0, microsecond=0)
                        return dt
            except Exception:
                continue
        
        return None
    except Exception as e:
        print(f"Error fetching latest timestamp for {symbol} from {table_name}: {e}")
        return None


def check_completeness(
    symbol: str,
    asset_class: str,
    reference_date: Optional[datetime] = None
) -> Dict:
    """
    Check completeness of 1-minute data for a symbol.
    Checks from 2024-01-05 00:00:00 up to the last ingested timestamp,
    and also checks up to UTC.now() to suggest ingestion.
    
    Args:
        symbol: Symbol to check (will be normalized for Gold to ^XAUUSD)
        asset_class: Asset class (Commodities, Currencies, Stocks, Indices)
        reference_date: Reference date (defaults to now)
    
    Returns:
        Dictionary with completeness metrics and analysis
    """
    # Normalize symbol for consistency (especially Gold)
    if asset_class == "Commodities":
        symbol = normalize_gold_symbol(symbol)
    """
    Check completeness of 1-minute data for a symbol.
    Checks from 2024-01-05 00:00:00 up to the last ingested timestamp,
    and also checks up to UTC.now() to suggest ingestion.
    
    Args:
        symbol: Symbol to check
        asset_class: Asset class (Commodities, Currencies, Stocks, Indices)
        reference_date: Reference date (defaults to now)
    
    Returns:
        Dictionary with completeness metrics and analysis
    """
    if reference_date is None:
        reference_date = datetime.now(pytz.UTC)
    elif reference_date.tzinfo is None:
        reference_date = pytz.UTC.localize(reference_date)
    else:
        reference_date = reference_date.astimezone(pytz.UTC)
    
    # Fixed start date: 2024-01-05 00:00:00 UTC
    start_dt = datetime(2024, 1, 5, 0, 0, 0, tzinfo=pytz.UTC)
    
    # Get latest ingested timestamp
    last_ingested = get_latest_ingested_timestamp(symbol, asset_class)
    
    # Current UTC time with safety buffer (15 minutes)
    now_utc = datetime.now(pytz.UTC)
    end_dt_check = now_utc - timedelta(minutes=15)
    
    # Use last_ingested as end_dt for completeness check, or end_dt_check if no data
    if last_ingested:
        end_dt = last_ingested
    else:
        end_dt = end_dt_check
    
    # Generate expected timestamps from start_dt to end_dt
    expected_timestamps = TradingHoursCalendar.generate_expected_timestamps(
        start_dt, end_dt, asset_class
    )
    
    # Get actual timestamps
    actual_timestamps = get_actual_timestamps(symbol, asset_class, start_dt, end_dt)
    
    # Calculate metrics
    expected_count = len(expected_timestamps)
    actual_count = len(actual_timestamps)
    missing_count = expected_count - actual_count
    completeness_pct = (actual_count / expected_count * 100) if expected_count > 0 else 0.0
    
    # Detect gaps
    gaps = detect_gaps(actual_timestamps, expected_timestamps)
    
    # Find first and last missing timestamps
    missing_timestamps = expected_timestamps - actual_timestamps
    first_missing = min(missing_timestamps) if missing_timestamps else None
    last_missing = max(missing_timestamps) if missing_timestamps else None
    
    # Identify missing weeks and provide ingestion suggestions
    missing_weeks = []
    if asset_class in ["Commodities", "Currencies"] and gaps:
        missing_weeks = identify_missing_weeks(gaps, asset_class)
    
    # Analyze gap patterns for commodities/currencies (weekly trading cycles)
    gap_analysis = None
    if asset_class in ["Commodities", "Currencies"] and gaps:
        full_week_count = sum(1 for week in missing_weeks if week.get("is_full_week", False))
        gap_analysis = {
            "total_gaps": len(gaps),
            "full_week_gaps": full_week_count,
            "partial_gaps": len(gaps) - full_week_count,
            "missing_weeks": missing_weeks[:20]  # Limit to first 20 for display
        }
    
    # Determine status
    if completeness_pct >= 99.5:
        status = "✅ Complete"
    elif completeness_pct >= 90.0:
        status = "⚠️ Partial"
    else:
        status = "❌ Incomplete"
    
    # Get date range of actual data
    actual_sorted = sorted(actual_timestamps) if actual_timestamps else []
    first_actual = actual_sorted[0] if actual_sorted else None
    last_actual = actual_sorted[-1] if actual_sorted else None
    
    # Calculate gap from last ingested to now (for ingestion suggestion)
    gap_to_now_minutes = None
    gap_to_now_hours = None
    gap_to_now_days = None
    if last_ingested:
        gap_delta = end_dt_check - last_ingested
        gap_to_now_minutes = int(gap_delta.total_seconds() / 60)
        gap_to_now_hours = gap_to_now_minutes / 60
        gap_to_now_days = gap_delta.days
    
    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "status": status,
        "completeness_percentage": round(completeness_pct, 2),
        "expected_minutes": expected_count,
        "actual_minutes": actual_count,
        "missing_minutes": missing_count,
        "date_range_checked": {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "days": (end_dt - start_dt).days
        },
        "date_range_actual": {
            "start": first_actual.isoformat() if first_actual else None,
            "end": last_actual.isoformat() if last_actual else None
        },
        "last_ingested": last_ingested.isoformat() if last_ingested else None,
        "current_utc": now_utc.isoformat(),
        "gap_to_now": {
            "minutes": gap_to_now_minutes,
            "hours": round(gap_to_now_hours, 2) if gap_to_now_hours else None,
            "days": gap_to_now_days
        },
        "gaps": [
            {"start": gap[0].isoformat(), "end": gap[1].isoformat(), 
             "duration_minutes": int((gap[1] - gap[0]).total_seconds() / 60) + 1}
            for gap in gaps
        ],
        "first_missing": first_missing.isoformat() if first_missing else None,
        "last_missing": last_missing.isoformat() if last_missing else None,
        "gap_count": len(gaps),
        "gap_analysis": gap_analysis,
        "missing_weeks": missing_weeks[:50]  # Include up to 50 missing weeks for ingestion suggestions
    }


def check_all_instruments(
    asset_class: Optional[str] = None,
    reference_date: Optional[datetime] = None
) -> Dict[str, Dict]:
    """
    Check completeness for all instruments.
    
    Args:
        asset_class: Optional filter for specific asset class
        reference_date: Reference date for calculating required range
    
    Returns:
        Dictionary mapping "symbol@asset_class" -> completeness results
    """
    instruments = enumerate_instruments(asset_class)
    results = {}
    
    for ac, symbols in instruments.items():
        for symbol in symbols:
            key = f"{symbol}@{ac}"
            try:
                result = check_completeness(symbol, ac, reference_date)
                results[key] = result
            except Exception as e:
                print(f"Error checking {symbol} ({ac}): {e}")
                results[key] = {
                    "symbol": symbol,
                    "asset_class": ac,
                    "status": "❌ Error",
                    "error": str(e)
                }
    
    return results


def explain_missing_data(result: Dict) -> List[str]:
    """
    Generate AI explanations for missing data.
    
    Args:
        result: Completeness check result dictionary
    
    Returns:
        List of explanation strings
    """
    explanations = []
    asset_class = result.get("asset_class", "")
    completeness = result.get("completeness_percentage", 0)
    gaps = result.get("gaps", [])
    
    # Explain missing data reasons
    if completeness < 100:
        if asset_class in ["Stocks", "Indices"]:
            explanations.append(
                "📅 **Market Hours**: Stocks and indices only trade during US market hours "
                "(typically 9:30 AM - 4:00 PM ET, with extended hours). Missing data outside these hours is expected."
            )
            explanations.append(
                "🏖️ **Weekends & Holidays**: No trading on weekends. US market holidays also cause gaps."
            )
        elif asset_class in ["Commodities", "Currencies"]:
            explanations.append(
                "🌍 **24/7 Trading**: Commodities and currencies trade continuously from Sunday evening to Friday evening (UTC). "
                "Missing data may indicate provider gaps or ingestion issues."
            )
        
        if gaps:
            explanations.append(
                f"🔍 **Detected Gaps**: Found {len(gaps)} gap(s) in the data. These may be due to: "
                "market holidays, data provider outages, ingestion failures, or Polygon plan limitations."
            )
        
        if completeness < 90:
            explanations.append(
                "⚠️ **Low Completeness**: Completeness below 90% may indicate significant data quality issues. "
                "Consider re-ingesting data or checking for provider limitations."
            )
    
    return explanations


if __name__ == "__main__":
    # Example usage
    print("Market Data Completeness Checker")
    print("=" * 60)
    
    # Check a specific instrument
    result = check_completeness("^XAUUSD", "Commodities")
    print(f"\nCompleteness Check for {result['symbol']} ({result['asset_class']}):")
    print(f"Status: {result['status']}")
    print(f"Completeness: {result['completeness_percentage']}%")
    print(f"Expected: {result['expected_minutes']} minutes")
    print(f"Actual: {result['actual_minutes']} minutes")
    print(f"Missing: {result['missing_minutes']} minutes")
    print(f"Gaps: {result['gap_count']}")
    
    explanations = explain_missing_data(result)
    if explanations:
        print("\nExplanations:")
        for exp in explanations:
            print(f"  - {exp}")

