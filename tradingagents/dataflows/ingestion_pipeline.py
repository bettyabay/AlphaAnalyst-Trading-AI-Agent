"""
Data ingestion pipeline for AlphaAnalyst Trading AI Agent
"""
import os
import time
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv

from ..database.config import get_supabase
from ..config.watchlist import WATCHLIST_STOCKS, get_watchlist_symbols
from .polygon_integration import PolygonDataClient

load_dotenv()


def get_1min_table_name_for_symbol(symbol: str) -> str:
    """
    Determine the appropriate 1-minute market data table based on symbol format.
    
    Args:
        symbol: Symbol string (e.g., 'AAPL', 'XAUUSD', 'I:SPX', 'C:EURUSD')
    
    Returns:
        Table name string:
        - 'market_data_stocks_1min' for regular stocks
        - 'market_data_commodities_1min' for commodities (symbols with '*' or commodities like XAUUSD)
        - 'market_data_indices_1min' for indices (symbols starting with '^' or 'I:')
        - 'market_data_currencies_1min' for currencies (symbols with '/' or 'C:' prefix)
    """
    symbol_upper = symbol.upper() if symbol else ""
    
    # Check for commodities (contains * or common commodity symbols)
    # Check this BEFORE currencies to ensure C:XAUUSD goes to commodities
    if "*" in symbol_upper or "XAU" in symbol_upper or "GOLD" in symbol_upper:
        return "market_data_commodities_1min"
    
    # Check for indices first (I: prefix or ^ prefix)
    elif symbol_upper.startswith("I:") or symbol_upper.startswith("^"):
        return "market_data_indices_1min"
    # Check for currencies (C: prefix or / separator)
    elif symbol_upper.startswith("C:") or "/" in symbol_upper:
        return "market_data_currencies_1min"
    # Default to stocks
    else:
        return "market_data_stocks_1min"


def convert_instrument_to_polygon_symbol(category: str, instrument: str) -> str:
    """
    Convert instrument name/category to Polygon API symbol format.
    
    Args:
        category: Category name (e.g., 'Commodities', 'Indices', 'Currencies', 'Stocks')
        instrument: Instrument name/symbol (e.g., 'GOLD', 'S&P 500', 'EUR/USD', 'AAPL')
    
    Returns:
        Polygon symbol string (e.g., 'XAUUSD', 'I:SPX', 'C:EURUSD', 'AAPL')
    """
    instrument_upper = instrument.upper().strip()
    category_upper = category.upper().strip()
    
    # Commodities
    if category_upper == "COMMODITIES":
        if "GOLD" in instrument_upper:
            return "C:XAUUSD"  # Gold spot price (treated as currency pair by Polygon)
        return instrument_upper
    
    # Indices
    elif category_upper == "INDICES":
        # NOTE: For 1-minute data, Polygon doesn't support indices (I:SPX, etc.)
        # The universal_ingestion function will auto-convert I:SPX to SPY
        # But we still return I:SPX here for consistency, and let the ingestion layer handle conversion
        if "S&P" in instrument_upper or "SPX" in instrument_upper or "SP500" in instrument_upper:
            return "I:SPX"  # S&P 500 index (will be converted to SPY for minute data)
        elif instrument_upper.startswith("^"):
            symbol_part = instrument_upper[1:].strip()
            if symbol_part:  # Make sure there's something after the ^
                return f"I:{symbol_part}"
            else:
                return instrument_upper  # Return as-is if invalid
        elif instrument_upper.startswith("I:"):
            # Already has I: prefix, return as-is
            return instrument_upper
        elif len(instrument_upper) > 1:  # Only add I: prefix if there's more than one character
            return f"I:{instrument_upper}"
        else:
            # Single character like "I" - return as-is (user should enter full symbol)
            return instrument_upper
    
    # Currencies
    elif category_upper == "CURRENCIES":
        if "/" in instrument_upper:
            return f"C:{instrument_upper.replace('/', '')}"  # EUR/USD -> C:EURUSD
        elif not instrument_upper.startswith("C:"):
            return f"C:{instrument_upper}"
        return instrument_upper
    
    # Stocks - return as is
    else:
        return instrument_upper


class DataIngestionPipeline:
    """Data ingestion pipeline for market data"""
    
    def __init__(self):
        self.polygon_client = PolygonDataClient()
        self.supabase = get_supabase()
        self.high_rate_plan = os.getenv("POLYGON_PREMIUM_PLAN", os.getenv("POLYGON_PAID_PLAN", "false")).lower() in {"1", "true", "yes", "premium"}
        
        # Polygon Rate Limit Throttle
        self._last_polygon_call = 0
        # Safe default for paid/starter plans (5 calls/min = 12s interval)
        # For free plan, it's 5 calls/min. For Starter, unlimited but practically rate limited.
        # We'll use 12s as a safe baseline to avoid hitting 429s.
        self._polygon_min_interval = 12

    def _polygon_throttle(self):
        """Ensure minimum interval between Polygon API calls"""
        elapsed = time.time() - self._last_polygon_call
        if elapsed < self._polygon_min_interval:
            sleep_time = self._polygon_min_interval - elapsed
            print(f"⏳ Throttling Polygon call for {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        self._last_polygon_call = time.time()
        
    def initialize_stocks(self) -> bool:
        """Initialize stocks in database"""
        try:
            if not self.supabase:
                print("Supabase not configured. Skipping stock initialization.")
                return False
                
            # For Supabase, we don't need to pre-create stocks
            # They'll be created automatically when we insert market data
            print("Stock initialization completed successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing stocks: {e}")
            return False
    
    def ingest_historical_data(
        self,
        symbol: str,
        days_back: int = 1825,
        interval: str = "daily",
        chunk_days: Optional[int] = None,
        max_retries: int = 5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resume_from_latest: bool = True,
        target_timezone: Optional[str] = None,
    ) -> bool:
        """Ingest historical data for a symbol, with chunked fetching and retry/backoff for intraday data.

        interval: 'daily' (default), '5min' to ingest 5-minute bars, or '1min' to ingest 1-minute bars.
        chunk_days: number of days per API call chunk (for intraday data)
        max_retries: max retries per chunk
        start_date: optional explicit start datetime (inclusive)
        end_date: optional explicit end datetime (inclusive)
        target_timezone: optional timezone string (e.g. 'Asia/Dubai') to convert data to. Default is UTC (None).
        
        Note: Polygon API provides max 2 years of 1-minute data. For 1min interval, days_back defaults to 730 (2 years).
        """
        # Adjust default days_back for 1-minute data (Polygon only provides 2 years)
        if interval in ("1min", "1-minute", "1m") and days_back == 1825:
            days_back = 730  # 2 years for 1-minute data
        try:
            if not self.supabase:
                print("Supabase not configured. Cannot ingest data.")
                return False

            def _normalize_date(value, default):
                if value is None:
                    return default
                if isinstance(value, datetime):
                    return value
                if isinstance(value, str):
                    try:
                        return datetime.fromisoformat(value)
                    except ValueError:
                        try:
                            return datetime.strptime(value, "%Y-%m-%d")
                        except ValueError:
                            raise ValueError(f"Invalid date format: {value}")
                raise ValueError(f"Unsupported date type: {type(value)}")

            try:
                # Determine if this is intraday data
                is_intraday = interval in ("5min", "5-minute", "5m", "1min", "1-minute", "1m")

                # Enforce coverage policy:
                #   - Intraday: ingest up to "now - 15m" (EAT buffer; UTC equivalent)
                #   - Daily   : ingest up to previous trading day close
                now_utc = datetime.utcnow()
                intraday_cutoff = now_utc - timedelta(minutes=15)
                prev_trading_day = self._previous_trading_day(now_utc.date())
                daily_default_end = datetime.combine(prev_trading_day, datetime.min.time())

                if end_date is None:
                    default_end = intraday_cutoff if is_intraday else daily_default_end
                else:
                    default_end = None
                
                end_date = _normalize_date(end_date, default_end)
                start_date = _normalize_date(start_date, end_date - timedelta(days=days_back))
            except ValueError as date_error:
                print(f"Invalid date provided: {date_error}")
                return False

            # Normalize dates: truncate start_date to midnight, handle end_date based on data type
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if is_intraday:
                # Clamp intraday end to the policy cutoff (now - 15m, UTC)
                now = datetime.utcnow()
                policy_cutoff = now - timedelta(minutes=15)
                if end_date > policy_cutoff:
                    end_date = policy_cutoff

                if end_date.date() < now.date():
                    # Past date: truncate to midnight
                    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
                elif end_date.date() == now.date():
                    # Today: keep the time component (already clamped), unless it was midnight
                    if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0:
                        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

                # Always re-fetch the recent window to close any gaps caused by partial days
                # BUT ONLY IF we are near the end date (i.e., updating recent data)
                # If we are fetching historical data (start_date is far in the past), do not override start_date
                recent_window_hours = 6
                recent_window_start = end_date - timedelta(hours=recent_window_hours)
                
                # Only apply recent window logic if the calculated start_date is very close to end_date
                # This prevents historical backfills (e.g. 5 years) from being truncated to just the last 6 hours
                if resume_from_latest and (end_date - start_date).days < 2:
                    print(f"ℹ️ Expanding intraday window to re-fetch last {recent_window_hours}h for {symbol} ({recent_window_start.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')})")
                    start_date = recent_window_start.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                # Daily data: always end on the previous trading day (midnight)
                prev_td = self._previous_trading_day(datetime.utcnow().date())
                # If user requested a future/too-recent date, clamp to previous trading day
                if end_date.date() > prev_td:
                    end_date = datetime.combine(prev_td, datetime.min.time())
                else:
                    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

            if start_date > end_date:
                print(f"Start date {start_date} is after end date {end_date}. Cannot ingest.")
                return False

            if interval in ("5min", "5-minute", "5m"):
                target_table = "market_data_5min"
                multiplier = 5
            elif interval in ("1min", "1-minute", "1m"):
                # Route to appropriate table based on symbol format
                # For 1min data, use symbol-based routing to determine correct table
                target_table = get_1min_table_name_for_symbol(symbol)
                multiplier = 1
            else:
                target_table = None
                multiplier = None
            
            if target_table and multiplier:
                chunk_days = self._resolve_chunk_days(interval, chunk_days)
                # For intraday data (5-minute and 1-minute): Check if we should resume from existing data
                should_resume = resume_from_latest and interval in ("5min", "5-minute", "5m", "1min", "1-minute", "1m")
                if should_resume:
                    try:
                        # Get the latest timestamp for this symbol
                        latest_result = self.supabase.table(target_table)\
                            .select("timestamp")\
                            .eq("symbol", symbol)\
                            .order("timestamp", desc=True)\
                            .limit(1)\
                            .execute()
                        
                        if latest_result.data and len(latest_result.data) > 0:
                            latest_ts_str = latest_result.data[0].get("timestamp")
                            if latest_ts_str:
                                try:
                                    if isinstance(latest_ts_str, str):
                                        latest_ts = datetime.fromisoformat(latest_ts_str.replace('Z', '+00:00'))
                                    else:
                                        latest_ts = latest_ts_str
                                    
                                    # Make latest_ts timezone-naive if it has timezone
                                    if latest_ts.tzinfo is not None:
                                        latest_ts = latest_ts.replace(tzinfo=None)
                                    
                                    # Resume from the next bar after the latest timestamp
                                    delta_minutes = multiplier if multiplier else 0
                                    resume_ts = latest_ts + timedelta(minutes=delta_minutes)
                                    resume_date = resume_ts.replace(hour=0, minute=0, second=0, microsecond=0)
                                    
                                    # Always resume from existing data if it exists and is before end_date
                                    # This ensures we continue from where data stops, regardless of initial start_date
                                    # Use UTC to match database timezone
                                    now = datetime.utcnow()
                                    interval_label = "1-min" if interval in ("1min", "1-minute", "1m") else "5-min"
                                    
                                    print(f"🔍 {symbol} {interval_label} - Latest DB timestamp: {latest_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                                    print(f"🔍 {symbol} {interval_label} - Calculated resume date: {resume_date.strftime('%Y-%m-%d')} (from {resume_ts.strftime('%Y-%m-%d %H:%M:%S')})")
                                    print(f"🔍 {symbol} {interval_label} - End date: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
                                    print(f"🔍 {symbol} {interval_label} - Today: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                                    
                                    # Check if resume timestamp is in the future (shouldn't happen normally, but handle gracefully)
                                    if resume_ts > now:
                                        print(f"⚠️ {symbol} {interval_label} resume ts {resume_ts.strftime('%Y-%m-%d %H:%M:%S')} is in the future (now: {now.strftime('%Y-%m-%d %H:%M:%S')}). Latest DB timestamp: {latest_ts.strftime('%Y-%m-%d %H:%M:%S')}.")
                                        print(f"{symbol} already has {interval} data up to {latest_ts.strftime('%Y-%m-%d %H:%M:%S')}, which is in the future. No new data to fetch. Skipping.")
                                        return True  # Already complete (or has future dates)
                                    
                                    # Check if resume_date is before or equal to end_date
                                    if resume_ts <= end_date:
                                        print(f"✅ Resuming {symbol} {interval_label} ingestion from {resume_ts.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')} (latest existing: {latest_ts.strftime('%Y-%m-%d %H:%M:%S')})")
                                        # Keep start_date at the beginning of that day to minimize queries while duplicates are filtered
                                        start_date = resume_date
                                    else:
                                        # Data is already up to date (latest timestamp is at or after end_date)
                                        print(f"ℹ️ {symbol} already has {interval} data up to {latest_ts.strftime('%Y-%m-%d %H:%M:%S')}, which is at or after the target end date ({end_date.strftime('%Y-%m-%d %H:%M:%S')}). Skipping.")
                                        return True  # Already complete
                                except Exception as e:
                                    print(f"Warning: Could not parse latest timestamp for {symbol}: {e}. Starting from beginning.")
                    except Exception as e:
                        print(f"Warning: Could not check existing data for {symbol}: {e}. Starting from beginning.")
                
                chunk_delta = timedelta(days=chunk_days)
                chunk_start = start_date
                all_success = True
                total_inserted = 0
                total_skipped = 0
                
                # Log the date range we're processing
                interval_label = "1-min" if interval in ("1min", "1-minute", "1m") else "5-min"
                print(f"📊 Starting {symbol} {interval_label} ingestion: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

                # Track existing timestamps per chunk (only load what we need, when we need it)
                # This avoids loading all records upfront
                existing_timestamps_cache = set()

                chunk_count = 0
                # Use UTC to match database timezone
                now = datetime.utcnow()
                last_timestamp_written = None

                while chunk_start <= end_date:
                    chunk_count += 1
                    chunk_end = min(chunk_start + chunk_delta - timedelta(days=1), end_date)
                    if chunk_end < chunk_start:
                        chunk_end = chunk_start

                    # For the last chunk (today), ensure we include the current time
                    # If chunk_end is today and end_date is also today with a time component,
                    # use end_date to ensure we get data up to "now"
                    if chunk_end.date() == now.date() and end_date.date() == now.date() and end_date > chunk_end:
                        # This is the last chunk for today, use end_date to get data up to current time
                        chunk_end = end_date
                    
                    start_str = chunk_start.strftime("%Y-%m-%d")
                    # For Polygon API, we still use date-only format, but for today's chunk,
                    # Polygon will return all available data up to the current time
                    end_str = chunk_end.strftime("%Y-%m-%d")
                    
                    # Log with time info if it's today's chunk
                    if chunk_end.date() == now.date():
                        print(f"📦 Processing chunk {chunk_count} for {symbol}: {start_str} to {end_str} (up to {chunk_end.strftime('%H:%M:%S')})")
                    else:
                        print(f"📦 Processing chunk {chunk_count} for {symbol}: {start_str} to {end_str}")
                    
                    # Skip if chunk is entirely in the future
                    if chunk_start > now:
                        print(f"⏭️ Skipping future chunk for {symbol}: {start_str} to {end_str} (today: {now.strftime('%Y-%m-%d %H:%M:%S')})")
                        chunk_start = chunk_end + timedelta(days=1)
                        continue

                    # For FX/Commodities (Gold), skip weekends as Polygon returns no data
                    # C:XAUUSD is treated as currency/commodity
                    if (symbol.startswith("C:") or "XAU" in symbol) and interval in ("1min", "1-minute", "1m"):
                        if chunk_start.weekday() >= 5:  # 5=Saturday, 6=Sunday
                            print(f"⏭️ Skipping weekend chunk for {symbol}: {start_str}")
                            chunk_start = chunk_end + timedelta(days=1)
                            continue

                    # Check existing records ONLY for this specific chunk date range
                    chunk_existing_timestamps = set()
                    try:
                        # Query only timestamps in this chunk's date range (exclusive upper bound)
                        chunk_start_iso = chunk_start.isoformat()
                        chunk_end_exclusive = (chunk_end + timedelta(days=1)).isoformat()

                        result = self.supabase.table(target_table)\
                            .select("timestamp")\
                            .eq("symbol", symbol)\
                            .gte("timestamp", chunk_start_iso)\
                            .lt("timestamp", chunk_end_exclusive)\
                            .execute()

                        if result.data:
                            for record in result.data:
                                ts = record.get("timestamp")
                                if ts:
                                    # Handle both string and datetime objects
                                    if isinstance(ts, str):
                                        ts_str = ts
                                    else:
                                        # If it's a datetime object, convert to ISO string
                                        ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                                    
                                    # Normalize for comparison (remove timezone)
                                    ts_normalized = ts_str.split('+')[0].split('Z')[0]
                                    # Also add without microseconds if present
                                    if '.' in ts_normalized:
                                        ts_normalized_no_us = ts_normalized.split('.')[0]
                                    else:
                                        ts_normalized_no_us = ts_normalized
                                    
                                    # Add both formats to cache
                                    chunk_existing_timestamps.add(ts_normalized)
                                    chunk_existing_timestamps.add(ts_normalized_no_us)
                                    existing_timestamps_cache.add(ts_normalized)
                                    existing_timestamps_cache.add(ts_normalized_no_us)
                    except Exception as e:
                        print(f"Warning: Could not check existing records for {symbol} chunk {start_str} to {end_str}: {e}")

                    attempt = 0
                    while attempt < max_retries:
                        try:
                            # Use global throttle for Polygon calls
                            self._polygon_throttle()
                            data = self.polygon_client.get_intraday_data(
                                symbol, 
                                start_str, 
                                end_str, 
                                multiplier=multiplier, 
                                timespan="minute", 
                                max_retries=1  # 🚨 IMPORTANT: Reduced retries, handle rate limits in loop
                            )
                            if data.empty:
                                # Check if dates are in the future (no need to print for future dates)
                                try:
                                    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                                    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
                                    # Use UTC to match database timezone
                                    now = datetime.utcnow()
                                    if start_dt > now or end_dt > now:
                                        # Future dates - skip this chunk
                                        print(f"Skipping future dates for {symbol} {start_str} to {end_str} (today: {now.strftime('%Y-%m-%d')})")
                                        break  # Don't retry empty data for future dates
                                    elif start_dt <= now and end_dt <= now:
                                        # Past dates with no data - might be weekend/holiday
                                        print(f"No trading data for {symbol} {start_str} to {end_str} (likely weekend/holiday)")
                                except Exception as e:
                                    print(f"No data for {symbol} {start_str} to {end_str} (error: {e})")
                                break  # Don't retry empty data

                            # Prepare rows and filter duplicates using chunk-specific existing timestamps
                            rows = []
                            new_rows = []
                            
                            # Prepare timezone objects if needed
                            target_tz = pytz.timezone(target_timezone) if target_timezone else None
                            
                            for _, row in data.iterrows():
                                ts = row["timestamp"]
                                
                                # Handle timezone conversion
                                if target_tz:
                                    # Ensure timestamp is timezone-aware (assume UTC if naive, as Polygon returns UTC)
                                    if ts.tzinfo is None:
                                        ts = ts.replace(tzinfo=pytz.UTC)
                                    # Convert to target timezone
                                    ts = ts.astimezone(target_tz)
                                
                                # Convert to ISO format
                                if hasattr(ts, 'isoformat'):
                                    timestamp_iso = ts.isoformat()
                                else:
                                    timestamp_iso = str(ts)
                                
                                # Normalize for comparison (remove microseconds and timezone for matching)
                                # Database might store with or without timezone, so normalize both
                                timestamp_normalized = timestamp_iso.split('+')[0].split('Z')[0]
                                # Also try without microseconds for matching
                                if '.' in timestamp_normalized:
                                    timestamp_normalized_no_us = timestamp_normalized.split('.')[0]
                                else:
                                    timestamp_normalized_no_us = timestamp_normalized

                                row_data = {
                                    "symbol": symbol,
                                    "timestamp": timestamp_iso,
                                    "open": float(row["open"]),
                                    "high": float(row["high"]),
                                    "low": float(row["low"]),
                                    "close": float(row["close"]),
                                    "volume": int(row["volume"]),
                                }
                                # Only add source if it's NOT a 1-minute table (which don't have source column)
                                if "1min" not in target_table:
                                    row_data["source"] = "polygon"

                                rows.append(row_data)

                                # Check both chunk-specific cache and global cache
                                # Check both with and without microseconds
                                is_duplicate = (
                                    timestamp_normalized in chunk_existing_timestamps or 
                                    timestamp_normalized in existing_timestamps_cache or
                                    timestamp_normalized_no_us in chunk_existing_timestamps or
                                    timestamp_normalized_no_us in existing_timestamps_cache
                                )
                                
                                if not is_duplicate:
                                    new_rows.append(row_data)
                                    # Add to both caches to avoid duplicates (add both formats)
                                    chunk_existing_timestamps.add(timestamp_normalized)
                                    chunk_existing_timestamps.add(timestamp_normalized_no_us)
                                    existing_timestamps_cache.add(timestamp_normalized)
                                    existing_timestamps_cache.add(timestamp_normalized_no_us)

                            # Bulk insert only new records (batch if too many to avoid timeouts)
                            if new_rows:
                                batch_size = 5000  # Insert in batches to avoid timeout
                                try:
                                    if len(new_rows) <= batch_size:
                                        # Use insert instead of upsert since we've already filtered duplicates
                                        result = self.supabase.table(target_table).insert(new_rows).execute()
                                        total_inserted += len(new_rows)
                                        total_skipped += (len(rows) - len(new_rows))
                                        print(f"✅ Inserted {len(new_rows)} new records for {symbol} {start_str} to {end_str} (skipped {len(rows) - len(new_rows)} duplicates)")
                                    else:
                                        # Insert in batches
                                        for i in range(0, len(new_rows), batch_size):
                                            batch = new_rows[i:i + batch_size]
                                            self.supabase.table(target_table).insert(batch).execute()
                                            total_inserted += len(batch)
                                        total_skipped += (len(rows) - len(new_rows))
                                        print(f"✅ Inserted {len(new_rows)} new records for {symbol} {start_str} to {end_str} in {len(range(0, len(new_rows), batch_size))} batches (skipped {len(rows) - len(new_rows)} duplicates)")
                                except Exception as e:
                                    # If insert fails (e.g., duplicate key constraint), filter out duplicates and retry
                                    error_str = str(e).lower()
                                    if "duplicate key" in error_str or "23505" in error_str:
                                        # Extract the duplicate timestamp from error if possible
                                        print(f"Duplicate detected for {symbol} {start_str} to {end_str}. Filtering duplicates and retrying...")
                                        
                                        # Re-check existing records more thoroughly and filter
                                        filtered_rows = []
                                        for row in new_rows:
                                            ts_iso = row["timestamp"]
                                            ts_normalized = ts_iso.split('+')[0].split('Z')[0]
                                            if '.' in ts_normalized:
                                                ts_normalized_no_us = ts_normalized.split('.')[0]
                                            else:
                                                ts_normalized_no_us = ts_normalized
                                            
                                            # Check if this timestamp exists
                                            if (ts_normalized not in chunk_existing_timestamps and 
                                                ts_normalized_no_us not in chunk_existing_timestamps and
                                                ts_normalized not in existing_timestamps_cache and
                                                ts_normalized_no_us not in existing_timestamps_cache):
                                                filtered_rows.append(row)
                                                # Add to cache
                                                chunk_existing_timestamps.add(ts_normalized)
                                                chunk_existing_timestamps.add(ts_normalized_no_us)
                                                existing_timestamps_cache.add(ts_normalized)
                                                existing_timestamps_cache.add(ts_normalized_no_us)
                                        
                                        if filtered_rows:
                                            try:
                                                # Try inserting filtered rows
                                                if len(filtered_rows) <= batch_size:
                                                    self.supabase.table(target_table).insert(filtered_rows).execute()
                                                    total_inserted += len(filtered_rows)
                                                    total_skipped += (len(new_rows) - len(filtered_rows))
                                                    print(f"Inserted {len(filtered_rows)} new records for {symbol} {start_str} to {end_str} after filtering {len(new_rows) - len(filtered_rows)} duplicates")
                                                else:
                                                    for i in range(0, len(filtered_rows), batch_size):
                                                        batch = filtered_rows[i:i + batch_size]
                                                        self.supabase.table(target_table).insert(batch).execute()
                                                        total_inserted += len(batch)
                                                    total_skipped += (len(new_rows) - len(filtered_rows))
                                                    print(f"Inserted {len(filtered_rows)} new records for {symbol} {start_str} to {end_str} in batches after filtering duplicates")
                                            except Exception as retry_error:
                                                print(f"Retry insert also failed for {symbol} {start_str} to {end_str}: {retry_error}")
                                                # Skip this chunk but continue
                                                total_skipped += len(new_rows)
                                        else:
                                            # All rows were duplicates
                                            total_skipped += len(new_rows)
                                            print(f"All {len(new_rows)} records for {symbol} {start_str} to {end_str} were duplicates, skipping")
                                    else:
                                        # Other error, try upsert as fallback
                                        print(f"Bulk insert failed for {symbol} {start_str} to {end_str}: {e}. Trying upsert...")
                                        try:
                                            # Try upsert in batches if needed
                                            if len(new_rows) <= batch_size:
                                                self.supabase.table(target_table).upsert(new_rows).execute()
                                                total_inserted += len(new_rows)
                                                print(f"Upserted {len(new_rows)} records for {symbol} {start_str} to {end_str}")
                                            else:
                                                for i in range(0, len(new_rows), batch_size):
                                                    batch = new_rows[i:i + batch_size]
                                                    self.supabase.table(target_table).upsert(batch).execute()
                                                    total_inserted += len(batch)
                                                print(f"Upserted {len(new_rows)} records for {symbol} {start_str} to {end_str} in batches")
                                        except Exception as upsert_error:
                                            print(f"Upsert also failed: {upsert_error}")
                                            all_success = False
                            else:
                                total_skipped += len(rows)
                                print(f"All {len(rows)} records for {symbol} {start_str} to {end_str} already exist, skipping")

                            break  # Success, break retry loop
                        except Exception as e:
                            # Check if it's a rate limit error that should stop processing
                            error_str = str(e).lower()
                            if "429" in error_str or "too many requests" in error_str:
                                print(f"🛑 Polygon rate limit hit for {symbol}. Cooling down 90s.")
                                import time
                                time.sleep(90)
                                all_success = False
                                break  # DO NOT retry same chunk more than once for Polygon
                            else:
                                attempt += 1
                                wait_time = 2 ** attempt
                                print(f"Error fetching chunk {start_str} to {end_str} for {symbol} (attempt {attempt}): {e}. Retrying in {wait_time}s...")
                                import time
                                time.sleep(wait_time)
                                if attempt >= max_retries:
                                    print(f"Max retries reached for {symbol} {start_str} to {end_str}. Skipping chunk.")
                                    all_success = False
                    
                    # Advance to next chunk: if chunk_end has time component (today's last chunk),
                    # set next chunk_start to start of next day
                    if chunk_end.hour != 0 or chunk_end.minute != 0 or chunk_end.second != 0:
                        # chunk_end has time component, advance to start of next day
                        chunk_start = (chunk_end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
                    else:
                        # chunk_end is at midnight, normal increment
                        chunk_start = chunk_end + timedelta(days=1)

                if total_inserted > 0:
                    last_ts_msg = f" (Last timestamp: {last_timestamp_written})" if last_timestamp_written else ""
                    print(f"✅ Completed {symbol}: Inserted {total_inserted} new records, skipped {total_skipped} duplicates{last_ts_msg}")
                    return True
                elif total_skipped > 0:
                    print(f"ℹ️ Completed {symbol}: All {total_skipped} records were duplicates or already exist")
                    return True
                else:
                    print(f"⚠️ No data ingested for {symbol}. No records found or all failed.")
                    return False

                return all_success
            else:
                # Daily data with resume-from-latest behavior
                if resume_from_latest:
                    try:
                        latest_daily = (
                            self.supabase.table("market_data")
                            .select("date")
                            .eq("symbol", symbol)
                            .order("date", desc=True)
                            .limit(1)
                            .execute()
                        )
                        latest_data = latest_daily.data if hasattr(latest_daily, "data") else []
                        if latest_data:
                            latest_date_raw = latest_data[0].get("date")
                            if latest_date_raw:
                                if isinstance(latest_date_raw, str):
                                    latest_date = datetime.fromisoformat(latest_date_raw).date()
                                else:
                                    latest_date = latest_date_raw
                                resume_date = latest_date + timedelta(days=1)
                                if resume_date > end_date.date():
                                    print(f"ℹ️ {symbol} daily already up to {latest_date}; target end {end_date.date()}. Skipping.")
                                    return True
                                if resume_date > start_date.date():
                                    start_date = datetime.combine(resume_date, datetime.min.time())
                                    print(f"✅ Resuming daily ingestion for {symbol} from {start_date.date()} to {end_date.date()}")
                    except Exception as e:
                        print(f"Warning: Could not determine latest daily for {symbol}: {e}. Using provided start date.")

                if start_date > end_date:
                    print(f"Start date {start_date.date()} is after end date {end_date.date()} for {symbol}. Nothing to ingest.")
                    return True

                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")
                data = self.polygon_client.get_historical_data(symbol, start_date_str, end_date_str)
                target_table = "market_data"
                if data.empty:
                    print(f"No data returned from Polygon API for {symbol} ({start_date_str} to {end_date_str})")
                    return False
                rows = []
                for _, row in data.iterrows():
                    row_data = {
                        "symbol": symbol,
                        "date": row["timestamp"].date().isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                    }
                    
                    # Only add source if it's NOT a 1-minute table (which don't have source column)
                    if "1min" not in target_table:
                        row_data["source"] = "polygon"
                        
                    rows.append(row_data)
                if rows:
                    try:
                        self.supabase.table(target_table).upsert(rows).execute()
                        records_processed = len(rows)
                        print(f"Successfully processed {records_processed} daily records for {symbol} (upserted)")
                        return True
                    except Exception as e:
                        print(f"Bulk upsert failed for {symbol}: {e}")
                        # Try individual upserts to handle duplicates gracefully
                        success_count = 0
                        duplicate_count = 0
                        for row in rows:
                            try:
                                self.supabase.table(target_table).upsert([row]).execute()
                                success_count += 1
                            except Exception as individual_error:
                                if "duplicate key" in str(individual_error).lower():
                                    duplicate_count += 1
                                    display_when = row.get('date') or row.get('timestamp') or 'unknown'
                                    print(f"Record already exists for {symbol} on {display_when} - skipping")
                                else:
                                    display_when = row.get('date') or row.get('timestamp') or 'unknown'
                                    print(f"Failed to upsert record for {symbol} on {display_when}: {individual_error}")
                        if success_count > 0 or duplicate_count > 0:
                            print(f"Processed {symbol}: {success_count} new daily records, {duplicate_count} duplicates skipped")
                            return True
                        return False
                else:
                    print(f"No data to insert for {symbol} ({start_date_str} to {end_date_str})")
                    return False
        except Exception as e:
            print(f"Error ingesting historical data for {symbol}: {e}")
            return False

    def _resolve_chunk_days(self, interval: str, chunk_days: Optional[int]) -> int:
        """
        Determine appropriate chunk size based on interval and plan.
        Polygon has strict limits for intraday data:
        - 1min: MAX 1-2 days per call (even on paid plans) to avoid 429s andtimeouts
        - 5min: 3-7 days safe
        """
        interval_key = interval.lower()

        # Polygon-safe limits
        if interval_key in ("1min", "1-minute", "1m"):
            return 1   # 🚨 NEVER more than 1 day for 1-minute data
        elif interval_key in ("5min", "5-minute", "5m"):
            return 7   # safe for 5-minute data
        else:
            return chunk_days or 7

    @staticmethod
    def _previous_trading_day(ref_date) -> datetime.date:
        """Return the most recent weekday before ref_date (Mon-Fri)."""
        prev = ref_date - timedelta(days=1)
        while prev.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            prev -= timedelta(days=1)
        return prev

    def ingest_symbol_full_stack(
        self,
        symbol: str,
        daily_range: Optional[Tuple[datetime, datetime]] = None,
        m5_range: Optional[Tuple[datetime, datetime]] = None,
        m1_range: Optional[Tuple[datetime, datetime]] = None,
        chunk_overrides: Optional[Dict[str, int]] = None,
    ) -> Dict[str, bool]:
        chunk_overrides = chunk_overrides or {}
        now = datetime.utcnow()
        intraday_cutoff = now - timedelta(minutes=15)
        prev_trading_day = self._previous_trading_day(now.date())
        daily_default_end = datetime.combine(prev_trading_day, datetime.min.time())
        defaults = {
            "daily": (
                daily_range[0] if daily_range else datetime(2023, 10, 1),
                daily_range[1] if daily_range else daily_default_end,
            ),
            "5min": (
                m5_range[0] if m5_range else datetime(2023, 10, 1),
                m5_range[1] if m5_range else intraday_cutoff,
            ),
            "1min": (
                m1_range[0] if m1_range else datetime(2025, 8, 1),
                m1_range[1] if m1_range else intraday_cutoff,
            ),
        }

        results: Dict[str, bool] = {}
        for interval_key, bounds in defaults.items():
            start, end = bounds
            kwargs = {
                "symbol": symbol,
                "interval": interval_key if interval_key != "daily" else "daily",
                "start_date": start,
                "end_date": end,
                "resume_from_latest": True,
            }
            if interval_key != "daily":
                kwargs["chunk_days"] = chunk_overrides.get(interval_key)
            success = self.ingest_historical_data(**kwargs)
            results[interval_key] = success
        return results
    
    def ingest_all_historical_data(self, days_back: int = 1825) -> Dict[str, bool]:  # Default to 5 years
        """Ingest historical data for all watchlist stocks"""
        results = {}
        symbols = get_watchlist_symbols()
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            results[symbol] = self.ingest_historical_data(symbol, days_back)
        
        return results
    
    def close(self):
        """Close any open connections or resources"""
        # Currently no resources to close, but method exists for consistency
        pass
