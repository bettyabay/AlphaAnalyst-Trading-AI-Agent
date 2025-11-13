"""
Data ingestion pipeline for AlphaAnalyst Trading AI Agent
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

from ..database.config import get_supabase
from ..config.watchlist import WATCHLIST_STOCKS, get_watchlist_symbols
from .polygon_integration import PolygonDataClient

load_dotenv()

class DataIngestionPipeline:
    """Data ingestion pipeline for market data"""
    
    def __init__(self):
        self.polygon_client = PolygonDataClient()
        self.supabase = get_supabase()
        
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
        chunk_days: int = 7,
        max_retries: int = 5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """Ingest historical data for a symbol, with chunked fetching and retry/backoff for intraday data.

        interval: 'daily' (default), '5min' to ingest 5-minute bars, or '1min' to ingest 1-minute bars.
        chunk_days: number of days per API call chunk (for intraday data)
        max_retries: max retries per chunk
        start_date: optional explicit start datetime (inclusive)
        end_date: optional explicit end datetime (inclusive)
        """
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
                end_date = _normalize_date(end_date, datetime.now())
                start_date = _normalize_date(start_date, end_date - timedelta(days=days_back))
            except ValueError as date_error:
                print(f"Invalid date provided: {date_error}")
                return False

            # Work with date boundaries only (midnight UTC naive)
            end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

            if start_date > end_date:
                print(f"Start date {start_date} is after end date {end_date}. Cannot ingest.")
                return False

            if interval in ("5min", "5-minute", "5m"):
                target_table = "market_data_5min"
                multiplier = 5
            elif interval in ("1min", "1-minute", "1m"):
                target_table = "market_data_1min"
                multiplier = 1
            else:
                target_table = None
                multiplier = None
            
            if target_table and multiplier:
                chunk_delta = timedelta(days=chunk_days)
                chunk_start = start_date
                all_success = True
                total_inserted = 0
                total_skipped = 0

                # Track existing timestamps per chunk (only load what we need, when we need it)
                # This avoids loading all records upfront
                existing_timestamps_cache = set()

                while chunk_start <= end_date:
                    chunk_end = min(chunk_start + chunk_delta - timedelta(days=1), end_date)
                    if chunk_end < chunk_start:
                        chunk_end = chunk_start

                    start_str = chunk_start.strftime("%Y-%m-%d")
                    end_str = chunk_end.strftime("%Y-%m-%d")

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
                                if ts and isinstance(ts, str):
                                    ts_normalized = ts.split('+')[0].split('Z')[0]
                                    chunk_existing_timestamps.add(ts_normalized)
                                    existing_timestamps_cache.add(ts_normalized)  # Cache for later chunks
                    except Exception as e:
                        print(f"Warning: Could not check existing records for {symbol} chunk {start_str} to {end_str}: {e}")

                    attempt = 0
                    while attempt < max_retries:
                        try:
                            data = self.polygon_client.get_intraday_data(symbol, start_str, end_str, multiplier=multiplier, max_retries=max_retries)
                            if data.empty:
                                # Check if dates are in the future (no need to print for future dates)
                                try:
                                    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                                    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
                                    now = datetime.now()
                                    if start_dt <= now and end_dt <= now:
                                        # Past dates with no data - might be weekend/holiday
                                        print(f"No trading data for {symbol} {start_str} to {end_str} (likely weekend/holiday)")
                                except Exception:
                                    print(f"No data for {symbol} {start_str} to {end_str}")
                                break  # Don't retry empty data

                            # Prepare rows and filter duplicates using chunk-specific existing timestamps
                            rows = []
                            new_rows = []
                            for _, row in data.iterrows():
                                timestamp_iso = row["timestamp"].isoformat()
                                timestamp_normalized = timestamp_iso.split('+')[0].split('Z')[0]

                                row_data = {
                                    "symbol": symbol,
                                    "timestamp": timestamp_iso,
                                    "open": float(row["open"]),
                                    "high": float(row["high"]),
                                    "low": float(row["low"]),
                                    "close": float(row["close"]),
                                    "volume": int(row["volume"]),
                                    "source": "polygon"
                                }
                                rows.append(row_data)

                                # Check both chunk-specific cache and global cache
                                if timestamp_normalized not in chunk_existing_timestamps and timestamp_normalized not in existing_timestamps_cache:
                                    new_rows.append(row_data)
                                    # Add to both caches to avoid duplicates
                                    chunk_existing_timestamps.add(timestamp_normalized)
                                    existing_timestamps_cache.add(timestamp_normalized)

                            # Bulk insert only new records (batch if too many to avoid timeouts)
                            if new_rows:
                                batch_size = 5000  # Insert in batches to avoid timeout
                                try:
                                    if len(new_rows) <= batch_size:
                                        # Use insert instead of upsert since we've already filtered duplicates
                                        self.supabase.table(target_table).insert(new_rows).execute()
                                        total_inserted += len(new_rows)
                                        total_skipped += (len(rows) - len(new_rows))
                                        print(f"Inserted {len(new_rows)} new records for {symbol} {start_str} to {end_str} (skipped {len(rows) - len(new_rows)} duplicates)")
                                    else:
                                        # Insert in batches
                                        for i in range(0, len(new_rows), batch_size):
                                            batch = new_rows[i:i + batch_size]
                                            self.supabase.table(target_table).insert(batch).execute()
                                            total_inserted += len(batch)
                                        total_skipped += (len(rows) - len(new_rows))
                                        print(f"Inserted {len(new_rows)} new records for {symbol} {start_str} to {end_str} in batches (skipped {len(rows) - len(new_rows)} duplicates)")
                                except Exception as e:
                                    # If insert fails (e.g., duplicate key constraint), try upsert as fallback
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
                                print(f"⚠️ Rate limit exceeded for {symbol}. Consider running with fewer stocks or waiting before continuing.")
                                # For rate limits, wait longer before retrying
                                attempt += 1
                                wait_time = min(60, 10 * (2 ** attempt))  # Cap at 60s, start at 10s
                                print(f"Waiting {wait_time}s before retry {attempt}/{max_retries}...")
                                import time
                                time.sleep(wait_time)
                                if attempt >= max_retries:
                                    print(f"Max retries reached for {symbol} {start_str} to {end_str} due to rate limits.")
                                    all_success = False
                                    # Continue to next chunk instead of breaking completely
                                    break
                            else:
                                attempt += 1
                                wait_time = 2 ** attempt
                                print(f"Error fetching chunk {start_str} to {end_str} for {symbol} (attempt {attempt}): {e}. Retrying in {wait_time}s...")
                                import time
                                time.sleep(wait_time)
                                if attempt >= max_retries:
                                    print(f"Max retries reached for {symbol} {start_str} to {end_str}. Skipping chunk.")
                                    all_success = False
                    chunk_start = chunk_end + timedelta(days=1)

                print(f"Completed {symbol}: Inserted {total_inserted} new records, skipped {total_skipped} duplicates")
                return all_success
            else:
                # Daily data (original logic)
                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")
                data = self.polygon_client.get_historical_data(symbol, start_date_str, end_date_str)
                target_table = "market_data"
                if data.empty:
                    print(f"No data returned from Polygon API for {symbol}")
                    return False
                rows = []
                for _, row in data.iterrows():
                    rows.append({
                        "symbol": symbol,
                        "date": row["timestamp"].date().isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                        "source": "polygon"
                    })
                if rows:
                    try:
                        resp = self.supabase.table(target_table).upsert(rows).execute()
                        records_processed = len(rows)
                        print(f"Successfully processed {records_processed} records for {symbol} (upserted)")
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
                            print(f"Processed {symbol}: {success_count} new records, {duplicate_count} duplicates skipped")
                            return True
                        return False
                else:
                    print(f"No data to insert for {symbol}")
                    return False
        except Exception as e:
            print(f"Error ingesting historical data for {symbol}: {e}")
            return False
    
    def ingest_all_historical_data(self, days_back: int = 1825) -> Dict[str, bool]:  # Default to 5 years
        """Ingest historical data for all watchlist stocks"""
        results = {}
        symbols = get_watchlist_symbols()
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            results[symbol] = self.ingest_historical_data(symbol, days_back)
        
        return results
    
    def get_data_completion_status(self) -> Dict[str, Dict]:
        """Get data completion status for all stocks"""
        status = {}
        
        if not self.supabase:
            print("Supabase not configured. Cannot get completion status.")
            return status
        
        for symbol in get_watchlist_symbols():
            try:
                # Count records for this symbol
                resp = self.supabase.table("market_data").select("date").eq("symbol", symbol).execute()
                data = resp.data if hasattr(resp, "data") else []
                hist_count = len(data) if isinstance(data, list) else 0
                
                # Get latest date
                latest_date = None
                if data:
                    dates = [item.get("date") for item in data if item.get("date")]
                    if dates:
                        latest_date = max(dates)
                
                status[symbol] = {
                    "symbol": symbol,
                    "name": WATCHLIST_STOCKS.get(symbol, symbol),
                    "historical_records": hist_count,
                    "latest_date": latest_date,
                    "is_complete": hist_count > 0,
                    "completion_percentage": min(100, (hist_count / 1260) * 100)  # Assuming 1260 trading days (5 years)
                }
            except Exception as e:
                print(f"Error getting status for {symbol}: {e}")
                status[symbol] = {
                    "symbol": symbol,
                    "name": WATCHLIST_STOCKS.get(symbol, symbol),
                    "historical_records": 0,
                    "latest_date": None,
                    "is_complete": False,
                    "completion_percentage": 0
                }
        
        return status
    
    def close(self):
        """Close any open connections or resources"""
        # Currently no resources to close, but method exists for consistency
        pass