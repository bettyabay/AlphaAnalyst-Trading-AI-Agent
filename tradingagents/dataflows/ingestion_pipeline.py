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
    
    def ingest_historical_data(self, symbol: str, days_back: int = 1825, interval: str = "daily", chunk_days: int = 7, max_retries: int = 5) -> bool:
        """Ingest historical data for a symbol, with chunked fetching and retry/backoff for 5-min data.

        interval: 'daily' (default) or '5min' to ingest 5-minute bars into a separate table.
        chunk_days: number of days per API call chunk (for 5min data)
        max_retries: max retries per chunk
        """
        try:
            if not self.supabase:
                print("Supabase not configured. Cannot ingest data.")
                return False

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            if interval in ("5min", "5-minute", "5m"):
                target_table = "market_data_5min"
                chunk_delta = timedelta(days=chunk_days)
                chunk_start = start_date
                all_success = True
                while chunk_start < end_date:
                    chunk_end = min(chunk_start + chunk_delta, end_date)
                    start_str = chunk_start.strftime("%Y-%m-%d")
                    end_str = chunk_end.strftime("%Y-%m-%d")
                    attempt = 0
                    while attempt < max_retries:
                        try:
                            data = self.polygon_client.get_intraday_data(symbol, start_str, end_str, multiplier=5)
                            if data.empty:
                                print(f"No data for {symbol} {start_str} to {end_str}")
                                break  # Don't retry empty data
                            rows = []
                            for _, row in data.iterrows():
                                rows.append({
                                    "symbol": symbol,
                                    "timestamp": row["timestamp"].isoformat(),
                                    "open": float(row["open"]),
                                    "high": float(row["high"]),
                                    "low": float(row["low"]),
                                    "close": float(row["close"]),
                                    "volume": int(row["volume"]),
                                    "source": "polygon"
                                })
                            if rows:
                                try:
                                    self.supabase.table(target_table).upsert(rows).execute()
                                    print(f"Upserted {len(rows)} records for {symbol} {start_str} to {end_str}")
                                except Exception as e:
                                    print(f"Bulk upsert failed for {symbol} {start_str} to {end_str}: {e}")
                                    # Try individual upserts
                                    for row in rows:
                                        try:
                                            self.supabase.table(target_table).upsert([row]).execute()
                                        except Exception as individual_error:
                                            print(f"Failed to upsert record for {symbol} on {row.get('timestamp')}: {individual_error}")
                            break  # Success, break retry loop
                        except Exception as e:
                            attempt += 1
                            wait_time = 2 ** attempt
                            print(f"Error fetching chunk {start_str} to {end_str} for {symbol} (attempt {attempt}): {e}. Retrying in {wait_time}s...")
                            import time
                            time.sleep(wait_time)
                            if attempt == max_retries:
                                print(f"Max retries reached for {symbol} {start_str} to {end_str}. Skipping chunk.")
                                all_success = False
                    chunk_start = chunk_end
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