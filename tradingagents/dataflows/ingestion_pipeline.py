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
    
    def ingest_historical_data(self, symbol: str, days_back: int = 1825) -> bool:  # Default to 5 years
        """Ingest historical data for a symbol"""
        try:
            if not self.supabase:
                print("Supabase not configured. Cannot ingest data.")
                return False
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Fetch data from Polygon
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            data = self.polygon_client.get_historical_data(symbol, start_date_str, end_date_str)
            
            if data.empty:
                print(f"No data returned from Polygon API for {symbol}")
                return False
            
            # Insert into Supabase market_data table
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
                    # Use upsert to handle duplicates gracefully
                    resp = self.supabase.table("market_data").upsert(rows).execute()
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
                            self.supabase.table("market_data").upsert([row]).execute()
                            success_count += 1
                        except Exception as individual_error:
                            if "duplicate key" in str(individual_error).lower():
                                duplicate_count += 1
                                print(f"Record already exists for {symbol} on {row['date']} - skipping")
                            else:
                                print(f"Failed to upsert record for {symbol} on {row['date']}: {individual_error}")
                    
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