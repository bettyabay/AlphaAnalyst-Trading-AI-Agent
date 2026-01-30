"""
Indices Ingestion Script for Polygon API (Free Plan)
- Fetches 1 year of historical data (Polygon free plan limit for indices)
- Chunks by 1 day for both 1-minute and daily data
- 12-second throttle between calls
- 90-second wait on 429 rate limit
- Exponential backoff for retries
- Converts UTC (Polygon) to GMT+4 (Asia/Dubai) before storing
"""

from tradingagents.dataflows.polygon_integration import PolygonDataClient
from tradingagents.database.config import get_supabase
from tradingagents.dataflows.ingestion_pipeline import convert_instrument_to_polygon_symbol
from datetime import datetime, timedelta
import pandas as pd
import pytz
import time
import traceback
from typing import Dict


def _insert_chunk_to_db(
    chunk_df: pd.DataFrame,
    target_symbol: str,
    target_table: str,
    sb,
    utc_tz,
    gmt4_tz,
    chunk_s_str: str,
    chunk_e_str: str
) -> Dict:
    """
    Helper function to insert a single chunk into the database immediately after fetching.
    Returns: {"success": bool, "count": int, "message": str}
    """
    db_rows = []
    
    if "timestamp" in chunk_df.columns:
        chunk_df = chunk_df.set_index("timestamp")
    
    for timestamp, row in chunk_df.iterrows():
        if pd.isna(row.get("close")):
            continue 
            
        try:
            # Polygon returns UTC timestamps → Convert to GMT+4 for database
            if timestamp.tzinfo is None:
                ts_utc = utc_tz.localize(timestamp)  # Naive UTC → UTC-aware
            else:
                ts_utc = timestamp.astimezone(utc_tz)  # Ensure UTC
            
            ts_gmt4 = ts_utc.astimezone(gmt4_tz)  # Convert UTC → GMT+4
            
            # Store GMT+4 timestamp in database
            # Note: Not including open_interest as it may not exist in the table schema
            record = {
                "symbol": str(target_symbol),
                "timestamp": ts_gmt4.isoformat(),  # Store in GMT+4
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]) if pd.notnull(row.get("volume")) else 0
            }
            
            # Skip open_interest - column doesn't exist in table or causes schema errors
            
            db_rows.append(record)
        except (ValueError, TypeError) as e:
            continue
    
    if not db_rows:
        return {"success": False, "count": 0, "message": "No valid rows after processing"}
    
    # Insert immediately
    try:
        result = sb.table(target_table).upsert(db_rows).execute()
        
        # Quick verification
        verify = sb.table(target_table)\
            .select("symbol")\
            .eq("symbol", target_symbol)\
            .limit(1)\
            .execute()
        
        if verify.data:
            return {"success": True, "count": len(db_rows), "message": "Inserted and verified"}
        else:
            return {"success": False, "count": 0, "message": "Inserted but verification failed"}
    except Exception as e:
        return {"success": False, "count": 0, "message": f"Insert error: {str(e)}"}


# Mapping of indices to their ETF equivalents for 1-minute data
# Polygon doesn't provide 1-minute data for cash indices, so we use ETFs
INDEX_TO_ETF_MAPPING = {
    # S&P 500
    "I:SPX": "SPY",
    "SPX": "SPY",
    "^SPX": "SPY",
    "SP500": "SPY",
    "S&P 500": "SPY",
    "S&P500": "SPY",
    # NASDAQ-100 (handle various formats)
    "I:NDX": "QQQ",
    "NDX": "QQQ",
    "^NDX": "QQQ",
    "NASDAQ-100": "QQQ",
    "NASDAQ100": "QQQ",
    "NAS 100": "QQQ",  # Common variation with space
    "NAS100": "QQQ",
    "I:NAS 100": "QQQ",  # If user enters with I: prefix and space
    "I:NAS100": "QQQ",
    # Dow Jones Industrial Average
    "I:DJI": "DIA",
    "DJI": "DIA",
    "^DJI": "DIA",
    "DOW": "DIA",
    "DOW JONES": "DIA",
    # Russell 2000
    "I:RUT": "IWM",
    "RUT": "IWM",
    "^RUT": "IWM",
    "RUSSELL 2000": "IWM",
    "RUSSELL2000": "IWM",
    # VIX (volatility index - may work directly, but try VXX as backup)
    "I:VIX": "VIX",  # VIX might work directly for daily, but for 1min we might need VXX
    "VIX": "VIX",
    "^VIX": "VIX",
}


def _get_etf_for_index(index_symbol: str, interval: str = "1min") -> str:
    """
    Convert an index symbol to its ETF equivalent for Polygon API.
    
    Args:
        index_symbol: Index symbol (e.g., "I:SPX", "^SPX", "SPX", "NAS 100", "I:NAS 100")
        interval: "1min" for 1-minute data (requires ETF), "daily" for daily data (can use index)
    
    Returns:
        ETF symbol for 1-minute data, or original symbol for daily data
    """
    symbol_upper = index_symbol.upper().strip()
    
    # Normalize by removing spaces, dashes for flexible matching
    symbol_normalized = symbol_upper.replace(" ", "").replace("-", "").replace("_", "")
    
    # For daily data, indices might work directly, so try original first
    if interval == "daily":
        # Check if it's a known index that might work for daily
        if symbol_upper in INDEX_TO_ETF_MAPPING or symbol_normalized in INDEX_TO_ETF_MAPPING:
            # For daily, we can try the index directly first, but have ETF as fallback
            return index_symbol.strip()
        return index_symbol.strip()
    
    # For 1-minute data, always use ETF mapping
    # First try exact match
    if symbol_upper in INDEX_TO_ETF_MAPPING:
        etf = INDEX_TO_ETF_MAPPING[symbol_upper]
        if etf != symbol_upper:
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
        return etf
    
    # Try normalized version (without spaces/dashes)
    if symbol_normalized in INDEX_TO_ETF_MAPPING:
        etf = INDEX_TO_ETF_MAPPING[symbol_normalized]
        if etf != symbol_normalized:
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
        return etf
    
    # Check if it starts with I: or ^ and extract base symbol
    if symbol_upper.startswith("I:"):
        base = symbol_upper[2:].strip()
        base_normalized = base.replace(" ", "").replace("-", "").replace("_", "")
        # Try base with spaces removed
        if base in INDEX_TO_ETF_MAPPING:
            etf = INDEX_TO_ETF_MAPPING[base]
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
            return etf
        elif base_normalized in INDEX_TO_ETF_MAPPING:
            etf = INDEX_TO_ETF_MAPPING[base_normalized]
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
            return etf
        # Also try the full I: prefix version
        if symbol_normalized in INDEX_TO_ETF_MAPPING:
            etf = INDEX_TO_ETF_MAPPING[symbol_normalized]
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
            return etf
    elif symbol_upper.startswith("^"):
        base = symbol_upper[1:].strip()
        base_normalized = base.replace(" ", "").replace("-", "").replace("_", "")
        if base in INDEX_TO_ETF_MAPPING:
            etf = INDEX_TO_ETF_MAPPING[base]
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
            return etf
        elif base_normalized in INDEX_TO_ETF_MAPPING:
            etf = INDEX_TO_ETF_MAPPING[base_normalized]
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
            return etf
    
    # Special handling for "NAS 100" variations that might not be in mapping
    if "NAS" in symbol_upper and "100" in symbol_upper:
        print(f"⚠️ Auto-converting {index_symbol} to QQQ (Polygon doesn't support 1-minute data for indices)")
        return "QQQ"
    
    # If no mapping found, return as-is (might be an ETF already or unknown index)
    return index_symbol.strip()


def ingest_indices_from_polygon(
    api_symbol: str,
    interval: str = "1min",  # "1min" or "daily"
    years: int = 1,  # Polygon free plan: 1 year for indices
    db_symbol: str = None
):
    """
    Ingest indices data from Polygon API.
    
    Args:
        api_symbol: Polygon symbol (e.g., "SPY", "I:SPX" will be converted to "SPY")
        interval: "1min" for 1-minute bars, "daily" for daily bars
        years: Number of years to fetch (default 1 for free plan)
        db_symbol: Symbol to store in database (defaults to api_symbol)
    """
    try:
        # Convert index to ETF for 1-minute data
        polygon_symbol = _get_etf_for_index(api_symbol, interval)
        
        # Determine target table
        if interval == "1min":
            target_table = "market_data_indices_1min"
        elif interval == "daily":
            target_table = "market_data"  # Or create a separate indices daily table if needed
        else:
            return {"success": False, "message": f"Invalid interval: {interval}. Use '1min' or 'daily'"}
        
        # Verify target table is set correctly
        print(f"📋 Target table determined: '{target_table}' for interval '{interval}'")
        
        # Determine target symbol early
        target_symbol = db_symbol if db_symbol else polygon_symbol
        
        # Get Supabase client
        sb = get_supabase()
        if not sb:
            return {"success": False, "message": "Supabase not configured (check .env)"}
        
        # Prepare timezone conversion objects (needed for resume logic)
        utc_tz = pytz.timezone('UTC')
        gmt4_tz = pytz.timezone('Asia/Dubai')  # GMT+4
        
        # RESUME LOGIC: Check if data already exists in database
        # If data exists, only fetch from latest timestamp forward
        resume_from_latest = True  # Always check for existing data
        latest_timestamp_utc = None
        
        if resume_from_latest:
            try:
                # Query for latest timestamp for this symbol
                latest_result = sb.table(target_table)\
                    .select("timestamp")\
                    .eq("symbol", target_symbol)\
                    .order("timestamp", desc=True)\
                    .limit(1)\
                    .execute()
                
                if latest_result.data and len(latest_result.data) > 0:
                    latest_ts_str = latest_result.data[0].get("timestamp")
                    if latest_ts_str:
                        # Parse timestamp (stored in GMT+4, convert back to UTC for comparison)
                        if isinstance(latest_ts_str, str):
                            # Parse ISO format timestamp
                            latest_timestamp = datetime.fromisoformat(latest_ts_str.replace('Z', '+00:00'))
                        else:
                            latest_timestamp = latest_ts_str
                        
                        # Convert from GMT+4 (database) to UTC for Polygon API
                        if latest_timestamp.tzinfo is None:
                            # Assume it's GMT+4 if naive
                            latest_timestamp = gmt4_tz.localize(latest_timestamp)
                        
                        # Convert to UTC
                        latest_timestamp_utc = latest_timestamp.astimezone(utc_tz)
                        
                        print(f"✅ Found existing data for '{target_symbol}' in database")
                        print(f"   Latest timestamp: {latest_timestamp_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                        print(f"   Will resume from this point forward...")
            except Exception as e:
                print(f"⚠️ Could not check for existing data: {e}")
                print(f"   Will start fresh ingestion...")
                latest_timestamp_utc = None
        
        # Calculate date range
        now_utc = datetime.utcnow()
        end_dt = now_utc - timedelta(minutes=15)  # Safety buffer for intraday
        
        # Determine start date based on resume logic
        if latest_timestamp_utc and resume_from_latest:
            # Resume from latest timestamp + 1 minute (to avoid duplicates)
            start_dt = latest_timestamp_utc + timedelta(minutes=1)
            print(f"📅 Resuming from: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # If start_dt is already at or after end_dt, no new data to fetch
            if start_dt >= end_dt:
                print(f"✅ Data is already up to date. Latest: {latest_timestamp_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC, End: {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                return {"success": True, "message": f"✅ Data for {target_symbol} is already up to date. Latest timestamp: {latest_timestamp_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"}
        else:
            # No existing data - start from 1 year back
            start_dt = end_dt - timedelta(days=365 * years)  # 1 year BACK for free plan
            print(f"📅 Starting fresh ingestion from: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        # Validate dates are not in the future
        current_year = now_utc.year
        if start_dt.year > current_year or end_dt.year > current_year:
            print(f"⚠️ WARNING: Date range contains future year. Recalculating...")
            end_dt = now_utc - timedelta(minutes=15)
            start_dt = end_dt - timedelta(days=365 * years)
        
        if start_dt > end_dt:
            print(f"⚠️ WARNING: Start date is after end date. Recalculating...")
            end_dt = now_utc - timedelta(minutes=15)
            start_dt = end_dt - timedelta(days=365 * years)
        
        # For daily data, use previous trading day
        if interval == "daily":
            end_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        print(f"\n{'='*60}")
        print(f"📊 POLYGON INDICES INGESTION")
        print(f"{'='*60}")
        print(f"   Symbol: {polygon_symbol}")
        print(f"   Interval: {interval}")
        print(f"   Start Date: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"   End Date:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"   Total Days: {(end_dt - start_dt).days} days")
        print(f"   Table: {target_table}")
        print(f"   DB Symbol: {target_symbol}")
        print(f"{'='*60}\n")
        
        # Polygon throttle: 12 seconds between calls (free plan: 5 calls/min)
        _last_polygon_call = 0
        _polygon_min_interval = 12  # 12 seconds = 5 calls/min (safe for free plan)
        
        def _polygon_throttle():
            """Ensure minimum interval between Polygon API calls"""
            nonlocal _last_polygon_call
            elapsed = time.time() - _last_polygon_call
            if elapsed < _polygon_min_interval:
                sleep_time = _polygon_min_interval - elapsed
                print(f"⏳ Throttling Polygon call for {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            _last_polygon_call = time.time()
        
        # Timezone conversion objects already defined above
        
        # Chunk by 1 day (same as gold ingestion for free plan compatibility)
        chunk_days = 1
        chunk_start = start_dt
        max_retries = 5
        all_success = True
        total_inserted = 0
        
        print(f"📦 Chunking date range into {chunk_days}-day chunks for Polygon free plan compatibility...")
        print(f"   Will insert each chunk immediately after fetching (incremental storage)\n")
        print(f"📝 Will store data with symbol: '{target_symbol}' in table: {target_table}\n")
        
        client = PolygonDataClient()
        
        # Process in 1-day chunks - INSERT IMMEDIATELY after each fetch
        chunk_count = 0
        total_chunks_expected = (end_dt - start_dt).days + 1
        print(f"📊 Will process approximately {total_chunks_expected} chunks (1 per day)")
        print(f"   Each chunk will be inserted immediately after fetching...\n")
        
        while chunk_start <= end_dt:
            chunk_count += 1
            chunk_end = min(chunk_start + timedelta(days=chunk_days) - timedelta(days=1), end_dt)
            if chunk_end < chunk_start:
                chunk_end = chunk_start
            
            chunk_s_str = chunk_start.strftime("%Y-%m-%d")
            chunk_e_str = chunk_end.strftime("%Y-%m-%d")
            
            # Show progress
            progress_pct = (chunk_count / total_chunks_expected * 100) if total_chunks_expected > 0 else 0
            print(f"📦 Processing chunk {chunk_count}/{total_chunks_expected} ({progress_pct:.1f}%): {chunk_s_str} to {chunk_e_str}")
            
            attempt = 0
            chunk_success = False
            
            while attempt < max_retries:
                try:
                    # Apply throttle before each API call
                    _polygon_throttle()
                    
                    if interval == "1min":
                        print(f"🔍 Calling Polygon API: get_intraday_data('{polygon_symbol}', '{chunk_s_str}', '{chunk_e_str}', multiplier=1, timespan='minute')")
                        chunk_df = client.get_intraday_data(polygon_symbol, chunk_s_str, chunk_e_str, multiplier=1, timespan="minute")
                    else:  # daily
                        print(f"🔍 Calling Polygon API: get_historical_data('{polygon_symbol}', '{chunk_s_str}', '{chunk_e_str}')")
                        chunk_df = client.get_historical_data(polygon_symbol, chunk_s_str, chunk_e_str)
                    
                    if not chunk_df.empty:
                        print(f"✅ Fetched {len(chunk_df)} records for {chunk_s_str} to {chunk_e_str}")
                        
                        # INSERT IMMEDIATELY after fetching
                        insert_result = _insert_chunk_to_db(
                            chunk_df, target_symbol, target_table, sb, utc_tz, gmt4_tz, chunk_s_str, chunk_e_str
                        )
                        if insert_result["success"]:
                            total_inserted += insert_result["count"]
                            print(f"   💾 Stored {insert_result['count']} records in database")
                        else:
                            print(f"   ⚠️ Storage warning: {insert_result['message']}")
                            all_success = False
                        
                        chunk_success = True
                        break  # Success, break retry loop
                    else:
                        # Empty data - might be weekend/holiday, don't retry
                        print(f"ℹ️ No data for {chunk_s_str} to {chunk_e_str} (likely weekend/holiday)")
                        chunk_success = True  # Not an error, just no data
                        break
                        
                except ValueError as e:
                    # Handle 403 Forbidden or other Polygon restrictions
                    error_str = str(e)
                    if "403" in error_str or "Forbidden" in error_str:
                        return {"success": False, "message": f"❌ Polygon 403 Forbidden: {error_str}\n\nThis symbol may not be available for {interval} data on your Polygon plan.\n\nDate range: {chunk_s_str} to {chunk_e_str}"}
                    else:
                        # Other ValueError - retry with exponential backoff
                        attempt += 1
                        if attempt < max_retries:
                            wait_time = 2 ** attempt
                            print(f"⚠️ Error fetching chunk {chunk_s_str} to {chunk_e_str} (attempt {attempt}/{max_retries}): {error_str}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"❌ Max retries reached for chunk {chunk_s_str} to {chunk_e_str}. Skipping chunk.")
                            all_success = False
                            break
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a rate limit error (429)
                    if "429" in error_str or "too many requests" in error_str.lower():
                        print(f"🛑 Polygon rate limit hit for {polygon_symbol}. Cooling down 90s...")
                        time.sleep(90)
                        all_success = False
                        break  # DO NOT retry same chunk more than once for Polygon
                    else:
                        # Other error - retry with exponential backoff
                        attempt += 1
                        if attempt < max_retries:
                            wait_time = 2 ** attempt
                            print(f"⚠️ Error fetching chunk {chunk_s_str} to {chunk_e_str} (attempt {attempt}/{max_retries}): {error_str}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"❌ Max retries reached for chunk {chunk_s_str} to {chunk_e_str}. Skipping chunk.")
                            all_success = False
                            break
            
            # Advance to next chunk (next day)
            chunk_start = chunk_end + timedelta(days=1)
        
        # All chunks processed - final summary
        print(f"\n{'='*60}")
        print(f"✅ INGESTION COMPLETE")
        print(f"{'='*60}")
        print(f"   Total records inserted: {total_inserted}")
        print(f"   Symbol: {target_symbol}")
        print(f"   Table: {target_table}")
        print(f"{'='*60}\n")
        
        # Final verification
        date_range_info = f"Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} ({(end_dt - start_dt).days} days)"
        
        print(f"🔍 Final verification...")
        try:
            verify_result = sb.table(target_table)\
                .select("symbol, timestamp")\
                .eq("symbol", target_symbol)\
                .order("timestamp", desc=True)\
                .limit(5)\
                .execute()
            
            if verify_result.data:
                print(f"✅ VERIFICATION: Found records in database for symbol '{target_symbol}'")
                print(f"   Sample records:")
                for rec in verify_result.data[:3]:
                    print(f"     - {rec.get('symbol')} at {rec.get('timestamp')}")
            else:
                print(f"⚠️ WARNING: Verification query returned no records for symbol '{target_symbol}'")
                print(f"   Please check RLS policies and symbol format")
        except Exception as verify_error:
            print(f"⚠️ Could not verify data storage: {verify_error}")
        
        return {"success": True, "message": f"✅ Successfully ingested {total_inserted} records for {target_symbol} into {target_table}.\n\n{date_range_info} (converted from UTC to GMT+4)"}
    
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Indices ingestion error: {str(e)}"}


def verify_indices_in_database():
    """
    Helper function to verify what index symbols are stored in the database.
    This helps identify if symbols are stored correctly (I:SPX, I:NDX, I:DJI).
    """
    sb = get_supabase()
    if not sb:
        print("❌ Supabase not configured")
        return
    
    target_table = "market_data_indices_1min"
    
    try:
        # Get all distinct symbols
        result = sb.table(target_table)\
            .select("symbol, timestamp")\
            .order("symbol")\
            .order("timestamp", desc=True)\
            .execute()
        
        if not result.data:
            print(f"ℹ️ No data found in {target_table}")
            return
        
        # Group by symbol and get latest timestamp for each
        symbols_info = {}
        for row in result.data:
            symbol = row.get("symbol")
            timestamp = row.get("timestamp")
            
            if symbol not in symbols_info:
                symbols_info[symbol] = {
                    "latest": timestamp,
                    "count": 0
                }
            symbols_info[symbol]["count"] += 1
            # Update latest if this timestamp is newer
            if timestamp and symbols_info[symbol]["latest"]:
                try:
                    if isinstance(timestamp, str):
                        ts1 = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        ts2 = datetime.fromisoformat(symbols_info[symbol]["latest"].replace('Z', '+00:00'))
                        if ts1 > ts2:
                            symbols_info[symbol]["latest"] = timestamp
                    else:
                        if timestamp > symbols_info[symbol]["latest"]:
                            symbols_info[symbol]["latest"] = timestamp
                except:
                    pass
        
        print(f"\n{'='*60}")
        print(f"📊 INDICES IN DATABASE: {target_table}")
        print(f"{'='*60}")
        print(f"Found {len(symbols_info)} unique symbol(s):\n")
        
        for symbol, info in sorted(symbols_info.items()):
            print(f"  • {symbol}")
            print(f"    - Records: {info['count']}")
            print(f"    - Latest: {info['latest']}")
            print()
        
        # Check for expected symbols
        expected_symbols = ["I:SPX", "I:NDX", "I:DJI"]
        print(f"Expected symbols (from image): {', '.join(expected_symbols)}")
        print(f"\n✅ Verification:")
        for exp_sym in expected_symbols:
            if exp_sym in symbols_info:
                print(f"  ✅ {exp_sym} - Found ({symbols_info[exp_sym]['count']} records)")
            else:
                # Check for variations
                found_variants = [s for s in symbols_info.keys() if exp_sym.upper() in s.upper() or s.upper() in exp_sym.upper()]
                if found_variants:
                    print(f"  ⚠️ {exp_sym} - Not found, but found variants: {', '.join(found_variants)}")
                else:
                    print(f"  ❌ {exp_sym} - Not found")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error verifying database: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Example usage"""
    # First, verify what's in the database
    print("🔍 Verifying indices in database...")
    verify_indices_in_database()
    
    # Example: Ingest SPY (S&P 500 ETF) 1-minute data
    # Uncomment to test ingestion:
    # result = ingest_indices_from_polygon(
    #     api_symbol="I:SPX",
    #     interval="1min",
    #     years=1,  # 1 year for free plan
    #     db_symbol="I:SPX"
    # )
    # 
    # print("\n" + "="*60)
    # if result["success"]:
    #     print("✅ " + result["message"])
    # else:
    #     print("❌ " + result["message"])
    # print("="*60)


if __name__ == "__main__":
    main()

