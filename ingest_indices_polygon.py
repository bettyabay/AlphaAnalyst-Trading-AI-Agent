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
        # Auto-convert I:SPX to SPY (Polygon doesn't support I:SPX minute data)
        if api_symbol.upper() in ["I:SPX", "SPX", "^SPX"]:
            polygon_symbol = "SPY"
            print(f"⚠️ Auto-converting I:SPX to SPY (Polygon doesn't support I:SPX minute data)")
        else:
            polygon_symbol = api_symbol.strip()
        
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
        
        # Calculate date range (1 year back from now)
        now_utc = datetime.utcnow()
        end_dt = now_utc - timedelta(minutes=15)  # Safety buffer for intraday
        
        # Calculate start date: go BACK in time (subtract days)
        start_dt = end_dt - timedelta(days=365 * years)  # 1 year BACK for free plan
        
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
        
        # Prepare timezone conversion objects
        utc_tz = pytz.timezone('UTC')
        gmt4_tz = pytz.timezone('Asia/Dubai')  # GMT+4
        
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


def main():
    """Example usage"""
    # Example: Ingest SPY (S&P 500 ETF) 1-minute data
    result = ingest_indices_from_polygon(
        api_symbol="SPY",
        interval="1min",
        years=1,  # 1 year for free plan
        db_symbol="SPY"
    )
    
    print("\n" + "="*60)
    if result["success"]:
        print("✅ " + result["message"])
    else:
        print("❌ " + result["message"])
    print("="*60)


if __name__ == "__main__":
    main()

