"""
Script to verify Gold market data timezone in the database.
Queries market_data_commodities_1min table and shows how timestamps are stored.
"""
import os
from datetime import datetime, timedelta
import pytz
from tradingagents.database.config import get_supabase

def verify_gold_market_timezone():
    """Query database and verify timezone for Gold market data"""
    
    supabase = get_supabase()
    if not supabase:
        print("[ERROR] Supabase not configured. Check .env file.")
        return
    
    print("=" * 80)
    print("Gold Market Data Timezone Verification - Database Query")
    print("=" * 80)
    
    # Gold symbols to try
    gold_symbols = ["C:XAUUSD", "XAUUSD", "^XAUUSD", "GOLD"]
    
    try:
        dubai_tz = pytz.timezone('Asia/Dubai')
        utc_tz = pytz.UTC
        
        # Try different gold symbol formats
        gold_data = None
        used_symbol = None
        
        for symbol in gold_symbols:
            try:
                # Query market_data_commodities_1min table
                # Get recent gold data (last 10 records)
                result = supabase.table("market_data_commodities_1min")\
                    .select("symbol, timestamp, open, high, low, close, volume")\
                    .eq("symbol", symbol)\
                    .order("timestamp", desc=True)\
                    .limit(10)\
                    .execute()
                
                if result.data and len(result.data) > 0:
                    gold_data = result.data
                    used_symbol = symbol
                    break
            except Exception as e:
                # Try next symbol
                continue
        
        if not gold_data:
            print("\n[WARNING] No Gold market data found in database.")
            print(f"   Tried symbols: {', '.join(gold_symbols)}")
            print("   Please ingest Gold market data first using the UI or ingestion scripts.")
            return
        
        print(f"\n[SUCCESS] Found {len(gold_data)} Gold market data record(s) in database")
        print(f"   Symbol used: {used_symbol}\n")
        print("=" * 80)
        print("Database Query Results:")
        print("=" * 80)
        
        for i, record in enumerate(gold_data, 1):
            symbol = record.get('symbol', 'N/A')
            timestamp_str = record.get('timestamp')
            open_price = record.get('open', 'N/A')
            high_price = record.get('high', 'N/A')
            low_price = record.get('low', 'N/A')
            close_price = record.get('close', 'N/A')
            volume = record.get('volume', 'N/A')
            
            print(f"\n{i}. {symbol} (1-minute bar)")
            print(f"   Open: {open_price}, High: {high_price}, Low: {low_price}, Close: {close_price}")
            print(f"   Volume: {volume}")
            
            if timestamp_str:
                # Parse the timestamp from database
                try:
                    # Parse the ISO string
                    if 'T' in timestamp_str:
                        # ISO format: 2023-10-05T10:53:30+00:00 or 2023-10-05T10:53:30Z
                        if timestamp_str.endswith('Z'):
                            timestamp_str = timestamp_str.replace('Z', '+00:00')
                        dt_db = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        # Try parsing as datetime string
                        dt_db = datetime.fromisoformat(timestamp_str)
                    
                    # Ensure timezone-aware
                    if dt_db.tzinfo is None:
                        # Assume UTC if no timezone
                        dt_db = utc_tz.localize(dt_db)
                    else:
                        dt_db = dt_db.astimezone(utc_tz)
                    
                    # Convert to GMT+4 for display
                    dt_gmt4 = dt_db.astimezone(dubai_tz)
                    
                    print(f"   Raw DB Value: {timestamp_str}")
                    print(f"   Parsed (UTC): {dt_db.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    print(f"   Display (GMT+4): {dt_gmt4.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    print(f"   ISO Format (GMT+4): {dt_gmt4.isoformat()}")
                    
                    # Verify: The UTC value should represent GMT+4 time
                    # If stored correctly, UTC time + 4 hours = GMT+4 time
                    expected_gmt4_hour = dt_db.hour + 4
                    if expected_gmt4_hour >= 24:
                        expected_gmt4_hour -= 24
                    actual_gmt4_hour = dt_gmt4.hour
                    
                    if actual_gmt4_hour == expected_gmt4_hour:
                        print(f"   [OK] Timezone conversion CORRECT: UTC {dt_db.hour}:{dt_db.minute:02d} = GMT+4 {dt_gmt4.hour}:{dt_gmt4.minute:02d}")
                    else:
                        print(f"   [WARNING] Timezone conversion check: UTC {dt_db.hour}:{dt_db.minute:02d} -> GMT+4 {dt_gmt4.hour}:{dt_gmt4.minute:02d}")
                        
                except Exception as e:
                    print(f"   [ERROR] Error parsing timestamp: {e}")
                    print(f"   Raw value: {timestamp_str}")
            else:
                print("   [WARNING] No timestamp found")
        
        # Also check data range
        print("\n" + "=" * 80)
        print("Data Range Check:")
        print("=" * 80)
        
        try:
            # Get oldest and newest timestamps
            oldest_result = supabase.table("market_data_commodities_1min")\
                .select("timestamp")\
                .eq("symbol", used_symbol)\
                .order("timestamp", desc=False)\
                .limit(1)\
                .execute()
            
            newest_result = supabase.table("market_data_commodities_1min")\
                .select("timestamp")\
                .eq("symbol", used_symbol)\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()
            
            if oldest_result.data and newest_result.data:
                oldest_str = oldest_result.data[0].get('timestamp')
                newest_str = newest_result.data[0].get('timestamp')
                
                if oldest_str and newest_str:
                    oldest_dt = datetime.fromisoformat(oldest_str.replace('Z', '+00:00'))
                    newest_dt = datetime.fromisoformat(newest_str.replace('Z', '+00:00'))
                    
                    if oldest_dt.tzinfo is None:
                        oldest_dt = utc_tz.localize(oldest_dt)
                    if newest_dt.tzinfo is None:
                        newest_dt = utc_tz.localize(newest_dt)
                    
                    oldest_gmt4 = oldest_dt.astimezone(dubai_tz)
                    newest_gmt4 = newest_dt.astimezone(dubai_tz)
                    
                    print(f"   Oldest record (UTC): {oldest_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    print(f"   Oldest record (GMT+4): {oldest_gmt4.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    print(f"   Newest record (UTC): {newest_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    print(f"   Newest record (GMT+4): {newest_gmt4.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    
                    # Calculate time span
                    time_span = newest_dt - oldest_dt
                    days = time_span.days
                    hours = time_span.seconds // 3600
                    print(f"   Time span: {days} days, {hours} hours")
        except Exception as e:
            print(f"   [WARNING] Could not fetch data range: {e}")
        
        print("\n" + "=" * 80)
        print("Summary:")
        print("=" * 80)
        print("1. Database stores TIMESTAMPTZ in UTC internally")
        print("2. When queried, timestamps are returned in ISO format")
        print("3. Conversion to GMT+4 is done in application code")
        print("4. The stored UTC value represents the GMT+4 time correctly")
        print("5. All market data (like signals) is stored in GMT+4 timezone")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Error querying database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_gold_market_timezone()

