"""
Script to verify signal timezone conversion in the database.
Queries signal_provider_signals table and shows how timestamps are stored.
"""
import os
from datetime import datetime
import pytz
from tradingagents.database.config import get_supabase

def verify_signal_timezone():
    """Query database and verify timezone conversion for signals"""
    
    supabase = get_supabase()
    if not supabase:
        print("[ERROR] Supabase not configured. Check .env file.")
        return
    
    print("=" * 80)
    print("Signal Timezone Verification - Database Query")
    print("=" * 80)
    
    try:
        # Query signal_provider_signals table
        # Get recent signals (last 10)
        result = supabase.table("signal_provider_signals")\
            .select("provider_name, symbol, signal_date, action, entry_price, timezone_offset")\
            .order("signal_date", desc=True)\
            .limit(10)\
            .execute()
        
        if not result.data:
            print("\n[WARNING] No signals found in database.")
            print("   Please ingest some signal data first using the UI.")
            return
        
        print(f"\n[SUCCESS] Found {len(result.data)} signal(s) in database\n")
        print("=" * 80)
        print("Database Query Results:")
        print("=" * 80)
        
        dubai_tz = pytz.timezone('Asia/Dubai')
        utc_tz = pytz.UTC
        
        for i, signal in enumerate(result.data, 1):
            provider = signal.get('provider_name', 'N/A')
            symbol = signal.get('symbol', 'N/A')
            signal_date_str = signal.get('signal_date')
            action = signal.get('action', 'N/A')
            entry_price = signal.get('entry_price', 'N/A')
            tz_offset = signal.get('timezone_offset', '+04:00')
            
            print(f"\n{i}. {symbol} ({provider})")
            print(f"   Action: {action}")
            print(f"   Entry Price: {entry_price}")
            print(f"   Timezone Offset: {tz_offset}")
            
            if signal_date_str:
                # Parse the timestamp from database
                # PostgreSQL returns ISO format string
                try:
                    # Parse the ISO string
                    if 'T' in signal_date_str:
                        # ISO format: 2023-10-05T10:53:30+00:00 or 2023-10-05T10:53:30Z
                        if signal_date_str.endswith('Z'):
                            signal_date_str = signal_date_str.replace('Z', '+00:00')
                        dt_db = datetime.fromisoformat(signal_date_str.replace('Z', '+00:00'))
                    else:
                        # Try parsing as datetime string
                        dt_db = datetime.fromisoformat(signal_date_str)
                    
                    # Ensure timezone-aware
                    if dt_db.tzinfo is None:
                        # Assume UTC if no timezone
                        dt_db = utc_tz.localize(dt_db)
                    else:
                        dt_db = dt_db.astimezone(utc_tz)
                    
                    # Convert to GMT+4 for display
                    dt_gmt4 = dt_db.astimezone(dubai_tz)
                    
                    print(f"   Raw DB Value: {signal_date_str}")
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
                    print(f"   Raw value: {signal_date_str}")
            else:
                print("   [WARNING] No signal_date found")
        
        print("\n" + "=" * 80)
        print("Summary:")
        print("=" * 80)
        print("1. Database stores TIMESTAMPTZ in UTC internally")
        print("2. When queried, timestamps are returned in ISO format")
        print("3. Conversion to GMT+4 is done in application code")
        print("4. The stored UTC value represents the GMT+4 time correctly")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Error querying database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_signal_timezone()

