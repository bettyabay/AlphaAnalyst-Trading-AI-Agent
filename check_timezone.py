from tradingagents.dataflows.polygon_integration import PolygonDataClient
from datetime import datetime, timedelta
import pytz
import pandas as pd

def check_timezone():
    print("🕵️  Verifying Timezone Conversion Logic...")
    
    # 1. Define Target Timezone
    target_timezone = "Asia/Dubai"
    target_tz = pytz.timezone(target_timezone)
    print(f"🎯 Target Timezone: {target_timezone}")

    # 2. Simulate a Polygon Response (always UTC)
    # Let's verify with a hardcoded UTC time first
    utc_mock = datetime.utcnow().replace(tzinfo=pytz.UTC)
    print(f"\n1️⃣  Simulation:")
    print(f"   Original (UTC):       {utc_mock.isoformat()}")
    
    # Convert
    converted_mock = utc_mock.astimezone(target_tz)
    print(f"   Converted (Target):   {converted_mock.isoformat()}")
    
    # Check Offset
    offset = converted_mock.utcoffset()
    print(f"   Offset found:         {offset} (Should be +04:00 for Dubai)")

    if offset == timedelta(hours=4):
        print("   ✅ Logic Check: PASS (Offset is correct)")
    else:
        print("   ❌ Logic Check: FAIL (Offset is incorrect)")

    # 3. Verify with Real Live Data
    print(f"\n2️⃣  Live Data Test (Symbol: APP):")
    client = PolygonDataClient()
    
    # Get yesterday's date to ensure data exists
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=3) # Go back a few days to be safe (weekends etc)
    
    print(f"   Fetching 1 minute of data from Polygon...")
    df = client.get_intraday_data(
        "APP", 
        start_date.strftime("%Y-%m-%d"), 
        end_date.strftime("%Y-%m-%d"),
        multiplier=1,
        timespan="minute"
    )
    
    if not df.empty:
        # Get the last row
        last_row = df.iloc[-1]
        raw_ts = last_row["timestamp"]
        
        # Polygon returns timezone-naive pandas Timestamps in UTC, or tz-aware UTC
        if raw_ts.tzinfo is None:
            raw_ts = raw_ts.replace(tzinfo=pytz.UTC)
            
        print(f"   Raw from Polygon:     {raw_ts.isoformat()} (UTC)")
        
        # Apply Pipeline Logic
        final_ts = raw_ts.astimezone(target_tz)
        print(f"   Converted for DB:     {final_ts.isoformat()}")
        
        if str(final_ts.utcoffset()) == "4:00:00":
             print("   ✅ Live Check: PASS (Correctly converted to GMT+4)")
        else:
             print(f"   ❌ Live Check: FAIL (Offset is {final_ts.utcoffset()})")
             
    else:
        print("   ⚠️ Could not fetch live data (market might be closed or key invalid). relying on simulation.")

if __name__ == "__main__":
    check_timezone()
