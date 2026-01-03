import pandas as pd
from tradingagents.database.config import get_supabase

def verify_data():
    print("Connecting to Supabase...")
    supabase = get_supabase()
    
    if not supabase:
        print("❌ Error: Supabase not configured (check .env)")
        return

    table_name = "market_data_commodities_1min"
    symbols_to_check = ["GOLD", "^XAUUSD", "C:XAUUSD"]
    
    print(f"\nChecking table: {table_name}")
    print("-" * 50)
    
    for symbol in symbols_to_check:
        print(f"\n🔍 Searching for symbol: '{symbol}'")
        
        try:
            # Count records
            response = supabase.table(table_name)\
                .select("timestamp", count="exact", head=True)\
                .eq("symbol", symbol)\
                .execute()
            
            count = response.count
            print(f"   Found {count} records.")
            
            if count > 0:
                # Fetch latest 3 records
                data_response = supabase.table(table_name)\
                    .select("timestamp, open, close")\
                    .eq("symbol", symbol)\
                    .order("timestamp", desc=True)\
                    .limit(3)\
                    .execute()
                
                print("   Latest 3 records:")
                for row in data_response.data:
                    print(f"   - {row['timestamp']}: Open={row['open']}, Close={row['close']}")
                    
        except Exception as e:
            print(f"   ❌ Error querying for {symbol}: {e}")

if __name__ == "__main__":
    verify_data()
