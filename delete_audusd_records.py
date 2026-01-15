"""
Script to delete AUDUSD records from the database for fresh ingestion.

This script deletes all records for AUDUSD from market_data_currencies_1min table,
trying multiple symbol formats that might exist in the database.
"""

from tradingagents.database.config import get_supabase

def delete_audusd_records():
    """Delete all AUDUSD records from market_data_currencies_1min table"""
    
    supabase = get_supabase()
    if not supabase:
        print("❌ Failed to connect to Supabase. Check your .env file.")
        return False
    
    table = "market_data_currencies_1min"
    
    # Try multiple symbol formats that might exist in the database
    # Based on the ingestion logic, AUDUSD might be stored as:
    # - C:AUDUSD (Polygon format)
    # - AUDUSD (without prefix)
    # - AUD/USD (with slash)
    symbols_to_try = ["C:AUDUSD", "AUDUSD", "AUD/USD"]
    
    total_deleted = 0
    
    for symbol in symbols_to_try:
        try:
            # First, check how many records exist (use symbol column which exists)
            count_result = supabase.table(table)\
                .select("symbol", count="exact")\
                .eq("symbol", symbol)\
                .execute()
            
            count = count_result.count if hasattr(count_result, 'count') else 0
            
            if count > 0:
                print(f"🔍 Found {count} records for symbol '{symbol}'")
                
                # Delete all records for this symbol
                # Note: Supabase delete() returns the deleted rows in the response
                delete_result = supabase.table(table)\
                    .delete()\
                    .eq("symbol", symbol)\
                    .execute()
                
                # Count deleted records (Supabase returns deleted rows in data)
                deleted_count = len(delete_result.data) if hasattr(delete_result, 'data') and delete_result.data else count
                total_deleted += deleted_count
                print(f"✅ Deleted {deleted_count} records for symbol '{symbol}'")
            else:
                print(f"ℹ️ No records found for symbol '{symbol}'")
                
        except Exception as e:
            print(f"⚠️ Error processing symbol '{symbol}': {e}")
            continue
    
    if total_deleted > 0:
        print(f"\n✅ Successfully deleted {total_deleted} total records for AUDUSD")
        print("💡 You can now run a fresh ingestion for AUDUSD")
        return True
    else:
        print("\nℹ️ No AUDUSD records found to delete")
        return False

if __name__ == "__main__":
    print("🗑️  Deleting AUDUSD records from market_data_currencies_1min...")
    print("=" * 60)
    delete_audusd_records()
    print("=" * 60)

