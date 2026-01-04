"""
Verify that Gold (C:XAUUSD) 1-minute data has no gaps throughout the ingestion period.

This script:
1. Fetches all Gold data from the database
2. Generates expected complete time series (every minute)
3. Identifies missing minutes/gaps
4. Reports statistics and gaps found
5. Handles GMT+4 timezone (Asia/Dubai) properly
"""
import pandas as pd
import pytz
from datetime import datetime, timedelta
from tradingagents.database.config import get_supabase
from typing import List, Tuple, Optional
import sys

def get_gold_data_from_db(symbol: str = "C:XAUUSD", table: str = "market_data_commodities_1min") -> Tuple[pd.DataFrame, str]:
    """Fetch all Gold data from database"""
    supabase = get_supabase()
    if not supabase:
        print("❌ Error: Supabase not configured")
        return pd.DataFrame(), symbol
    
    print(f"📥 Fetching all {symbol} data from {table}...")
    
    # Try multiple symbol formats
    symbols_to_try = [symbol, "C:XAUUSD", "^XAUUSD", "GOLD"]
    
    all_data = []
    found_symbol = None
    
    for try_sym in symbols_to_try:
        try:
            # First, get count
            count_resp = supabase.table(table)\
                .select("id", count="exact")\
                .eq("symbol", try_sym)\
                .execute()
            
            count = count_resp.count if hasattr(count_resp, 'count') else 0
            
            if count > 0:
                print(f"   Found {count} records for symbol '{try_sym}'")
                found_symbol = try_sym
                
                # Fetch all data in batches (Supabase limit is 1000 per query)
                offset = 0
                limit = 1000
                total_fetched = 0
                
                while True:
                    response = supabase.table(table)\
                        .select("timestamp, symbol, open, high, low, close, volume")\
                        .eq("symbol", try_sym)\
                        .order("timestamp", desc=False)\
                        .range(offset, offset + limit - 1)\
                        .execute()
                    
                    if not response.data or len(response.data) == 0:
                        break
                    
                    all_data.extend(response.data)
                    total_fetched += len(response.data)
                    
                    if len(response.data) < limit:
                        break
                    
                    offset += limit
                    if total_fetched % 10000 == 0:
                        print(f"   Fetched {total_fetched} records...")
                
                break
        except Exception as e:
            print(f"   ⚠️ Error querying symbol '{try_sym}': {e}")
            continue
    
    if not all_data:
        print(f"❌ No data found for any symbol variant")
        return pd.DataFrame(), symbol
    
    print(f"✅ Fetched {len(all_data)} total records for symbol '{found_symbol}'")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df, found_symbol


def generate_expected_timeseries(start: datetime, end: datetime, timezone_str: str = "Asia/Dubai") -> pd.DatetimeIndex:
    """Generate expected complete 1-minute time series"""
    tz = pytz.timezone(timezone_str)
    
    # Ensure start and end are timezone-aware in GMT+4
    if start.tzinfo is None:
        start = tz.localize(start)
    else:
        start = start.astimezone(tz)
    
    if end.tzinfo is None:
        end = tz.localize(end)
    else:
        end = end.astimezone(tz)
    
    # Round to minute boundaries
    start = start.replace(second=0, microsecond=0)
    end = end.replace(second=0, microsecond=0)
    
    # Generate expected timestamps (every minute)
    expected_ts = pd.date_range(start=start, end=end, freq='1min', tz=tz)
    
    return expected_ts


def find_gaps(actual_df: pd.DataFrame, expected_ts: pd.DatetimeIndex, symbol: str) -> Tuple[List[datetime], dict]:
    """Find missing minutes in the data"""
    if actual_df.empty:
        return list(expected_ts), {
            'total_expected': len(expected_ts),
            'total_actual': 0,
            'missing_count': len(expected_ts),
            'completeness_pct': 0.0
        }
    
    # Ensure timestamp column is timezone-aware
    if actual_df['timestamp'].dt.tz is None:
        # Assume GMT+4 if naive
        tz = pytz.timezone("Asia/Dubai")
        actual_df['timestamp'] = actual_df['timestamp'].dt.tz_localize(tz)
    else:
        # Convert to GMT+4 if needed
        tz = pytz.timezone("Asia/Dubai")
        actual_df['timestamp'] = actual_df['timestamp'].dt.tz_convert(tz)
    
    # Round actual timestamps to minute boundaries
    actual_df['timestamp_rounded'] = actual_df['timestamp'].dt.floor('1min')
    
    # Get unique actual timestamps
    actual_ts_set = set(actual_df['timestamp_rounded'].unique())
    expected_ts_set = set(expected_ts)
    
    # Find missing timestamps
    missing_ts = sorted(expected_ts_set - actual_ts_set)
    
    # Calculate statistics
    total_expected = len(expected_ts)
    total_actual = len(actual_ts_set)
    missing_count = len(missing_ts)
    completeness_pct = (total_actual / total_expected * 100) if total_expected > 0 else 0.0
    
    stats = {
        'total_expected': total_expected,
        'total_actual': total_actual,
        'missing_count': missing_count,
        'completeness_pct': completeness_pct,
        'duplicate_count': len(actual_df) - total_actual
    }
    
    return missing_ts, stats


def analyze_gaps(missing_ts: List[datetime]) -> dict:
    """Analyze gaps to find continuous missing periods"""
    if not missing_ts:
        return {'gap_periods': [], 'largest_gap_minutes': 0, 'total_gaps': 0}
    
    gap_periods = []
    current_gap_start = missing_ts[0]
    current_gap_end = missing_ts[0]
    largest_gap = 1
    
    for i in range(1, len(missing_ts)):
        # Check if this timestamp is 1 minute after the previous
        time_diff = (missing_ts[i] - missing_ts[i-1]).total_seconds() / 60
        
        if time_diff == 1:
            # Continuous gap
            current_gap_end = missing_ts[i]
        else:
            # Gap ended, save it
            gap_duration = (current_gap_end - current_gap_start).total_seconds() / 60 + 1
            gap_periods.append({
                'start': current_gap_start,
                'end': current_gap_end,
                'duration_minutes': int(gap_duration)
            })
            largest_gap = max(largest_gap, gap_duration)
            
            # Start new gap
            current_gap_start = missing_ts[i]
            current_gap_end = missing_ts[i]
    
    # Don't forget the last gap
    if current_gap_start:
        gap_duration = (current_gap_end - current_gap_start).total_seconds() / 60 + 1
        gap_periods.append({
            'start': current_gap_start,
            'end': current_gap_end,
            'duration_minutes': int(gap_duration)
        })
        largest_gap = max(largest_gap, gap_duration)
    
    return {
        'gap_periods': gap_periods,
        'largest_gap_minutes': int(largest_gap),
        'total_gaps': len(gap_periods)
    }


def verify_gold_completeness(symbol: str = "C:XAUUSD", max_gaps_to_show: int = 20):
    """Main verification function"""
    print("=" * 80)
    print("🔍 GOLD DATA COMPLETENESS VERIFICATION")
    print("=" * 80)
    print()
    
    # Fetch data
    df, found_symbol = get_gold_data_from_db(symbol)
    
    if df.empty:
        print("❌ No data found. Cannot verify completeness.")
        return False
    
    # Get date range
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    
    print(f"\n📊 Data Range:")
    print(f"   Start: {start_time}")
    print(f"   End:   {end_time}")
    print(f"   Duration: {(end_time - start_time).days} days, {(end_time - start_time).seconds // 3600} hours")
    print()
    
    # Generate expected time series
    print("🔄 Generating expected complete time series...")
    expected_ts = generate_expected_timeseries(start_time, end_time, "Asia/Dubai")
    print(f"   Expected: {len(expected_ts)} minutes")
    print()
    
    # Find gaps
    print("🔍 Analyzing for gaps...")
    missing_ts, stats = find_gaps(df, expected_ts, found_symbol)
    
    # Analyze gaps
    gap_analysis = analyze_gaps(missing_ts)
    
    # Print results
    print("=" * 80)
    print("📈 COMPLETENESS STATISTICS")
    print("=" * 80)
    print(f"Total Expected Minutes: {stats['total_expected']:,}")
    print(f"Total Actual Minutes:    {stats['total_actual']:,}")
    print(f"Missing Minutes:         {stats['missing_count']:,}")
    print(f"Completeness:            {stats['completeness_pct']:.4f}%")
    print(f"Duplicate Records:       {stats['duplicate_count']:,}")
    print()
    
    print("=" * 80)
    print("🔎 GAP ANALYSIS")
    print("=" * 80)
    print(f"Total Gap Periods:       {gap_analysis['total_gaps']}")
    print(f"Largest Gap:             {gap_analysis['largest_gap_minutes']} minutes ({gap_analysis['largest_gap_minutes']/60:.2f} hours)")
    print()
    
    if missing_ts:
        print(f"⚠️  GAPS FOUND! Showing first {min(max_gaps_to_show, len(missing_ts))} missing timestamps:")
        print()
        
        # Show individual missing timestamps (limited)
        for i, ts in enumerate(missing_ts[:max_gaps_to_show]):
            print(f"   {i+1}. {ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        if len(missing_ts) > max_gaps_to_show:
            print(f"\n   ... and {len(missing_ts) - max_gaps_to_show} more missing timestamps")
        
        print()
        
        # Show gap periods
        if gap_analysis['gap_periods']:
            print("📅 Gap Periods (continuous missing periods):")
            print()
            for i, gap in enumerate(gap_analysis['gap_periods'][:max_gaps_to_show]):
                print(f"   Gap {i+1}:")
                print(f"      Start:  {gap['start'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"      End:    {gap['end'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"      Duration: {gap['duration_minutes']} minutes ({gap['duration_minutes']/60:.2f} hours)")
                print()
            
            if len(gap_analysis['gap_periods']) > max_gaps_to_show:
                print(f"   ... and {len(gap_analysis['gap_periods']) - max_gaps_to_show} more gap periods")
    else:
        print("✅ NO GAPS FOUND! Data is 100% complete.")
        print()
    
    # Final verdict
    print("=" * 80)
    if stats['missing_count'] == 0:
        print("✅ VERDICT: DATA IS COMPLETE - NO MISSING MINUTES")
        return True
    else:
        print(f"❌ VERDICT: DATA HAS GAPS - {stats['missing_count']:,} MISSING MINUTES ({100 - stats['completeness_pct']:.2f}% missing)")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Gold data completeness")
    parser.add_argument("--symbol", type=str, default="C:XAUUSD", help="Symbol to check (default: C:XAUUSD)")
    parser.add_argument("--max-gaps", type=int, default=20, help="Max gaps to display (default: 20)")
    
    args = parser.parse_args()
    
    success = verify_gold_completeness(symbol=args.symbol, max_gaps_to_show=args.max_gaps)
    
    sys.exit(0 if success else 1)

