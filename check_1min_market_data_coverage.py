"""
Utility script to check 1-minute market data coverage for all stocks in the watchlist.
Shows which stocks have data, date ranges, and record counts.
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Optional

from tradingagents.database.config import get_supabase
from tradingagents.config.watchlist import WATCHLIST_STOCKS, get_watchlist_symbols

load_dotenv()

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365
ONE_MIN_INTERVALS_PER_DAY = 390  # 6.5 trading hours


def get_symbol_record_count(table_name: str, symbol: str, supabase) -> Optional[int]:
    """
    Use Supabase count metadata to get record volumes without downloading data.
    """
    try:
        result = supabase.table(table_name)\
            .select("id", count="exact", head=True)\
            .eq("symbol", symbol)\
            .execute()
        if hasattr(result, "count") and result.count is not None:
            return result.count
        if result.data is not None:
            return len(result.data)
    except Exception:
        return None
    return None


def estimate_trading_days(date_range_days: Optional[int]) -> Optional[int]:
    if not isinstance(date_range_days, int) or date_range_days < 0:
        return None
    return max(1, int((date_range_days + 1) * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR))


def evaluate_data_quality(record_count: Optional[int],
                          trading_days: Optional[int],
                          expected_rows_per_day: int = ONE_MIN_INTERVALS_PER_DAY) -> Dict[str, Optional[int]]:
    if record_count is None or trading_days is None:
        return {
            "expected_records": None,
            "missing_records": None,
            "quality": "Unknown"
        }

    expected_records = trading_days * expected_rows_per_day
    missing_records = expected_records - record_count

    if missing_records <= 0:
        quality = "Complete"
        missing_records = 0
    else:
        gap_ratio = missing_records / expected_records if expected_records else 0
        if gap_ratio < 0.02:
            quality = "Minor gap"
        elif gap_ratio < 0.15:
            quality = "Major gap"
        else:
            quality = "Severe gap"

    return {
        "expected_records": expected_records,
        "missing_records": missing_records,
        "quality": quality
    }


def get_1min_market_data_coverage(symbol: str, supabase) -> Dict:
    """
    Get 1-minute market data coverage statistics for a single symbol.
    
    Returns:
        Dictionary with symbol, record_count, earliest_date, latest_date, 
        date_range_days, and data_gap_info
    """
    try:
        # Get min and max timestamps for this symbol
        result = supabase.table("market_data_1min")\
            .select("timestamp")\
            .eq("symbol", symbol)\
            .order("timestamp", desc=False)\
            .limit(1)\
            .execute()
        
        earliest_timestamp = None
        if result.data and len(result.data) > 0:
            earliest_timestamp = result.data[0].get("timestamp")
        
        result = supabase.table("market_data_1min")\
            .select("timestamp")\
            .eq("symbol", symbol)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        
        latest_timestamp = None
        if result.data and len(result.data) > 0:
            latest_timestamp = result.data[0].get("timestamp")
        
        # Calculate date range first (needed for estimation)
        date_range_days = None
        earliest_dt = None
        latest_dt = None
        if earliest_timestamp and latest_timestamp:
            try:
                # Parse timestamps
                if isinstance(earliest_timestamp, str):
                    earliest_dt = datetime.fromisoformat(earliest_timestamp.replace('Z', '+00:00'))
                else:
                    earliest_dt = earliest_timestamp
                
                if isinstance(latest_timestamp, str):
                    latest_dt = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
                else:
                    latest_dt = latest_timestamp
                
                date_range_days = (latest_dt - earliest_dt).days
            except Exception as e:
                date_range_days = f"Error: {str(e)}"
        
        # Get total count using pagination (more reliable for large datasets)
        record_count = 0
        if earliest_timestamp and latest_timestamp:
            try:
                page_size = 1000
                offset = 0
                total_counted = 0
                
                # Count by paginating - but limit to reasonable number of pages
                max_pages_to_count = 1000  # Count up to 1M records, then estimate
                
                while offset < (max_pages_to_count * page_size):
                    count_result = supabase.table("market_data_1min")\
                        .select("id")\
                        .eq("symbol", symbol)\
                        .range(offset, offset + page_size - 1)\
                        .execute()
                    
                    if not count_result.data:
                        break
                    
                    batch_count = len(count_result.data)
                    total_counted += batch_count
                    
                    # If we got fewer records than page_size, we've reached the end
                    if batch_count < page_size:
                        record_count = total_counted
                        break
                    
                    offset += page_size
                
                # If we hit the max pages limit, estimate based on date range
                if offset >= (max_pages_to_count * page_size):
                    # Estimate: ~390 records per trading day for 1-min data
                    # (390 minutes per trading day)
                    if date_range_days and isinstance(date_range_days, int):
                        # Rough estimate: assume 252 trading days per year
                        trading_days = int(date_range_days * 252 / 365)
                        estimated_records = trading_days * 390
                        record_count = f"~{estimated_records:,} (estimated, >{total_counted:,} confirmed)"
                    else:
                        record_count = f">{total_counted:,} (counting stopped at {total_counted:,})"
                else:
                    record_count = total_counted
                    
            except Exception as e:
                # Fallback: try to get at least a sample count
                try:
                    sample_result = supabase.table("market_data_1min")\
                        .select("timestamp")\
                        .eq("symbol", symbol)\
                        .limit(100)\
                        .execute()
                    if sample_result.data:
                        record_count = f">={len(sample_result.data)} (sampled)"
                    else:
                        record_count = 0
                except:
                    record_count = f"Error counting: {str(e)}"
        else:
            record_count = 0
        
        trading_days = estimate_trading_days(date_range_days) if isinstance(date_range_days, int) else None
        record_count_for_quality = record_count if isinstance(record_count, int) else None
        quality_info = evaluate_data_quality(record_count_for_quality, trading_days, ONE_MIN_INTERVALS_PER_DAY)

        return {
            "symbol": symbol,
            "record_count": record_count,
            "earliest_date": earliest_timestamp,
            "latest_date": latest_timestamp,
            "date_range_days": date_range_days,
            "has_data": earliest_timestamp is not None and latest_timestamp is not None,
            "expected_records": quality_info["expected_records"],
            "missing_records": quality_info["missing_records"],
            "data_quality": quality_info["quality"]
        }
        
    except Exception as e:
        return {
            "symbol": symbol,
            "record_count": 0,
            "earliest_date": None,
            "latest_date": None,
            "date_range_days": None,
            "has_data": False,
            "expected_records": None,
            "missing_records": None,
            "data_quality": "Error",
            "error": str(e)
        }


def get_quick_1min_coverage_summary() -> pd.DataFrame:
    """
    Quick coverage check - shows date ranges without counting records.
    Much faster for large datasets.
    
    Returns:
        DataFrame with coverage statistics (date ranges only, no counts)
    """
    supabase = get_supabase()
    if not supabase:
        print("Error: Supabase not configured. Please check your .env file.")
        return pd.DataFrame()
    
    symbols = get_watchlist_symbols()
    coverage_data = []
    
    print(f"Quick check: Getting date ranges and counts for {len(symbols)} stocks (1-minute data)...")
    print("=" * 80)
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Checking {symbol}...", end=" ")
        try:
            # Get earliest timestamp
            result = supabase.table("market_data_1min")\
                .select("timestamp")\
                .eq("symbol", symbol)\
                .order("timestamp", desc=False)\
                .limit(1)\
                .execute()
            
            earliest_timestamp = result.data[0].get("timestamp") if result.data else None
            
            # Get latest timestamp
            result = supabase.table("market_data_1min")\
                .select("timestamp")\
                .eq("symbol", symbol)\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()
            
            latest_timestamp = result.data[0].get("timestamp") if result.data else None
            
            if earliest_timestamp and latest_timestamp:
                # Parse and format dates
                try:
                    if isinstance(earliest_timestamp, str):
                        earliest_dt = datetime.fromisoformat(earliest_timestamp.replace('Z', '+00:00'))
                    else:
                        earliest_dt = earliest_timestamp
                    
                    if isinstance(latest_timestamp, str):
                        latest_dt = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
                    else:
                        latest_dt = latest_timestamp
                    
                    date_range_days = (latest_dt - earliest_dt).days
                    earliest_str = earliest_dt.strftime("%Y-%m-%d")
                    latest_str = latest_dt.strftime("%Y-%m-%d")
                    
                    record_count = get_symbol_record_count("market_data_1min", symbol, supabase)
                    trading_days = estimate_trading_days(date_range_days)
                    quality_info = evaluate_data_quality(record_count, trading_days, ONE_MIN_INTERVALS_PER_DAY)

                    coverage_data.append({
                        "Symbol": symbol,
                        "Stock Name": WATCHLIST_STOCKS.get(symbol, "Unknown"),
                        "Earliest Date": earliest_str,
                        "Latest Date": latest_str,
                        "Date Range (Days)": date_range_days,
                        "Record Count": record_count if record_count is not None else "Unknown",
                        "Expected Records": quality_info["expected_records"],
                        "Missing Records": quality_info["missing_records"],
                        "Data Quality": quality_info["quality"],
                        "Has Data": "Yes"
                    })
                    record_str = record_count if record_count is not None else "?"
                    print(f"✓ {earliest_str} to {latest_str} ({date_range_days} days, {record_str} records)")
                except Exception as e:
                    coverage_data.append({
                        "Symbol": symbol,
                        "Stock Name": WATCHLIST_STOCKS.get(symbol, "Unknown"),
                        "Earliest Date": "Error",
                        "Latest Date": "Error",
                        "Date Range (Days)": "Error",
                        "Record Count": "Error",
                        "Expected Records": None,
                        "Missing Records": None,
                        "Data Quality": "Error",
                        "Has Data": "Error",
                        "Error": str(e)
                    })
                    print(f"✗ Error: {e}")
            else:
                coverage_data.append({
                    "Symbol": symbol,
                    "Stock Name": WATCHLIST_STOCKS.get(symbol, "Unknown"),
                    "Earliest Date": "No data",
                    "Latest Date": "No data",
                    "Date Range (Days)": 0,
                    "Record Count": 0,
                    "Expected Records": 0,
                    "Missing Records": None,
                    "Data Quality": "No data",
                    "Has Data": "No"
                })
                print(f"✗ No data")
        except Exception as e:
            coverage_data.append({
                "Symbol": symbol,
                "Stock Name": WATCHLIST_STOCKS.get(symbol, "Unknown"),
                "Earliest Date": "Error",
                "Latest Date": "Error",
                "Date Range (Days)": "Error",
                "Record Count": "Error",
                "Expected Records": None,
                "Missing Records": None,
                "Data Quality": "Error",
                "Has Data": "Error",
                "Error": str(e)
            })
            print(f"✗ Error: {e}")
    
    print("=" * 80)
    return pd.DataFrame(coverage_data)


def get_all_stocks_1min_coverage() -> pd.DataFrame:
    """
    Get 1-minute market data coverage for all stocks in the watchlist.
    
    Returns:
        DataFrame with coverage statistics for all stocks
    """
    supabase = get_supabase()
    if not supabase:
        print("Error: Supabase not configured. Please check your .env file.")
        return pd.DataFrame()
    
    symbols = get_watchlist_symbols()
    coverage_data = []
    
    print(f"Checking 1-minute market data coverage for {len(symbols)} stocks...")
    print("=" * 80)
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Checking {symbol}...", end=" ")
        coverage = get_1min_market_data_coverage(symbol, supabase)
        coverage_data.append(coverage)
        
        if coverage["has_data"]:
            print(f"✓ Found {coverage['record_count']} records")
        else:
            print(f"✗ No data found")
    
    print("=" * 80)
    print("\nCreating coverage report...\n")
    
    # Create DataFrame
    df_data = []
    for coverage in coverage_data:
        earliest_str = ""
        latest_str = ""
        if coverage.get("earliest_date"):
            try:
                if isinstance(coverage["earliest_date"], str):
                    dt = datetime.fromisoformat(coverage["earliest_date"].replace('Z', '+00:00'))
                else:
                    dt = coverage["earliest_date"]
                earliest_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                earliest_str = str(coverage["earliest_date"])[:19]
        
        if coverage.get("latest_date"):
            try:
                if isinstance(coverage["latest_date"], str):
                    dt = datetime.fromisoformat(coverage["latest_date"].replace('Z', '+00:00'))
                else:
                    dt = coverage["latest_date"]
                latest_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                latest_str = str(coverage["latest_date"])[:19]
        
        df_data.append({
            "Symbol": coverage["symbol"],
            "Stock Name": WATCHLIST_STOCKS.get(coverage["symbol"], "Unknown"),
            "Record Count": coverage["record_count"],
            "Earliest Date": earliest_str if earliest_str else "No data",
            "Latest Date": latest_str if latest_str else "No data",
            "Date Range (Days)": coverage.get("date_range_days", "N/A"),
            "Expected Records": coverage.get("expected_records"),
            "Missing Records": coverage.get("missing_records"),
            "Data Quality": coverage.get("data_quality", "Unknown"),
            "Has Data": "Yes" if coverage["has_data"] else "No",
            "Error": coverage.get("error", "")
        })
    
    df = pd.DataFrame(df_data)
    return df


if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("1-MINUTE MARKET DATA COVERAGE REPORT")
    print("=" * 80)
    print()
    
    # Check if user wants full report (with counts) or quick report (date ranges only)
    use_full_report = "--full" in sys.argv or "-f" in sys.argv
    
    if use_full_report:
        print("Running FULL report (with record counts - may be slow for large datasets)...")
        print()
        # Get coverage for all stocks with counts
        df = get_all_stocks_1min_coverage()
    else:
        print("Running QUICK report (date ranges + counts, fast)...")
        print("(Use --full or -f flag for exhaustive pagination counts)")
        print()
        # Get quick coverage (date ranges + lightweight counts)
        df = get_quick_1min_coverage_summary()
    
    if not df.empty:
        # Display summary
        print("\n" + "=" * 80)
        print("COVERAGE SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        print()
        
        # Save to CSV
        output_file = "1min_market_data_coverage_report_quick.csv" if not use_full_report else "1min_market_data_coverage_report_full.csv"
        df.to_csv(output_file, index=False)
        print(f"✓ Report saved to {output_file}")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        total_stocks = len(df)
        stocks_with_data = len(df[df["Has Data"] == "Yes"])
        stocks_without_data = total_stocks - stocks_with_data
        
        print(f"Total stocks in watchlist: {total_stocks}")
        print(f"Stocks with data: {stocks_with_data}")
        print(f"Stocks without data: {stocks_without_data}")
        
        if stocks_with_data > 0:
            # Calculate average date range (always available)
            date_ranges = []
            for days in df[df["Has Data"] == "Yes"]["Date Range (Days)"]:
                try:
                    if isinstance(days, (int, float)) and days > 0:
                        date_ranges.append(days)
                except:
                    pass
            
            if date_ranges:
                avg_days = sum(date_ranges) / len(date_ranges)
                print(f"Average date range: {avg_days:.0f} days")
                print(f"Min date range: {min(date_ranges):.0f} days")
                print(f"Max date range: {max(date_ranges):.0f} days")
            
            # Calculate average record count (only for full reports)
            if "Record Count" in df.columns:
                record_counts = []
                for count in df[df["Has Data"] == "Yes"]["Record Count"]:
                    try:
                        if isinstance(count, (int, float)):
                            record_counts.append(count)
                        elif isinstance(count, str) and count.replace(',', '').isdigit():
                            record_counts.append(int(count.replace(',', '')))
                    except:
                        pass
                
                if record_counts:
                    avg_records = sum(record_counts) / len(record_counts)
                    print(f"Average records per stock (with data): {avg_records:,.0f}")
                    print(f"Min records: {min(record_counts):,}")
                    print(f"Max records: {max(record_counts):,}")
        
        print("\n" + "=" * 80)
        print("DETAILED VIEW (stocks with data)")
        print("=" * 80)
        
        # Show detailed info for stocks with data
        stocks_with_data_df = df[df["Has Data"] == "Yes"]
        if not stocks_with_data_df.empty:
            # Select columns based on what's available
            cols_to_show = ["Symbol", "Stock Name"]
            if "Record Count" in stocks_with_data_df.columns:
                cols_to_show.append("Record Count")
            cols_to_show.extend(["Expected Records", "Missing Records", "Data Quality",
                                 "Earliest Date", "Latest Date", "Date Range (Days)"])
            existing_cols = [col for col in cols_to_show if col in stocks_with_data_df.columns]
            print(stocks_with_data_df[existing_cols].to_string(index=False))
        
        print("\n" + "=" * 80)
    else:
        print("Error: Could not generate coverage report")

