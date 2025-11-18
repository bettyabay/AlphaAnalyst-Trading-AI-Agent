"""
Utility script to check daily market data coverage for all stocks in the watchlist.
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


def get_daily_market_data_coverage(symbol: str, supabase) -> Dict:
    """
    Get daily market data coverage statistics for a single symbol.
    
    Returns:
        Dictionary with symbol, record_count, earliest_date, latest_date, 
        date_range_days, and data_gap_info
    """
    try:
        # Get min and max dates for this symbol
        result = supabase.table("market_data")\
            .select("date")\
            .eq("symbol", symbol)\
            .order("date", desc=False)\
            .limit(1)\
            .execute()
        
        earliest_date = None
        if result.data and len(result.data) > 0:
            earliest_date = result.data[0].get("date")
        
        result = supabase.table("market_data")\
            .select("date")\
            .eq("symbol", symbol)\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        latest_date = None
        if result.data and len(result.data) > 0:
            latest_date = result.data[0].get("date")
        
        # Calculate date range first (needed for estimation)
        date_range_days = None
        earliest_dt = None
        latest_dt = None
        if earliest_date and latest_date:
            try:
                # Parse dates
                if isinstance(earliest_date, str):
                    earliest_dt = datetime.strptime(earliest_date, "%Y-%m-%d")
                else:
                    earliest_dt = earliest_date
                
                if isinstance(latest_date, str):
                    latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
                else:
                    latest_dt = latest_date
                
                date_range_days = (latest_dt - earliest_dt).days
            except Exception as e:
                date_range_days = f"Error: {str(e)}"
        
        # Get total count using pagination (more reliable for large datasets)
        record_count = 0
        if earliest_date and latest_date:
            try:
                page_size = 1000
                offset = 0
                total_counted = 0
                
                # Count by paginating - but limit to reasonable number of pages
                max_pages_to_count = 100  # Count up to 100K records, then estimate
                
                while offset < (max_pages_to_count * page_size):
                    count_result = supabase.table("market_data")\
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
                    # Estimate: ~252 trading days per year
                    if date_range_days and isinstance(date_range_days, int):
                        # Rough estimate: assume 252 trading days per year
                        trading_days = int(date_range_days * 252 / 365)
                        estimated_records = trading_days
                        record_count = f"~{estimated_records:,} (estimated, >{total_counted:,} confirmed)"
                    else:
                        record_count = f">{total_counted:,} (counting stopped at {total_counted:,})"
                else:
                    record_count = total_counted
                    
            except Exception as e:
                # Fallback: try to get at least a sample count
                try:
                    sample_result = supabase.table("market_data")\
                        .select("date")\
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
        
        return {
            "symbol": symbol,
            "record_count": record_count,
            "earliest_date": earliest_date,
            "latest_date": latest_date,
            "date_range_days": date_range_days,
            "has_data": earliest_date is not None and latest_date is not None
        }
        
    except Exception as e:
        return {
            "symbol": symbol,
            "record_count": 0,
            "earliest_date": None,
            "latest_date": None,
            "date_range_days": None,
            "has_data": False,
            "error": str(e)
        }


def get_quick_daily_coverage_summary() -> pd.DataFrame:
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
    
    print(f"Quick check: Getting date ranges for {len(symbols)} stocks (daily data)...")
    print("=" * 80)
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Checking {symbol}...", end=" ")
        try:
            # Get earliest date
            result = supabase.table("market_data")\
                .select("date")\
                .eq("symbol", symbol)\
                .order("date", desc=False)\
                .limit(1)\
                .execute()
            
            earliest_date = result.data[0].get("date") if result.data else None
            
            # Get latest date
            result = supabase.table("market_data")\
                .select("date")\
                .eq("symbol", symbol)\
                .order("date", desc=True)\
                .limit(1)\
                .execute()
            
            latest_date = result.data[0].get("date") if result.data else None
            
            if earliest_date and latest_date:
                # Parse and format dates
                try:
                    if isinstance(earliest_date, str):
                        earliest_dt = datetime.strptime(earliest_date, "%Y-%m-%d")
                    else:
                        earliest_dt = earliest_date
                    
                    if isinstance(latest_date, str):
                        latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
                    else:
                        latest_dt = latest_date
                    
                    date_range_days = (latest_dt - earliest_dt).days
                    earliest_str = earliest_dt.strftime("%Y-%m-%d")
                    latest_str = latest_dt.strftime("%Y-%m-%d")
                    
                    coverage_data.append({
                        "Symbol": symbol,
                        "Stock Name": WATCHLIST_STOCKS.get(symbol, "Unknown"),
                        "Earliest Date": earliest_str,
                        "Latest Date": latest_str,
                        "Date Range (Days)": date_range_days,
                        "Has Data": "Yes"
                    })
                    print(f"✓ {earliest_str} to {latest_str} ({date_range_days} days)")
                except Exception as e:
                    coverage_data.append({
                        "Symbol": symbol,
                        "Stock Name": WATCHLIST_STOCKS.get(symbol, "Unknown"),
                        "Earliest Date": "Error",
                        "Latest Date": "Error",
                        "Date Range (Days)": "Error",
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
                "Has Data": "Error",
                "Error": str(e)
            })
            print(f"✗ Error: {e}")
    
    print("=" * 80)
    return pd.DataFrame(coverage_data)


def get_all_stocks_daily_coverage() -> pd.DataFrame:
    """
    Get daily market data coverage for all stocks in the watchlist.
    
    Returns:
        DataFrame with coverage statistics for all stocks
    """
    supabase = get_supabase()
    if not supabase:
        print("Error: Supabase not configured. Please check your .env file.")
        return pd.DataFrame()
    
    symbols = get_watchlist_symbols()
    coverage_data = []
    
    print(f"Checking daily market data coverage for {len(symbols)} stocks...")
    print("=" * 80)
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Checking {symbol}...", end=" ")
        coverage = get_daily_market_data_coverage(symbol, supabase)
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
                    dt = datetime.strptime(coverage["earliest_date"], "%Y-%m-%d")
                else:
                    dt = coverage["earliest_date"]
                earliest_str = dt.strftime("%Y-%m-%d")
            except:
                earliest_str = str(coverage["earliest_date"])[:10]
        
        if coverage.get("latest_date"):
            try:
                if isinstance(coverage["latest_date"], str):
                    dt = datetime.strptime(coverage["latest_date"], "%Y-%m-%d")
                else:
                    dt = coverage["latest_date"]
                latest_str = dt.strftime("%Y-%m-%d")
            except:
                latest_str = str(coverage["latest_date"])[:10]
        
        df_data.append({
            "Symbol": coverage["symbol"],
            "Stock Name": WATCHLIST_STOCKS.get(coverage["symbol"], "Unknown"),
            "Record Count": coverage["record_count"],
            "Earliest Date": earliest_str if earliest_str else "No data",
            "Latest Date": latest_str if latest_str else "No data",
            "Date Range (Days)": coverage.get("date_range_days", "N/A"),
            "Has Data": "Yes" if coverage["has_data"] else "No",
            "Error": coverage.get("error", "")
        })
    
    df = pd.DataFrame(df_data)
    return df


if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("DAILY MARKET DATA COVERAGE REPORT")
    print("=" * 80)
    print()
    
    # Check if user wants full report (with counts) or quick report (date ranges only)
    use_full_report = "--full" in sys.argv or "-f" in sys.argv
    
    if use_full_report:
        print("Running FULL report (with record counts - may be slow for large datasets)...")
        print()
        # Get coverage for all stocks with counts
        df = get_all_stocks_daily_coverage()
    else:
        print("Running QUICK report (date ranges only - fast)...")
        print("(Use --full or -f flag for full report with record counts)")
        print()
        # Get quick coverage (date ranges only, no counts)
        df = get_quick_daily_coverage_summary()
    
    if not df.empty:
        # Display summary
        print("\n" + "=" * 80)
        print("COVERAGE SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        print()
        
        # Save to CSV
        output_file = "daily_market_data_coverage_report_quick.csv" if not use_full_report else "daily_market_data_coverage_report_full.csv"
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
                
                # Calculate expected trading days (252 per year)
                avg_trading_days = int(avg_days * 252 / 365)
                print(f"Expected trading days (approx): {avg_trading_days} days")
            
            # Calculate average record count (only for full reports)
            if "Record Count" in df.columns:
                record_counts = []
                for count in df[df["Has Data"] == "Yes"]["Record Count"]:
                    try:
                        if isinstance(count, (int, float)):
                            record_counts.append(count)
                        elif isinstance(count, str) and count.replace(',', '').replace('~', '').replace('>', '').replace('(', '').isdigit():
                            # Extract number from strings like "~504 (estimated)"
                            import re
                            numbers = re.findall(r'\d+', count.replace(',', ''))
                            if numbers:
                                record_counts.append(int(numbers[0]))
                    except:
                        pass
                
                if record_counts:
                    avg_records = sum(record_counts) / len(record_counts)
                    print(f"Average records per stock (with data): {avg_records:,.0f}")
                    print(f"Min records: {min(record_counts):,}")
                    print(f"Max records: {max(record_counts):,}")
                    
                    # Calculate expected total
                    expected_per_symbol = avg_trading_days if date_ranges else 504  # 2 years = ~504 trading days
                    expected_total = expected_per_symbol * total_stocks
                    actual_total = sum(record_counts)
                    print(f"\nExpected total records (19 symbols × {expected_per_symbol} trading days): {expected_total:,}")
                    print(f"Actual total records: {actual_total:,}")
                    if actual_total < expected_total * 0.8:
                        print(f"⚠️  WARNING: Only {actual_total/expected_total*100:.1f}% of expected records found!")
        
        print("\n" + "=" * 80)
        print("DETAILED VIEW (stocks with data)")
        print("=" * 80)
        
        # Show detailed info for stocks with data
        stocks_with_data_df = df[df["Has Data"] == "Yes"]
        if not stocks_with_data_df.empty:
            # Select columns based on what's available
            cols_to_show = ["Symbol", "Stock Name", "Earliest Date", "Latest Date", "Date Range (Days)"]
            if "Record Count" in stocks_with_data_df.columns:
                cols_to_show.insert(2, "Record Count")  # Insert after "Stock Name"
            print(stocks_with_data_df[cols_to_show].to_string(index=False))
        
        print("\n" + "=" * 80)
    else:
        print("Error: Could not generate coverage report")

