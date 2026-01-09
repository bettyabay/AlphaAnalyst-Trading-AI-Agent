import pandas as pd
from tradingagents.database.config import get_supabase
from tradingagents.dataflows.polygon_integration import PolygonDataClient
from tradingagents.dataflows.ingestion_pipeline import convert_instrument_to_polygon_symbol
import io
import traceback
from datetime import datetime, timedelta
import pytz

def ingest_from_polygon_api(api_symbol, asset_class, start_date=None, end_date=None, years=2, db_symbol=None, auto_resume=True):
    """
    Ingest data directly from Polygon API into the appropriate Supabase table.
    Handles timezone conversion to GMT+4 (Dubai).
    Automatically resumes from latest timestamp if auto_resume=True.
    
    Default: Fetches last 2 years of 1-minute data (from 2024 to now).
    """
    """
    Ingest data directly from Polygon API into the appropriate Supabase table.
    Handles timezone conversion to GMT+4 (Dubai).
    Automatically resumes from latest timestamp if auto_resume=True.
    
    Args:
        api_symbol: Polygon symbol (e.g. "AAPL", "C:XAUUSD") or Barchart symbol to be converted
        asset_class: "Commodities", "Indices", "Currencies", "Stocks"
        start_date: Optional start date (datetime or YYYYMMDD string). Ignored if auto_resume=True and data exists.
        end_date: Optional end date (datetime or YYYYMMDD string). Defaults to now_utc - 15min if not provided.
        years: Number of years to fetch if start_date is not provided (only used if no existing data)
        db_symbol: Symbol to store in DB (defaults to api_symbol if None)
        auto_resume: If True, automatically resume from latest timestamp in DB. If False, use start_date.
        
    Returns:
        dict: {"success": bool, "message": str}
    """
    # Determine target table
    table_map = {
        "Commodities": "market_data_commodities_1min",
        "Indices": "market_data_indices_1min",
        "Currencies": "market_data_currencies_1min",
        "Stocks": "market_data_stocks_1min"
    }
    
    target_table = table_map.get(asset_class)
    if not target_table:
        return {"success": False, "message": f"Unknown asset class: {asset_class}"}

    try:
        print(f"🚀 Starting ingestion for symbol: '{api_symbol}', asset_class: '{asset_class}', auto_resume: {auto_resume}")
        
        sb = get_supabase()
        if not sb:
            return {"success": False, "message": "Supabase not configured (check .env)"}
        
        print(f"✅ Supabase connection established")
        
        # Get current UTC time once (critical for date validation)
        now_utc = datetime.utcnow()
        current_year = now_utc.year
        print(f"📅 Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}, Current year: {current_year}")
        
        # Validate input symbol first - strip all whitespace including newlines
        api_symbol = api_symbol.strip().replace('\n', '').replace('\r', '').replace('\t', '') if api_symbol else ""
        
        # CRITICAL: Check for truncated/incomplete symbols BEFORE processing
        if api_symbol in ["C", "I", "C:", "I:"]:
            examples = {
                "Commodities": "C:XAUUSD (for Gold)",
                "Indices": "I:SPX (for S&P 500), I:DJI (for Dow Jones)",
                "Currencies": "C:EURUSD, C:GBPUSD, C:USDJPY",
                "Stocks": "AAPL, MSFT, GOOGL"
            }
            example = examples.get(asset_class, "I:SPX, C:EURUSD, AAPL")
            return {"success": False, "message": f"❌ Symbol '{api_symbol}' is incomplete/truncated. It appears only the prefix was entered.\n\nFor {asset_class}, please enter a complete symbol like: {example}\n\n💡 If you're seeing this error repeatedly, the symbol input field may have been corrupted. Try refreshing the page and re-entering the symbol."}
        
        if not api_symbol or len(api_symbol) <= 1:
            # Provide helpful examples based on asset class
            examples = {
                "Commodities": "C:XAUUSD (for Gold)",
                "Indices": "I:SPX (for S&P 500), I:DJI (for Dow Jones)",
                "Currencies": "C:EURUSD, C:GBPUSD, C:USDJPY",
                "Stocks": "AAPL, MSFT, GOOGL"
            }
            example = examples.get(asset_class, "I:SPX, C:EURUSD, AAPL")
            return {"success": False, "message": f"❌ Invalid API symbol: '{api_symbol}'. Please enter a full symbol.\n\nExamples for {asset_class}: {example}\n\nNote: For indices, use format 'I:SYMBOL' (e.g., I:SPX). For currencies, use 'C:PAIR' (e.g., C:EURUSD)."}
        
        # Determine end date (default to now_utc - 15min for intraday safety buffer)
        if not end_date:
            end_dt = now_utc - timedelta(minutes=15)
        elif isinstance(end_date, str):
            # Try multiple date formats
            try:
                end_dt = datetime.strptime(end_date, "%Y%m%d")
            except ValueError:
                try:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                except ValueError:
                    try:
                        end_dt = datetime.fromisoformat(end_date)
                    except ValueError:
                        return {"success": False, "message": f"Invalid end_date format: '{end_date}'. Use YYYY-MM-DD or YYYYMMDD"}
            
            # If end_date is today, cap it to now_utc - 15 minutes for safety
            # This prevents requesting incomplete day data that causes 403 errors
            today_date = now_utc.date()
            if end_dt.date() == today_date and end_dt > (now_utc - timedelta(minutes=15)):
                end_dt = now_utc - timedelta(minutes=15)
                print(f"📅 End date is today - capping to current time minus 15min safety buffer: {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        else:
            end_dt = end_date
            # If end_date is today, cap it to now_utc - 15 minutes for safety
            today_date = now_utc.date()
            if end_dt.date() == today_date and end_dt > (now_utc - timedelta(minutes=15)):
                end_dt = now_utc - timedelta(minutes=15)
                print(f"📅 End date is today - capping to current time minus 15min safety buffer: {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        # Ensure end_date is not in the future (return error instead of clamping)
        # Allow same day (end_dt can be today, just not in the future)
        # Check if year is more than 1 year ahead (catches obvious errors, but allows current year)
        if end_dt.year > current_year + 1:
            return {"success": False, "message": f"❌ End date year {end_dt.year} is too far in the future (current year: {current_year}). End date: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}, Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}. This may be caused by bad data in the database or incorrect date parsing."}
        elif end_dt > now_utc + timedelta(hours=1):  # Allow up to 1 hour in the future (timezone buffer)
            return {"success": False, "message": f"❌ End date {end_dt.strftime('%Y-%m-%d %H:%M:%S')} is more than 1 hour in the future. Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}. This may be caused by bad data in the database or incorrect date parsing."}
        
        # Debug: Log end_dt to help diagnose
        print(f"🔍 End date after parsing: {end_dt.strftime('%Y-%m-%d %H:%M:%S')} (year: {end_dt.year}, current year: {current_year})")
        
        # Auto-resume: Check for latest timestamp in database
        start_dt = None
        print(f"🔍 Auto-resume enabled: {auto_resume}")
        if auto_resume:
            print(f"🔄 Starting auto-resume check for symbol '{api_symbol}' in asset class '{asset_class}'...")
            try:
                # First, convert symbol to Polygon format to match what we store in DB
                # We'll determine the polygon symbol early for auto-resume lookup
                polygon_symbol_for_resume = convert_instrument_to_polygon_symbol(asset_class, api_symbol)
                if not polygon_symbol_for_resume:
                    polygon_symbol_for_resume = api_symbol
                
                # Clean the symbol
                if polygon_symbol_for_resume:
                    polygon_symbol_for_resume = polygon_symbol_for_resume.strip().replace('\n', '').replace('\r', '').replace('\t', '')
                
                # Also handle legacy Barchart symbols
                if api_symbol == "GC*1":
                    polygon_symbol_for_resume = "C:XAUUSD"
                elif api_symbol == "^SPX" or api_symbol == "$SPX":
                    polygon_symbol_for_resume = "I:SPX"
                
                # If input already has correct format (I:SPX, C:EURUSD), use it directly
                if api_symbol and ":" in api_symbol and api_symbol.startswith(("I:", "C:")):
                    api_symbol_cleaned = api_symbol.strip().replace('\n', '').replace('\r', '').replace('\t', '')
                    parts = api_symbol_cleaned.split(":", 1)
                    if len(parts) == 2 and len(parts[1].strip()) > 0:
                        polygon_symbol_for_resume = api_symbol_cleaned
                
                target_symbol = db_symbol if db_symbol else polygon_symbol_for_resume
                # Try multiple symbol formats (what might be in DB vs what we're using)
                symbols_to_try = [target_symbol, polygon_symbol_for_resume, api_symbol]
                if target_symbol == "^XAUUSD" or polygon_symbol_for_resume == "C:XAUUSD":
                    symbols_to_try.extend(["C:XAUUSD", "GOLD", "^XAUUSD"])
                elif target_symbol == "^SPX" or polygon_symbol_for_resume == "I:SPX":
                    symbols_to_try.extend(["I:SPX", "S&P 500", "^SPX", "SPY"])
                elif asset_class == "Currencies" and polygon_symbol_for_resume.startswith("C:"):
                    # For currencies, also try without the C: prefix
                    symbols_to_try.append(polygon_symbol_for_resume[2:])  # e.g., "EURUSD" from "C:EURUSD"
                
                latest_ts = None
                found_symbol = None
                print(f"🔍 Auto-resume: Checking database for latest timestamp...")
                print(f"🔍 Target table: {target_table}")
                print(f"🔍 Symbols to try: {symbols_to_try}")
                
                # Debug: List all symbols in the table to see what's actually stored
                try:
                    all_symbols_result = sb.table(target_table).select("symbol").limit(100).execute()
                    if all_symbols_result.data:
                        unique_symbols = list(set([row.get("symbol") for row in all_symbols_result.data]))
                        print(f"🔍 Sample symbols found in {target_table}: {unique_symbols[:10]}")  # Show first 10
                except Exception as e:
                    print(f"⚠️ Could not list symbols in table: {e}")
                
                for try_sym in symbols_to_try:
                    try:
                        print(f"🔍 Checking symbol: '{try_sym}' in table {target_table}...")
                        result = sb.table(target_table)\
                            .select("timestamp")\
                            .eq("symbol", try_sym)\
                            .order("timestamp", desc=True)\
                            .limit(1)\
                            .execute()
                        
                        print(f"🔍 Query result: {len(result.data) if result.data else 0} records found")
                        
                        if result.data and len(result.data) > 0:
                            latest_ts_str = result.data[0].get("timestamp")
                            print(f"🔍 Found latest timestamp string: {latest_ts_str}")
                            
                            if latest_ts_str:
                                # Parse latest timestamp from database (stored in GMT+4)
                                try:
                                    if isinstance(latest_ts_str, str):
                                        # Handle ISO format strings
                                        latest_ts_gmt4 = datetime.fromisoformat(latest_ts_str.replace('Z', '+00:00'))
                                    else:
                                        latest_ts_gmt4 = latest_ts_str
                                    
                                    # Ensure GMT+4 timezone
                                    if latest_ts_gmt4.tzinfo is None:
                                        latest_ts_gmt4 = pytz.timezone('Asia/Dubai').localize(latest_ts_gmt4)
                                    else:
                                        latest_ts_gmt4 = latest_ts_gmt4.astimezone(pytz.timezone('Asia/Dubai'))
                                    
                                    # Convert GMT+4 → UTC for Polygon API query
                                    latest_ts_utc = latest_ts_gmt4.astimezone(pytz.UTC).replace(tzinfo=None)
                                    
                                    print(f"🔍 Parsed timestamp - GMT+4: {latest_ts_gmt4.strftime('%Y-%m-%d %H:%M:%S')}, UTC: {latest_ts_utc.strftime('%Y-%m-%d %H:%M:%S')}")
                                    print(f"🔍 Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}, Current year: {current_year}")
                                    
                                    # Validate timestamp from database is not in the future
                                    # Allow up to 1 hour in the future (timezone/clock drift buffer)
                                    if latest_ts_utc.year > current_year + 1:
                                        print(f"⚠️ Database timestamp year {latest_ts_utc.year} is too far in the future (current: {current_year}). Ignoring.")
                                        continue
                                    elif latest_ts_utc > now_utc + timedelta(hours=1):
                                        print(f"⚠️ Database timestamp {latest_ts_utc.strftime('%Y-%m-%d %H:%M:%S')} is more than 1 hour in the future. Ignoring.")
                                        continue
                                    
                                    # Found valid timestamp - use it
                                    start_dt = latest_ts_utc + timedelta(minutes=1)  # Resume from next minute
                                    found_symbol = try_sym
                                    
                                    print(f"✅ Auto-resume: Found latest data for '{found_symbol}'")
                                    print(f"🔄 Latest (GMT+4): {latest_ts_gmt4.strftime('%Y-%m-%d %H:%M:%S')}")
                                    print(f"🔄 Resuming from: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                                    break
                                except Exception as parse_error:
                                    print(f"⚠️ Error parsing timestamp '{latest_ts_str}': {parse_error}")
                                    continue
                    except Exception as e:
                        print(f"⚠️ Error querying database for symbol '{try_sym}': {e}")
                        continue
                
                # If no existing data found, use default (2 years back)
                # Default: Last 2 years of data (from 2024 to now)
                if start_dt is None:
                    print(f"ℹ️ No existing data found in database for any of the symbol formats: {symbols_to_try}")
                    print(f"ℹ️ Starting fresh ingestion from 2 years ago...")
                    safe_end = now_utc - timedelta(minutes=15)
                    start_dt = safe_end - timedelta(days=730)  # 2 years = 730 days
                    end_dt = safe_end  # Ensure end_dt is also safe
                    print(f"📅 Using default 2-year range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
                    print(f"📅 Calculated start_dt: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC (year: {start_dt.year})")
                    print(f"📅 Calculated end_dt: {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC (year: {end_dt.year})")
                    
                    # CRITICAL: Double-check that start_dt is not in the future
                    if start_dt.year > current_year or start_dt > now_utc:
                        print(f"❌ ERROR: Calculated start_dt is invalid! start_dt.year={start_dt.year}, current_year={current_year}, start_dt={start_dt}, now_utc={now_utc}")
                        # Force to 2 years ago
                        start_dt = now_utc - timedelta(days=730)
                        end_dt = now_utc - timedelta(minutes=15)
                        print(f"🔧 Corrected to: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
                else:
                    print(f"✅ Auto-resume successful! Will continue from {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    
                    # Validate start_dt immediately after calculation
                    if start_dt and (start_dt.year > current_year + 1 or start_dt > now_utc + timedelta(hours=1)):
                        print(f"⚠️ Calculated start_dt {start_dt.strftime('%Y-%m-%d %H:%M:%S')} is invalid (year: {start_dt.year}, current: {current_year}). Using safe default.")
                        safe_end = now_utc - timedelta(minutes=15)
                        start_dt = safe_end - timedelta(days=730)
                        end_dt = safe_end
                        print(f"📅 Reset to safe default: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
            except Exception as e:
                # Fallback to default if auto-resume fails (2 years)
                print(f"❌ Auto-resume exception: {e}")
                traceback.print_exc()
                safe_end = now_utc - timedelta(minutes=15)
                start_dt = safe_end - timedelta(days=730)  # 2 years
                end_dt = safe_end
                print(f"⚠️ Auto-resume failed, using default 2-year range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        else:
            # Manual start date (auto_resume=False)
            print(f"ℹ️ Auto-resume is disabled. Using manual start_date parameter.")
            if not start_date:
                # Default to 2 years back (730 days) - last 2 years from 2024 to now
                start_dt = end_dt - timedelta(days=730)
            elif isinstance(start_date, str):
                # Try multiple date formats
                try:
                    start_dt = datetime.strptime(start_date, "%Y%m%d")
                except ValueError:
                    try:
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    except ValueError:
                        start_dt = datetime.fromisoformat(start_date)
            else:
                start_dt = start_date
            
            # Ensure start_date is not in the future and is before end_date
            if start_dt.year > current_year or start_dt > now_utc:
                print(f"⚠️ Start date {start_dt.strftime('%Y-%m-%d')} is invalid. Recalculating from safe end_dt.")
                safe_end = now_utc - timedelta(minutes=15)
                start_dt = safe_end - timedelta(days=730)  # 2 years default
                end_dt = safe_end
            if start_dt > end_dt:
                print(f"⚠️ Start date {start_dt.strftime('%Y-%m-%d')} is after end date {end_dt.strftime('%Y-%m-%d')}. Recalculating.")
                safe_end = now_utc - timedelta(minutes=15)
                start_dt = safe_end - timedelta(days=730)  # 2 years default
                end_dt = safe_end
        
        # Validate dates BEFORE formatting - allow same day but not far future
        # Allow current year and next year (catches obvious errors like 2030+ but allows today)
        if end_dt.year > current_year + 1:
            return {"success": False, "message": f"❌ End date year {end_dt.year} is too far in the future (current: {current_year}). Date: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}. This may be caused by bad data in the database. Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}"}
        
        if start_dt.year > current_year + 1:
            return {"success": False, "message": f"❌ Start date year {start_dt.year} is too far in the future (current: {current_year}). Date: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}. This may be caused by bad data in the database. Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}"}
        
        # Allow up to 1 hour in the future (timezone/clock drift buffer)
        if end_dt > now_utc + timedelta(hours=1):
            return {"success": False, "message": f"❌ End date ({end_dt.strftime('%Y-%m-%d %H:%M:%S')}) is more than 1 hour in the future. Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}"}
        
        if start_dt > end_dt:
            return {"success": False, "message": f"❌ Invalid date range: start ({start_dt.strftime('%Y-%m-%d %H:%M:%S')}) is after end ({end_dt.strftime('%Y-%m-%d %H:%M:%S')})"}
        
        # Final validation of start_dt and end_dt before proceeding (same checks)
        if start_dt.year > current_year + 1:
            return {"success": False, "message": f"❌ Start date year {start_dt.year} is too far in the future (current: {current_year}). Date: {start_dt.strftime('%Y-%m-%d')}. This may be caused by bad data in the database."}
        
        if end_dt.year > current_year + 1:
            return {"success": False, "message": f"❌ End date year {end_dt.year} is too far in the future (current: {current_year}). Date: {end_dt.strftime('%Y-%m-%d')}. This may be caused by bad data in the database."}
        
        # Check if start_date is after end_date (data already up to date)
        if start_dt >= end_dt:
            latest_msg = f"Latest: {(start_dt - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')}" if auto_resume else ""
            print(f"⚠️ WARNING: start_dt ({start_dt.strftime('%Y-%m-%d %H:%M:%S')}) >= end_dt ({end_dt.strftime('%Y-%m-%d %H:%M:%S')})")
            print(f"⚠️ This usually means data is already up to date, but checking if this is correct...")
            # If start_dt is today and end_dt is today, this might be a bug - don't return early
            if start_dt.date() == end_dt.date() == now_utc.date():
                print(f"⚠️ Both dates are today - this might indicate auto-resume didn't find data. Checking database...")
                # Don't return early - let it try to fetch (will likely get 403 for today, but that's better than silently failing)
            else:
                return {"success": True, "message": f"Data for {api_symbol} is already up to date. {latest_msg} End: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}"}
            
        # Convert symbol to Polygon format using the conversion function
        # This handles currencies (EUR/USD -> C:EURUSD), indices, stocks, etc.
        polygon_symbol = convert_instrument_to_polygon_symbol(asset_class, api_symbol)
        
        # CRITICAL: Validate conversion didn't create an invalid symbol (e.g., "C:C" from input "C")
        if polygon_symbol and ":" in polygon_symbol:
            parts = polygon_symbol.split(":", 1)
            if len(parts) == 2 and parts[0] == parts[1]:  # e.g., "C:C" from input "C"
                examples = {
                    "Commodities": "C:XAUUSD (for Gold)",
                    "Indices": "I:SPX (for S&P 500), I:DJI (for Dow Jones)",
                    "Currencies": "C:EURUSD, C:GBPUSD, C:USDJPY",
                    "Stocks": "AAPL, MSFT, GOOGL"
                }
                example = examples.get(asset_class, "I:SPX, C:EURUSD, AAPL")
                return {"success": False, "message": f"❌ Symbol conversion created invalid symbol '{polygon_symbol}' from input '{api_symbol}'.\n\nThis usually means you entered only the prefix (e.g., 'C' instead of 'C:EURUSD').\n\nFor {asset_class}, please enter a complete symbol like: {example}"}
        
        # Also handle legacy Barchart symbols if passed directly
        if api_symbol == "GC*1":
            polygon_symbol = "C:XAUUSD"
        elif api_symbol == "^SPX" or api_symbol == "$SPX":
            polygon_symbol = "I:SPX"
            
        # CRITICAL: Polygon does NOT support 1-minute data for indices (I:SPX, I:DJI, etc.)
        # Auto-convert I:SPX to SPY (ETF that tracks S&P 500) for minute data
        # SPY is tradable and Polygon provides full minute history
        if polygon_symbol and polygon_symbol.startswith("I:"):
            index_name = polygon_symbol[2:]  # Remove "I:" prefix
            if index_name == "SPX":
                # Use SPY instead of I:SPX for minute data (industry standard workaround)
                polygon_symbol = "SPY"
                print(f"⚠️ Polygon doesn't support I:SPX minute data. Auto-converting to SPY (ETF that tracks S&P 500).")
            else:
                # For other indices, warn but try anyway
                print(f"⚠️ WARNING: Polygon may not support 1-minute data for {polygon_symbol}. Consider using an ETF or futures contract instead.")
        
        # CRITICAL: Clean symbol immediately after conversion (remove all whitespace/newlines)
        if polygon_symbol:
            polygon_symbol = polygon_symbol.strip().replace('\n', '').replace('\r', '').replace('\t', '')
        
        # Ensure symbol is not modified/truncated - preserve original if conversion fails
        if not polygon_symbol:
            return {"success": False, "message": f"❌ Symbol conversion failed. Input: '{api_symbol}', Asset Class: '{asset_class}'"}
        
        # If input already has correct format (I:SPX, C:EURUSD), use it directly (already cleaned above)
        # BUT: Only use it if it's a complete symbol (not just "C:" or "I:")
        if api_symbol and ":" in api_symbol and api_symbol.startswith(("I:", "C:")):
            api_symbol_cleaned = api_symbol.strip().replace('\n', '').replace('\r', '').replace('\t', '')
            # Only use if it's a complete symbol (has content after the colon)
            parts = api_symbol_cleaned.split(":", 1)  # Split only on first colon
            if len(parts) == 2 and len(parts[1].strip()) > 0:  # e.g., "C:EURUSD" has content after ":"
                polygon_symbol = api_symbol_cleaned
                print(f"✅ Using provided symbol format: '{polygon_symbol}'")
            else:
                # Symbol is incomplete (e.g., just "C:"), use converted symbol instead
                print(f"⚠️ Provided symbol '{api_symbol_cleaned}' is incomplete, using converted symbol: '{polygon_symbol}'")
            # Otherwise, keep the converted symbol from convert_instrument_to_polygon_symbol
        
        # Validate polygon symbol (should not be empty or just a prefix)
        # Strip all whitespace including newlines (double-check)
        polygon_symbol_clean = polygon_symbol.strip().replace('\n', '').replace('\r', '').replace('\t', '') if polygon_symbol else ""
        
        # Debug: Log symbol conversion
        print(f"🔍 Symbol conversion: '{api_symbol}' → '{polygon_symbol}' (clean: '{polygon_symbol_clean}')")
        invalid_symbols = ["I:", "C:", "I", "C", ""]
        
        # CRITICAL: Check if symbol was truncated to just prefix (e.g., "C" or "C:")
        if polygon_symbol_clean in ["C", "I"] or polygon_symbol_clean in ["C:", "I:"]:
            examples = {
                "Commodities": "C:XAUUSD (for Gold)",
                "Indices": "I:SPX (for S&P 500), I:DJI (for Dow Jones)",
                "Currencies": "C:EURUSD, C:GBPUSD, C:USDJPY",
                "Stocks": "AAPL, MSFT, GOOGL"
            }
            example = examples.get(asset_class, "I:SPX, C:EURUSD, AAPL")
            return {"success": False, "message": f"❌ Symbol '{polygon_symbol_clean}' is incomplete (only prefix). Original input: '{api_symbol}'.\n\nFor {asset_class}, please enter a complete symbol like: {example}\n\n💡 If you're seeing this repeatedly, try refreshing the page and re-entering the symbol."}
        
        # Check if symbol is too short or invalid
        if len(polygon_symbol_clean) <= 1:
            examples = {
                "Commodities": "C:XAUUSD",
                "Indices": "I:SPX",
                "Currencies": "C:EURUSD",
                "Stocks": "AAPL"
            }
            example = examples.get(asset_class, "I:SPX")
            return {"success": False, "message": f"❌ Symbol too short: '{polygon_symbol}' (from '{api_symbol}').\n\nFor {asset_class}, please enter a full symbol like: {example}\n\nIf you entered just '{api_symbol}', you need to include the full symbol format."}
        
        if polygon_symbol_clean in invalid_symbols:
            examples = {
                "Commodities": "C:XAUUSD (for Gold)",
                "Indices": "I:SPX (for S&P 500), I:DJI (for Dow Jones)",
                "Currencies": "C:EURUSD, C:GBPUSD",
                "Stocks": "AAPL, MSFT, GOOGL"
            }
            example = examples.get(asset_class, "I:SPX, C:EURUSD, AAPL")
            return {"success": False, "message": f"❌ Invalid Polygon symbol: '{polygon_symbol}' (from '{api_symbol}').\n\nFor {asset_class}, please use:\n{example}\n\nNote: '{api_symbol}' is incomplete. You need the full symbol format."}
        
        # Ensure symbol wasn't truncated (safety check)
        if api_symbol and ":" in api_symbol and ":" not in polygon_symbol_clean:
            return {"success": False, "message": f"❌ Symbol conversion error: '{api_symbol}' was converted to '{polygon_symbol}'. Please check the symbol format."}
        
        # Debug: Show final start_dt value before formatting
        print(f"\n{'='*60}")
        print(f"🔍 FINAL DATE RANGE AFTER AUTO-RESUME")
        print(f"{'='*60}")
        print(f"   start_dt: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC (year: {start_dt.year})")
        print(f"   end_dt: {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC (year: {end_dt.year})")
        print(f"   current_year: {current_year}")
        print(f"   now_utc: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"{'='*60}\n")
        
        # Format dates for API call
        s_str = start_dt.strftime("%Y-%m-%d")
        e_str = end_dt.strftime("%Y-%m-%d")
        
        # Final date validation before API call - CRITICAL CHECKS (MUST PREVENT API CALL)
        now_str = now_utc.strftime("%Y-%m-%d")
        
        # Check year first (catches 2026 dates immediately)
        if end_dt.year > current_year or start_dt.year > current_year:
            return {"success": False, "message": f"❌ BLOCKED: Date range contains future year. Start: {s_str} (year: {start_dt.year}), End: {e_str} (year: {end_dt.year}). Current year: {current_year}. API call prevented."}
        
        # Check if dates are in the future
        if end_dt > now_utc or start_dt > now_utc:
            return {"success": False, "message": f"❌ BLOCKED: Date range contains future dates. Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}, End: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}. Current: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}. API call prevented."}
        
        # Check formatted date strings
        if e_str > now_str or s_str > now_str:
            return {"success": False, "message": f"❌ BLOCKED: Date strings are in the future. Start: {s_str}, End: {e_str}. Current: {now_str}. API call prevented."}
        
        # Final safety check before API call
        if not polygon_symbol or len(polygon_symbol.strip()) <= 1:
            return {"success": False, "message": f"❌ CRITICAL: Symbol validation failed. polygon_symbol='{polygon_symbol}' is invalid. Original input was '{api_symbol}'"}
            
        # Fetch data from Polygon
        # Ensure symbol is clean before API call (final cleanup)
        polygon_symbol_final = polygon_symbol_clean if polygon_symbol_clean else polygon_symbol.strip().replace('\n', '').replace('\r', '').replace('\t', '')
        
        # CRITICAL DEBUG: If start_dt is today, something is wrong - check database directly
        if start_dt.date() == now_utc.date() and auto_resume:
            print(f"\n{'='*60}")
            print(f"🔍 DEBUGGING: Start date is TODAY but auto-resume is enabled!")
            print(f"{'='*60}")
            print(f"   Checking database directly for symbol '{polygon_symbol_final}' in table '{target_table}'...")
            try:
                # Direct query to see what's actually in the database
                direct_result = sb.table(target_table)\
                    .select("symbol, timestamp")\
                    .eq("symbol", polygon_symbol_final)\
                    .order("timestamp", desc=True)\
                    .limit(5)\
                    .execute()
                
                if direct_result.data and len(direct_result.data) > 0:
                    print(f"   ✅ Found {len(direct_result.data)} records for '{polygon_symbol_final}'")
                    latest_record = direct_result.data[0]
                    latest_ts = latest_record.get("timestamp")
                    print(f"   Latest timestamp in DB: {latest_ts}")
                    print(f"   Latest symbol in DB: {latest_record.get('symbol')}")
                    print(f"   ⚠️ PROBLEM: Data exists but auto-resume didn't use it!")
                else:
                    print(f"   ❌ No records found for '{polygon_symbol_final}' in database")
                    print(f"   This explains why start_dt is today - no data found, should default to 2 years ago")
            except Exception as e:
                print(f"   ❌ Error querying database: {e}")
            print(f"{'='*60}\n")
        
        # Display exact date range being used for Polygon API
        print(f"\n{'='*60}")
        print(f"📊 POLYGON API INGESTION - EXACT DATE RANGE")
        print(f"{'='*60}")
        print(f"   Symbol: {polygon_symbol_final}")
        print(f"   Asset Class: {asset_class}")
        print(f"   Start Date: {s_str} ({start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC)")
        print(f"   End Date:   {e_str} ({end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC)")
        print(f"   Date Range: {s_str} to {e_str}")
        print(f"   Total Days: {(end_dt - start_dt).days} days")
        print(f"   Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Auto-resume was: {'ENABLED' if auto_resume else 'DISABLED'}")
        if start_dt.date() == now_utc.date():
            print(f"   ⚠️ WARNING: Start date is TODAY ({start_dt.date()}) - auto-resume may not have found existing data!")
            print(f"   ⚠️ Expected: Start date should be ~2 years ago ({now_utc.date() - timedelta(days=730)}) if no data found")
        print(f"{'='*60}\n")
        
        # CRITICAL: For Polygon free plan, we need to chunk by 1 day and add rate limiting
        # Same logic as gold ingestion: 1 day chunks, 12s throttle, 90s wait on 429
        import time
        
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
        
        # Helper function to process and insert a single chunk dataframe
        def process_and_insert_chunk(chunk_df, chunk_date_str, target_symbol, target_table):
            """Process a chunk dataframe and insert it into the database immediately"""
            if chunk_df.empty:
                return 0
            
            db_rows = []
            utc_tz = pytz.timezone('UTC')
            gmt4_tz = pytz.timezone('Asia/Dubai')
            
            # Ensure timestamp is the index
            if "timestamp" in chunk_df.columns:
                chunk_df = chunk_df.set_index("timestamp")
            
            for timestamp, row in chunk_df.iterrows():
                if pd.isna(row.get("close")):
                    continue
                    
                try:
                    # Polygon returns UTC timestamps → Convert to GMT+4 for database
                    if timestamp.tzinfo is None:
                        ts_utc = utc_tz.localize(timestamp)
                    else:
                        ts_utc = timestamp.astimezone(utc_tz)
                    
                    ts_gmt4 = ts_utc.astimezone(gmt4_tz)
                    
                    record = {
                        "symbol": str(target_symbol),
                        "timestamp": ts_gmt4.isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]) if pd.notnull(row.get("volume")) else 0
                    }
                    db_rows.append(record)
                except (ValueError, TypeError) as e:
                    continue
            
            if not db_rows:
                return 0
            
            # Insert immediately
            sb = get_supabase()
            if not sb:
                print(f"❌ Supabase not configured - cannot insert chunk for {chunk_date_str}")
                return 0
            
            chunk_size = 1000
            total_inserted = 0
            
            for i in range(0, len(db_rows), chunk_size):
                chunk = db_rows[i:i+chunk_size]
                try:
                    result_upsert = sb.table(target_table).upsert(chunk).execute()
                    total_inserted += len(chunk)
                except Exception as e:
                    print(f"❌ Database error inserting chunk {chunk_date_str} (batch {i//chunk_size + 1}): {str(e)}")
                    raise  # Re-raise to handle at higher level
            
            return total_inserted
        
        # Chunk by 1 day (same as gold ingestion for free plan compatibility)
        chunk_days = 1
        chunk_start = start_dt
        max_retries = 5
        all_success = True
        skipped_chunks = []  # Track chunks that were skipped due to 403 (date-specific issues)
        total_inserted_all_chunks = 0  # Track total records inserted across all chunks
        target_symbol = db_symbol if db_symbol else polygon_symbol_final
        
        print(f"📦 Chunking date range into {chunk_days}-day chunks for Polygon free plan compatibility...")
        total_chunks = (end_dt - start_dt).days + 1
        print(f"📊 Total chunks to process: {total_chunks} days")
        
        client = PolygonDataClient()
        
        chunk_number = 0
        # Process in 1-day chunks
        while chunk_start <= end_dt:
            chunk_number += 1
            chunk_end = min(chunk_start + timedelta(days=chunk_days) - timedelta(days=1), end_dt)
            if chunk_end < chunk_start:
                chunk_end = chunk_start
            
            # For today's date, cap the end time to current time (not end of day)
            # This prevents 403 errors when requesting full day data that hasn't completed yet
            today_date = now_utc.date()
            if chunk_end.date() == today_date and chunk_end > now_utc:
                chunk_end = now_utc - timedelta(minutes=15)  # Safety buffer: 15 minutes before now
                print(f"📅 Today's chunk detected - capping end time to current time: {chunk_end.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # CRITICAL: Validate chunk dates are not too far in the future before making API call
            # Allow same day (up to 1 hour buffer for timezone/clock drift)
            if chunk_start.year > current_year + 1 or chunk_end.year > current_year + 1:
                return {"success": False, "message": f"❌ BLOCKED: Chunk date contains future year. Start: {chunk_start.strftime('%Y-%m-%d')} (year: {chunk_start.year}), End: {chunk_end.strftime('%Y-%m-%d')} (year: {chunk_end.year}). Current year: {current_year}. API call prevented."}
            if chunk_start > now_utc + timedelta(hours=1) or chunk_end > now_utc + timedelta(hours=1):
                return {"success": False, "message": f"❌ BLOCKED: Chunk date is more than 1 hour in the future. Start: {chunk_start.strftime('%Y-%m-%d %H:%M:%S')}, End: {chunk_end.strftime('%Y-%m-%d %H:%M:%S')}. Current: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}. API call prevented."}
            
            # Use date strings for API call (Polygon accepts YYYY-MM-DD format)
            chunk_s_str = chunk_start.strftime("%Y-%m-%d")
            chunk_e_str = chunk_end.strftime("%Y-%m-%d")
            
            print(f"📦 Processing chunk {chunk_number}/{total_chunks}: {chunk_s_str} to {chunk_e_str}")
            
            attempt = 0
            chunk_success = False
            
            while attempt < max_retries:
                try:
                    # Apply throttle before each API call
                    _polygon_throttle()
                    
                    print(f"🔍 Calling Polygon API: get_intraday_data('{polygon_symbol_final}', '{chunk_s_str}', '{chunk_e_str}', multiplier=1, timespan='minute')")
                    chunk_df = client.get_intraday_data(polygon_symbol_final, chunk_s_str, chunk_e_str, multiplier=1, timespan="minute")
                    
                    if not chunk_df.empty:
                        print(f"✅ Fetched {len(chunk_df)} records for {chunk_s_str} to {chunk_e_str}")
                        
                        # Insert immediately after fetching
                        try:
                            inserted_count = process_and_insert_chunk(chunk_df, chunk_s_str, target_symbol, target_table)
                            if inserted_count > 0:
                                total_inserted_all_chunks += inserted_count
                                print(f"💾 Inserted {inserted_count} records for {chunk_s_str} to {chunk_e_str} (total so far: {total_inserted_all_chunks})")
                            else:
                                print(f"⚠️ No records inserted for {chunk_s_str} to {chunk_e_str} (all may be duplicates or invalid)")
                        except Exception as e:
                            print(f"❌ Error inserting chunk {chunk_s_str} to {chunk_e_str}: {str(e)}")
                            # Continue to next chunk even if insertion fails for this one
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
                        if polygon_symbol_final.startswith("I:") or api_symbol.upper() in ["I:SPX", "SPX", "^SPX"]:
                            return {"success": False, "message": f"❌ Polygon 403 Forbidden: Polygon does NOT provide 1-minute data for indices (I:SPX) due to licensing restrictions.\n\n✅ **SOLUTION**: Use SPY instead of I:SPX. SPY is an ETF that tracks the S&P 500.\n\nDate range: {chunk_s_str} to {chunk_e_str}"}
                        else:
                            # Check if symbol is truncated (just "C" or "I")
                            # Use api_symbol for display to avoid truncation issues
                            symbol_display = api_symbol.strip().replace('\n', '').replace('\r', '').replace('\t', '') if api_symbol else (polygon_symbol_final if polygon_symbol_final else "UNKNOWN")
                            if symbol_display in ["C", "I", "C:", "I:"]:
                                return {"success": False, "message": f"❌ Polygon 403 Forbidden: Symbol '{symbol_display}' is incomplete/truncated.\n\n💡 **SOLUTION**: The symbol was truncated. For currencies, use full format like 'C:EURUSD' (not just 'C').\n\nOriginal input: '{api_symbol}'\nConverted symbol: '{polygon_symbol_final}'\nAsset class: {asset_class}\n\nDate range: {chunk_s_str} to {chunk_e_str}"}
                            else:
                                # Check if this is "today" - might be before market opens
                                today_str = datetime.utcnow().strftime("%Y-%m-%d")
                                is_today = chunk_s_str == today_str
                                is_commodity = asset_class == "Commodities"
                                is_currency = asset_class == "Currencies"
                                is_24_7_market = is_commodity or is_currency  # Commodities and currencies trade 24/7
                                
                                # For commodities (like gold) and currencies (forex), markets are open 24/7, 
                                # so don't skip today's data automatically - but if we get a 403, skip and continue
                                # Only skip if we've already fetched some data (meaning this is a subsequent chunk that failed)
                                # For stocks: If we've already fetched some data, OR if it's today's date, skip the chunk
                                # (today's data might not be available yet before market opens)
                                # For 24/7 markets on today: Skip the chunk (likely Polygon plan limitation for this pair)
                                if total_inserted_all_chunks > 0 or (is_today and not is_24_7_market) or (is_today and is_24_7_market):
                                    if is_today and not is_24_7_market:
                                        reason = "today's data not yet available (market may not have opened yet)"
                                    elif is_today and is_24_7_market:
                                        # For 24/7 markets, 403 usually means Polygon plan limitation, not timing
                                        reason = "Polygon plan limitation (this currency pair may not be available for 1-minute data on free plan)"
                                    else:
                                        reason = "no data for this date (weekend/holiday or data not yet available)"
                                    print(f"⚠️ 403 Forbidden for {chunk_s_str} to {chunk_e_str} ({reason}). Skipping chunk and continuing...")
                                    skipped_chunks.append(f"{chunk_s_str} to {chunk_e_str}")
                                    chunk_success = True  # Mark as handled, continue to next chunk
                                    break
                                else:
                                    # No data fetched yet and not today - might be a symbol-wide issue
                                    # Check if it's a currency pair
                                    if polygon_symbol_final.startswith("C:") or asset_class == "Currencies":
                                        return {"success": False, "message": f"❌ Polygon 403 Forbidden: {error_str}\n\n⚠️ **POLYGON PLAN LIMITATION**: This currency pair may not be available for 1-minute data on your Polygon free plan.\n\n**Symbol**: {symbol_display}\n**Original input**: '{api_symbol}'\n**Date range**: {chunk_s_str} to {chunk_e_str}\n\n💡 **Possible Solutions**:\n1. Check if your Polygon plan includes forex minute data\n2. Try a different currency pair (e.g., GBPUSD works)\n3. Use daily data instead of 1-minute data\n4. Upgrade your Polygon plan for access to more currency pairs\n\n📚 **Note**: Polygon's free plan has limited currency pair coverage. Some pairs like EUR/USD may require a paid plan for 1-minute data."}
                                    else:
                                        return {"success": False, "message": f"❌ Polygon 403 Forbidden: {error_str}\n\nThis symbol may not be available for 1-minute data on your Polygon plan.\n\nSymbol: {symbol_display}\nOriginal input: '{api_symbol}'\nDate range: {chunk_s_str} to {chunk_e_str}"}
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
                        attempt += 1
                        if attempt < max_retries:
                            print(f"🛑 Polygon rate limit hit for {polygon_symbol_final}. Cooling down 90s before retry {attempt}/{max_retries}...")
                            time.sleep(90)
                            continue  # Retry after cooling down
                        else:
                            print(f"❌ Rate limit exceeded for {polygon_symbol_final} after {max_retries} attempts. Skipping chunk.")
                            all_success = False
                            break  # Max retries reached, skip this chunk
                    elif "403" in error_str or "Forbidden" in error_str:
                        if polygon_symbol_final.startswith("I:") or api_symbol.upper() in ["I:SPX", "SPX", "^SPX"]:
                            return {"success": False, "message": f"❌ Polygon 403 Forbidden: Polygon does NOT provide 1-minute data for indices (I:SPX) due to licensing restrictions.\n\n✅ **SOLUTION**: Use SPY instead of I:SPX. SPY is an ETF that tracks the S&P 500.\n\nDate range: {chunk_s_str} to {chunk_e_str}"}
                        else:
                            # Check if symbol is truncated (just "C" or "I")
                            # Use api_symbol for display to avoid truncation issues
                            symbol_display = api_symbol.strip().replace('\n', '').replace('\r', '').replace('\t', '') if api_symbol else (polygon_symbol_final if polygon_symbol_final else "UNKNOWN")
                            if symbol_display in ["C", "I", "C:", "I:"]:
                                return {"success": False, "message": f"❌ Polygon 403 Forbidden: Symbol '{symbol_display}' is incomplete/truncated.\n\n💡 **SOLUTION**: The symbol was truncated. For currencies, use full format like 'C:EURUSD' (not just 'C').\n\nOriginal input: '{api_symbol}'\nConverted symbol: '{polygon_symbol_final}'\nAsset class: {asset_class}\n\nDate range: {chunk_s_str} to {chunk_e_str}"}
                            else:
                                # Check if this is "today" - might be before market opens
                                today_str = datetime.utcnow().strftime("%Y-%m-%d")
                                is_today = chunk_s_str == today_str
                                is_commodity = asset_class == "Commodities"
                                is_currency = asset_class == "Currencies"
                                is_24_7_market = is_commodity or is_currency  # Commodities and currencies trade 24/7
                                
                                # For commodities (like gold) and currencies (forex), markets are open 24/7, 
                                # so don't skip today's data automatically - but if we get a 403, skip and continue
                                # Only skip if we've already fetched some data (meaning this is a subsequent chunk that failed)
                                # For stocks: If we've already fetched some data, OR if it's today's date, skip the chunk
                                # (today's data might not be available yet before market opens)
                                # For 24/7 markets on today: Skip the chunk (likely Polygon plan limitation for this pair)
                                if total_inserted_all_chunks > 0 or (is_today and not is_24_7_market) or (is_today and is_24_7_market):
                                    if is_today and not is_24_7_market:
                                        reason = "today's data not yet available (market may not have opened yet)"
                                    elif is_today and is_24_7_market:
                                        # For 24/7 markets, 403 usually means Polygon plan limitation, not timing
                                        reason = "Polygon plan limitation (this currency pair may not be available for 1-minute data on free plan)"
                                    else:
                                        reason = "no data for this date (weekend/holiday or data not yet available)"
                                    print(f"⚠️ 403 Forbidden for {chunk_s_str} to {chunk_e_str} ({reason}). Skipping chunk and continuing...")
                                    skipped_chunks.append(f"{chunk_s_str} to {chunk_e_str}")
                                    chunk_success = True  # Mark as handled, continue to next chunk
                                    break
                                else:
                                    # No data fetched yet and not today - might be a symbol-wide issue
                                    # Check if it's a currency pair
                                    if polygon_symbol_final.startswith("C:") or asset_class == "Currencies":
                                        return {"success": False, "message": f"❌ Polygon 403 Forbidden: {error_str}\n\n⚠️ **POLYGON PLAN LIMITATION**: This currency pair may not be available for 1-minute data on your Polygon free plan.\n\n**Symbol**: {symbol_display}\n**Original input**: '{api_symbol}'\n**Date range**: {chunk_s_str} to {chunk_e_str}\n\n💡 **Possible Solutions**:\n1. Check if your Polygon plan includes forex minute data\n2. Try a different currency pair (e.g., GBPUSD works)\n3. Use daily data instead of 1-minute data\n4. Upgrade your Polygon plan for access to more currency pairs\n\n📚 **Note**: Polygon's free plan has limited currency pair coverage. Some pairs like EUR/USD may require a paid plan for 1-minute data."}
                                    else:
                                        return {"success": False, "message": f"❌ Polygon 403 Forbidden: {error_str}\n\nThis symbol may not be available for 1-minute data on your Polygon plan.\n\nSymbol: {symbol_display}\nOriginal input: '{api_symbol}'\nDate range: {chunk_s_str} to {chunk_e_str}"}
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
        
        # All chunks processed - prepare final summary
        print(f"\n{'='*60}")
        print(f"📊 INGESTION COMPLETE")
        print(f"{'='*60}")
        
        # Since we're inserting as we go, we don't need to combine and re-insert
        # Just prepare the success message
        if total_inserted_all_chunks == 0:
            # Re-validate dates and symbol before showing error (in case something went wrong)
            if end_dt.year > current_year or end_dt > now_utc:
                return {"success": False, "message": f"❌ Date validation error: End date {e_str} (year: {end_dt.year}) is in the future. This should have been caught earlier. Current date: {now_utc.strftime('%Y-%m-%d')}"}
            
            # Ensure symbol is properly cleaned for display (remove all whitespace including newlines)
            # Use api_symbol first to avoid truncation issues
            api_symbol_clean = api_symbol.strip().replace('\n', '').replace('\r', '').replace('\t', '') if api_symbol else "UNKNOWN"
            polygon_symbol_clean = polygon_symbol_final.strip().replace('\n', '').replace('\r', '').replace('\t', '') if polygon_symbol_final else ""
            
            # Determine the best symbol to display
            # Priority: 1) Full polygon_symbol if available, 2) api_symbol if not truncated, 3) Extract from polygon_symbol
            if polygon_symbol_clean and len(polygon_symbol_clean) > 2:
                # Use polygon_symbol if it's complete (e.g., "C:EURUSD")
                if ":" in polygon_symbol_clean:
                    # Extract the actual symbol part (e.g., "C:EURUSD" -> "EURUSD")
                    symbol_display = polygon_symbol_clean.split(":", 1)[1]
                else:
                    symbol_display = polygon_symbol_clean
            elif api_symbol_clean and api_symbol_clean not in ["C", "I", "C:", "I:", "UNKNOWN"]:
                # Use api_symbol if it's not truncated
                symbol_display = api_symbol_clean
            elif polygon_symbol_clean:
                # Fallback to polygon_symbol even if short
                if ":" in polygon_symbol_clean:
                    symbol_display = polygon_symbol_clean.split(":", 1)[1]
                else:
                    symbol_display = polygon_symbol_clean
            else:
                # Last resort
                symbol_display = api_symbol_clean if api_symbol_clean != "UNKNOWN" else "UNKNOWN"
            
            # Check if this might be a Polygon restriction issue
            if symbol_display.startswith("I:") or api_symbol_clean.upper() in ["I:SPX", "SPX", "^SPX"]:
                return {"success": False, "message": f"❌ No data returned from Polygon for '{symbol_display}' ({s_str} to {e_str}).\n\n⚠️ **POLYGON RESTRICTION**: Polygon does NOT provide 1-minute data for indices (I:SPX, I:DJI, etc.) due to licensing restrictions.\n\n✅ **SOLUTION**: Use SPY instead of I:SPX. SPY is an ETF that tracks the S&P 500 and Polygon provides full minute history for it.\n\n💡 **Recommendation**: Change your symbol from 'I:SPX' to 'SPY' in the Polygon Symbol field.\n\nDebug: api_symbol='{api_symbol_clean}', polygon_symbol='{symbol_display}', asset_class='{asset_class}'"}
            
            return {"success": False, "message": f"❌ No data available for '{symbol_display}' on {s_str}."}
                 
        # All data was inserted as we went - prepare final success message
        resume_msg = f" (resumed from latest)" if auto_resume and start_dt else ""
        timezone_note = " (converted from UTC to GMT+4)"
        date_range_info = f"Date range: {s_str} to {e_str} ({(end_dt - start_dt).days} days)"
        
        # Add info about skipped chunks if any
        skipped_info = ""
        if skipped_chunks:
            today_str = datetime.utcnow().strftime("%Y-%m-%d")
            skipped_today = [chunk for chunk in skipped_chunks if chunk.startswith(today_str)]
            is_24_7 = asset_class == "Commodities" or asset_class == "Currencies"
            if skipped_today and is_24_7:
                # For 24/7 markets, today's skip is likely a Polygon plan limitation
                skipped_info = f"\n\n⚠️ Note: {len(skipped_chunks)} date chunk(s) were skipped: {', '.join(skipped_chunks)}\n💡 Today's data may not be available due to Polygon plan limitations for this {asset_class.lower()} pair. Try a different pair (e.g., GBPUSD for currencies) or upgrade your Polygon plan."
            elif skipped_today:
                skipped_info = f"\n\n⚠️ Note: {len(skipped_chunks)} date chunk(s) were skipped: {', '.join(skipped_chunks)}\n💡 Today's data ({today_str}) may not be available yet if the market hasn't opened. Try again later after market hours."
            else:
                skipped_info = f"\n\n⚠️ Note: {len(skipped_chunks)} date chunk(s) were skipped (no data available): {', '.join(skipped_chunks)}"
        
        if total_inserted_all_chunks > 0:
            success_message = f"✅ Successfully ingested {total_inserted_all_chunks} records for {target_symbol} into {target_table}.\n\n{date_range_info}{timezone_note}.{resume_msg}{skipped_info}"
        else:
            # Even if no new records (all duplicates), still show success
            success_message = f"✅ Ingestion completed for {target_symbol}.\n\n{date_range_info}{timezone_note}.\n\nℹ️ Note: All records were duplicates or already exist in the database.{resume_msg}{skipped_info}"
        
        print(f"🎉 {success_message}")
        return {"success": True, "message": success_message}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"API Ingestion error: {str(e)}"}

def ingest_market_data(uploaded_file, asset_class, default_symbol=None, source_timezone="America/New_York"):
    """
    Ingest uploaded Market Data (CSV/XLS/XLSX) into the appropriate Supabase table.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        asset_class: "Commodities", "Indices", "Currencies", "Stocks"
        default_symbol: Symbol to use if not present in the file
        source_timezone: Timezone of the data in the file (default: US Eastern).
                         Will be converted to GMT+4 (Dubai).
        
    Returns:
        dict: {"success": bool, "message": str}
    """
    if uploaded_file is None:
        return {"success": False, "message": "No file uploaded"}

    # Timezone configuration
    try:
        src_tz = pytz.timezone(source_timezone)
        dst_tz = pytz.timezone('Asia/Dubai')
    except pytz.UnknownTimeZoneError:
        return {"success": False, "message": f"Unknown timezone: {source_timezone}"}

    # Determine target table
    table_map = {
        "Commodities": "market_data_commodities_1min",
        "Indices": "market_data_indices_1min",
        "Currencies": "market_data_currencies_1min",
        "Stocks": "market_data_stocks_1min"
    }
    
    target_table = table_map.get(asset_class)
    if not target_table:
        return {"success": False, "message": f"Unknown asset class: {asset_class}"}

    try:
        # Read file
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            # Read CSV file as text first to handle Barchart header comments
            uploaded_file.seek(0)  # Reset file pointer
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset again
            
            # Decode if bytes
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8', errors='ignore')
            
            lines = file_content.split('\n')
            
            # Find the actual header row (skip comment lines like "Downloaded from Barchart.com...")
            header_row_idx = 0
            for i, line in enumerate(lines):
                line_lower = line.lower()
                # Skip lines that are clearly comments/headers
                if any(keyword in line_lower for keyword in ['downloaded from', 'barchart', 'as of']):
                    continue
                # Check if this line looks like a header (contains common column names)
                if any(col in line_lower for col in ['timestamp', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']):
                    header_row_idx = i
                    break
            
            # Read CSV starting from the header row using StringIO
            csv_content = '\n'.join(lines[header_row_idx:])
            df = pd.read_csv(io.StringIO(csv_content))
        else:
            try:
                df = pd.read_excel(uploaded_file)
            except ImportError:
                return {"success": False, "message": "Missing optional dependency 'openpyxl'. Use pip or conda to install openpyxl."}
            except Exception as e:
                return {"success": False, "message": f"Error reading Excel file: {str(e)}"}
        
        if df.empty:
            return {"success": False, "message": "File is empty"}
            
        # Standardize columns
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Rename common variations to match DB schema
        rename_map = {
            "tradingday": "trading_day",
            "openinterest": "open_interest",
            "time": "timestamp",
            "date": "timestamp",
            "last": "close",
            "latest": "close",
            "price": "close",
            "close_price": "close"
        }
        
        # Manual check for 'latest' if rename doesn't pick it up for some reason
        if "latest" in df.columns and "close" not in df.columns:
             df = df.rename(columns={"latest": "close"})

        df = df.rename(columns=rename_map)
        
        # If symbol is missing, use default_symbol
        if "symbol" not in df.columns:
            if default_symbol:
                df["symbol"] = default_symbol
            else:
                return {"success": False, "message": "Symbol column missing and no default symbol provided"}
            
        # Ensure required columns
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        
        if missing:
             return {"success": False, "message": f"Missing columns in file: {missing}. Found: {list(df.columns)}"}
             
        # Clean invalid rows first (e.g. footer lines like "Downloaded from...")
        if "timestamp" in df.columns:
            df = df[~df["timestamp"].astype(str).str.contains("Downloaded from", case=False, na=False)]
        
        # Handle timestamp parsing
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                # Coerce errors to NaT, then drop NaT rows
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
                df = df.dropna(subset=["timestamp"])
            except Exception as e:
                 return {"success": False, "message": f"Error parsing timestamp: {str(e)}"}
        
        # Prepare for DB
        records = []
        for _, row in df.iterrows():
            try:
                # Timezone Conversion
                ts = row["timestamp"]
                # If naive, assume source_timezone
                if ts.tzinfo is None:
                    ts_localized = src_tz.localize(ts)
                else:
                    ts_localized = ts.astimezone(src_tz)
                
                # Convert to GMT+4
                ts_target = ts_localized.astimezone(dst_tz)
                
                record = {
                    "symbol": str(row["symbol"]),
                    "timestamp": ts_target.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"])
                }
                
                if "open_interest" in row and pd.notnull(row["open_interest"]):
                    record["open_interest"] = int(row["open_interest"])
                
                records.append(record)
            except (ValueError, TypeError):
                continue
        
        if not records:
            return {"success": False, "message": "No valid records found after processing"}

        # Upsert to Supabase
        sb = get_supabase()
        if not sb:
            return {"success": False, "message": "Supabase client not initialized"}
            
        # Chunking for large files
        chunk_size = 1000
        total_rows = len(records)
        
        for i in range(0, total_rows, chunk_size):
            chunk = records[i:i + chunk_size]
            try:
                sb.table(target_table).upsert(chunk).execute()
            except Exception as e:
                # Retry once
                try:
                    sb.table(target_table).upsert(chunk).execute()
                except Exception as e2:
                    return {"success": False, "message": f"Error upserting chunk {i}-{i+chunk_size}: {str(e2)}"}
                    
        return {"success": True, "message": f"Successfully ingested {total_rows} rows into {target_table} for {default_symbol or 'multiple symbols'}"}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Ingestion error: {str(e)}"}