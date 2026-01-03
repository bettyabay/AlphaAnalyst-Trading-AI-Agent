import pandas as pd
from tradingagents.database.config import get_supabase
from tradingagents.dataflows.polygon_integration import PolygonDataClient
import io
import traceback
from datetime import datetime, timedelta
import pytz

def ingest_from_polygon_api(api_symbol, asset_class, start_date=None, end_date=None, years=5, db_symbol=None):
    """
    Ingest data directly from Polygon API into the appropriate Supabase table.
    Handles timezone conversion to GMT+4 (Dubai).
    
    Args:
        api_symbol: Polygon symbol (e.g. "AAPL", "C:XAUUSD") or Barchart symbol to be converted
        asset_class: "Commodities", "Indices", "Currencies", "Stocks"
        start_date: Optional start date (datetime or YYYYMMDD string)
        end_date: Optional end date (datetime or YYYYMMDD string)
        years: Number of years to fetch if start_date is not provided
        db_symbol: Symbol to store in DB (defaults to api_symbol if None)
        
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
        # Determine dates
        if not end_date:
            end_dt = datetime.now()
        elif isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, "%Y%m%d")
        else:
            end_dt = end_date
            
        if not start_date:
            start_dt = end_dt.replace(year=end_dt.year - years)
        elif isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, "%Y%m%d")
        else:
            start_dt = start_date
            
        s_str = start_dt.strftime("%Y-%m-%d")
        e_str = end_dt.strftime("%Y-%m-%d")
        
        # Convert symbol if needed (handling legacy Barchart symbols if passed)
        polygon_symbol = api_symbol
        if api_symbol == "GC*1":
            polygon_symbol = "C:XAUUSD"
        elif api_symbol == "^SPX" or api_symbol == "$SPX":
            polygon_symbol = "I:SPX"
            
        # Fetch data from Polygon
        client = PolygonDataClient()
        # Use 1-minute interval
        df = client.get_intraday_data(polygon_symbol, s_str, e_str, multiplier=1, timespan="minute")
        
        if df.empty:
            return {"success": False, "message": f"No data returned from Polygon for {polygon_symbol} ({s_str}-{e_str})"}
            
        # Prepare for DB
        db_rows = []
        target_symbol = db_symbol if db_symbol else polygon_symbol
        
        # Timezone configuration
        # Polygon data is in UTC
        src_tz = pytz.timezone('UTC') 
        dst_tz = pytz.timezone('Asia/Dubai')       
        
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        
        for timestamp, row in df.iterrows():
            if pd.isna(row.get("close")):
                continue 
                
            try:
                # Handle timezone conversion
                # timestamp from Polygon client is naive UTC (from unix ms)
                # 1. Localize to Source Timezone (UTC)
                if timestamp.tzinfo is None:
                    ts_localized = src_tz.localize(timestamp)
                else:
                    ts_localized = timestamp.astimezone(src_tz)
                
                # 2. Convert to Target Timezone (GMT+4)
                ts_target = ts_localized.astimezone(dst_tz)
                
                record = {
                    "symbol": str(target_symbol),
                    "timestamp": ts_target.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]) if pd.notnull(row.get("volume")) else 0
                }
                
                # Polygon doesn't typically provide open interest in agg bars
                # So we skip open_interest
                
                db_rows.append(record)
            except (ValueError, TypeError):
                continue

        if not db_rows:
            return {"success": False, "message": "No valid data rows after processing."}

        # Batch insert
        sb = get_supabase()
        if not sb:
             return {"success": False, "message": "Supabase not configured (check .env)"}
             
        chunk_size = 1000
        total_inserted = 0
        
        for i in range(0, len(db_rows), chunk_size):
            chunk = db_rows[i:i+chunk_size]
            try:
                sb.table(target_table).upsert(chunk).execute()
                total_inserted += len(chunk)
            except Exception as e:
                return {"success": False, "message": f"Database error at chunk {i}: {str(e)}"}
                 
        return {"success": True, "message": f"Successfully ingested {total_inserted} records for {target_symbol} ({s_str}-{e_str}) into {target_table}."}

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
