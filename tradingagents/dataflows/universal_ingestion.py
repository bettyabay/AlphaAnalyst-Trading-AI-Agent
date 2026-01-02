import pandas as pd
from tradingagents.database.config import get_supabase
import io
import traceback

def ingest_market_data(uploaded_file, asset_class, default_symbol=None):
    """
    Ingest uploaded Market Data (CSV/XLS/XLSX) into the appropriate Supabase table.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        asset_class: "Commodities", "Indices", "Currencies", "Stocks"
        default_symbol: Symbol to use if not present in the file
        
    Returns:
        dict: {"success": bool, "message": str}
    """
    if uploaded_file is None:
        return {"success": False, "message": "No file uploaded"}

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
        
        # Convert UTC to GMT+4 (Barchart exports are typically UTC or Exchange Time, assuming UTC based on user input)
        # Check if timezone naive, if so localize to UTC then convert. If already aware, convert.
        if df["timestamp"].dt.tz is None:
             df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=4)
        else:
             # If already tz-aware, convert to GMT+4 (Etc/GMT-4 which is +4, or just add offset if we strip tz)
             # Simpler to just add 4 hours to the underlying UTC time if we treat it as naive-ish or just want to shift
             df["timestamp"] = df["timestamp"].dt.tz_convert(None) + pd.Timedelta(hours=4)

        # Select only valid columns
        valid_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "open_interest"]
        cols_to_keep = [c for c in valid_cols if c in df.columns]
        df = df[cols_to_keep]
        
        # Replace NaN with None for JSON compatibility
        df = df.where(pd.notnull(df), None)
        
        # Convert to records
        records = df.to_dict(orient="records")
        
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
