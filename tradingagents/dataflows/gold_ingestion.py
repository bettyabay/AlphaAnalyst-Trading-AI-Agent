import pandas as pd
from tradingagents.database.config import get_supabase
import io

def ingest_gold_data(uploaded_file):
    """
    Ingest uploaded Gold Data (CSV/XLS/XLSX) into market_data_commodities_1min table.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        dict: {"success": bool, "message": str}
    """
    if uploaded_file is None:
        return {"success": False, "message": "No file uploaded"}

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
        
        # Debug: Print columns to help diagnosis
        print(f"DEBUG: Columns after lower/strip: {list(df.columns)}")

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
        
        # If symbol is missing, assume it's GOLD (^XAUUSD)
        if "symbol" not in df.columns:
            df["symbol"] = "^XAUUSD"
            
        # Ensure required columns
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        
        if missing:
             return {"success": False, "message": f"Missing columns in file: {missing}. Found: {list(df.columns)}"}
             
        # Filter out any rows that are clearly not data (e.g., comment lines, headers)
        # These might have been included if they somehow passed the header skip
        if "timestamp" in df.columns:
            # Convert timestamp column, but first filter out invalid rows
            # Try to identify and remove rows where timestamp is not a valid date
            original_len = len(df)
            
            # Convert timestamp with errors='coerce' to mark invalid ones as NaT
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', format='mixed')
            
            # Remove rows where timestamp parsing failed (NaT values)
            df = df.dropna(subset=["timestamp"])
            
            if len(df) == 0:
                return {"success": False, "message": "No valid timestamp data found in file. Please check file format."}
            elif len(df) < original_len:
                print(f"Warning: Removed {original_len - len(df)} rows with invalid timestamps")

        # Prepare data for Supabase
        db_rows = []
        for _, row in df.iterrows():
            try:
                record = {
                    "symbol": str(row["symbol"]),
                    "timestamp": row["timestamp"].isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]) if pd.notnull(row["volume"]) else 0,
                    "open_interest": int(row["open_interest"]) if "open_interest" in df.columns and pd.notnull(row["open_interest"]) else None
                }
                db_rows.append(record)
            except (ValueError, TypeError) as e:
                return {"success": False, "message": f"Data type error in row: {row}. Error: {str(e)}"}
            
        # Batch insert
        supabase = get_supabase()
        if not supabase:
             return {"success": False, "message": "Supabase not configured (check .env)"}
             
        # Upsert in chunks
        chunk_size = 1000
        total_inserted = 0
        
        for i in range(0, len(db_rows), chunk_size):
            chunk = db_rows[i:i+chunk_size]
            try:
                # Upsert into market_data_commodities_1min
                result = supabase.table("market_data_commodities_1min").upsert(chunk).execute()
                total_inserted += len(chunk)
            except Exception as e:
                error_msg = str(e)
                # Provide more detailed error information
                if "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
                    return {"success": False, "message": f"Table 'market_data_commodities_1min' does not exist. Please create it first. Error: {error_msg}"}
                elif "column" in error_msg.lower() and "does not exist" in error_msg.lower():
                    return {"success": False, "message": f"Column mismatch. Check table schema. Error: {error_msg}"}
                else:
                    return {"success": False, "message": f"Database error at chunk starting at row {i}: {error_msg}"}
                 
        return {"success": True, "message": f"Successfully ingested {total_inserted} records into market_data_commodities_1min."}
        
    except Exception as e:
        return {"success": False, "message": f"Error ingesting file: {str(e)}"}
