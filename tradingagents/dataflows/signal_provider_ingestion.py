import pandas as pd
from tradingagents.database.config import get_supabase
import io
from datetime import datetime
from typing import Dict, Optional


def validate_signal_provider_data(
    df: pd.DataFrame,
    provider_name: str,
    symbol: str,
    timezone_offset: str = "+04:00"
) -> Dict[str, any]:
    """
    Validate signal provider data before ingestion.
    
    Args:
        df: DataFrame with signal data
        provider_name: Name of the signal provider
        symbol: Trading symbol/currency pair
        timezone_offset: Timezone offset (default GMT+4)
        
    Returns:
        dict: {"valid": bool, "message": str, "warnings": list, "data_summary": dict}
    """
    warnings = []
    data_summary = {}
    
    # Check required columns
    required_cols = ["Date", "Action", "Currency Pair"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {
            "valid": False,
            "message": f"Missing required columns: {', '.join(missing_cols)}",
            "warnings": [],
            "data_summary": {}
        }
    
    # Check provider name
    if not provider_name or len(provider_name.strip()) == 0:
        return {
            "valid": False,
            "message": "Provider name is required",
            "warnings": [],
            "data_summary": {}
        }
    
    # Check symbol
    if not symbol or len(symbol.strip()) == 0:
        return {
            "valid": False,
            "message": "Symbol is required",
            "warnings": [],
            "data_summary": {}
        }
    
    # Check data structure
    if df.empty:
        return {
            "valid": False,
            "message": "File contains no data rows",
            "warnings": [],
            "data_summary": {}
        }
    
    # Validate Date column
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        invalid_dates = df["Date"].isna().sum()
        if invalid_dates > 0:
            warnings.append(f"{invalid_dates} rows have invalid dates")
    except Exception as e:
        return {
            "valid": False,
            "message": f"Error parsing dates: {str(e)}",
            "warnings": [],
            "data_summary": {}
        }
    
    # Validate Action column
    valid_actions = ["buy", "sell", "Buy", "Sell", "BUY", "SELL"]
    invalid_actions = df[~df["Action"].isin(valid_actions)]
    if len(invalid_actions) > 0:
        warnings.append(f"{len(invalid_actions)} rows have invalid actions (expected Buy/Sell)")
    
    # Check timezone - assume GMT+4 as standard
    if timezone_offset != "+04:00":
        warnings.append(f"Using timezone offset: {timezone_offset} (standard is GMT+4)")
    
    # Data summary
    data_summary = {
        "total_rows": len(df),
        "date_range": {
            "start": df["Date"].min().isoformat() if not df["Date"].isna().all() else None,
            "end": df["Date"].max().isoformat() if not df["Date"].isna().all() else None
        },
        "action_counts": df["Action"].value_counts().to_dict() if "Action" in df.columns else {},
        "symbols": df["Currency Pair"].unique().tolist() if "Currency Pair" in df.columns else []
    }
    
    return {
        "valid": True,
        "message": "Data validation passed",
        "warnings": warnings,
        "data_summary": data_summary
    }


def ingest_signal_provider_data(
    uploaded_file,
    provider_name: str,
    symbol: str,
    timezone_offset: str = "+04:00"
) -> Dict[str, str]:
    """
    Ingest signal provider data from Excel file into signal_provider_signals table.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        provider_name: Name of the signal provider (e.g., "PipXpert")
        symbol: Trading symbol/currency pair
        timezone_offset: Timezone offset (default GMT+4)
        
    Returns:
        dict: {"success": bool, "message": str}
    """
    if uploaded_file is None:
        return {"success": False, "message": "No file uploaded"}
    
    if not provider_name or len(provider_name.strip()) == 0:
        return {"success": False, "message": "Provider name is required"}
    
    if not symbol or len(symbol.strip()) == 0:
        return {"success": False, "message": "Symbol is required"}
    
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Read Excel file
        name = uploaded_file.name.lower()
        if name.endswith('.xlsx') or name.endswith('.xls'):
            try:
                df = pd.read_excel(uploaded_file)
            except ImportError:
                return {"success": False, "message": "Missing optional dependency 'openpyxl'. Use pip or conda to install openpyxl."}
            except Exception as e:
                return {"success": False, "message": f"Error reading Excel file: {str(e)}"}
        elif name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            return {"success": False, "message": "Unsupported file format. Please use Excel (.xlsx, .xls) or CSV (.csv)"}
        
        # Reset file pointer again after reading
        uploaded_file.seek(0)
        
        if df.empty:
            return {"success": False, "message": "File is empty"}
        
        # Standardize column names (handle case variations)
        df.columns = [str(c).strip() for c in df.columns]
        
        # Map common column name variations
        column_mapping = {
            "currency pair": "Currency Pair",
            "currencypair": "Currency Pair",
            "pair": "Currency Pair",
            "symbol": "Currency Pair",
            "date": "Date",
            "datetime": "Date",
            "timestamp": "Date",
            "action": "Action",
            "signal": "Action",
            "entry price": "Entry Price",
            "entryprice": "Entry Price",
            "entry": "Entry Price"
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name.lower() in [c.lower() for c in df.columns] and new_name not in df.columns:
                old_col = [c for c in df.columns if c.lower() == old_name.lower()][0]
                df = df.rename(columns={old_col: new_name})
        
        # Validate data
        validation = validate_signal_provider_data(df, provider_name, symbol, timezone_offset)
        if not validation["valid"]:
            return {"success": False, "message": validation["message"]}
        
        # Convert Date to datetime and apply timezone
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        # Remove rows with invalid dates
        df = df.dropna(subset=["Date"])
        
        if df.empty:
            return {"success": False, "message": "No valid date data found after parsing"}
        
        # Convert datetime columns to datetime type if they exist
        datetime_columns = ["SL Hit DateTime", "TP1 Hit DateTime", "TP2 Hit DateTime", "TP3 Hit DateTime"]
        for dt_col in datetime_columns:
            if dt_col in df.columns:
                df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
        
        # Prepare data for Supabase
        db_rows = []
        for _, row in df.iterrows():
            try:
                # Normalize action to lowercase
                action = str(row.get("Action", "")).strip().lower()
                if action not in ["buy", "sell"]:
                    continue  # Skip invalid actions
                
                # Get currency pair (use provided symbol or from data)
                currency_pair = str(symbol).upper().strip() if symbol else str(row.get("Currency Pair", "")).upper().strip()
                if not currency_pair:
                    continue
                
                # Build record
                record = {
                    "provider_name": provider_name.strip(),
                    "symbol": currency_pair,
                    "signal_date": row["Date"].isoformat(),
                    "action": action,
                    "entry_price": float(row["Entry Price"]) if pd.notnull(row.get("Entry Price")) else None,
                    "target_1": float(row["Target 1"]) if pd.notnull(row.get("Target 1")) else None,
                    "target_2": float(row["Target 2"]) if pd.notnull(row.get("Target 2")) else None,
                    "target_3": float(row["Target 3"]) if pd.notnull(row.get("Target 3")) else None,
                    "stop_loss": float(row["Stop Loss"]) if pd.notnull(row.get("Stop Loss")) else None,
                    "timezone_offset": timezone_offset,
                    "created_at": datetime.now().isoformat()
                }
                
                # Handle datetime columns properly
                datetime_fields = {
                    "SL Hit DateTime": "sl_hit_datetime",
                    "TP1 Hit DateTime": "tp1_hit_datetime",
                    "TP2 Hit DateTime": "tp2_hit_datetime",
                    "TP3 Hit DateTime": "tp3_hit_datetime"
                }
                
                for dt_col, field_name in datetime_fields.items():
                    if dt_col in df.columns:
                        dt_val = row.get(dt_col)
                        if pd.notnull(dt_val):
                            # Already converted to datetime, just format it
                            if isinstance(dt_val, pd.Timestamp):
                                record[field_name] = dt_val.isoformat()
                            else:
                                try:
                                    dt_val = pd.to_datetime(dt_val)
                                    if pd.notnull(dt_val):
                                        record[field_name] = dt_val.isoformat()
                                except Exception:
                                    pass
                
                db_rows.append(record)
            except (ValueError, TypeError) as e:
                # Skip rows with data type errors but continue processing
                continue
        
        if not db_rows:
            return {"success": False, "message": "No valid records to insert after processing"}
        
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
                # Upsert into signal_provider_signals
                result = supabase.table("signal_provider_signals").upsert(chunk).execute()
                total_inserted += len(chunk)
            except Exception as e:
                error_msg = str(e)
                # Provide more detailed error information
                if "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
                    return {"success": False, "message": f"Table 'signal_provider_signals' does not exist. Please create it first. Error: {error_msg}"}
                elif "column" in error_msg.lower() and "does not exist" in error_msg.lower():
                    return {"success": False, "message": f"Column mismatch. Check table schema. Error: {error_msg}"}
                else:
                    return {"success": False, "message": f"Database error at chunk starting at row {i}: {error_msg}"}
        
        return {
            "success": True,
            "message": f"Successfully ingested {total_inserted} records for provider '{provider_name}' into signal_provider_signals."
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error ingesting file: {str(e)}"}

