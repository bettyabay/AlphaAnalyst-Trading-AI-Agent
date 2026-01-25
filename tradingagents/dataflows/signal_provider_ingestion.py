import pandas as pd
from tradingagents.database.config import get_supabase
import io
from datetime import datetime
from typing import Dict, Optional
import pytz


def validate_signal_provider_data(
    df: pd.DataFrame,
    provider_name: str,
    timezone_offset: str = "+04:00"
) -> Dict[str, any]:
    """
    Validate signal provider data before ingestion.
    
    Args:
        df: DataFrame with signal data
        provider_name: Name of the signal provider
        timezone_offset: Timezone offset (default GMT+4)
        
    Returns:
        dict: {"valid": bool, "message": str, "warnings": list, "data_summary": dict}
    """
    warnings = []
    data_summary = {}
    
    # Standardize column names (handle case variations) first
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
        "entry": "Entry Price",
        "entry price min": "Entry Price",
        "stop loss": "Stop Loss",
        "target 1": "Target 1",
        "target 2": "Target 2",
        "target 3": "Target 3"
    }
    
    for old_name, new_name in column_mapping.items():
        # Case insensitive check
        matches = [c for c in df.columns if c.lower() == old_name.lower()]
        if matches and new_name not in df.columns:
            df.rename(columns={matches[0]: new_name}, inplace=True)
    
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
    timezone_offset: str = "+04:00",
    source_timezone: str = "UTC",
    progress_callback=None
) -> Dict[str, str]:
    """
    Ingest signal provider data from Excel file into signal_provider_signals table.
    Converts timestamps from source_timezone to GMT+4 (Asia/Dubai).
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        provider_name: Name of the signal provider (e.g., "PipXpert")
        timezone_offset: Timezone offset string for metadata (default GMT+4, kept for backward compatibility)
        source_timezone: Timezone of the dates in the Excel file (default UTC). Will be converted to GMT+4 (Asia/Dubai).
        
    Returns:
        dict: {"success": bool, "message": str}
    """
    if uploaded_file is None:
        return {"success": False, "message": "No file uploaded"}
    
    if not provider_name or len(provider_name.strip()) == 0:
        return {"success": False, "message": "Provider name is required"}
    
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Read file with fallback logic
        name = uploaded_file.name.lower()
        df = None
        
        try:
            if name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                # Try Excel - read all columns as strings to prevent truncation
                try:
                    # Use dtype=str and keep_default_na=False to preserve full cell values
                    # This prevents pandas from truncating or converting values
                    df = pd.read_excel(uploaded_file, engine='openpyxl', dtype=str, keep_default_na=False)
                except Exception as e_xls:
                     # Fallback to CSV
                     uploaded_file.seek(0)
                     try:
                         df = pd.read_csv(uploaded_file)
                     except:
                         raise e_xls # Raise original Excel error if CSV also fails
        except ImportError:
             return {"success": False, "message": "Missing optional dependency 'openpyxl'. Use pip or conda to install openpyxl."}
        except Exception as e:
             return {"success": False, "message": f"Error reading file: {str(e)}"}

        # Reset file pointer again after reading
        uploaded_file.seek(0)
        
        if df.empty:
            return {"success": False, "message": "File is empty"}
        
        # Validation and column mapping happens inside validate now
        validation = validate_signal_provider_data(df, provider_name, timezone_offset)
        if not validation["valid"]:
            return {"success": False, "message": validation["message"]}
        
        # Timezone configuration for conversion
        try:
            src_tz = pytz.timezone(source_timezone)
            dst_tz = pytz.timezone('Asia/Dubai')  # GMT+4
        except pytz.UnknownTimeZoneError:
            return {"success": False, "message": f"Unknown timezone: {source_timezone}"}
        
        # Convert Date to datetime
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
        skipped_count = 0
        skip_reasons = {}  # Track why rows are skipped
        
        # Progress callback for UI updates
        def log_progress(message, level="info"):
            if progress_callback:
                progress_callback(message, level)
            else:
                print(message)
        
        log_progress(f"Processing {len(df)} rows from Excel file...", "info")
        
        for row_idx, row in df.iterrows():
            try:
                # Normalize action to lowercase
                action = str(row.get("Action", "")).strip().lower()
                if action not in ["buy", "sell"]:
                    skip_reasons['Invalid action'] = skip_reasons.get('Invalid action', 0) + 1
                    continue  # Skip invalid actions
                
                # Get currency pair (use provided symbol or from data)
                currency_pair_raw = row.get("Currency Pair", "")
                
                # Debug: Log the raw value to see what we're getting
                if currency_pair_raw and len(str(currency_pair_raw).strip()) <= 3:
                    print(f"DEBUG: Row {len(db_rows)+1} - Raw Currency Pair value: {repr(currency_pair_raw)} (type: {type(currency_pair_raw)})")
                
                currency_pair_raw = str(currency_pair_raw).strip()
                if not currency_pair_raw:
                    skip_reasons['Empty Currency Pair'] = skip_reasons.get('Empty Currency Pair', 0) + 1
                    continue
                
                # Convert to uppercase and validate
                currency_pair = currency_pair_raw.upper().strip()
                
                # Remove any newlines or special characters that might cause issues
                currency_pair = currency_pair.replace('\n', '').replace('\r', '').replace('\t', '')
                
                # Validate symbol length - reject symbols that are too short (likely truncated)
                # Minimum valid symbol is 3 chars (e.g., "SPX"), but most are 6-8 chars
                if len(currency_pair) < 3:
                    skipped_count += 1
                    skip_reasons['Symbol too short'] = skip_reasons.get('Symbol too short', 0) + 1
                    if skipped_count <= 5:  # Only print first 5 to avoid spam
                        print(f"WARNING: Row {row_idx+2} - Skipping invalid symbol '{repr(currency_pair)}' (length: {len(currency_pair)})")
                    continue
                
                # Reject common invalid symbols
                invalid_symbols = {'C', 'I', 'X', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5', 'TP', 'SL', 'TARGET', 'ENTRY'}
                if currency_pair in invalid_symbols:
                    skipped_count += 1
                    skip_reasons['Invalid symbol pattern'] = skip_reasons.get('Invalid symbol pattern', 0) + 1
                    if skipped_count <= 5:
                        print(f"WARNING: Row {row_idx+2} - Skipping invalid symbol '{repr(currency_pair)}'")
                    continue
                
                # If symbol starts with "C:" but is only 2-3 chars, it's likely truncated
                if currency_pair.startswith('C:') and len(currency_pair) <= 3:
                    skipped_count += 1
                    skip_reasons['Truncated C: symbol'] = skip_reasons.get('Truncated C: symbol', 0) + 1
                    if skipped_count <= 5:
                        print(f"WARNING: Row {row_idx+2} - Skipping truncated symbol '{repr(currency_pair)}'")
                    continue
                
                # Normalize symbol format: add C: prefix for currency pairs if not present
                # This ensures consistency with database format
                if len(currency_pair) >= 6 and len(currency_pair) <= 7:
                    # Common currency pairs (6-7 chars)
                    if not currency_pair.startswith('C:') and not currency_pair.startswith('I:') and not currency_pair.startswith('^'):
                        # Check if it's a known currency pair
                        known_pairs = [
                            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
                            'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURCHF', 'AUDNZD', 'EURAUD',
                            'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'EURNZD', 'EURCAD', 'AUDCAD',
                            'AUDCHF', 'CADCHF', 'CADJPY', 'CHFJPY', 'NZDJPY', 'NZDCHF', 'NZDCAD',
                            'XAUUSD', 'XAGUSD', 'XPDUSD', 'XPTUSD'  # Precious metals
                        ]
                        if currency_pair in known_pairs or currency_pair.startswith('XAU') or currency_pair.startswith('XAG'):
                            currency_pair = f"C:{currency_pair}"
                
                # Convert Date from source timezone to GMT+4
                # Example: Excel date "2023-10-05 10:53:30" (UTC) -> "2023-10-05 14:53:30+04:00" (GMT+4)
                # PostgreSQL stores TIMESTAMPTZ in UTC internally, but the value represents GMT+4 time
                # When queried with GMT+4 timezone, it will display as "2023-10-05 14:53:30" (GMT+4)
                signal_date_raw = row["Date"]
                signal_date = None
                if pd.notnull(signal_date_raw):
                    # Convert pandas Timestamp to datetime if needed
                    if isinstance(signal_date_raw, pd.Timestamp):
                        dt_val = signal_date_raw.to_pydatetime()
                    else:
                        dt_val = signal_date_raw
                    
                    # Localize to source timezone if naive
                    if dt_val.tzinfo is None:
                        dt_localized = src_tz.localize(dt_val)
                    else:
                        dt_localized = dt_val.astimezone(src_tz)
                    # Convert to GMT+4 (Asia/Dubai) - this is the target timezone for all signals
                    signal_date = dt_localized.astimezone(dst_tz)
                else:
                    # Date is missing or invalid
                    skip_reasons['Missing/Invalid Date'] = skip_reasons.get('Missing/Invalid Date', 0) + 1
                    if skipped_count <= 5:
                        print(f"WARNING: Row {row_idx+2} - Missing or invalid date")
                    continue
                
                # Validate that we have a valid signal_date
                if signal_date is None:
                    skip_reasons['Date conversion failed'] = skip_reasons.get('Date conversion failed', 0) + 1
                    if skipped_count <= 5:
                        print(f"WARNING: Row {row_idx+2} - Date conversion failed")
                    continue
                
                # Final validation check - ensure symbol is valid before creating record
                if len(currency_pair) < 3 or currency_pair in {'C', 'I', 'X', 'TP1', 'TP2', 'TP3', 'TP4', 'TP5', 'TP', 'SL'}:
                    skipped_count += 1
                    skip_reasons['Final validation failed'] = skip_reasons.get('Final validation failed', 0) + 1
                    if skipped_count <= 5:
                        print(f"ERROR: Row {row_idx+2} - Invalid symbol '{repr(currency_pair)}' detected. Skipping.")
                    continue
                
                # Helper function to safely convert to float (handles empty strings from dtype=str)
                def safe_float(value):
                    """Convert value to float, handling empty strings and NaN"""
                    if value is None:
                        return None
                    if isinstance(value, str):
                        value = value.strip()
                        if value == '' or value.lower() in ['nan', 'none', 'null', '']:
                            return None
                    try:
                        if pd.notnull(value) and value != '':
                            return float(value)
                    except (ValueError, TypeError):
                        pass
                    return None
                
                # Build record - handle potential conversion errors
                try:
                    record = {
                        "provider_name": provider_name.strip(),
                        "symbol": currency_pair,
                        "signal_date": signal_date.isoformat(),
                        "action": action,
                        "entry_price": safe_float(row.get("Entry Price")),
                        "entry_price_max": safe_float(row.get("Entry Price Max")),
                        "target_1": safe_float(row.get("Target 1")),
                        "target_2": safe_float(row.get("Target 2")),
                        "target_3": safe_float(row.get("Target 3")),
                        "target_4": safe_float(row.get("Target 4")),
                        "target_5": safe_float(row.get("Target 5")),
                        "stop_loss": safe_float(row.get("Stop Loss")),
                        "timezone_offset": timezone_offset,
                        "created_at": datetime.now().isoformat()
                    }
                except (ValueError, TypeError) as conv_error:
                    skip_reasons['Value conversion error'] = skip_reasons.get('Value conversion error', 0) + 1
                    skipped_count += 1
                    if skipped_count <= 5:
                        print(f"WARNING: Row {row_idx+2} - Error converting values: {str(conv_error)}")
                    continue
                
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
                            # Convert to datetime if not already
                            if isinstance(dt_val, pd.Timestamp):
                                dt_ts = dt_val
                            else:
                                try:
                                    dt_ts = pd.to_datetime(dt_val)
                                    if pd.isna(dt_ts):
                                        continue
                                except Exception:
                                    continue
                            
                            # Convert from source timezone to GMT+4
                            # Localize to source timezone if naive
                            if dt_ts.tzinfo is None:
                                dt_localized = src_tz.localize(dt_ts.to_pydatetime())
                            else:
                                dt_localized = dt_ts.to_pydatetime().astimezone(src_tz)
                            # Convert to GMT+4 (Asia/Dubai)
                            dt_target = dt_localized.astimezone(dst_tz)
                            record[field_name] = dt_target.isoformat()
                
                db_rows.append(record)
            except (ValueError, TypeError) as e:
                # Skip rows with data type errors but continue processing
                skipped_count += 1
                skip_reasons['Data type error'] = skip_reasons.get('Data type error', 0) + 1
                if skipped_count <= 5:
                    print(f"ERROR: Row {row_idx+2} - Data type error: {str(e)}")
                continue
            except Exception as e:
                # Catch any other unexpected errors
                skipped_count += 1
                skip_reasons['Unexpected error'] = skip_reasons.get('Unexpected error', 0) + 1
                if skipped_count <= 5:
                    print(f"ERROR: Row {row_idx+2} - Unexpected error: {str(e)}")
                continue
        
        # Print summary of skipped rows
        if skipped_count > 0:
            print(f"\n{'='*80}")
            print(f"SKIP SUMMARY: {skipped_count} row(s) were skipped:")
            for reason, count in skip_reasons.items():
                print(f"  - {reason}: {count} row(s)")
            print(f"{'='*80}\n")
        
        if not db_rows:
            return {
                "success": False, 
                "message": f"No valid records to insert after processing. {skipped_count} row(s) were skipped due to validation errors (e.g., invalid symbols like 'C', missing required fields). Please check your Excel file and ensure the 'Currency Pair' column contains valid trading symbols (e.g., EURUSD, XAUUSD, GBPUSD)."
            }
        
        # Batch insert
        supabase = get_supabase()
        if not supabase:
            return {"success": False, "message": "Supabase not configured (check .env)"}
        
        # Insert records one by one to handle duplicates properly
        # This allows us to insert new signals even if some in the batch are duplicates
        total_inserted = 0
        total_skipped = 0
        warning_msg = ""
        
        log_progress(f"Inserting {len(db_rows)} records into database...", "info")
        
        # Progress tracking
        progress_messages = []
        
        for idx, record in enumerate(db_rows):
            # Update progress every 50 records or at milestones
            if (idx + 1) % 50 == 0 or idx == 0 or idx == len(db_rows) - 1:
                progress_pct = ((idx + 1) / len(db_rows)) * 100
                progress_msg = f"Processing: {idx + 1}/{len(db_rows)} records ({progress_pct:.1f}%)"
                log_progress(progress_msg, "info")
                progress_messages.append(progress_msg)
            try:
                # Insert one record at a time
                result = supabase.table("signal_provider_signals").insert(record).execute()
                if result.data:
                    total_inserted += 1
                else:
                    # If no data returned but no error, assume it was inserted
                    total_inserted += 1
            except Exception as e:
                error_msg = str(e)
                
                # Check for duplicate key errors
                if "duplicate key" in error_msg.lower() or "23505" in error_msg:
                    # This signal already exists - skip it
                    total_skipped += 1
                    if total_skipped <= 5:  # Only log first 5 duplicates
                        dup_msg = f"Skipped duplicate: {record.get('symbol')} {record.get('action')} on {record.get('signal_date')}"
                        log_progress(dup_msg, "warning")
                        progress_messages.append(dup_msg)
                else:
                    # Some other error - log it but continue
                    print(f"  Error inserting {record.get('symbol')} on {record.get('signal_date')}: {error_msg[:200]}")
                    total_skipped += 1
        
        # Final summary
        summary_msg = f"Insertion complete: {total_inserted} inserted, {total_skipped} skipped"
        log_progress(summary_msg, "success")
        progress_messages.append(summary_msg)
        
        # Build success message with details
        message_parts = [f"Successfully ingested {total_inserted} record(s) for provider '{provider_name}'"]
        if skipped_count > 0:
            message_parts.append(f"{skipped_count} row(s) were skipped due to validation errors")
        if total_skipped > 0:
            message_parts.append(f"{total_skipped} duplicate signal(s) were skipped (already exist in database)")
        if warning_msg:
            message_parts.append(warning_msg.strip())
        
        return {
            "success": True,
            "message": ". ".join(message_parts) + ".",
            "details": {
                "total_processed": len(df),
                "valid_records": len(db_rows),
                "inserted": total_inserted,
                "skipped_validation": skipped_count,
                "skipped_duplicates": total_skipped,
                "skip_reasons": skip_reasons,
                "progress_messages": progress_messages[-10:] if len(progress_messages) > 10 else progress_messages  # Last 10 messages
            }
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error ingesting file: {str(e)}"}

