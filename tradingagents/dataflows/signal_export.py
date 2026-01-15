"""
Signal Export Service
Handles exporting signals from database to Excel and other formats.
"""
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import pytz
from io import BytesIO

from tradingagents.database.config import get_supabase
from tradingagents.database.db_service import get_provider_signals


def export_signals_to_excel(
    symbol: Optional[str] = None,
    provider: Optional[str] = None,
    limit: Optional[int] = None,
    include_all_providers: bool = True
) -> BytesIO:
    """
    Export signals from database to Excel format.
    
    Args:
        symbol: Filter by symbol (optional)
        provider: Filter by provider name (optional)
        limit: Maximum number of records to export (None = all)
        include_all_providers: If True, export all providers; if False, only specified provider
        
    Returns:
        BytesIO object containing Excel file
    """
    supabase = get_supabase()
    if not supabase:
        raise ValueError("Supabase not configured")
    
    # Build query
    query = supabase.table("signal_provider_signals").select("*")
    
    if symbol:
        query = query.eq("symbol", symbol.upper())
    if provider:
        query = query.eq("provider_name", provider)
    
    query = query.order("signal_date", desc=True)
    
    if limit:
        query = query.limit(limit)
    else:
        # For large exports, fetch in chunks
        query = query.limit(100000)  # Supabase limit
    
    result = query.execute()
    signals = result.data if result.data else []
    
    if not signals:
        # Return empty Excel file
        df = pd.DataFrame()
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Signals')
        output.seek(0)
        return output
    
    # Convert to DataFrame
    df = pd.DataFrame(signals)
    
    # Reorder columns for better Excel format
    column_order = [
        'provider_name',
        'symbol',
        'signal_date',
        'action',
        'entry_price',
        'stop_loss',
        'target_1',
        'target_2',
        'target_3',
        'target_4',
        'target_5',
        'sl_hit_datetime',
        'tp1_hit_datetime',
        'tp2_hit_datetime',
        'tp3_hit_datetime',
        'timezone_offset',
        'created_at',
        'updated_at'
    ]
    
    # Only include columns that exist in the DataFrame
    available_columns = [col for col in column_order if col in df.columns]
    # Add any remaining columns
    remaining_columns = [col for col in df.columns if col not in available_columns]
    final_columns = available_columns + remaining_columns
    
    df = df[final_columns]
    
    # Rename columns to Excel-friendly names
    column_mapping = {
        'provider_name': 'Provider Name',
        'symbol': 'Currency Pair',
        'signal_date': 'Date',
        'action': 'Action',
        'entry_price': 'Entry Price',
        'stop_loss': 'Stop Loss',
        'target_1': 'Target 1',
        'target_2': 'Target 2',
        'target_3': 'Target 3',
        'target_4': 'Target 4',
        'target_5': 'Target 5',
        'sl_hit_datetime': 'SL Hit DateTime',
        'tp1_hit_datetime': 'TP1 Hit DateTime',
        'tp2_hit_datetime': 'TP2 Hit DateTime',
        'tp3_hit_datetime': 'TP3 Hit DateTime',
        'timezone_offset': 'Timezone Offset',
        'created_at': 'Created At',
        'updated_at': 'Updated At'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Convert datetime columns to readable format
    datetime_columns = ['Date', 'SL Hit DateTime', 'TP1 Hit DateTime', 'TP2 Hit DateTime', 'TP3 Hit DateTime', 'Created At', 'Updated At']
    for col in datetime_columns:
        if col in df.columns:
            # Convert ISO format strings to datetime, then format
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Format as readable datetime string
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
    
    # Capitalize Action column
    if 'Action' in df.columns:
        df['Action'] = df['Action'].str.capitalize()
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Signals')
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Signals']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            )
            # Limit max width to 50
            max_length = min(max_length, 50)
            worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
    
    output.seek(0)
    return output


def export_telegram_messages_to_excel(
    messages: List[Dict], 
    channel_username: str,
    format_as_market_data: bool = False,
    format_as_signal_provider: bool = False
) -> BytesIO:
    """
    Export raw Telegram messages to Excel format.
    Useful for reviewing all message formats before parsing.
    
    Args:
        messages: List of message dictionaries from fetch_all_channel_messages
        channel_username: Channel username for filename
        format_as_market_data: If True, export parsed signals in market data format (symbol, timestamp, open, high, low, close, volume)
        format_as_signal_provider: If True, export in signal provider format (Date, Action, Currency Pair, Entry Price, Stop Loss, Target 1, etc.)
        
    Returns:
        BytesIO object containing Excel file
    """
    if not messages:
        # Return empty Excel file
        df = pd.DataFrame()
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Messages')
        output.seek(0)
        return output
    
    # If format_as_signal_provider, convert signals to signal provider format (for upload via Signal Provider section)
    if format_as_signal_provider:
        signal_provider_rows = []
        for msg in messages:
            parsed_signal = msg.get('parsed_signal')
            if not parsed_signal:
                continue
            
            # Extract signal data
            symbol = parsed_signal.get('symbol', '').upper()
            signal_date = parsed_signal.get('signal_date') or msg.get('date')
            action = parsed_signal.get('action', '').capitalize()  # Buy or Sell
            entry_price = parsed_signal.get('entry_price')
            stop_loss = parsed_signal.get('stop_loss')
            target_1 = parsed_signal.get('target_1')
            target_2 = parsed_signal.get('target_2')
            target_3 = parsed_signal.get('target_3')
            target_4 = parsed_signal.get('target_4')
            target_5 = parsed_signal.get('target_5')
            
            if not symbol or not signal_date or not action or entry_price is None:
                continue
            
            # Parse timestamp and convert to timezone-naive for Excel
            try:
                if isinstance(signal_date, str):
                    ts = pd.to_datetime(signal_date, errors='coerce')
                else:
                    ts = pd.to_datetime(signal_date, errors='coerce')
                
                if pd.isna(ts):
                    continue
                
                # Convert to timezone-naive (Excel doesn't support timezone-aware)
                if isinstance(ts, pd.Timestamp):
                    if ts.tz is not None:
                        ts = ts.tz_localize(None)
                    # Format as datetime for Excel
                    date_value = ts
                elif hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                    # Python datetime with timezone
                    date_value = ts.replace(tzinfo=None)
                else:
                    # Already timezone-naive
                    date_value = ts
            except Exception as e:
                # Fallback: skip this record
                continue
            
            # Build row in signal provider format
            row = {
                'Date': date_value,
                'Action': action,
                'Currency Pair': symbol,
                'Entry Price': float(entry_price) if entry_price is not None else None
            }
            
            # Add optional fields
            if stop_loss is not None:
                row['Stop Loss'] = float(stop_loss)
            if target_1 is not None:
                row['Target 1'] = float(target_1)
            if target_2 is not None:
                row['Target 2'] = float(target_2)
            if target_3 is not None:
                row['Target 3'] = float(target_3)
            if target_4 is not None:
                row['Target 4'] = float(target_4)
            if target_5 is not None:
                row['Target 5'] = float(target_5)
            
            signal_provider_rows.append(row)
        
        if not signal_provider_rows:
            # Return empty Excel file
            df = pd.DataFrame()
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Signals')
            output.seek(0)
            return output
        
        df = pd.DataFrame(signal_provider_rows)
        
        # Ensure column order matches signal provider format
        column_order = ['Date', 'Action', 'Currency Pair', 'Entry Price', 'Stop Loss', 
                       'Target 1', 'Target 2', 'Target 3', 'Target 4', 'Target 5']
        available_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns
        df = df[final_columns]
        
        # Sort by Date
        df = df.sort_values('Date')
        
    # If format_as_market_data, convert signals to market data format
    elif format_as_market_data:
        market_data_rows = []
        for msg in messages:
            parsed_signal = msg.get('parsed_signal')
            if not parsed_signal:
                continue
            
            # Extract signal data
            symbol = parsed_signal.get('symbol', '').upper()
            signal_date = parsed_signal.get('signal_date') or msg.get('date')
            entry_price = parsed_signal.get('entry_price')
            stop_loss = parsed_signal.get('stop_loss')
            targets = [
                parsed_signal.get('target_1'),
                parsed_signal.get('target_2'),
                parsed_signal.get('target_3'),
                parsed_signal.get('target_4'),
                parsed_signal.get('target_5')
            ]
            # Filter out None values
            targets = [t for t in targets if t is not None]
            
            # Map to market data format:
            # symbol: from signal
            # timestamp: signal_date
            # open: entry_price
            # high: highest target (or entry_price if no targets)
            # low: stop_loss (or entry_price if no stop_loss)
            # close: entry_price (signal entry point)
            # volume: 0 (signals don't have volume)
            
            if not symbol or not signal_date or entry_price is None:
                continue
            
            # Calculate high (max of targets or entry_price)
            high = max(targets) if targets else entry_price
            
            # Calculate low (stop_loss or entry_price)
            low = stop_loss if stop_loss is not None else entry_price
            
            # Parse timestamp and convert to timezone-naive for Excel
            try:
                if isinstance(signal_date, str):
                    # Parse ISO format string
                    ts = pd.to_datetime(signal_date, errors='coerce')
                else:
                    ts = pd.to_datetime(signal_date, errors='coerce')
                
                if pd.isna(ts):
                    continue
                
                # Convert to timezone-naive (Excel doesn't support timezone-aware)
                if isinstance(ts, pd.Timestamp):
                    if ts.tz is not None:
                        ts = ts.tz_localize(None)
                    # Format as string for Excel
                    timestamp_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                elif hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                    # Python datetime with timezone
                    ts_naive = ts.replace(tzinfo=None)
                    timestamp_str = ts_naive.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Already timezone-naive
                    timestamp_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
            except Exception as e:
                # Fallback: use string representation
                timestamp_str = str(signal_date)
            
            market_data_rows.append({
                'symbol': symbol,
                'timestamp': timestamp_str,
                'open': float(entry_price),
                'high': float(high),
                'low': float(low),
                'close': float(entry_price),
                'volume': 0
            })
        
        if not market_data_rows:
            # Return empty Excel file
            df = pd.DataFrame()
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Signals')
            output.seek(0)
            return output
        
        df = pd.DataFrame(market_data_rows)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
    else:
        # Original format: export all message details
        # Convert to DataFrame, but remove timezone-aware datetime objects
        # Excel doesn't support timezone-aware datetimes
        messages_clean = []
        for msg in messages:
            msg_copy = msg.copy()
            # Remove parsed_signal (it's a nested dict, handle separately if needed)
            if 'parsed_signal' in msg_copy:
                del msg_copy['parsed_signal']
            messages_clean.append(msg_copy)
        
        df = pd.DataFrame(messages_clean)
        
        # Reorder columns
        column_order = ['message_id', 'date', 'text', 'has_media', 'is_reply', 'views', 'forwards']
        available_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns
        
        df = df[final_columns]
        
        # Rename columns
        column_mapping = {
            'message_id': 'Message ID',
            'date': 'Date (UTC)',
            'text': 'Message Text',
            'has_media': 'Has Media',
            'is_reply': 'Is Reply',
            'views': 'Views',
            'forwards': 'Forwards'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Format date column
        if 'Date (UTC)' in df.columns:
            try:
                df['Date (UTC)'] = pd.to_datetime(df['Date (UTC)'], errors='coerce')
                df['Date (UTC)'] = df['Date (UTC)'].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
    
    # Create Excel file in memory
    output = BytesIO()
    if format_as_signal_provider:
        sheet_name = 'Signals'
    elif format_as_market_data:
        sheet_name = 'Signals'
    else:
        sheet_name = 'Messages'
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        
        # Auto-adjust column widths
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns):
            if format_as_signal_provider:
                # Signal provider format: adjust widths
                if col == 'Date':
                    worksheet.column_dimensions[chr(65 + idx)].width = 20
                elif col == 'Currency Pair':
                    worksheet.column_dimensions[chr(65 + idx)].width = 15
                elif col == 'Action':
                    worksheet.column_dimensions[chr(65 + idx)].width = 10
                else:
                    worksheet.column_dimensions[chr(65 + idx)].width = 12
            elif format_as_market_data:
                # Market data format: standard widths
                if col == 'timestamp':
                    worksheet.column_dimensions[chr(65 + idx)].width = 20
                elif col == 'symbol':
                    worksheet.column_dimensions[chr(65 + idx)].width = 15
                else:
                    worksheet.column_dimensions[chr(65 + idx)].width = 12
            else:
                # Original format: adjust based on content
                if col == 'Message Text':
                    # Make text column wider
                    worksheet.column_dimensions[chr(65 + idx)].width = 100
                else:
                    max_length = max(
                        df[col].astype(str).map(len).max() if not df[col].empty else 0,
                        len(str(col))
                    )
                    max_length = min(max_length, 50)
                    worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
    
    output.seek(0)
    return output

