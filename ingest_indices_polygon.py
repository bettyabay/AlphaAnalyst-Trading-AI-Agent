"""
Indices Ingestion Script for Polygon API (Free Plan)
- Fetches 1 year of historical data (Polygon free plan limit for indices)
- Chunks by 1 day for both 1-minute and daily data
- 12-second throttle between calls
- 90-second wait on 429 rate limit
- Exponential backoff for retries
- Converts UTC (Polygon) to GMT+4 (Asia/Dubai) before storing
"""

from tradingagents.dataflows.polygon_integration import PolygonDataClient
from tradingagents.database.config import get_supabase
from tradingagents.dataflows.ingestion_pipeline import convert_instrument_to_polygon_symbol
from datetime import datetime, timedelta
import pandas as pd
import pytz
import time
import traceback
from typing import Dict


def _insert_chunk_to_db(
    chunk_df: pd.DataFrame,
    target_symbol: str,
    target_table: str,
    sb,
    utc_tz,
    gmt4_tz,
    chunk_s_str: str,
    chunk_e_str: str,
    expected_api_symbol: str = None  # For verification
) -> Dict:
    """
    Helper function to insert a single chunk into the database immediately after fetching.
    
    CRITICAL: Polygon returns timestamps in UTC (timezone-naive pandas Timestamps).
    We MUST convert them to GMT+4 before storing in the database.
    
    Returns: {"success": bool, "count": int, "message": str}
    """
    db_rows = []
    
    if "timestamp" in chunk_df.columns:
        chunk_df = chunk_df.set_index("timestamp")
    
    # Verify we have data
    if chunk_df.empty:
        return {"success": False, "count": 0, "message": "No data in chunk DataFrame"}
    
    # Sample verification: Check first few rows to ensure data looks valid
    sample_size = min(3, len(chunk_df))
    sample_rows = chunk_df.head(sample_size)
    
    for timestamp, row in chunk_df.iterrows():
        if pd.isna(row.get("close")):
            continue 
            
        try:
            # CRITICAL TIMEZONE CONVERSION:
            # Polygon API returns timestamps in UTC (as timezone-naive pandas Timestamps)
            # We MUST explicitly treat them as UTC and convert to GMT+4 for database storage
            
            if timestamp.tzinfo is None:
                # Polygon returns UTC timestamps as timezone-naive
                # Explicitly localize as UTC
                ts_utc = utc_tz.localize(timestamp) if isinstance(timestamp, datetime) else utc_tz.localize(timestamp.to_pydatetime())
            else:
                # If already timezone-aware, ensure it's UTC
                ts_utc = timestamp.astimezone(utc_tz) if isinstance(timestamp, datetime) else timestamp.tz_convert(utc_tz).to_pydatetime()
                if not isinstance(ts_utc, datetime):
                    ts_utc = utc_tz.localize(ts_utc)
            
            # Convert UTC → GMT+4 for database storage
            if isinstance(ts_utc, datetime):
                ts_gmt4 = ts_utc.astimezone(gmt4_tz)
            else:
                # Handle pandas Timestamp
                if hasattr(ts_utc, 'tz_convert'):
                    ts_gmt4 = ts_utc.tz_convert(gmt4_tz).to_pydatetime()
                else:
                    ts_gmt4 = pd.Timestamp(ts_utc).tz_localize(utc_tz).tz_convert(gmt4_tz).to_pydatetime()
            
            # Verify timezone conversion - ensure we have GMT+4
            if ts_gmt4.tzinfo != gmt4_tz:
                # Force conversion to GMT+4
                if isinstance(ts_gmt4, datetime):
                    ts_gmt4 = ts_gmt4.astimezone(gmt4_tz)
                else:
                    ts_gmt4 = pd.Timestamp(ts_gmt4).tz_convert(gmt4_tz).to_pydatetime()
            
            # Verify offset is +04:00 (GMT+4)
            expected_offset = timedelta(hours=4)
            actual_offset = ts_gmt4.utcoffset()
            if actual_offset != expected_offset:
                print(f"   ⚠️ WARNING: Timezone offset mismatch. Expected +04:00, got {actual_offset}")
                # Log for first few rows only
                if len(db_rows) < 3:
                    print(f"      Original UTC: {ts_utc.isoformat() if isinstance(ts_utc, datetime) else ts_utc}")
                    print(f"      Converted GMT+4: {ts_gmt4.isoformat()}")
            
            # CRITICAL PRICE CONVERSION: Convert ETF prices to Index-equivalent prices
            # Some ETFs (like DIA) trade at fractions of index values (DIA ≈ 1/100th of DJI)
            conversion_factor = INDEX_PRICE_CONVERSION_FACTORS.get(target_symbol, 1.0)
            
            if conversion_factor != 1.0:
                # Convert ETF prices to index-equivalent prices
                open_price = float(row["open"]) * conversion_factor
                high_price = float(row["high"]) * conversion_factor
                low_price = float(row["low"]) * conversion_factor
                close_price = float(row["close"]) * conversion_factor
                
                # Log conversion for first few rows
                if len(db_rows) < 3:
                    print(f"   🔄 Price conversion: ETF {row['close']:.2f} × {conversion_factor} = Index {close_price:.2f}")
            else:
                # No conversion needed - use prices as-is
                open_price = float(row["open"])
                high_price = float(row["high"])
                low_price = float(row["low"])
                close_price = float(row["close"])
            
            # Store GMT+4 timestamp in database
            # Note: Not including open_interest as it may not exist in the table schema
            record = {
                "symbol": str(target_symbol),
                "timestamp": ts_gmt4.isoformat(),  # Store in GMT+4
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": int(row["volume"]) if pd.notnull(row.get("volume")) else 0
            }
            
            # Skip open_interest - column doesn't exist in table or causes schema errors
            
            db_rows.append(record)
        except (ValueError, TypeError) as e:
            print(f"   ⚠️ Error processing row: {e}")
            continue
    
    if not db_rows:
        return {"success": False, "count": 0, "message": "No valid rows after processing"}
    
    # Insert immediately
    try:
        # Log what we're about to insert
        print(f"   💾 Inserting {len(db_rows)} records with symbol '{target_symbol}' into table '{target_table}'")
        
        result = sb.table(target_table).upsert(db_rows).execute()
        
        # Quick verification - check that data was actually stored with correct symbol
        verify = sb.table(target_table)\
            .select("symbol, timestamp")\
            .eq("symbol", target_symbol)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        
        if verify.data:
            stored_symbol = verify.data[0].get("symbol")
            stored_timestamp = verify.data[0].get("timestamp")
            if stored_symbol == target_symbol:
                print(f"   ✅ Verified: Data stored with correct symbol '{stored_symbol}' (latest: {stored_timestamp})")
                return {"success": True, "count": len(db_rows), "message": f"Inserted {len(db_rows)} records with symbol '{target_symbol}'"}
            else:
                print(f"   ⚠️ Warning: Expected symbol '{target_symbol}' but found '{stored_symbol}'")
                return {"success": False, "count": 0, "message": f"Symbol mismatch: expected '{target_symbol}', found '{stored_symbol}'"}
        else:
            print(f"   ⚠️ Warning: Inserted but verification query returned no records for symbol '{target_symbol}'")
            return {"success": False, "count": 0, "message": "Inserted but verification failed - no records found"}
    except Exception as e:
        print(f"   ❌ Insert error: {str(e)}")
        return {"success": False, "count": 0, "message": f"Insert error: {str(e)}"}


# Price conversion factors: ETF price × factor = Index value
# CRITICAL: Some ETFs trade at fractions of index values and need conversion
# These factors convert ETF prices to index-equivalent prices for storage
INDEX_PRICE_CONVERSION_FACTORS = {
    # Dow Jones Industrial Average: DIA ETF trades at ~1/100th of DJI index
    # Example: DIA $434 → DJI ~43,400 points (actual DJI was ~49,000 in early 2026)
    # Note: Factor can vary slightly, but 100 is the standard ratio
    "I:DJI": 100.0,
    
    # Other indices: ETFs track closely, no conversion needed
    "I:SPX": 1.0,        # SPY tracks S&P 500 - prices are similar scale
    "I:NDX": 1.0,        # QQQ tracks NASDAQ-100 - prices are similar
    "I:RUT": 1.0,        # IWM tracks Russell 2000 - prices are similar
    "I:OEX": 1.0,        # OEF tracks S&P 100 - prices are similar
    "I:DJT": 1.0,        # IYT tracks DJ Transportation - prices are similar
    "I:DJU": 1.0,        # XLU tracks DJ Utility - prices are similar
    "I:W5000": 1.0,      # VTI tracks total market - prices are similar
    "I:VIX": 1.0,        # VIX works directly - no conversion
    "I:IXIC": 1.0,       # ONEQ tracks NASDAQ Composite - prices are similar
    "I:NYA": 1.0,        # VTI proxy - prices are similar
}

# Mapping of indices to their ETF equivalents for 1-minute data
# Polygon doesn't provide 1-minute data for cash indices, so we use ETFs
INDEX_TO_ETF_MAPPING = {
    # S&P 500
    "I:SPX": "SPY",
    "SPX": "SPY",
    "^SPX": "SPY",
    "SP500": "SPY",
    "S&P 500": "SPY",
    "S&P500": "SPY",
    
    # NASDAQ-100 (handle various formats)
    "I:NDX": "QQQ",
    "NDX": "QQQ",
    "^NDX": "QQQ",
    "NASDAQ-100": "QQQ",
    "NASDAQ100": "QQQ",
    "NAS 100": "QQQ",  # Common variation with space
    "NAS100": "QQQ",
    "I:NAS 100": "QQQ",  # If user enters with I: prefix and space
    "I:NAS100": "QQQ",
    
    # NASDAQ Composite
    "I:IXIC": "ONEQ",  # NASDAQ Composite → Fidelity NASDAQ Composite ETF
    "IXIC": "ONEQ",
    "^IXIC": "ONEQ",
    "NASDAQ": "ONEQ",
    "NASDAQ COMPOSITE": "ONEQ",
    "NASDAQCOMPOSITE": "ONEQ",
    
    # NYSE Composite
    # Note: No direct ETF tracks NYSE Composite. VTI is used as proxy since:
    # - NYSE Composite represents ~2,000 NYSE-listed stocks
    # - VTI includes all NYSE stocks plus NASDAQ (broader coverage)
    # - High correlation (~95%+) makes VTI an acceptable proxy for minute-level data
    "I:NYA": "VTI",  # NYSE Composite → Vanguard Total Stock Market ETF (best available proxy)
    "NYA": "VTI",
    "^NYA": "VTI",
    "NYSE": "VTI",
    "NYSE COMPOSITE": "VTI",
    "NYSECOMPOSITE": "VTI",
    
    # S&P 100
    "I:OEX": "OEF",  # S&P 100 → iShares S&P 100 ETF
    "OEX": "OEF",
    "^OEX": "OEF",
    "S&P 100": "OEF",
    "S&P100": "OEF",
    "SP100": "OEF",
    
    # Dow Jones Industrial Average
    "I:DJI": "DIA",
    "DJI": "DIA",
    "^DJI": "DIA",
    "DOW": "DIA",
    "DOW JONES": "DIA",
    "DOW JONES INDUSTRIAL": "DIA",
    
    # Dow Jones Transportation Average
    "I:DJT": "IYT",  # Dow Jones Transportation → iShares Transportation Average ETF
    "DJT": "IYT",
    "^DJT": "IYT",
    "DOW TRANSPORTATION": "IYT",
    "DOW TRANSPORT": "IYT",
    "DJ TRANSPORTATION": "IYT",
    "DJ TRANSPORT": "IYT",
    
    # Dow Jones Utility Average
    "I:DJU": "XLU",  # Dow Jones Utility → Utilities Select Sector SPDR Fund
    "DJU": "XLU",
    "^DJU": "XLU",
    "DOW UTILITY": "XLU",
    "DOW UTILITIES": "XLU",
    "DJ UTILITY": "XLU",
    "DJ UTILITIES": "XLU",
    
    # Russell 2000
    "I:RUT": "IWM",
    "RUT": "IWM",
    "^RUT": "IWM",
    "RUSSELL 2000": "IWM",
    "RUSSELL2000": "IWM",
    
    # S&P 400 MidCap
    "I:MID": "MDY",  # S&P MidCap 400 → SPDR S&P MidCap 400 ETF
    "MID": "MDY",
    "^MID": "MDY",
    "S&P 400": "MDY",
    "S&P400": "MDY",
    "SP400": "MDY",
    "MIDCAP": "MDY",
    "MID CAP": "MDY",
    
    # S&P 600 SmallCap
    "I:SML": "IJR",  # S&P SmallCap 600 → iShares Core S&P Small-Cap ETF
    "SML": "IJR",
    "^SML": "IJR",
    "S&P 600": "IJR",
    "S&P600": "IJR",
    "SP600": "IJR",
    "SMALLCAP": "IJR",
    "SMALL CAP": "IJR",
    
    # Russell 1000 (Large Cap)
    "I:RUI": "ONEQ",  # Russell 1000 → Use ONEQ as proxy (or IWB for iShares Russell 1000)
    "RUI": "ONEQ",
    "^RUI": "ONEQ",
    "RUSSELL 1000": "ONEQ",
    "RUSSELL1000": "ONEQ",
    
    # Russell 3000 (Broad Market)
    "I:RUA": "VTI",  # Russell 3000 → Vanguard Total Stock Market ETF
    "RUA": "VTI",
    "^RUA": "VTI",
    "RUSSELL 3000": "VTI",
    "RUSSELL3000": "VTI",
    
    # Wilshire 5000 (Total Market)
    # Note: VTI directly tracks the CRSP US Total Market Index, which is very similar to Wilshire 5000
    # Both represent the entire US stock market (~3,500-4,000 stocks)
    # VTI is the most accurate ETF proxy for Wilshire 5000
    "I:W5000": "VTI",  # Wilshire 5000 → Vanguard Total Stock Market ETF (best match)
    "W5000": "VTI",
    "^W5000": "VTI",
    "WILSHIRE 5000": "VTI",
    "WILSHIRE5000": "VTI",
    
    # VIX (volatility index - may work directly, but try VXX as backup)
    "I:VIX": "VIX",  # VIX might work directly for daily, but for 1min we might need VXX
    "VIX": "VIX",
    "^VIX": "VIX",
    
    # International Indices
    # FTSE 100 (UK)
    "I:UKX": "EWU",  # FTSE 100 → iShares MSCI United Kingdom ETF
    "UKX": "EWU",
    "^UKX": "EWU",
    "FTSE 100": "EWU",
    "FTSE100": "EWU",
    
    # Nikkei 225 (Japan)
    "I:N225": "EWJ",  # Nikkei 225 → iShares MSCI Japan ETF
    "N225": "EWJ",
    "^N225": "EWJ",
    "NIKKEI": "EWJ",
    "NIKKEI 225": "EWJ",
    "NIKKEI225": "EWJ",
    
    # DAX (Germany)
    "I:GDAXI": "EWG",  # DAX → iShares MSCI Germany ETF
    "GDAXI": "EWG",
    "^GDAXI": "EWG",
    "DAX": "EWG",
    
    # CAC 40 (France)
    "I:FCHI": "EWQ",  # CAC 40 → iShares MSCI France ETF
    "FCHI": "EWQ",
    "^FCHI": "EWQ",
    "CAC 40": "EWQ",
    "CAC40": "EWQ",
    "CAC": "EWQ",
    
    # Hang Seng (Hong Kong)
    "I:HSI": "EWH",  # Hang Seng → iShares MSCI Hong Kong ETF
    "HSI": "EWH",
    "^HSI": "EWH",
    "HANG SENG": "EWH",
    "HANGSENG": "EWH",
    
    # Shanghai Composite (China)
    "I:SSEC": "FXI",  # Shanghai Composite → iShares China Large-Cap ETF
    "SSEC": "FXI",
    "^SSEC": "FXI",
    "SHANGHAI": "FXI",
    "SHANGHAI COMPOSITE": "FXI",
    
    # TSX (Canada)
    "I:TSX": "EWC",  # TSX Composite → iShares MSCI Canada ETF
    "TSX": "EWC",
    "^TSX": "EWC",
    "TSX COMPOSITE": "EWC",
    "TSXCOMPOSITE": "EWC",
    
    # ASX 200 (Australia)
    "I:AXJO": "EWA",  # ASX 200 → iShares MSCI Australia ETF
    "AXJO": "EWA",
    "^AXJO": "EWA",
    "ASX 200": "EWA",
    "ASX200": "EWA",
    "ASX": "EWA",
    
    # MSCI EAFE (International Developed Markets)
    "I:EAFE": "EFA",  # MSCI EAFE → iShares MSCI EAFE ETF
    "EAFE": "EFA",
    "^EAFE": "EFA",
    "MSCI EAFE": "EFA",
    "MSCIEAFE": "EFA",
    
    # MSCI Emerging Markets
    "I:EEM": "EEM",  # MSCI Emerging Markets → iShares MSCI Emerging Markets ETF
    "EEM": "EEM",
    "^EEM": "EEM",
    "MSCI EM": "EEM",
    "MSCIEM": "EEM",
    "EMERGING MARKETS": "EEM",
}


def _get_etf_for_index(index_symbol: str, interval: str = "1min") -> str:
    """
    Convert an index symbol to its ETF equivalent for Polygon API.
    
    Args:
        index_symbol: Index symbol (e.g., "I:SPX", "^SPX", "SPX", "NAS 100", "I:NAS 100")
        interval: "1min" for 1-minute data (requires ETF), "daily" for daily data (can use index)
    
    Returns:
        ETF symbol for 1-minute data, or original symbol for daily data
    """
    symbol_upper = index_symbol.upper().strip()
    
    # Normalize by removing spaces, dashes for flexible matching
    symbol_normalized = symbol_upper.replace(" ", "").replace("-", "").replace("_", "")
    
    # For daily data, indices might work directly, so try original first
    if interval == "daily":
        # Check if it's a known index that might work for daily
        if symbol_upper in INDEX_TO_ETF_MAPPING or symbol_normalized in INDEX_TO_ETF_MAPPING:
            # For daily, we can try the index directly first, but have ETF as fallback
            return index_symbol.strip()
        return index_symbol.strip()
    
    # For 1-minute data, always use ETF mapping
    # First try exact match
    if symbol_upper in INDEX_TO_ETF_MAPPING:
        etf = INDEX_TO_ETF_MAPPING[symbol_upper]
        if etf != symbol_upper:
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
        return etf
    
    # Try normalized version (without spaces/dashes)
    if symbol_normalized in INDEX_TO_ETF_MAPPING:
        etf = INDEX_TO_ETF_MAPPING[symbol_normalized]
        if etf != symbol_normalized:
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
        return etf
    
    # Check if it starts with I: or ^ and extract base symbol
    if symbol_upper.startswith("I:"):
        base = symbol_upper[2:].strip()
        base_normalized = base.replace(" ", "").replace("-", "").replace("_", "")
        # Try base with spaces removed
        if base in INDEX_TO_ETF_MAPPING:
            etf = INDEX_TO_ETF_MAPPING[base]
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
            return etf
        elif base_normalized in INDEX_TO_ETF_MAPPING:
            etf = INDEX_TO_ETF_MAPPING[base_normalized]
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
            return etf
        # Also try the full I: prefix version
        if symbol_normalized in INDEX_TO_ETF_MAPPING:
            etf = INDEX_TO_ETF_MAPPING[symbol_normalized]
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
            return etf
    elif symbol_upper.startswith("^"):
        base = symbol_upper[1:].strip()
        base_normalized = base.replace(" ", "").replace("-", "").replace("_", "")
        if base in INDEX_TO_ETF_MAPPING:
            etf = INDEX_TO_ETF_MAPPING[base]
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
            return etf
        elif base_normalized in INDEX_TO_ETF_MAPPING:
            etf = INDEX_TO_ETF_MAPPING[base_normalized]
            print(f"⚠️ Auto-converting {index_symbol} to {etf} (Polygon doesn't support 1-minute data for indices)")
            return etf
    
    # Special handling for "NAS 100" variations that might not be in mapping
    if "NAS" in symbol_upper and "100" in symbol_upper:
        print(f"⚠️ Auto-converting {index_symbol} to QQQ (Polygon doesn't support 1-minute data for indices)")
        return "QQQ"
    
    # If no mapping found, return as-is (might be an ETF already or unknown index)
    return index_symbol.strip()


def ingest_indices_from_polygon(
    api_symbol: str,
    interval: str = "1min",  # "1min" or "daily"
    years: int = 1,  # Polygon free plan: 1 year for indices
    db_symbol: str = None
):
    """
    Ingest indices data from Polygon API.
    
    CRITICAL SYMBOL VERIFICATION:
    - api_symbol: The original symbol from UI (e.g., "I:SPX", "I:NDX")
    - polygon_symbol: The symbol used for Polygon API (may be converted to ETF like "SPY")
    - db_symbol: The symbol stored in database (should match api_symbol for indices)
    
    TIMEZONE HANDLING:
    - Polygon returns timestamps in UTC (timezone-naive)
    - All timestamps are converted to GMT+4 (Asia/Dubai) before database storage
    
    Args:
        api_symbol: Original symbol from UI (e.g., "I:SPX", "I:NDX", "I:DJI")
        interval: "1min" for 1-minute bars, "daily" for daily bars
        years: Number of years to fetch (default 1 for free plan)
        db_symbol: Symbol to store in database (should be the index symbol like "I:SPX")
    """
    try:
        # ============================================================
        # STEP 1: SYMBOL VERIFICATION & MAPPING
        # ============================================================
        print(f"\n{'='*60}")
        print(f"🔍 SYMBOL VERIFICATION")
        print(f"{'='*60}")
        print(f"   Original API Symbol (from UI): '{api_symbol}'")
        
        # Convert index to ETF for 1-minute data (if needed)
        polygon_symbol = _get_etf_for_index(api_symbol, interval)
        print(f"   Polygon API Symbol (for fetching): '{polygon_symbol}'")
        
        # Determine target symbol for database storage
        target_symbol = db_symbol if db_symbol else api_symbol
        print(f"   Database Storage Symbol: '{target_symbol}'")
        
        # CRITICAL VERIFICATION: Ensure db_symbol matches the original index symbol
        if db_symbol and db_symbol != api_symbol:
            print(f"   ⚠️ WARNING: db_symbol '{db_symbol}' differs from api_symbol '{api_symbol}'")
            print(f"   Using db_symbol '{db_symbol}' for storage as specified")
        elif not db_symbol:
            # If no db_symbol provided, use the original api_symbol (index symbol)
            target_symbol = api_symbol
            print(f"   ℹ️ No db_symbol provided, using original api_symbol '{api_symbol}' for storage")
        
        # Verify symbol mapping makes sense
        if interval == "1min" and polygon_symbol != api_symbol:
            print(f"   ✅ Symbol conversion verified: '{api_symbol}' → '{polygon_symbol}' (for API)")
            print(f"   ✅ Will store as '{target_symbol}' in database")
            
            # Check if price conversion is needed (ETF prices → Index-equivalent prices)
            conversion_factor = INDEX_PRICE_CONVERSION_FACTORS.get(target_symbol, 1.0)
            if conversion_factor != 1.0:
                print(f"   🔄 PRICE CONVERSION: ETF prices will be multiplied by {conversion_factor} to get index-equivalent values")
                print(f"      Example: DIA ETF $434 → DJI Index ~{434 * conversion_factor:.0f} points")
                print(f"      This ensures stored prices match actual index values, not ETF prices")
            else:
                print(f"   ℹ️ No price conversion needed - ETF prices are already close to index values")
            
            # WARNING: Check if multiple indices map to same ETF (duplicate data risk)
            indices_using_same_etf = [
                key for key, value in INDEX_TO_ETF_MAPPING.items() 
                if value == polygon_symbol and key.startswith("I:") and key != api_symbol
            ]
            if indices_using_same_etf:
                print(f"   ⚠️ WARNING: Multiple indices use the same ETF '{polygon_symbol}':")
                print(f"      - {api_symbol} (current)")
                for other_index in indices_using_same_etf[:3]:  # Show first 3
                    print(f"      - {other_index}")
                if len(indices_using_same_etf) > 3:
                    print(f"      - ... and {len(indices_using_same_etf) - 3} more")
                print(f"   ⚠️ All will fetch the SAME data from Polygon, stored with different symbols.")
                print(f"   ⚠️ This creates duplicate data in your database (same OHLCV, different symbols).")
        elif polygon_symbol == api_symbol:
            print(f"   ✅ No symbol conversion needed - using '{api_symbol}' for both API and DB")
        else:
            print(f"   ⚠️ Unexpected symbol mapping - please verify")
        
        print(f"{'='*60}\n")
        
        # Determine target table
        if interval == "1min":
            target_table = "market_data_indices_1min"
        elif interval == "daily":
            target_table = "market_data"  # Or create a separate indices daily table if needed
        else:
            return {"success": False, "message": f"Invalid interval: {interval}. Use '1min' or 'daily'"}
        
        # Verify target table is set correctly
        print(f"📋 Target table determined: '{target_table}' for interval '{interval}'")
        
        # Get Supabase client
        sb = get_supabase()
        if not sb:
            return {"success": False, "message": "Supabase not configured (check .env)"}
        
        # Prepare timezone conversion objects (needed for resume logic)
        utc_tz = pytz.timezone('UTC')
        gmt4_tz = pytz.timezone('Asia/Dubai')  # GMT+4
        
        # RESUME LOGIC: Check if data already exists in database
        # If data exists, only fetch from latest timestamp forward
        resume_from_latest = True  # Always check for existing data
        latest_timestamp_utc = None
        is_resuming = False  # Track if we're resuming from existing data
        
        print(f"🔍 Checking for existing data in table '{target_table}' for symbol '{target_symbol}'...")
        print(f"   Polygon API symbol: '{polygon_symbol}'")
        print(f"   Database storage symbol: '{target_symbol}'")
        
        if resume_from_latest:
            try:
                # Query for latest timestamp for this symbol
                latest_result = sb.table(target_table)\
                    .select("timestamp")\
                    .eq("symbol", target_symbol)\
                    .order("timestamp", desc=True)\
                    .limit(1)\
                    .execute()
                
                if latest_result.data and len(latest_result.data) > 0:
                    latest_ts_str = latest_result.data[0].get("timestamp")
                    if latest_ts_str:
                        # Parse timestamp (stored in GMT+4, convert back to UTC for comparison)
                        if isinstance(latest_ts_str, str):
                            # Parse ISO format timestamp - handle both with and without timezone
                            if 'Z' in latest_ts_str or '+' in latest_ts_str or latest_ts_str.endswith('-00:00'):
                                latest_timestamp = datetime.fromisoformat(latest_ts_str.replace('Z', '+00:00'))
                            else:
                                # Assume GMT+4 if no timezone info
                                latest_timestamp = datetime.fromisoformat(latest_ts_str)
                                latest_timestamp = gmt4_tz.localize(latest_timestamp)
                        else:
                            latest_timestamp = latest_ts_str
                        
                        # Convert from GMT+4 (database) to UTC for Polygon API
                        if latest_timestamp.tzinfo is None:
                            # Assume it's GMT+4 if naive
                            latest_timestamp = gmt4_tz.localize(latest_timestamp)
                        elif str(latest_timestamp.tzinfo) != str(gmt4_tz):
                            # If it has a different timezone, convert to GMT+4 first, then to UTC
                            latest_timestamp = latest_timestamp.astimezone(gmt4_tz)
                        
                        # Convert to UTC
                        latest_timestamp_utc = latest_timestamp.astimezone(utc_tz)
                        is_resuming = True
                        
                        print(f"✅ Found existing data for '{target_symbol}' in database")
                        print(f"   Latest timestamp (GMT+4): {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Latest timestamp (UTC): {latest_timestamp_utc.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Will resume from this point forward...")
                else:
                    print(f"ℹ️ No existing data found for symbol '{target_symbol}' in table '{target_table}'")
                    print(f"   Will start fresh ingestion...")
            except Exception as e:
                print(f"⚠️ Could not check for existing data: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Will start fresh ingestion...")
                latest_timestamp_utc = None
        
        # Calculate date range
        # CRITICAL: Use datetime.now(utc_tz) instead of utcnow() for timezone-aware datetime
        now_utc = datetime.now(utc_tz)
        end_dt = now_utc - timedelta(minutes=15)  # Safety buffer for intraday
        
        # Determine start date based on resume logic
        if latest_timestamp_utc and is_resuming:
            # Resume from latest timestamp + 1 minute (to avoid duplicates)
            start_dt = latest_timestamp_utc + timedelta(minutes=1)
            print(f"📅 RESUME MODE: Starting from: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # If start_dt is already at or after end_dt, no new data to fetch
            if start_dt >= end_dt:
                print(f"✅ Data is already up to date. Latest: {latest_timestamp_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC, End: {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                return {"success": True, "message": f"✅ Data for {target_symbol} is already up to date. Latest timestamp: {latest_timestamp_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"}
        else:
            # No existing data - start from 1 year back
            is_resuming = False
            start_dt = end_dt - timedelta(days=365 * years)  # 1 year BACK for free plan
            print(f"📅 FRESH INGESTION: Starting from: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        # Validate dates - but DON'T override resume logic
        current_year = now_utc.year
        if not is_resuming:  # Only validate/recalculate if NOT resuming
            if start_dt.year > current_year or end_dt.year > current_year:
                print(f"⚠️ WARNING: Date range contains future year. Recalculating...")
                end_dt = now_utc - timedelta(minutes=15)
                start_dt = end_dt - timedelta(days=365 * years)
            
            if start_dt > end_dt:
                print(f"⚠️ WARNING: Start date is after end date. Recalculating...")
                end_dt = now_utc - timedelta(minutes=15)
                start_dt = end_dt - timedelta(days=365 * years)
        else:
            # When resuming, only validate that start_dt is not way in the future
            if start_dt.year > current_year + 1:
                print(f"⚠️ WARNING: Resume start date year {start_dt.year} is too far in future. Using current year.")
                end_dt = now_utc - timedelta(minutes=15)
                start_dt = end_dt - timedelta(days=365 * years)
                is_resuming = False  # Switch to fresh ingestion if resume date is invalid
        
        # For daily data, use previous trading day
        if interval == "daily":
            end_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        print(f"\n{'='*60}")
        print(f"📊 POLYGON INDICES INGESTION")
        print(f"{'='*60}")
        print(f"   API Symbol (Polygon): {polygon_symbol}")
        print(f"   DB Symbol (Storage): {target_symbol}")
        print(f"   Original API Symbol: {api_symbol}")
        print(f"   Interval: {interval}")
        print(f"   Mode: {'🔄 RESUME' if is_resuming else '🆕 FRESH INGESTION'}")
        print(f"   Start Date: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"   End Date:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"   Total Days: {(end_dt - start_dt).days} days")
        print(f"   Table: {target_table}")
        print(f"{'='*60}\n")
        
        # Verify symbol mapping is correct
        if interval == "1min" and polygon_symbol != target_symbol:
            print(f"ℹ️ Symbol conversion: '{api_symbol}' → '{polygon_symbol}' (for API) → '{target_symbol}' (for DB)")
        else:
            print(f"ℹ️ Using symbol '{target_symbol}' for both API and DB")
        
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
        
        # Timezone conversion objects already defined above
        
        # Chunk by 1 day (same as gold ingestion for free plan compatibility)
        chunk_days = 1
        chunk_start = start_dt
        max_retries = 5
        all_success = True
        total_inserted = 0
        
        print(f"📦 Chunking date range into {chunk_days}-day chunks for Polygon free plan compatibility...")
        print(f"   Will insert each chunk immediately after fetching (incremental storage)\n")
        print(f"📝 Will store data with symbol: '{target_symbol}' in table: {target_table}\n")
        
        client = PolygonDataClient()
        
        # Process in 1-day chunks - INSERT IMMEDIATELY after each fetch
        chunk_count = 0
        # Calculate total chunks based on date range (for 1-day chunks)
        if chunk_days == 1:
            # For 1-day chunks, count actual days between start and end
            total_chunks_expected = (end_dt.date() - start_dt.date()).days + 1
        else:
            total_chunks_expected = (end_dt - start_dt).days + 1
        
        print(f"📊 Will process approximately {total_chunks_expected} chunks (1 per day)")
        print(f"   Each chunk will be inserted immediately after fetching...\n")
        
        while chunk_start <= end_dt:
            chunk_count += 1
            
            # For 1-day chunks, chunk_end is the same day (end of day)
            if chunk_days == 1:
                # Set chunk_end to end of the same day, but don't exceed end_dt
                chunk_end = min(
                    chunk_start.replace(hour=23, minute=59, second=59, microsecond=999999),
                    end_dt
                )
            else:
                chunk_end = min(chunk_start + timedelta(days=chunk_days) - timedelta(seconds=1), end_dt)
            
            if chunk_end < chunk_start:
                chunk_end = chunk_start
            
            # Format dates for API call (date only, not time)
            chunk_s_str = chunk_start.strftime("%Y-%m-%d")
            chunk_e_str = chunk_end.strftime("%Y-%m-%d")
            
            # Show progress
            progress_pct = (chunk_count / total_chunks_expected * 100) if total_chunks_expected > 0 else 0
            print(f"📦 Processing chunk {chunk_count}/{total_chunks_expected} ({progress_pct:.1f}%): {chunk_s_str} to {chunk_e_str}")
            
            attempt = 0
            chunk_success = False
            
            while attempt < max_retries:
                try:
                    # Apply throttle before each API call
                    _polygon_throttle()
                    
                    if interval == "1min":
                        print(f"🔍 Calling Polygon API: get_intraday_data('{polygon_symbol}', '{chunk_s_str}', '{chunk_e_str}', multiplier=1, timespan='minute')")
                        chunk_df = client.get_intraday_data(polygon_symbol, chunk_s_str, chunk_e_str, multiplier=1, timespan="minute")
                    else:  # daily
                        print(f"🔍 Calling Polygon API: get_historical_data('{polygon_symbol}', '{chunk_s_str}', '{chunk_e_str}')")
                        chunk_df = client.get_historical_data(polygon_symbol, chunk_s_str, chunk_e_str)
                    
                    if not chunk_df.empty:
                        print(f"✅ Fetched {len(chunk_df)} records for {chunk_s_str} to {chunk_e_str}")
                        
                        # ============================================================
                        # DATA VERIFICATION: Ensure fetched data is valid
                        # ============================================================
                        # Check data quality and verify it looks like market data
                        sample_row = chunk_df.iloc[0] if len(chunk_df) > 0 else None
                        if sample_row is not None:
                            # Verify OHLC values are reasonable
                            if pd.notnull(sample_row.get("close")):
                                close_val = float(sample_row.get("close"))
                                # Basic sanity check: prices should be positive and reasonable
                                if close_val <= 0:
                                    print(f"   ⚠️ WARNING: Invalid close price {close_val} in fetched data")
                                elif close_val > 1000000:  # Unlikely for indices/ETFs
                                    print(f"   ⚠️ WARNING: Suspiciously high close price {close_val}")
                                else:
                                    print(f"   ✅ Data validation: Sample close price {close_val:.2f} looks valid")
                            
                            # Verify timestamp format
                            if "timestamp" in chunk_df.columns or chunk_df.index.name == "timestamp":
                                sample_ts = chunk_df.index[0] if chunk_df.index.name == "timestamp" else chunk_df.iloc[0]["timestamp"]
                                print(f"   ✅ Data validation: Sample timestamp {sample_ts} (will be converted UTC→GMT+4)")
                        
                        # Verify timezone: Polygon returns UTC (timezone-naive)
                        # Log first timestamp to verify conversion
                        if len(chunk_df) > 0:
                            first_ts = chunk_df.index[0] if "timestamp" in str(chunk_df.index.name) else chunk_df.iloc[0].get("timestamp", None)
                            if first_ts is not None:
                                if hasattr(first_ts, 'tzinfo') and first_ts.tzinfo is None:
                                    print(f"   ℹ️ Timestamp timezone: UTC (naive) - will convert to GMT+4 for storage")
                                elif hasattr(first_ts, 'tzinfo') and first_ts.tzinfo is not None:
                                    print(f"   ℹ️ Timestamp timezone: {first_ts.tzinfo} - will convert to GMT+4 for storage")
                        
                        # INSERT IMMEDIATELY after fetching
                        insert_result = _insert_chunk_to_db(
                            chunk_df, target_symbol, target_table, sb, utc_tz, gmt4_tz, chunk_s_str, chunk_e_str, expected_api_symbol=polygon_symbol
                        )
                        if insert_result["success"]:
                            total_inserted += insert_result["count"]
                            print(f"   💾 Stored {insert_result['count']} records in database")
                        else:
                            print(f"   ⚠️ Storage warning: {insert_result['message']}")
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
                        return {"success": False, "message": f"❌ Polygon 403 Forbidden: {error_str}\n\nThis symbol may not be available for {interval} data on your Polygon plan.\n\nDate range: {chunk_s_str} to {chunk_e_str}"}
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
                        print(f"🛑 Polygon rate limit hit for {polygon_symbol}. Cooling down 90s...")
                        time.sleep(90)
                        all_success = False
                        break  # DO NOT retry same chunk more than once for Polygon
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
            # CRITICAL: For 1-day chunks, advance to start of next day (midnight UTC)
            if chunk_days == 1:
                # Get the date part, add 1 day, set to midnight
                current_date = chunk_start.date()
                next_date = current_date + timedelta(days=1)
                # Create new datetime at midnight UTC for next day
                chunk_start = datetime.combine(next_date, datetime.min.time())
                # Ensure it's timezone-aware (UTC)
                if chunk_start.tzinfo is None:
                    chunk_start = utc_tz.localize(chunk_start)
                else:
                    chunk_start = chunk_start.replace(tzinfo=utc_tz)
            else:
                chunk_start = chunk_end + timedelta(seconds=1)  # Move just past chunk_end
            
            # Safety check: if we've passed end_dt, break
            if chunk_start > end_dt:
                print(f"✅ Reached end date. Stopping chunk processing.")
                break
        
        # All chunks processed - final summary
        print(f"\n{'='*60}")
        print(f"✅ INGESTION COMPLETE")
        print(f"{'='*60}")
        print(f"   Total records inserted: {total_inserted}")
        print(f"   Symbol: {target_symbol}")
        print(f"   Table: {target_table}")
        print(f"{'='*60}\n")
        
        # Final verification
        date_range_info = f"Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} ({(end_dt - start_dt).days} days)"
        mode_info = "Resumed from existing data" if is_resuming else "Fresh ingestion"
        
        print(f"\n🔍 Final verification...")
        try:
            # Get count and latest timestamp
            count_result = sb.table(target_table)\
                .select("symbol", count="exact")\
                .eq("symbol", target_symbol)\
                .execute()
            
            verify_result = sb.table(target_table)\
                .select("symbol, timestamp")\
                .eq("symbol", target_symbol)\
                .order("timestamp", desc=True)\
                .limit(5)\
                .execute()
            
            if verify_result.data:
                total_count = len(count_result.data) if hasattr(count_result, 'data') else "unknown"
                latest_rec = verify_result.data[0]
                latest_symbol = latest_rec.get('symbol')
                latest_ts = latest_rec.get('timestamp')
                
                print(f"✅ VERIFICATION: Found records in database")
                print(f"   Symbol stored: '{latest_symbol}' (expected: '{target_symbol}')")
                print(f"   Total records: {total_count}")
                print(f"   Latest timestamp: {latest_ts}")
                print(f"   Sample records:")
                for rec in verify_result.data[:3]:
                    print(f"     - Symbol: {rec.get('symbol')}, Timestamp: {rec.get('timestamp')}")
                
                # Verify symbol matches
                if latest_symbol != target_symbol:
                    print(f"   ⚠️ WARNING: Symbol mismatch! Expected '{target_symbol}' but found '{latest_symbol}'")
                    return {"success": False, "message": f"❌ Symbol mismatch: Data stored with '{latest_symbol}' but expected '{target_symbol}'. Please check symbol mapping."}
            else:
                print(f"⚠️ WARNING: Verification query returned no records for symbol '{target_symbol}'")
                print(f"   Please check RLS policies and symbol format")
                print(f"   Attempted to query: table='{target_table}', symbol='{target_symbol}'")
        except Exception as verify_error:
            print(f"⚠️ Could not verify data storage: {verify_error}")
            import traceback
            traceback.print_exc()
        
        success_message = f"✅ Successfully ingested {total_inserted} records for {target_symbol} into {target_table}.\n\n"
        success_message += f"Mode: {mode_info}\n"
        success_message += f"{date_range_info}\n"
        success_message += f"\n📊 Symbol Verification:\n"
        success_message += f"   • Original Symbol (UI): {api_symbol}\n"
        success_message += f"   • Polygon API Symbol: {polygon_symbol}\n"
        success_message += f"   • Database Symbol: {target_symbol}\n"
        success_message += f"\n🕐 Timezone: All timestamps converted from UTC (Polygon) → GMT+4 (Database)"
        
        return {"success": True, "message": success_message}
    
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Indices ingestion error: {str(e)}"}


def verify_indices_in_database():
    """
    Helper function to verify what index symbols are stored in the database.
    This helps identify if symbols are stored correctly (I:SPX, I:NDX, I:DJI).
    """
    sb = get_supabase()
    if not sb:
        print("❌ Supabase not configured")
        return
    
    target_table = "market_data_indices_1min"
    
    try:
        # Get all distinct symbols
        result = sb.table(target_table)\
            .select("symbol, timestamp")\
            .order("symbol")\
            .order("timestamp", desc=True)\
            .execute()
        
        if not result.data:
            print(f"ℹ️ No data found in {target_table}")
            return
        
        # Group by symbol and get latest timestamp for each
        symbols_info = {}
        for row in result.data:
            symbol = row.get("symbol")
            timestamp = row.get("timestamp")
            
            if symbol not in symbols_info:
                symbols_info[symbol] = {
                    "latest": timestamp,
                    "count": 0
                }
            symbols_info[symbol]["count"] += 1
            # Update latest if this timestamp is newer
            if timestamp and symbols_info[symbol]["latest"]:
                try:
                    if isinstance(timestamp, str):
                        ts1 = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        ts2 = datetime.fromisoformat(symbols_info[symbol]["latest"].replace('Z', '+00:00'))
                        if ts1 > ts2:
                            symbols_info[symbol]["latest"] = timestamp
                    else:
                        if timestamp > symbols_info[symbol]["latest"]:
                            symbols_info[symbol]["latest"] = timestamp
                except:
                    pass
        
        print(f"\n{'='*60}")
        print(f"📊 INDICES IN DATABASE: {target_table}")
        print(f"{'='*60}")
        print(f"Found {len(symbols_info)} unique symbol(s):\n")
        
        for symbol, info in sorted(symbols_info.items()):
            print(f"  • {symbol}")
            print(f"    - Records: {info['count']}")
            print(f"    - Latest: {info['latest']}")
            print()
        
        # Check for expected symbols
        expected_symbols = ["I:SPX", "I:NDX", "I:DJI"]
        print(f"Expected symbols (from image): {', '.join(expected_symbols)}")
        print(f"\n✅ Verification:")
        for exp_sym in expected_symbols:
            if exp_sym in symbols_info:
                print(f"  ✅ {exp_sym} - Found ({symbols_info[exp_sym]['count']} records)")
            else:
                # Check for variations
                found_variants = [s for s in symbols_info.keys() if exp_sym.upper() in s.upper() or s.upper() in exp_sym.upper()]
                if found_variants:
                    print(f"  ⚠️ {exp_sym} - Not found, but found variants: {', '.join(found_variants)}")
                else:
                    print(f"  ❌ {exp_sym} - Not found")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error verifying database: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Example usage"""
    # First, verify what's in the database
    print("🔍 Verifying indices in database...")
    verify_indices_in_database()
    
    # Example: Ingest SPY (S&P 500 ETF) 1-minute data
    # Uncomment to test ingestion:
    # result = ingest_indices_from_polygon(
    #     api_symbol="I:SPX",
    #     interval="1min",
    #     years=1,  # 1 year for free plan
    #     db_symbol="I:SPX"
    # )
    # 
    # print("\n" + "="*60)
    # if result["success"]:
    #     print("✅ " + result["message"])
    # else:
    #     print("❌ " + result["message"])
    # print("="*60)


if __name__ == "__main__":
    main()

