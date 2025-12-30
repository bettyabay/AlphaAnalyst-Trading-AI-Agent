import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

BARCHART_API_KEY = os.getenv("BARCHART_API_KEY")
BARCHART_URL = "https://ondemand.websol.barchart.com/getHistory.json"

def fetch_barchart_data(
    symbol: str,
    start_date: str,  # YYYYMMDD
    end_date: str,    # YYYYMMDD
    interval: int = 1,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical data from BarChart OnDemand API in chunks to ensure completeness.
    """
    key = api_key or BARCHART_API_KEY
    if not key:
        raise ValueError("BarChart API Key is required. Set BARCHART_API_KEY env var or pass explicitly.")

    # Convert strings to datetime
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    all_data = []
    current_start = start_dt
    
    # Chunk size: 5 days to be safe with 1-minute data limits (approx 7200 records per chunk)
    # Barchart limits vary by plan, but smaller chunks are safer.
    CHUNK_DAYS = 5
    
    print(f"Starting collection for {symbol} from {start_date} to {end_date}...")
    
    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS), end_dt)
        
        # Format for API
        s_str = current_start.strftime("%Y%m%d")
        e_str = current_end.strftime("%Y%m%d")
        
        params = {
            "apikey": key,
            "symbol": symbol,
            "type": "minutes",
            "interval": interval,
            "startDate": s_str,
            "endDate": e_str,
            "order": "asc",
            "volume": "sum",
            "nearby": 1,
            "jerq": "true"
        }
        
        try:
            response = requests.get(BARCHART_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "status" in data and data["status"].get("code") != 200:
                print(f"Error fetching {s_str}-{e_str}: {data['status'].get('message')}")
            elif "results" in data and data["results"]:
                chunk_df = pd.DataFrame(data["results"])
                all_data.append(chunk_df)
                print(f"Fetched {len(chunk_df)} records for {s_str}-{e_str}")
            else:
                print(f"No data for {s_str}-{e_str}")
                
        except Exception as e:
            print(f"Exception fetching {s_str}-{e_str}: {e}")
            
        # Move to next chunk
        current_start = current_end + timedelta(days=1) # Avoid overlap? getHistory is inclusive.
        # Actually, if we do current_end + 1 day, we might miss the rest of current_end if current_end was partial?
        # Better: startDate and endDate are inclusive days.
        # So next chunk should start at current_end + 1 day.
        
        # Rate limit pause
        time.sleep(0.5)

    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    
    # Clean up and format
    if not final_df.empty:
        # Columns usually: symbol, timestamp, tradingDay, open, high, low, close, volume, openInterest
        # Ensure timestamp is datetime
        if "timestamp" in final_df.columns:
            final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])
            final_df = final_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
            final_df = final_df.set_index("timestamp")
    
    return final_df

def ensure_continuous_timestamps(df: pd.DataFrame, interval_minutes: int = 1) -> pd.DataFrame:
    """
    Reindex DataFrame to ensure continuous 1-minute timestamps.
    Fills missing periods with NaNs or forward fills (optional).
    For 'Ensure timestamps are continuous', we usually just want the index to be complete.
    """
    if df.empty:
        return df
        
    start_time = df.index.min()
    end_time = df.index.max()
    
    # Create complete range
    full_range = pd.date_range(start=start_time, end=end_time, freq=f"{interval_minutes}min")
    
    # Reindex
    df_continuous = df.reindex(full_range)
    
    # Optional: Fill logic. 
    # Usually for OHLC, we might forward fill Close, but Open/High/Low might be NaN or same as Close.
    # Or just leave as NaN to indicate no trading.
    # The prompt says "Ensure timestamps are continuous and complete".
    # I will leave NaNs where there is no data, so the user knows it's missing (e.g. weekends/holidays).
    # But usually "continuous" means the time series has no gaps in the index.
    
    return df_continuous

if __name__ == "__main__":
    # Test execution
    # This block allows running this file directly to perform Step 1.1
    
    # Configuration
    SYMBOL = "GC*1" # Nearby Gold Futures
    INTERVAL = 1
    YEARS = 5
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * YEARS)
    
    s_str = start_date.strftime("%Y%m%d")
    e_str = end_date.strftime("%Y%m%d")
    
    print(f"--- STEP 1.1: GATHER GOLD MARKET DATA ---")
    print(f"Symbol: {SYMBOL}")
    print(f"Period: {s_str} to {e_str} ({YEARS} years)")
    
    if not BARCHART_API_KEY:
        print("ERROR: BARCHART_API_KEY not found in environment.")
        print("Please set it in your .env file.")
    else:
        df = fetch_barchart_data(SYMBOL, s_str, e_str, INTERVAL)
        
        if not df.empty:
            print(f"Total records fetched: {len(df)}")
            
            # Ensure continuity
            df_cont = ensure_continuous_timestamps(df, INTERVAL)
            print(f"Total records after continuity check: {len(df_cont)}")
            
            # Save
            os.makedirs("data", exist_ok=True)
            filename = f"data/gold_1min_{s_str}_{e_str}.csv"
            df_cont.to_csv(filename)
            print(f"Data saved to {filename}")
        else:
            print("No data fetched.")
