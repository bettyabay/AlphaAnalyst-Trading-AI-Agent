"""
Polygon.io data integration for AlphaAnalyst Trading AI Agent
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import time

load_dotenv()

class PolygonDataClient:
    """Polygon.io API client for market data"""
    
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        self.base_url = "https://api.polygon.io"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.last_request_time = 0
        # Polygon free plan: 5 calls/min = 12 seconds between calls
        # Use 12 seconds as safe default to avoid 429 errors
        self.min_request_interval = 12.0  # 12 seconds = 5 calls/min (safe for free plan)
        
        # Warn if API key is missing
        if not self.api_key:
            print("⚠️ WARNING: POLYGON_API_KEY not found in environment variables. API calls will fail.")
            print("   Get a key from: https://polygon.io/dashboard/api-keys")
        
    def get_stock_details(self, symbol: str) -> Dict:
        """Get stock details and company information"""
        url = f"{self.base_url}/v3/reference/tickers/{symbol}"
        params = {"apikey": self.api_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching stock details for {symbol}: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, max_retries: int = 5) -> pd.DataFrame:
        """Get historical price data for a symbol with rate limiting"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {"apikey": self.api_key, "adjusted": "true", "sort": "asc"}
        
        for attempt in range(max_retries):
            try:
                # Rate limiting: ensure minimum time between requests
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_request_interval:
                    time.sleep(self.min_request_interval - elapsed)
                
                response = requests.get(url, params=params)
                self.last_request_time = time.time()
                
                # Handle rate limiting (429 status code)
                if response.status_code == 429:
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                    if attempt < max_retries - 1:
                        print(f"Rate limited (429) for {symbol}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded for {symbol} after {max_retries} attempts")
                        return pd.DataFrame()  # Return empty DataFrame instead of raising
                
                response.raise_for_status()
                data = response.json()
                
                if data.get("results"):
                    df = pd.DataFrame(data["results"])
                    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                    df = df.rename(columns={
                        "o": "open", "h": "high", "l": "low", 
                        "c": "close", "v": "volume"
                    })
                    return df[["timestamp", "open", "high", "low", "close", "volume"]]
                else:
                    print(f"No data returned for {symbol}. Response: {data}")
                    return pd.DataFrame()
                    
            except requests.exceptions.HTTPError as e:
                response_status = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
                
                # Handle 401 Unauthorized (invalid API key)
                if response_status == 401:
                    error_msg = (
                        f"❌ POLYGON_API_KEY authentication failed (401 Unauthorized). "
                        f"Please check your API key in the .env file.\n"
                        f"Get a key from: https://polygon.io/dashboard/api-keys"
                    )
                    print(error_msg)
                    # Don't retry 401 errors - they won't succeed
                    raise ValueError(error_msg) from e
                
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 60)
                    print(f"HTTP 429 error for {symbol}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error fetching historical data for {symbol}: {e}")
                    return pd.DataFrame()
            except Exception as e:
                print(f"Error fetching historical data for {symbol}: {e}")
                return pd.DataFrame()
        
        return pd.DataFrame()  # Return empty if all retries failed

    def get_intraday_data(self, symbol: str, start_date: str, end_date: str, multiplier: int = 5, timespan: str = "minute", max_retries: int = 5) -> pd.DataFrame:
        """Get intraday (e.g. 5-min) aggregated price data for a symbol
        
        Uses the Polygon aggregated range endpoint with a multiplier (e.g. 5) and timespan (e.g. 'minute').
        Returns a DataFrame with columns: timestamp, open, high, low, close, volume
        
        Handles rate limiting with exponential backoff and pagination (next_url).
        """
        # Sanitize timespan for Polygon API
        # Polygon expects 'minute', 'hour', 'day', etc.
        # Common misconfigurations like '1min', '1-min' should be mapped to 'minute'
        if timespan in ('1min', '1-min', '5min', '5-min', '15min', '15-min'):
             timespan = 'minute'

        # Base request URL
        base_request_url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        # Initial params - limit 50000 is the max for Polygon (default 5000)
        params = {"apikey": self.api_key, "adjusted": "true", "sort": "asc", "limit": 50000}

        # Check if dates are in the future
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            now = datetime.now()
            
            if start_dt > now or end_dt > now:
                # Dates are in the future, skip silently (no data expected)
                return pd.DataFrame()
        except:
            pass  # If date parsing fails, continue with API call

        all_results = []
        next_url = base_request_url
        
        while next_url:
            current_url = next_url
            # If we are using next_url, params are embedded in the URL, but we might need to re-add apikey 
            # if it's not preserved (Polygon usually includes it in next_url but safer to be explicit if needed, 
            # though usually next_url works as is. However, we should be careful not to duplicate params).
            # Polygon documentation says next_url is complete. Let's try using it directly, but ensure apikey is present.
            # Actually, standard practice is to just add apikey if missing. 
            # But let's stick to using params dict for the base request and simple appending for next_url if needed.
            
            # For the base request, use 'params'. For next_url, it's a full URL.
            current_params = params if current_url == base_request_url else {"apikey": self.api_key}

            success = False
            for attempt in range(max_retries):
                try:
                    # Rate limiting: ensure minimum time between requests
                    elapsed = time.time() - self.last_request_time
                    if elapsed < self.min_request_interval:
                        time.sleep(self.min_request_interval - elapsed)
                    
                    response = requests.get(current_url, params=current_params)
                    self.last_request_time = time.time()
                    
                    # Handle rate limiting (429 status code)
                    if response.status_code == 429:
                        wait_time = 2 ** attempt  # Exponential backoff
                        if attempt < max_retries - 1:
                            print(f"Rate limited (429) for {symbol}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"Rate limit exceeded for {symbol} after {max_retries} attempts")
                            raise requests.exceptions.HTTPError(f"429 Client Error: Too Many Requests")
                    
                    response.raise_for_status()
                    data = response.json()

                    if data.get("results"):
                        all_results.extend(data["results"])
                    
                    # Check for pagination
                    next_url = data.get("next_url")
                    success = True
                    break

                except requests.exceptions.HTTPError as e:
                    # Handle 401 Unauthorized
                    response_status = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
                    if response_status == 401:
                         # ... (existing 401 logic) ... 
                         raise
                    
                    # Handle 403 Forbidden (Polygon restriction - e.g., indices minute data)
                    if response_status == 403:
                        error_msg = f"403 Forbidden: Polygon does not allow access to this data. Symbol: {symbol}"
                        if symbol.startswith("I:"):
                            error_msg += f"\n⚠️ Polygon does NOT provide 1-minute data for indices (I:SPX, I:DJI, etc.) due to licensing restrictions.\n💡 Use SPY instead of I:SPX for S&P 500 minute data."
                        print(error_msg)
                        raise ValueError(error_msg) from e
                    
                    if response_status == 429:
                         # Handled above
                         pass
                    else:
                        if attempt == max_retries - 1:
                            print(f"Error fetching intraday data: {e}")
                        time.sleep(1 * (attempt + 1))
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Error fetching intraday data: {e}")
                    time.sleep(1 * (attempt + 1))
            
            if not success:
                # If we failed to fetch a page after all retries, stop pagination to avoid infinite loops or partial data gaps
                print(f"Failed to fetch page for {symbol}. Stopping pagination.")
                break

        if all_results:
            df = pd.DataFrame(all_results)
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(columns={
                "o": "open", "h": "high", "l": "low", 
                "c": "close", "v": "volume"
            })
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        else:
             # Logic for checking if it was a valid empty result (future date) or error
             # Reusing existing check logic briefly
             try:
                 start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                 now = datetime.now()
                 if start_dt <= now:
                      # Only print if we expected data (past date) and got none
                      # And only if it wasn't a handled 429
                      pass
             except:
                 pass
             return pd.DataFrame()
    
    def get_real_time_price(self, symbol: str) -> Dict:
        """Get real-time price for a symbol"""
        url = f"{self.base_url}/v2/last/trade/{symbol}"
        params = {"apikey": self.api_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching real-time price for {symbol}: {e}")
            return {}
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        url = f"{self.base_url}/v1/marketstatus/now"
        params = {"apikey": self.api_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching market status: {e}")
            return {}
    
    def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get news for a specific symbol"""
        url = f"{self.base_url}/v2/reference/news"
        params = {
            "apikey": self.api_key,
            "ticker": symbol,
            "limit": limit,
            "order": "desc"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []
    
    def get_recent_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get recent price data for a symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            return self.get_historical_data(symbol, start_date_str, end_date_str)
        except Exception as e:
            print(f"Error fetching recent data for {symbol}: {e}")
            return pd.DataFrame()
    
    def batch_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            data = self.get_historical_data(symbol, start_date, end_date)
            if not data.empty:
                results[symbol] = data
            time.sleep(0.1)  # Rate limiting
            
        return results

