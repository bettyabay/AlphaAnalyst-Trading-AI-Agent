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
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.last_request_time = 0
        self.min_request_interval = 0.25  # Minimum 250ms between requests (4 requests/sec)
        
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
        
        Handles rate limiting with exponential backoff.
        """
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {"apikey": self.api_key, "adjusted": "true", "sort": "asc"}

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
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                    if attempt < max_retries - 1:
                        print(f"Rate limited (429) for {symbol} {start_date} to {end_date}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded for {symbol} {start_date} to {end_date} after {max_retries} attempts")
                        raise requests.exceptions.HTTPError(f"429 Client Error: Too Many Requests")
                
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
                    # Check if this is a future date or truly no data
                    try:
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                        now = datetime.now()
                        
                        if start_dt > now or end_dt > now:
                            # Future dates - no data expected, don't print
                            return pd.DataFrame()
                        else:
                            # Past dates but no data - might be weekend/holiday or data issue
                            status_msg = data.get("status", "")
                            if status_msg != "OK":
                                print(f"No intraday data for {symbol} {start_date} to {end_date}: {status_msg}")
                            return pd.DataFrame()
                    except:
                        # Can't parse dates, just return empty
                        return pd.DataFrame()

            except requests.exceptions.HTTPError as e:
                # Check if response exists and if it's a 429
                response_status = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
                if response_status == 429:
                    # Already handled above, but catch here too
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"Rate limited (429) for {symbol} {start_date} to {end_date}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded for {symbol} {start_date} to {end_date} after {max_retries} attempts")
                        raise
                else:
                    # Other HTTP errors
                    if attempt == max_retries - 1:
                        print(f"Error fetching intraday data for {symbol}: {e}")
                        return pd.DataFrame()
                    # Retry other HTTP errors
                    wait_time = 1 * (attempt + 1)
                    time.sleep(wait_time)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error fetching intraday data for {symbol}: {e}")
                    return pd.DataFrame()
                # For other errors, wait a bit and retry
                wait_time = 1 * (attempt + 1)
                time.sleep(wait_time)
        
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

