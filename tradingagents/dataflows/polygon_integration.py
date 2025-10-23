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
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data for a symbol"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {"apikey": self.api_key, "adjusted": "true", "sort": "asc"}
        
        try:
            response = requests.get(url, params=params)
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
                
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
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

