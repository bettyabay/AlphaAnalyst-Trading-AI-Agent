"""
Watchlist configuration for AlphaAnalyst Trading AI Agent
Phase 1: 19 High-Beta US Stocks
"""
from typing import Dict, List

# 19 High-Beta US Stocks for Phase 1
WATCHLIST_STOCKS = {
    "COIN": "Coinbase Global, Inc.",
    "TSLA": "Tesla, Inc.",
    "NVDA": "NVIDIA Corporation",
    "AVGO": "Broadcom Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "LRCX": "Lam Research Corporation",
    "CRWD": "CrowdStrike Holdings, Inc.",
    "HOOD": "Robinhood Markets, Inc.",
    "APP": "AppLovin Corporation",
    "MU": "Micron Technology, Inc.",
    "SNPS": "Synopsys, Inc.",
    "CDNS": "Cadence Design Systems, Inc.",
    "VST": "Vistra Corp.",
    "DASH": "DoorDash, Inc.",
    "DELL": "Dell Technologies Inc.",
    "NOW": "ServiceNow, Inc.",
    "PANW": "Palo Alto Networks, Inc.",
    "AXON": "Axon Enterprise, Inc.",
    "URI": "United Rentals, Inc."
}

# Market configuration
MARKET_CONFIG = {
    "market_focus": "US Stocks Only (MVP Phase)",
    "trading_session": "US Market Hours (9:30 AM - 4:00 PM EST)",
    "primary_trading_window": "First 2 Hours (9:30-11:30 AM EST)",
    "risk_management": "1% per trade = $100 max risk on $10,000 account"
}

# Tech stack integration
TECH_STACK = {
    "primary_analysis": "TradingView Premium (real-time data, scanners, news)",
    "api_data_source": "Polygon.io + Alpaca Markets (for historical & real-time data)",
    "execution_platform": "Alpaca Paper Trading or ThinkOrSwim PaperMoney",
    "news_sources": "Benzinga Pro, Bloomberg Markets, Real-time Twitter sentiment"
}

def get_watchlist_symbols() -> List[str]:
    """Get list of watchlist symbols"""
    return list(WATCHLIST_STOCKS.keys())

def get_stock_info(symbol: str) -> Dict[str, str]:
    """Get stock information for a symbol"""
    return {
        "symbol": symbol,
        "name": WATCHLIST_STOCKS.get(symbol, "Unknown"),
        "sector": "Technology"  # Most are tech stocks, can be expanded later
    }

def get_all_stock_info() -> List[Dict[str, str]]:
    """Get information for all watchlist stocks"""
    return [get_stock_info(symbol) for symbol in WATCHLIST_STOCKS.keys()]
