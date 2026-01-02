import pandas as pd
from tradingagents.dataflows.universal_ingestion import ingest_from_barchart_api

def ingest_gold_from_api(symbol="GC*1", start_date=None, end_date=None, years=5):
    """
    Ingest Gold data directly from BarChart API.
    Wrapper around universal_ingestion.ingest_from_barchart_api for backward compatibility.
    
    Args:
        symbol: BarChart symbol (default GC*1 for nearby Gold Futures)
        start_date: Optional start date (datetime or YYYYMMDD string). If None, calculated from years.
        end_date: Optional end date (datetime or YYYYMMDD string). If None, defaults to now.
        years: Number of years to fetch if start_date is not provided.
        
    Returns:
        dict: {"success": bool, "message": str}
    """
    return ingest_from_barchart_api(
        api_symbol=symbol,
        asset_class="Commodities",
        start_date=start_date,
        end_date=end_date,
        years=years,
        db_symbol=symbol # Use the same symbol for DB
    )

def ingest_gold_data(file_path):
    """
    Legacy function for file-based ingestion.
    Now deprecated in favor of universal_ingestion.ingest_market_data.
    """
    return {"success": False, "message": "Deprecated. Use universal_ingestion.ingest_market_data instead."}
