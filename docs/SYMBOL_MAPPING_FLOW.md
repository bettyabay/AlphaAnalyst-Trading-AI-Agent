# Symbol Mapping Flow: UI → Polygon API

This document explains how indices symbols are mapped from the UI selection to Polygon's exact API symbol.

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. UI SELECTION                                                 │
│    User selects: "S&P 500" from dropdown                      │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. INITIAL CONVERSION (app.py)                                  │
│    convert_instrument_to_polygon_symbol("Indices", "S&P 500")   │
│    → Returns: "I:SPX"                                           │
│                                                                 │
│    Logic in tradingagents/dataflows/ingestion_pipeline.py:    │
│    - Detects "S&P" or "SPX" or "SP500" in instrument name     │
│    - Returns "I:SPX" (Polygon format for S&P 500 index)        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. UI TEXT INPUT (app.py line 1307)                            │
│    default_api_symbol = "I:SPX"                                │
│    User can edit this field if needed                           │
│    User clicks "Fetch & Ingest from API"                       │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. SYMBOL CLEANING (app.py line 1314)                          │
│    api_symbol_cleaned = "I:SPX" (whitespace removed)          │
│    effective_db_symbol = "I:SPX" (for database storage)        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. INGESTION FUNCTION CALL (app.py line 1349)                  │
│    ingest_indices_from_polygon(                                 │
│        api_symbol="I:SPX",      # Original from UI             │
│        interval="1min",                                         │
│        db_symbol="I:SPX"        # For database storage          │
│    )                                                             │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. ETF CONVERSION (ingest_indices_polygon.py line 194)         │
│    _get_etf_for_index("I:SPX", "1min")                         │
│                                                                 │
│    Mapping Logic:                                               │
│    1. Check INDEX_TO_ETF_MAPPING dictionary                    │
│    2. "I:SPX" → Found → Returns "SPY"                         │
│                                                                 │
│    INDEX_TO_ETF_MAPPING = {                                    │
│        "I:SPX": "SPY",    # S&P 500 → SPY ETF                  │
│        "I:NDX": "QQQ",    # NASDAQ-100 → QQQ ETF               │
│        "I:DJI": "DIA",    # Dow Jones → DIA ETF                 │
│        "I:RUT": "IWM",    # Russell 2000 → IWM ETF             │
│        ...                                                     │
│    }                                                            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. POLYGON API CALL                                             │
│    polygon_symbol = "SPY"                                      │
│    client.get_intraday_data("SPY", ...)                        │
│                                                                 │
│    Why SPY? Polygon doesn't provide 1-minute data for          │
│    cash indices (I:SPX). ETFs like SPY track the index         │
│    and provide minute-level data.                              │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. DATA STORAGE                                                 │
│    target_symbol = "I:SPX" (original from UI)                  │
│    Data stored with symbol "I:SPX" in database                 │
│                                                                 │
│    Important: Even though we fetch SPY data from Polygon,     │
│    we store it with the original index symbol "I:SPX" so       │
│    users can query by the index name they selected.             │
└─────────────────────────────────────────────────────────────────┘
```

## Key Mapping Dictionary

The `INDEX_TO_ETF_MAPPING` dictionary in `ingest_indices_polygon.py` (lines 157-191) contains all the mappings:

```python
INDEX_TO_ETF_MAPPING = {
    # === US MAJOR INDICES ===
    
    # S&P 500 variations → SPY
    "I:SPX": "SPY",
    "SPX": "SPY",
    "^SPX": "SPY",
    "SP500": "SPY",
    "S&P 500": "SPY",
    "S&P500": "SPY",
    
    # NASDAQ-100 variations → QQQ
    "I:NDX": "QQQ",
    "NDX": "QQQ",
    "^NDX": "QQQ",
    "NASDAQ-100": "QQQ",
    "NAS 100": "QQQ",
    "NAS100": "QQQ",
    "I:NAS 100": "QQQ",
    "I:NAS100": "QQQ",
    
    # NASDAQ Composite → ONEQ
    "I:IXIC": "ONEQ",
    "IXIC": "ONEQ",
    "^IXIC": "ONEQ",
    "NASDAQ": "ONEQ",
    "NASDAQ COMPOSITE": "ONEQ",
    
    # NYSE Composite → VTI
    "I:NYA": "VTI",
    "NYA": "VTI",
    "^NYA": "VTI",
    "NYSE": "VTI",
    "NYSE COMPOSITE": "VTI",
    "NYSECOMPOSITE": "VTI",
    
    # S&P 100 → OEF
    "I:OEX": "OEF",
    "OEX": "OEF",
    "^OEX": "OEF",
    "S&P 100": "OEF",
    "S&P100": "OEF",
    "SP100": "OEF",
    
    # Dow Jones Industrial Average → DIA
    "I:DJI": "DIA",
    "DJI": "DIA",
    "^DJI": "DIA",
    "DOW": "DIA",
    "DOW JONES": "DIA",
    "DOW JONES INDUSTRIAL": "DIA",
    
    # Dow Jones Transportation Average → IYT
    "I:DJT": "IYT",
    "DJT": "IYT",
    "^DJT": "IYT",
    "DOW TRANSPORTATION": "IYT",
    "DOW TRANSPORT": "IYT",
    "DJ TRANSPORTATION": "IYT",
    "DJ TRANSPORT": "IYT",
    
    # Dow Jones Utility Average → XLU
    "I:DJU": "XLU",
    "DJU": "XLU",
    "^DJU": "XLU",
    "DOW UTILITY": "XLU",
    "DOW UTILITIES": "XLU",
    "DJ UTILITY": "XLU",
    "DJ UTILITIES": "XLU",
    
    # Russell 2000 variations → IWM
    "I:RUT": "IWM",
    "RUT": "IWM",
    "^RUT": "IWM",
    "RUSSELL 2000": "IWM",
    "RUSSELL2000": "IWM",
    
    # === US SECTOR/CAP INDICES ===
    
    # S&P 400 MidCap → MDY
    "I:MID": "MDY",
    "MID": "MDY",
    "^MID": "MDY",
    "S&P 400": "MDY",
    "S&P400": "MDY",
    "SP400": "MDY",
    "MIDCAP": "MDY",
    "MID CAP": "MDY",
    
    # S&P 600 SmallCap → IJR
    "I:SML": "IJR",
    "SML": "IJR",
    "^SML": "IJR",
    "S&P 600": "IJR",
    "S&P600": "IJR",
    "SP600": "IJR",
    "SMALLCAP": "IJR",
    "SMALL CAP": "IJR",
    
    # Russell 1000 → ONEQ
    "I:RUI": "ONEQ",
    "RUI": "ONEQ",
    "^RUI": "ONEQ",
    "RUSSELL 1000": "ONEQ",
    "RUSSELL1000": "ONEQ",
    
    # Russell 3000 → VTI
    "I:RUA": "VTI",
    "RUA": "VTI",
    "^RUA": "VTI",
    "RUSSELL 3000": "VTI",
    "RUSSELL3000": "VTI",
    
    # Wilshire 5000 → VTI
    "I:W5000": "VTI",
    "W5000": "VTI",
    "^W5000": "VTI",
    "WILSHIRE 5000": "VTI",
    "WILSHIRE5000": "VTI",
    
    # === VOLATILITY ===
    
    # VIX variations → VIX (works directly)
    "I:VIX": "VIX",
    "VIX": "VIX",
    "^VIX": "VIX",
    
    # === INTERNATIONAL INDICES ===
    
    # FTSE 100 (UK) → EWU
    "I:UKX": "EWU",
    "UKX": "EWU",
    "^UKX": "EWU",
    "FTSE 100": "EWU",
    "FTSE100": "EWU",
    
    # Nikkei 225 (Japan) → EWJ
    "I:N225": "EWJ",
    "N225": "EWJ",
    "^N225": "EWJ",
    "NIKKEI": "EWJ",
    "NIKKEI 225": "EWJ",
    "NIKKEI225": "EWJ",
    
    # DAX (Germany) → EWG
    "I:GDAXI": "EWG",
    "GDAXI": "EWG",
    "^GDAXI": "EWG",
    "DAX": "EWG",
    
    # CAC 40 (France) → EWQ
    "I:FCHI": "EWQ",
    "FCHI": "EWQ",
    "^FCHI": "EWQ",
    "CAC 40": "EWQ",
    "CAC40": "EWQ",
    "CAC": "EWQ",
    
    # Hang Seng (Hong Kong) → EWH
    "I:HSI": "EWH",
    "HSI": "EWH",
    "^HSI": "EWH",
    "HANG SENG": "EWH",
    "HANGSENG": "EWH",
    
    # Shanghai Composite (China) → FXI
    "I:SSEC": "FXI",
    "SSEC": "FXI",
    "^SSEC": "FXI",
    "SHANGHAI": "FXI",
    "SHANGHAI COMPOSITE": "FXI",
    
    # TSX (Canada) → EWC
    "I:TSX": "EWC",
    "TSX": "EWC",
    "^TSX": "EWC",
    "TSX COMPOSITE": "EWC",
    "TSXCOMPOSITE": "EWC",
    
    # ASX 200 (Australia) → EWA
    "I:AXJO": "EWA",
    "AXJO": "EWA",
    "^AXJO": "EWA",
    "ASX 200": "EWA",
    "ASX200": "EWA",
    "ASX": "EWA",
    
    # MSCI EAFE (International Developed) → EFA
    "I:EAFE": "EFA",
    "EAFE": "EFA",
    "^EAFE": "EFA",
    "MSCI EAFE": "EFA",
    "MSCIEAFE": "EFA",
    
    # MSCI Emerging Markets → EEM
    "I:EEM": "EEM",
    "EEM": "EEM",
    "^EEM": "EEM",
    "MSCI EM": "EEM",
    "MSCIEM": "EEM",
    "EMERGING MARKETS": "EEM",
}
```

## Mapping Function Logic

The `_get_etf_for_index()` function (lines 194-269) uses a multi-step matching process:

1. **Exact Match**: Check if the symbol (uppercase) exists in the dictionary
2. **Normalized Match**: Remove spaces, dashes, underscores and check again
3. **Prefix Handling**: 
   - If starts with "I:", extract base symbol (e.g., "I:SPX" → "SPX")
   - If starts with "^", extract base symbol (e.g., "^SPX" → "SPX")
4. **Special Cases**: Handle variations like "NAS 100" → "QQQ"
5. **Fallback**: Return original symbol if no mapping found

## Why This Mapping is Necessary

**Problem**: Polygon API doesn't provide 1-minute data for cash indices (I:SPX, I:NDX, etc.) on free/standard plans.

**Solution**: Use ETFs that track these indices:
- SPY tracks S&P 500 (I:SPX)
- QQQ tracks NASDAQ-100 (I:NDX)
- DIA tracks Dow Jones (I:DJI)
- IWM tracks Russell 2000 (I:RUT)

**Result**: 
- Fetch data using ETF symbol (SPY, QQQ, etc.)
- Store data with original index symbol (I:SPX, I:NDX, etc.)
- Users query by index name, not ETF name

## Example: Complete Flow for "S&P 500"

```
UI Selection:        "S&P 500"
                     ↓
Initial Conversion:  "I:SPX" (via convert_instrument_to_polygon_symbol)
                     ↓
User Input:          "I:SPX" (can be edited)
                     ↓
Cleaning:            "I:SPX" (whitespace removed)
                     ↓
Ingestion Call:      api_symbol="I:SPX", db_symbol="I:SPX"
                     ↓
ETF Conversion:      "I:SPX" → "SPY" (via INDEX_TO_ETF_MAPPING)
                     ↓
Polygon API:         Fetches SPY data
                     ↓
Database Storage:    Stores with symbol "I:SPX"
```

## Verification Points

The system verifies the mapping at multiple points:

1. **At Start**: Logs original symbol, Polygon symbol, and DB symbol
2. **After Fetch**: Validates data quality
3. **Before Storage**: Confirms symbol to store
4. **After Storage**: Verifies stored symbol matches expected

## Adding New Index Mappings

To add a new index mapping:

1. Add entry to `INDEX_TO_ETF_MAPPING` dictionary
2. Add conversion logic in `convert_instrument_to_polygon_symbol()` if needed
3. Test the mapping with the ingestion function

Example:
```python
# In INDEX_TO_ETF_MAPPING
"I:VIX": "VIX",  # VIX works directly, or use "VXX" for futures

# In convert_instrument_to_polygon_symbol()
elif "VIX" in base_symbol:
    return "I:VIX"
```

