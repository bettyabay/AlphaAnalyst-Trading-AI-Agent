# End-to-End Code Flow Documentation

This document provides a comprehensive trace of the code logic from UI selection to database operations and response generation for three main features:

1. **Instrument Selection to Gold Ingestion**
2. **Signal Provider Flow**
3. **KPI Calculator Flow**

---

## 1. Instrument Selection to Gold Ingestion Flow

### 1.1 UI Layer - Selection (`app.py` lines 849-961)

#### Step 1: Category Selection
```python
# Location: app.py line 869-874
categories = ["Commodities", "Indices", "Currencies", "Stocks", "Add..."]
selected_category = st.selectbox(
    "Financial Instrument Category",
    options=categories,
    index=0,
    key="select_financial_category"
)
```
**Logic:**
- User selects category from dropdown
- Default is "Commodities" (index=0)
- Selection stored in Streamlit session state with key `select_financial_category`

#### Step 2: Instrument Selection
```python
# Location: app.py line 920-932
instruments_map = {
    "Commodities": ["GOLD", "Add..."],
    "Indices": ["S&P 500", "Add..."],
    "Currencies": ["EUR/USD", "Add..."],
    "Stocks": [*WATCHLIST_STOCKS.keys(), "Add..."] + (st.session_state.get("custom_instruments", []) or [])
}
current_instruments = instruments_map.get(selected_category, ["Add..."])
selected_instrument_item = st.selectbox(
    "Financial Instrument",
    options=current_instruments,
    index=0,
    key="select_financial_instrument_item"
)
```
**Logic:**
- Maps category to available instruments
- For "Commodities": Shows ["GOLD", "Add..."]
- User selection stored in `selected_instrument_item`

#### Step 3: Gold Upload UI Display
```python
# Location: app.py line 951-961
if selected_category == "Commodities" and selected_instrument_item == "GOLD":
    with st.expander("Gold Data (BarChart)"):
        st.info("Ingest 5-year 1-minute Gold data from BarChart export.")
        gold_file = st.file_uploader("Upload Gold 1-min Data (CSV/Excel)", 
                                     type=["csv", "xls", "xlsx"], 
                                     key="gold_upload_in_dropdown")
        if gold_file:
            if st.button("Ingest Gold Data", key="ingest_gold_btn_dropdown"):
                result = ingest_gold_data(gold_file)
                if result.get("success"):
                    st.success(result.get("message"))
                else:
                    st.error(f"Failed: {result.get('message')}")
```
**Logic:**
- Condition check: `selected_category == "Commodities" AND selected_instrument_item == "GOLD"`
- If true, shows expandable panel with file uploader
- Accepts CSV, XLS, XLSX files
- On button click, calls `ingest_gold_data(gold_file)`

### 1.2 Data Processing Layer - Gold Ingestion (`gold_ingestion.py`)

#### Step 4: File Reading and Validation
```python
# Location: gold_ingestion.py lines 18-57
def ingest_gold_data(uploaded_file):
    # Input validation
    if uploaded_file is None:
        return {"success": False, "message": "No file uploaded"}
    
    # File type detection
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        # CSV handling with Barchart comment skipping
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        # ... comment detection logic ...
        df = pd.read_csv(io.StringIO(csv_content))
    else:
        # Excel handling
        df = pd.read_excel(uploaded_file)
```
**Logic Flow:**
1. **Input Check**: Validates file exists
2. **File Type Detection**: Checks extension (.csv, .xls, .xlsx)
3. **CSV Special Handling**:
   - Reads file as text
   - Detects Barchart comment headers (lines containing "Downloaded from Barchart.com")
   - Finds actual header row (contains: timestamp, date, time, open, high, low, close, volume)
   - Skips comment rows before header
4. **Excel Handling**: Direct pandas read

#### Step 5: Column Standardization
```python
# Location: gold_ingestion.py lines 59-81
# Standardize columns
df.columns = [str(c).lower().strip() for c in df.columns]

# Rename common variations
rename_map = {
    "tradingday": "trading_day",
    "openinterest": "open_interest",
    "time": "timestamp",
    "date": "timestamp",
    "last": "close",
    "latest": "close",
    "price": "close",
    "close_price": "close"
}
df = df.rename(columns=rename_map)

# Default symbol assignment
if "symbol" not in df.columns:
    df["symbol"] = "^XAUUSD"
```
**Logic:**
- Converts all column names to lowercase
- Maps common column name variations to standard names
- Assigns default symbol "^XAUUSD" if missing

#### Step 6: Data Validation
```python
# Location: gold_ingestion.py lines 87-110
# Check required columns
required = ["timestamp", "open", "high", "low", "close", "volume"]
missing = [c for c in required if c not in df.columns]
if missing:
    return {"success": False, "message": f"Missing columns: {missing}"}

# Timestamp validation and cleaning
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', format='mixed')
df = df.dropna(subset=["timestamp"])  # Remove invalid timestamps
```
**Logic:**
- Validates all required columns exist
- Converts timestamp to datetime with error handling
- Removes rows with invalid timestamps
- Returns error if no valid data remains

#### Step 7: Data Transformation
```python
# Location: gold_ingestion.py lines 112-128
db_rows = []
for _, row in df.iterrows():
    try:
        record = {
            "symbol": str(row["symbol"]),
            "timestamp": row["timestamp"].isoformat(),  # Convert to ISO format
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]) if pd.notnull(row["volume"]) else 0,
            "open_interest": int(row["open_interest"]) if "open_interest" in df.columns and pd.notnull(row["open_interest"]) else None
        }
        db_rows.append(record)
    except (ValueError, TypeError) as e:
        return {"success": False, "message": f"Data type error: {str(e)}"}
```
**Logic:**
- Iterates through each row
- Converts data types (float for prices, int for volume)
- Formats timestamp as ISO string
- Handles missing values (volume defaults to 0, open_interest to None)
- Returns error on type conversion failure

#### Step 8: Database Connection
```python
# Location: gold_ingestion.py lines 130-133
supabase = get_supabase()
if not supabase:
    return {"success": False, "message": "Supabase not configured (check .env)"}
```
**Logic:**
- Calls `get_supabase()` from `tradingagents.database.config`
- Checks environment variables: `SUPABASE_URL` and `SUPABASE_KEY`
- Returns error if not configured

#### Step 9: Batch Database Insert
```python
# Location: gold_ingestion.py lines 135-155
chunk_size = 1000
total_inserted = 0

for i in range(0, len(db_rows), chunk_size):
    chunk = db_rows[i:i+chunk_size]
    try:
        result = supabase.table("market_data_commodities_1min").upsert(chunk).execute()
        total_inserted += len(chunk)
    except Exception as e:
        error_msg = str(e)
        # Error handling with specific messages
        if "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
            return {"success": False, "message": f"Table does not exist. Error: {error_msg}"}
        elif "column" in error_msg.lower() and "does not exist" in error_msg.lower():
            return {"success": False, "message": f"Column mismatch. Error: {error_msg}"}
        else:
            return {"success": False, "message": f"Database error at chunk {i}: {error_msg}"}

return {"success": True, "message": f"Successfully ingested {total_inserted} records into market_data_commodities_1min."}
```
**Logic:**
- Processes data in chunks of 1000 rows (Supabase batch limit)
- Uses `upsert()` to handle duplicates (based on unique constraint: symbol + timestamp)
- Tracks total inserted count
- Provides specific error messages for common issues:
  - Table doesn't exist
  - Column mismatch
  - Other database errors
- Returns success message with count

### 1.3 Response Generation
```python
# Location: app.py lines 957-961
result = ingest_gold_data(gold_file)
if result.get("success"):
    st.success(result.get("message"))  # Green success message
else:
    st.error(f"Failed: {result.get('message')}")  # Red error message
```
**Logic:**
- Checks `result["success"]` boolean
- Displays success message in green if True
- Displays error message in red if False
- Message content comes from `result["message"]`

---

## 2. Signal Provider Flow

### 2.1 UI Layer - Provider Selection (`app.py` lines 962-1127)

#### Step 1: Provider Dropdown
```python
# Location: app.py lines 963-970
signal_provider_options = ["PipXpert", "Add..."]
merged_providers = signal_provider_options + (st.session_state.get("signal_providers", []) or [])
selected_provider = st.selectbox(
    "Signal Provider",
    options=merged_providers,
    index=0,
    key="select_signal_provider"
)
```
**Logic:**
- Base options: ["PipXpert", "Add..."]
- Merges with session state custom providers
- User selection stored in `selected_provider`

#### Step 2: PipXpert Upload UI
```python
# Location: app.py lines 975-1048
if selected_provider == "PipXpert":
    with st.expander("PipXpert Signal Data"):
        provider_name = "PipXpert"
        symbol = st.text_input("Symbol/Currency Pair", key="pipxpert_symbol")
        timezone_offset = st.selectbox("Timezone", options=["+04:00", ...], index=0)
        signal_file = st.file_uploader("Upload Signal Data (Excel)", type=["xls", "xlsx"])
        
        if signal_file:
            df_preview = _read_df(signal_file)
            # Preview and validation
            validation = validate_signal_provider_data(df_preview, provider_name, symbol, timezone_offset)
            
            if validation["valid"]:
                st.success("✓ Data validation passed")
                # Show summary
                if st.button("✅ Ingest Signal Data"):
                    result = ingest_signal_provider_data(signal_file, provider_name, symbol, timezone_offset)
```
**Logic:**
- Shows expandable panel when PipXpert selected
- Collects: symbol, timezone (default GMT+4), file
- Previews file using `_read_df()` helper
- Validates before allowing submission
- Shows validation results and data summary
- Calls ingestion on button click

### 2.2 Validation Layer (`signal_provider_ingestion.py` lines 8-107)

#### Step 3: Data Validation
```python
# Location: signal_provider_ingestion.py lines 8-107
def validate_signal_provider_data(df, provider_name, symbol, timezone_offset):
    # Check required columns
    required_cols = ["Date", "Action", "Currency Pair"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {"valid": False, "message": f"Missing required columns: {missing_cols}"}
    
    # Validate provider name and symbol
    if not provider_name or not symbol:
        return {"valid": False, "message": "Provider name and symbol are required"}
    
    # Validate Date column
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates > 0:
        warnings.append(f"{invalid_dates} rows have invalid dates")
    
    # Validate Action column (Buy/Sell)
    valid_actions = ["buy", "sell", "Buy", "Sell", "BUY", "SELL"]
    invalid_actions = df[~df["Action"].isin(valid_actions)]
    if len(invalid_actions) > 0:
        warnings.append(f"{len(invalid_actions)} rows have invalid actions")
    
    # Timezone check
    if timezone_offset != "+04:00":
        warnings.append(f"Using timezone offset: {timezone_offset} (standard is GMT+4)")
    
    # Generate summary
    data_summary = {
        "total_rows": len(df),
        "date_range": {"start": ..., "end": ...},
        "action_counts": df["Action"].value_counts().to_dict(),
        "symbols": df["Currency Pair"].unique().tolist()
    }
    
    return {"valid": True, "message": "Data validation passed", "warnings": warnings, "data_summary": data_summary}
```
**Logic:**
- Validates required columns exist
- Checks provider name and symbol are provided
- Validates date format (converts to datetime, counts invalid)
- Validates action values (must be Buy/Sell variants)
- Checks timezone (warns if not GMT+4)
- Generates summary statistics
- Returns validation result with warnings and summary

### 2.3 Data Processing Layer (`signal_provider_ingestion.py` lines 110-295)

#### Step 4: File Reading
```python
# Location: signal_provider_ingestion.py lines 137-156
uploaded_file.seek(0)  # Reset file pointer
name = uploaded_file.name.lower()

if name.endswith('.xlsx') or name.endswith('.xls'):
    df = pd.read_excel(uploaded_file)
elif name.endswith('.csv'):
    df = pd.read_csv(uploaded_file)
else:
    return {"success": False, "message": "Unsupported file format"}
```
**Logic:**
- Resets file pointer (in case file was read for preview)
- Detects file type
- Reads Excel or CSV accordingly

#### Step 5: Column Mapping
```python
# Location: signal_provider_ingestion.py lines 161-183
# Standardize column names
df.columns = [str(c).strip() for c in df.columns]

# Map common variations
column_mapping = {
    "currency pair": "Currency Pair",
    "date": "Date",
    "action": "Action",
    "entry price": "Entry Price",
    # ... more mappings
}

for old_name, new_name in column_mapping.items():
    if old_name.lower() in [c.lower() for c in df.columns] and new_name not in df.columns:
        old_col = [c for c in df.columns if c.lower() == old_name.lower()][0]
        df = df.rename(columns={old_col: new_name})
```
**Logic:**
- Strips whitespace from column names
- Maps common column name variations to standard names
- Case-insensitive matching
- Only renames if target column doesn't already exist

#### Step 6: Data Validation (Re-run)
```python
# Location: signal_provider_ingestion.py lines 185-188
validation = validate_signal_provider_data(df, provider_name, symbol, timezone_offset)
if not validation["valid"]:
    return {"success": False, "message": validation["message"]}
```
**Logic:**
- Re-validates after column mapping
- Returns early if validation fails

#### Step 7: Data Transformation
```python
# Location: signal_provider_ingestion.py lines 190-259
# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df = df.dropna(subset=["Date"])

# Convert datetime columns
datetime_columns = ["SL Hit DateTime", "TP1 Hit DateTime", "TP2 Hit DateTime", "TP3 Hit DateTime"]
for dt_col in datetime_columns:
    if dt_col in df.columns:
        df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')

# Prepare records
db_rows = []
for _, row in df.iterrows():
    action = str(row.get("Action", "")).strip().lower()
    if action not in ["buy", "sell"]:
        continue  # Skip invalid actions
    
    currency_pair = str(symbol).upper().strip() if symbol else str(row.get("Currency Pair", "")).upper().strip()
    
    record = {
        "provider_name": provider_name.strip(),
        "symbol": currency_pair,
        "signal_date": row["Date"].isoformat(),
        "action": action,
        "entry_price": float(row["Entry Price"]) if pd.notnull(row.get("Entry Price")) else None,
        "target_1": float(row["Target 1"]) if pd.notnull(row.get("Target 1")) else None,
        "target_2": float(row["Target 2"]) if pd.notnull(row.get("Target 2")) else None,
        "target_3": float(row["Target 3"]) if pd.notnull(row.get("Target 3")) else None,
        "stop_loss": float(row["Stop Loss"]) if pd.notnull(row.get("Stop Loss")) else None,
        "timezone_offset": timezone_offset,
        "created_at": datetime.now().isoformat()
    }
    
    # Handle datetime hit columns
    for dt_col, field_name in datetime_fields.items():
        if dt_col in df.columns:
            dt_val = row.get(dt_col)
            if pd.notnull(dt_val) and isinstance(dt_val, pd.Timestamp):
                record[field_name] = dt_val.isoformat()
    
    db_rows.append(record)
```
**Logic:**
- Converts dates to datetime, removes invalid
- Normalizes action to lowercase ("buy"/"sell")
- Skips rows with invalid actions
- Uses provided symbol or extracts from Currency Pair column
- Converts prices to float, handles None values
- Formats all datetimes as ISO strings
- Builds record dictionary matching database schema

#### Step 8: Database Insert
```python
# Location: signal_provider_ingestion.py lines 264-292
supabase = get_supabase()
if not supabase:
    return {"success": False, "message": "Supabase not configured"}

chunk_size = 1000
total_inserted = 0

for i in range(0, len(db_rows), chunk_size):
    chunk = db_rows[i:i+chunk_size]
    try:
        result = supabase.table("signal_provider_signals").upsert(chunk).execute()
        total_inserted += len(chunk)
    except Exception as e:
        # Error handling similar to gold ingestion
        return {"success": False, "message": f"Database error: {error_msg}"}

return {"success": True, "message": f"Successfully ingested {total_inserted} records for provider '{provider_name}' into signal_provider_signals."}
```
**Logic:**
- Same chunking and error handling as gold ingestion
- Inserts into `signal_provider_signals` table
- Uses upsert to handle duplicates (unique: provider_name + symbol + signal_date + action)

### 2.4 Response Generation
```python
# Location: app.py lines 1036-1039
result = ingest_signal_provider_data(signal_file, provider_name, symbol, timezone_offset)
if result.get("success"):
    st.success(result.get("message"))
else:
    st.error(f"Failed: {result.get('message')}")
```
**Logic:**
- Same pattern as gold ingestion
- Success/error display based on result dictionary

---

## 3. KPI Calculator Flow

### 3.1 UI Layer - KPI Matrix Selection (`app.py` lines 1158-1290)

#### Step 1: Instrument Selection
```python
# Location: app.py lines 1165-1187
available_instruments = []
if selected_category == "Commodities" and selected_instrument_item == "GOLD":
    available_instruments = ["GOLD (^XAUUSD)"]
elif selected_category and selected_instrument_item and selected_instrument_item != "Add...":
    available_instruments = [f"{selected_instrument_item}"]

# Always include Gold as option
if "GOLD (^XAUUSD)" not in available_instruments:
    available_instruments.insert(0, "GOLD (^XAUUSD)")

selected_matrix_instrument = st.selectbox(
    "Financial Instrument",
    options=available_instruments,
    index=0,
    key="kpi_matrix_instrument"
)
```
**Logic:**
- Builds instrument list based on current selections
- Always includes Gold as first option
- User selects instrument for KPI calculation

#### Step 2: Provider Selection (Optional)
```python
# Location: app.py lines 1189-1201
available_providers = ["None"] + merged_providers
if "Add..." in available_providers:
    available_providers.remove("Add...")

selected_matrix_provider = st.selectbox(
    "Signal Provider",
    options=available_providers,
    index=0,
    key="kpi_matrix_provider"
)
```
**Logic:**
- Lists all providers plus "None" option
- Removes "Add..." from list
- Optional selection (for context only)

#### Step 3: KPI Selection
```python
# Location: app.py lines 1203-1212
available_kpis = [k for k in kpi_options if k != "Add..."]
selected_matrix_kpi = st.selectbox(
    "KPI Indicator",
    options=available_kpis,
    index=0,
    key="kpi_matrix_kpi"
)
```
**Logic:**
- Filters out "Add..." from KPI options
- User selects which KPI to calculate

#### Step 4: Calculate Button
```python
# Location: app.py lines 1219-1288
if st.button("Calculate KPI", key="calculate_kpi_btn", type="primary"):
    # Symbol extraction
    symbol = "^XAUUSD" if "GOLD" in selected_matrix_instrument.upper() else selected_matrix_instrument
    
    # Database connection
    supabase = get_supabase()
    if supabase:
        # Table selection
        if "^" in symbol or symbol.upper() == "GOLD":
            table_name = "market_data_commodities_1min"
        else:
            table_name = "market_data_1min"
        
        # Fetch data
        result = supabase.table(table_name)\
            .select("timestamp,open,high,low,close,volume")\
            .eq("symbol", symbol.upper() if not symbol.startswith("^") else symbol)\
            .order("timestamp", desc=True)\
            .limit(200)\
            .execute()
```
**Logic:**
- Extracts symbol from instrument selection
- Determines table based on symbol (Gold uses commodities table)
- Fetches last 200 bars (for calculation accuracy)
- Orders by timestamp descending (most recent first)

### 3.2 Data Fetching and Preparation
```python
# Location: app.py lines 1239-1252
if result.data and len(result.data) > 0:
    # Convert to DataFrame
    df = pd.DataFrame(result.data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')  # Sort ascending for calculations
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
```
**Logic:**
- Converts Supabase response to pandas DataFrame
- Converts timestamp to datetime
- Sorts ascending (oldest to newest) for proper calculation
- Renames columns to match KPI calculator expectations (capitalized)

### 3.3 KPI Calculation (`kpi_calculator.py`)

#### Step 5: KPI Router
```python
# Location: app.py line 1255
kpi_result = calculate_kpi(selected_matrix_kpi, df)
```
**Calls:** `kpi_calculator.py` line 188-221

```python
# Location: kpi_calculator.py lines 188-221
def calculate_kpi(kpi_name: str, df: pd.DataFrame, **kwargs):
    kpi_name = kpi_name.upper().strip()
    
    if kpi_name == "ATR":
        period = kwargs.get('period', 14)
        return calculate_atr(df, period)
    elif kpi_name == "VOLUME":
        period = kwargs.get('period', 20)
        return calculate_volume(df, period)
    elif kpi_name == "VAMP" or kpi_name == "VWAP":
        period = kwargs.get('period', 20)
        return calculate_vamp(df, period)
    elif kpi_name == "EMA":
        period = kwargs.get('period', 15)
        column = kwargs.get('column', 'Close')
        return calculate_ema(df, period, column)
    elif kpi_name == "SMA":
        period = kwargs.get('period', 50)
        column = kwargs.get('column', 'Close')
        return calculate_sma(df, period, column)
```
**Logic:**
- Normalizes KPI name to uppercase
- Routes to specific calculation function
- Passes default parameters if not provided

#### Step 6: ATR Calculation
```python
# Location: kpi_calculator.py lines 11-40
def calculate_atr(df: pd.DataFrame, period: int = 14):
    if df.empty or len(df) < period:
        return None
    
    high = pd.to_numeric(df['High'], errors='coerce')
    low = pd.to_numeric(df['Low'], errors='coerce')
    close = pd.to_numeric(df['Close'], errors='coerce')
    
    # True Range calculation
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    
    # Maximum of the three
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ATR = Moving average of True Range
    atr = true_range.rolling(window=period).mean()
    
    return float(atr.iloc[-1]) if not atr.empty and pd.notna(atr.iloc[-1]) else None
```
**Logic:**
- Validates sufficient data (need at least `period` rows)
- Calculates True Range: max of (High-Low, |High-PrevClose|, |Low-PrevClose|)
- ATR = rolling mean of True Range over `period`
- Returns latest value

#### Step 7: Volume Calculation
```python
# Location: kpi_calculator.py lines 43-78
def calculate_volume(df: pd.DataFrame, period: int = 20):
    volume = pd.to_numeric(df['Volume'], errors='coerce')
    current_volume = float(volume.iloc[-1])
    
    if len(volume) >= period:
        avg_volume = float(volume.tail(period).mean())
    else:
        avg_volume = float(volume.mean())
    
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else None
    
    return {
        "current_volume": current_volume,
        "avg_volume": avg_volume,
        "volume_ratio": volume_ratio
    }
```
**Logic:**
- Gets latest volume value
- Calculates average over `period` (or all available if less)
- Computes ratio: current / average
- Returns dictionary with all three metrics

#### Step 8: VAMP/VWAP Calculation
```python
# Location: kpi_calculator.py lines 81-120
def calculate_vamp(df: pd.DataFrame, period: int = 20):
    close = pd.to_numeric(df['Close'], errors='coerce')
    volume = pd.to_numeric(df['Volume'], errors='coerce')
    
    # Get last N periods
    tail_df = df.tail(period)
    close_tail = pd.to_numeric(tail_df['Close'], errors='coerce')
    volume_tail = pd.to_numeric(tail_df['Volume'], errors='coerce')
    
    # VWAP = sum(Price * Volume) / sum(Volume)
    typical_price = close_tail
    price_volume = typical_price * volume_tail
    sum_price_volume = price_volume.sum()
    sum_volume = volume_tail.sum()
    
    vwap = sum_price_volume / sum_volume if sum_volume > 0 else None
    return float(vwap) if pd.notna(vwap) else None
```
**Logic:**
- Gets last `period` rows
- Calculates volume-weighted average price
- Formula: Σ(Price × Volume) / Σ(Volume)
- Returns single float value

#### Step 9: EMA Calculation
```python
# Location: kpi_calculator.py lines 123-153
def calculate_ema(df: pd.DataFrame, period: int = 15, column: str = 'Close'):
    prices = pd.to_numeric(df[column], errors='coerce')
    
    # EMA using pandas exponential weighted moving
    ema = prices.ewm(span=period, adjust=False).mean()
    latest_ema = ema.iloc[-1]
    
    return float(latest_ema) if pd.notna(latest_ema) else None
```
**Logic:**
- Uses pandas `ewm()` (exponential weighted moving)
- `span=period` sets the decay factor
- `adjust=False` uses standard EMA formula
- Returns latest EMA value

#### Step 10: SMA Calculation
```python
# Location: kpi_calculator.py lines 156-185
def calculate_sma(df: pd.DataFrame, period: int = 50, column: str = 'Close'):
    prices = pd.to_numeric(df[column], errors='coerce')
    
    # SMA = Simple moving average
    sma = prices.rolling(window=period).mean()
    latest_sma = sma.iloc[-1]
    
    return float(latest_sma) if pd.notna(latest_sma) else None
```
**Logic:**
- Uses pandas `rolling().mean()`
- Simple average of last `period` values
- Returns latest SMA value

### 3.4 Response Generation
```python
# Location: app.py lines 1257-1278
if kpi_result is not None:
    st.success(f"✓ {selected_matrix_kpi} calculated successfully")
    
    # Display based on result type
    if isinstance(kpi_result, dict):
        # Volume returns dictionary
        st.markdown("### KPI Results")
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Current Volume", f"{kpi_result.get('current_volume', 0):,.0f}")
        with col_res2:
            st.metric("Average Volume", f"{kpi_result.get('avg_volume', 0):,.0f}")
        with col_res3:
            st.metric("Volume Ratio", f"{kpi_result.get('volume_ratio', 0):.2f}")
    else:
        # Single value result (ATR, VAMP, EMA, SMA)
        st.markdown("### KPI Result")
        st.metric(selected_matrix_kpi, f"{kpi_result:.4f}")
    
    # Show context if provider selected
    if selected_matrix_provider and selected_matrix_provider != "None":
        st.info(f"**Context:** KPI calculated for {selected_matrix_instrument} with signal provider {selected_matrix_provider}")
else:
    st.warning(f"Could not calculate {selected_matrix_kpi}. Please ensure sufficient data is available.")
```
**Logic:**
- Checks if result is not None
- **Volume**: Displays three metrics in columns (current, average, ratio)
- **Other KPIs**: Displays single metric value
- Shows context information if provider selected
- Shows warning if calculation failed

---

## Database Schema Reference

### `market_data_commodities_1min`
- **Unique Constraint**: `(symbol, timestamp)`
- **Columns**: `id`, `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`, `open_interest`, `created_at`, `updated_at`

### `signal_provider_signals`
- **Unique Constraint**: `(provider_name, symbol, signal_date, action)`
- **Columns**: `id`, `provider_name`, `symbol`, `signal_date`, `action`, `entry_price`, `target_1`, `target_2`, `target_3`, `stop_loss`, `sl_hit_datetime`, `tp1_hit_datetime`, `tp2_hit_datetime`, `tp3_hit_datetime`, `timezone_offset`, `created_at`, `updated_at`

---

## Error Handling Patterns

All three flows follow similar error handling:

1. **Input Validation**: Check for None/empty inputs early
2. **File Validation**: Check file type, structure, required columns
3. **Data Validation**: Validate data types, formats, ranges
4. **Database Errors**: Specific messages for table/column issues
5. **Type Conversion Errors**: Catch ValueError/TypeError with context
6. **User Feedback**: Always return `{"success": bool, "message": str}`

---

## Key Design Patterns

1. **Chunked Processing**: Database inserts in chunks of 1000
2. **Upsert Strategy**: Use upsert to handle duplicates gracefully
3. **Column Mapping**: Flexible column name handling for user files
4. **Timezone Standardization**: GMT+4 as default, with validation
5. **Error Context**: Detailed error messages for debugging
6. **Preview Before Submit**: Validate and show summary before ingestion
7. **Session State**: Store user selections and custom additions

---

## Summary

Each flow follows this pattern:
1. **UI Selection** → User selects options in Streamlit
2. **File Upload** → User uploads file (if applicable)
3. **Validation** → System validates input and data
4. **Transformation** → Data converted to database format
5. **Database Operation** → Chunked upsert to Supabase
6. **Response** → Success/error message displayed to user

All flows are independent but share common patterns for consistency and maintainability.

