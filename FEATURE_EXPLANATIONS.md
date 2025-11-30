# Feature Explanations

## 1. Promptable Feature Lab - Logic Flow (Step by Step)

### Overview
The Promptable Feature Lab is a tool that automatically calculates technical indicators from your stored stock data and creates a prompt for an AI assistant to analyze the stock.

### Step-by-Step Flow:

**Step 1: User Input**
- You select a stock symbol (e.g., "AAPL") from a dropdown menu
- You type in a question or instruction in a text box (e.g., "Highlight whether momentum across daily/5m/1m is aligned")

**Step 2: Data Collection**
- The system fetches three types of data from your database:
  - **Daily data**: Last 500 days of daily price bars
  - **5-minute data**: Last 14 days of 5-minute price bars
  - **1-minute data**: Last 3 days of 1-minute price bars

**Step 3: Feature Calculation**
For each timeframe, the system calculates technical indicators:

**Daily Features:**
- Number of records and date range
- Last closing price
- Percentage returns over 5, 20, and 60 days
- Simple Moving Averages (SMA) for 20, 50, and 200 days
- Average True Range (ATR) - measures volatility
- Volatility percentage over 20 days
- Volume ratio (comparing recent volume to prior volume)

**5-Minute Features:**
- Number of records and time range
- Last closing price
- RSI (Relative Strength Index) - measures if stock is overbought/oversold
- Percentage return over recent periods
- Volume ratio

**1-Minute Features:**
- Same as 5-minute but for 1-minute intervals

**Step 4: Sample Data Collection**
- The system also collects the most recent 5 rows of actual price data (Open, High, Low, Close, Volume) for each timeframe
- This gives the AI context about recent trading activity

**Step 5: Context Building**
- All the calculated features and sample data are formatted into a readable text document
- This includes:
  - Coverage dates for each timeframe
  - All calculated indicators
  - Recent price data tables

**Step 6: Prompt Creation**
- The system creates a final prompt that includes:
  - Instructions for the AI (telling it to act as an equities analyst)
  - The formatted context with all features
  - Your specific question/instruction

**Step 7: AI Processing (Optional)**
- If you have a GROQ_API_KEY configured, the system automatically sends the prompt to an AI model
- The AI analyzes the features and responds with:
  - **Observations**: What it notices about the data
  - **Signals**: Trading signals it identifies
  - **Risks**: Potential risks it sees
  - **Actions**: Recommended actions

**Step 8: Display Results**
- The system shows you:
  - A table of all calculated features
  - The formatted context document
  - The final prompt (so you can copy it to another AI if needed)
  - The AI's response (if available)

### Where Prompts Are Used:
- **Location**: `tradingagents/dataflows/feature_lab.py`
- **Method**: `_build_prompt()` (line 356) - Creates the final prompt template
- **Method**: `_call_llm()` (line 368) - Sends prompt to AI if API key is available
- **UI Location**: `app.py` lines 1095-1142 - The user interface for this feature

---

## 2. Data Coverage Guardrail & Feature Lab - Head Gap, Tail Gap, and Status

### Overview
The Data Coverage Guardrail checks if your database has all the required historical data for analysis. It identifies gaps (missing data periods) and can automatically fill them.

### Required Coverage Windows:
- **1-minute data**: From August 1, 2025 → current time minus 15 minutes
- **5-minute data**: From October 1, 2023 → current time minus 15 minutes  
- **Daily data**: From October 1, 2023 → previous trading day close

### Key Concepts:

#### **Head Gap (head_gap_min)**
**What it is:**
- The amount of missing data at the **beginning** of the required time period
- Measured in minutes (even for daily data, it's converted to minutes)

**How it's calculated:**
1. The system checks what's the earliest date in your database for a stock
2. It compares this to the required start date
3. If your database starts later than required, there's a head gap
4. **Formula**: `head_gap = (database_start_date - required_start_date) in minutes`

**Example:**
- Required: Data from Oct 1, 2023
- Your database: Starts on Oct 15, 2023
- Head gap: 14 days = 20,160 minutes (14 days × 24 hours × 60 minutes)

**Special case:**
- If there's NO data in the database at all, the head gap equals the entire required time period

#### **Tail Gap (tail_gap_min)**
**What it is:**
- The amount of missing data at the **end** of the required time period
- Measured in minutes

**How it's calculated:**
1. The system checks what's the latest date in your database for a stock
2. It compares this to the required end date
3. If your database ends earlier than required, there's a tail gap
4. **Formula**: `tail_gap = (required_end_date - database_end_date) in minutes`

**Example:**
- Required: Data up to "now minus 15 minutes" (e.g., 2:45 PM today)
- Your database: Latest data is from 1:00 PM today
- Tail gap: 1 hour 45 minutes = 105 minutes

**Special case:**
- If there's NO data in the database at all, the tail gap equals the entire required time period

#### **Status**
**What it is:**
- A simple label that tells you if your data is ready for analysis

**How it's calculated:**
The system checks three conditions:

1. **"missing"** status:
   - If there's no data at all in the database (no start date OR no end date)
   - This means you need to ingest data from scratch

2. **"ready"** status:
   - If BOTH head_gap ≤ 0 AND tail_gap ≤ 0
   - This means you have all required data (or more)
   - Your data is complete and ready for analysis

3. **"partial"** status:
   - If you have some data, but either head_gap > 0 OR tail_gap > 0
   - This means you're missing data at the beginning, end, or both
   - You can still do some analysis, but results may be incomplete

**Visual Example:**
```
Required Period:  [========== Oct 1, 2023 to Now ==========]
Your Database:         [==== Oct 15, 2023 to Yesterday ====]
                        ↑                    ↑
                   Head Gap              Tail Gap
                   (missing start)      (missing end)
                   Status: "partial"
```

### How the System Uses This:

**Step 1: Coverage Audit**
- Click "Run Coverage Audit" button
- System checks all stocks in your watchlist
- For each stock and each timeframe (daily, 5-min, 1-min), it calculates:
  - Head gap
  - Tail gap
  - Status
  - Number of rows in database

**Step 2: Display Results**
- Shows a table with all the gaps and statuses
- You can see which stocks need data backfilling

**Step 3: Auto Backfill (Optional)**
- Click "Auto Backfill Missing Data" button
- System automatically identifies what's missing
- Creates ingestion tasks to fill:
  - Head gaps (missing early data)
  - Tail gaps (missing recent data)
- Runs the data ingestion pipeline to fetch and store missing data

### Where Calculations Happen:
- **Location**: `tradingagents/dataflows/data_guardrails.py`
- **Method**: `build_symbol_report()` (line 248) - Calculates gaps for one stock
- **Method**: `_minutes_gap()` (line 104) - Converts time difference to minutes
- **Method**: `status()` (line 57) - Determines if data is "missing", "ready", or "partial"
- **UI Location**: `app.py` lines 1040-1093 - The user interface for this feature

### Why This Matters:
- Ensures you have complete data before running analysis
- Prevents errors from missing historical data
- Automatically identifies and fixes data gaps
- Saves time by showing exactly what's missing instead of guessing

