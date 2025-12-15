# Data Coverage Guardrail & Feature Lab - End-to-End Logic

## Overview
This feature ensures data quality before running analysis by checking coverage windows and validating data freshness for the QUANTUMTRADER automated screening system.

---

## Part 1: Data Coverage Guardrails

### Step 1: Define Required Coverage Windows
**Location:** `tradingagents/dataflows/data_guardrails.py` - `DataCoverageService._requirements()`

The system defines three required coverage windows:

1. **1-minute bars:**
   - Start: August 1, 2025 (UTC)
   - End: Current time minus 15 minutes (using EAT timezone reference)
   - Purpose: QUANTUMTRADER needs last 20 consecutive 1-minute bars

2. **5-minute bars:**
   - Start: October 1, 2023 (UTC)
   - End: Current time minus 15 minutes
   - Purpose: Trend analysis and intermediate timeframe context

3. **Daily bars:**
   - Start: October 1, 2023 (UTC)
   - End: Previous trading day close
   - Purpose: Long-term trend context and daily pattern recognition

**Why 15 minutes buffer?** 
- Prevents using incomplete/realtime bars
- Ensures data has been fully processed
- Accounts for API ingestion delays

### Step 2: Coverage Audit
**Location:** `app.py` lines 1016-1021

**User Action:** Click "🔍 Run Coverage Audit"

**Process:**
1. Calls `coverage_service.build_watchlist_report()`
2. For each symbol in watchlist:
   - Queries Supabase to find earliest and latest timestamps in each table
   - Calculates gaps:
     - **Head gap:** Missing data at the start of required window
     - **Tail gap:** Missing data at the end (should be up to current time - 15 min)
   - Counts rows in the required date range
   - Creates `CoverageRecord` objects with status: `ready`, `partial`, or `missing`

3. Stores results in `st.session_state.coverage_records`

**Output:** DataFrame showing:
- Symbol, interval, required start/end
- Actual DB start/end
- Row counts
- Gap sizes (head/tail in minutes)
- Status (ready/partial/missing)

### Step 3: Auto Backfill
**Location:** `app.py` lines 1025-1088, `data_guardrails.py` lines 336-365

**User Action:** Click "🛠️ Auto Backfill Missing Data"

**Process:**
1. Checks if coverage audit has been run
2. Filters records where `needs_backfill() == True` (status != "ready")
3. For each missing record, creates backfill tasks:
   - **Head gap task:** Ingest from required start → existing DB start
   - **Tail gap task:** Ingest from existing DB end → required end
4. Executes tasks using `DataIngestionPipeline.ingest_historical_data()`
5. Logs each task execution (success/failure)
6. Displays backfill activity results

**Note:** Requires `POLYGON_API_KEY` environment variable

---

## Part 2: QUANTUMTRADER v0.1 Prompt

### Step 6: QUANTUMTRADER Data Freshness Check
**Location:** `app.py` lines 1186-1505

**Process Flow:**

#### 6a. Command Timestamp Label (Auto)
- System stamps current UTC time as a label in the prompt output
- **CRITICAL:** Label only; never used as a data filter
- Always fetches the most recent data from the database
- No manual timestamp entry required

#### 6b. Direct Database Query (Debug Mode)
**Location:** `app.py` lines 1195-1236

Queries Supabase directly to show:
- Top 5 records from `market_data_1min` table for selected symbol
- Top 5 records from `market_data_5min` table
- Available symbols in each table (for troubleshooting)

#### 6c. Fetch Latest Bar
**Location:** `tradingagents/dataflows/market_data_service.py` - `fetch_latest_bar()`

For each interval:
1. Query Supabase table for symbol
2. Order by timestamp DESC, limit 1
3. Return: `{timestamp, open, high, low, close, volume, source}`

#### 6d. Freshness Validation
**Location:** `app.py` lines 1241-1320

**Freshness Rules:**
- **Normal market hours:** Data must be < 20 minutes old
- **Weekend/after hours:** Relaxed to 48 hours (2880 minutes)

**Check Logic:**
```
age_minutes = (current_utc_time - latest_bar_timestamp) / 60 seconds

if age_minutes > freshness_cutoff:
    data_ready = False
    Show warning with age details
```

**Market Hours Detection:**
- Weekend: Saturday (day 5) or Sunday (day 6)
- After hours: UTC hour < 14 or >= 21
- US market: 14:30-21:00 UTC (9:30 AM - 4:00 PM ET)

#### 6e. Quick Ingest Button
**Location:** `app.py` lines 1384-1479

**User Action:** Click "📥 Ingest Latest Data"

**Process:**
1. Get current state (before ingestion)
2. Set `end_date = current UTC time`
3. Call `pipeline.ingest_historical_data()`:
   - 1-minute: `days_back=3`, `resume_from_latest=True`
   - 5-minute: `days_back=7`, `resume_from_latest=True`
4. Wait 2 seconds for DB commit
5. Check state (after ingestion)
6. Compare timestamps:
   - If updated → Success, check freshness again
   - If unchanged → Warning (no new data or already up to date)
7. If fresh → Show success, trigger page refresh

#### 6f. Override Option
**Location:** `app.py` lines 1334-1340, 1492-1500

If market is closed but data is old:
- Show override button: "✅ Override & Allow QUANTUMTRADER"
- Sets `st.session_state.quantum_data_override = True`
- Allows QUANTUMTRADER to run despite age warnings
- User can cancel override later

**Effective Data Ready:**
```python
if override_active:
    effective_data_ready = True
else:
    effective_data_ready = data_ready
```

### Step 7: Run QUANTUMTRADER
**Location:** `app.py` lines 1506-1513, `feature_lab.py` - `run_quantum_screen()`

**Trigger:** Button "⚙️ Run QUANTUMTRADER Prompt" (disabled if `effective_data_ready == False`)

**Process:**

#### 7a. Data Fetching & Validation
**Location:** `feature_lab.py` lines 220-282

1. **Fetch 1-minute data:** Last 3 days
   - If empty → Try 30 days
   - If still empty → Raise error with diagnostic info
   - If < 20 bars → Raise error (need consecutive bars)

2. **Fetch 5-minute data:** Last 7 days
   - Similar validation logic
   - Need at least 6 consecutive bars

3. **Fetch daily data:** Last 60 days (optional but recommended)

#### 7b. Compute QUANTUMTRADER Metrics
**Location:** `feature_lab.py` lines 474-593

**Metrics Calculated:**

1. **Volume Score (0-10):**
   - Current volume vs 20-period MA
   - Volume ratio = current / MA20
   - Scoring: >2.0 = 10, 1.5-2.0 = 8, 1.0-1.5 = 5, etc.

2. **VWAP Score (0-10):**
   - Calculate VWAP from last 20 1-minute bars
   - Distance from VWAP = (price - VWAP) / VWAP * 100
   - Scoring: Near VWAP (±0.5%) = 10, far away = lower score

3. **Momentum Score (0-10):**
   - Rate of change (ROC) over 5 minutes
   - ROC = (current_price - price_5min_ago) / price_5min_ago * 100
   - Scoring: Positive momentum = higher score

4. **Catalyst Score (0-10):**
   - Currently fixed at 6.0 (placeholder)
   - Future: Could incorporate news/sector momentum

5. **Composite Score (weighted average):**
   ```
   composite = 0.3 * volume_score 
             + 0.3 * vwap_score 
             + 0.3 * momentum_score 
             + 0.1 * catalyst_score
   ```

6. **Additional Metrics:**
   - ATR(14) from 5-minute bars
   - RSI(14) from 1-minute bars
   - High/Low of last 120 minutes
   - EMA(15) current and previous (from 5-minute bars)
   - Daily metrics: SMAs, returns, ATR, RSI, trend, ranges

7. **Verdict:**
   - PASS if composite_score >= 6.0
   - FAIL otherwise

#### 7c. Build Data Extraction
**Location:** `feature_lab.py` lines 594-665

**Extracts:**
1. **Last 20 1-minute bars:** OHLCV table
2. **Last 6 5-minute bars:** OHLCV table
3. **Last 10 daily bars:** OHLCV table (if available)

**Validation Checks:**
- Timestamp alignment: 1-min and 5-min end within 5 minutes of each other
- Sequence OK: Bars are consecutive (no gaps)
- Freshness: Latest bar within 60 seconds
- Volume OK: All volume values >= 0

**Formats:**
- Window labels: "HH:MM-HH:MM (YYYY-MM-DD)"
- OHLCV tables: Markdown format

#### 7d. Build QUANTUMTRADER Prompt
**Location:** `feature_lab.py` lines 738-875

**Prompt Structure:**

```
QUANTUMTRADER v0.1 - DATA EXTRACTION TEST
Command: EXTRACT_RAW_DATA [SYMBOL] [TIMESTAMP_LABEL]

Required Data Output:
• Daily OHLCV: Last 10 periods (window)
  [OHLCV table]

• 5-minute OHLCV: Last 6 periods (window)
  [OHLCV table]

• 1-minute OHLCV: Last 20 periods (window)
  [OHLCV table]

Daily Context:
• Daily Trend: BULLISH/BEARISH/NEUTRAL
• Daily SMAs: SMA20, SMA50, SMA200
• Daily Returns: 5d, 20d
• Daily RSI(14), ATR(14)
• Daily Range (20d): High, Low

Intraday Metrics:
• VWAP
• ATR(14)[5min]
• EMA(20)[15min]: Current, Previous
• RSI(14)[1min]
• 120-minute high/low
• Volume MA(20)[1min]

Validation Checks:
• Timestamp alignment: OK/NOT OK
• Sequence: OK/NOT OK
• Freshness: X seconds ago
• Volume: OK/NOT OK

SCORING METRICS:
• Volume Score: X/10
• VWAP Score: X/10
• Momentum Score: X/10
• Catalyst Score: X/10
• Composite Score: X/10
• Verdict: PASS/FAIL

[Calculation details...]
```

#### 7e. Return Results
**Location:** `feature_lab.py` lines 294-306

Returns dictionary:
```python
{
    "metrics": {...},  # All calculated scores
    "prompt": "...",   # Full formatted prompt
    "summary": "...",  # One-line summary
    "timestamp": "...", # Command timestamp label
    "extraction": {...} # Raw data extraction
}
```

### Step 8: Display QUANTUMTRADER Results
**Location:** `app.py` lines 1515-1521

**Display:**
1. Success message with summary
2. Metrics DataFrame (all scores in table format)
3. Full prompt payload (copyable markdown)

### Step 9: Trade Decision Engine (Optional)
**Location:** `app.py` lines 1523-1676

**User Action:** Click "🎯 Evaluate Trade Decision"

**Process:**
1. Creates `TradeDecisionEngine` instance
2. Calls `evaluate_trade_decision(symbol, timestamp)`
3. Evaluates 5 conditions:
   - Composite Score ≥ 6.5
   - All Phase 1 gates passed
   - R:R ratio ≥ 1:2 achievable
   - Position size ≤ $2,000 calculable
   - No conflicting daily trend
4. Determines direction (UP/DOWN) based on 5-min trend alignment
5. Checks risk management overlay:
   - Max daily loss: $400
   - Max concurrent trades: 3
   - Auto-close at 4:00 PM ET

**Output:**
- Trade Decision: TRADE YES / NO TRADE
- Direction: UP / DOWN / NEUTRAL
- Reason: Text explanation
- Condition breakdown table (PASS/FAIL per condition)
- Risk metrics overlay
- Trade recommendation (if TRADE YES):
  - Entry price, Stop loss, Targets (2)
  - R:R ratio, Shares, Exposure, Risk amount

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Coverage Audit                                           │
│    └─> Query Supabase → Calculate gaps → Report status     │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Auto Backfill (if gaps found)                            │
│    └─> Create tasks → Ingest via Polygon API → Log results │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Feature Lab (Custom Prompt)                              │
│    └─> Fetch data → Compute features → Build prompt → LLM  │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. QUANTUMTRADER Freshness Check                            │
│    └─> Query latest bars → Check age → Validate readiness  │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Quick Ingest (if stale)                                  │
│    └─> Ingest latest 3 days (1min) + 7 days (5min)         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Run QUANTUMTRADER                                        │
│    └─> Fetch data → Compute metrics → Build prompt         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Trade Decision Engine (optional)                         │
│    └─> Evaluate conditions → Determine direction → Output  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Functions & Locations

| Function | File | Purpose |
|----------|------|---------|
| `DataCoverageService._requirements()` | `data_guardrails.py:107` | Define required coverage windows |
| `DataCoverageService.build_watchlist_report()` | `data_guardrails.py:287` | Audit coverage for all symbols |
| `DataCoverageService.backfill_missing()` | `data_guardrails.py:336` | Backfill gaps automatically |
| `FeatureLab.run()` | `feature_lab.py:164` | Generate custom feature packet |
| `FeatureLab.run_quantum_screen()` | `feature_lab.py:220` | Run QUANTUMTRADER analysis |
| `FeatureLab._compute_quantum_metrics()` | `feature_lab.py:474` | Calculate all scores |
| `fetch_latest_bar()` | `market_data_service.py:186` | Get most recent bar from DB |
| `TradeDecisionEngine.evaluate_trade_decision()` | `trading_engine.py` | Evaluate trade conditions |

---

## Important Notes

1. **Command Timestamp is Label-Only:**
   - Does NOT filter data
   - System always uses most recent data from database
   - Only appears in prompt output for documentation

2. **Freshness Thresholds:**
   - Market hours: 20 minutes
   - Off hours: 48 hours (2880 minutes)

3. **Data Requirements:**
   - QUANTUMTRADER needs minimum 20 consecutive 1-minute bars
   - Data must be within freshness window (or override enabled)

4. **Timezones:**
   - Database: UTC
   - Coverage windows: EAT (Africa/Nairobi) reference for current time calculation
   - Display: UTC

5. **Error Handling:**
   - Diagnostic queries show what data exists in DB
   - Clear error messages guide user to ingest missing data
   - Backfill logs show success/failure per task

