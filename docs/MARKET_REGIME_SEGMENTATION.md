# Market Regime Segmentation Module

## Overview

The Market Regime Segmentation Module adds a "Context Analysis" layer to the Alpha Analyst Trading App's signal analysis capabilities. It answers the critical question:

> **"Does this signal provider perform better in trending markets or ranging markets?"**

## Features

- **Regime Classification**: Automatically classifies market conditions into 4 regimes:
  - Trending - High Vol
  - Trending - Low Vol
  - Ranging - High Vol
  - Ranging - Low Vol

- **Performance Segmentation**: Calculates win rates, profit factors, and other metrics for each regime

- **Timezone-Safe Merging**: Ensures accurate signal-to-regime matching with proper timezone handling

- **Interactive UI**: Streamlit-based interface with heatmaps and configurable parameters

## Architecture

### Module Structure

```
tradingagents/dataflows/signal_analyzer.py
├── calculate_regimes()          # Step 1: Feature Engineering
├── define_regime()               # Step 2: Regime Classification
├── merge_signals_with_regimes()  # Step 3: Data Merging
└── calculate_metrics_by_regime() # Step 4: Aggregation & Metrics
```

### Data Flow

```
Market Data (OHLCV)
    ↓
[Step 1] Calculate Indicators (ADX, SMA, ATR)
    ↓
[Step 2] Classify Regimes (Trending/Ranging, High/Low Vol)
    ↓
[Step 3] Merge with Signals (Timezone-aware)
    ↓
[Step 4] Calculate Performance Metrics by Regime
    ↓
Display Results (Heatmap, Table, Insights)
```

## Technical Implementation

### Step 1: Feature Engineering

**Function**: `calculate_regimes(market_df, adx_period=14, sma_short=50, sma_long=200, atr_period=14, atr_ma_period=50)`

Calculates technical indicators:
- **ADX (Average Directional Index)**: Measures trend strength (0-100)
- **SMA (Simple Moving Average)**: 50 and 200-period for trend direction
- **ATR (Average True Range)**: Measures volatility
- **ATR_MA**: Moving average of ATR for volatility comparison

**Implementation Notes**:
- Uses `pandas_ta` library if available, falls back to manual calculations
- Normalizes ATR as percentage of price for cross-instrument comparison
- Handles missing data gracefully

### Step 2: Regime Classification

**Function**: `define_regime(market_df, adx_threshold=25)`

Classifies each candle into a regime based on:

| Condition | Classification |
|-----------|---------------|
| ADX > threshold | Trending |
| ADX ≤ threshold | Ranging |
| ATR > ATR_MA | High Vol |
| ATR ≤ ATR_MA | Low Vol |

**Example Output**:
- "Trending - High Vol"
- "Ranging - Low Vol"

### Step 3: Data Merging (CRITICAL)

**Function**: `merge_signals_with_regimes(signals_df, market_df, entry_time_col='signal_date', timezone='UTC')`

Merges signals with market regimes using `pd.merge_asof` with backward direction.

**Timezone Handling**:
```python
# CRITICAL: Both DataFrames converted to same timezone before merge
signals['signal_date'] = pd.to_datetime(signals['signal_date']).dt.tz_convert('UTC')
market.index = market.index.tz_convert('UTC')

# Merge with backward direction (no look-ahead bias)
merged = pd.merge_asof(
    signals,
    market[['Regime', 'ADX', 'ATR']],
    left_on='signal_date',
    right_index=True,
    direction='backward',  # Match to AT OR BEFORE entry time
    tolerance=pd.Timedelta('1 hour')
)
```

**Why This Matters**:
- Prevents look-ahead bias (using future data)
- Ensures accurate regime matching at signal entry time
- Handles timezone mismatches between signal and market data

### Step 4: Aggregation & Metrics

**Function**: `calculate_metrics_by_regime(signals_df, pnl_col='pips_made', final_status_col='final_status')`

Calculates per-regime metrics:

| Metric | Formula | Description |
|--------|---------|-------------|
| Total Trades | COUNT(*) | Number of signals in regime |
| Win Rate % | (Wins / Total) × 100 | Percentage of winning trades |
| Profit Factor | Gross Wins / Gross Losses | Risk-reward efficiency |
| Avg PnL | MEAN(pips_made) | Average profit/loss per trade |

## Streamlit UI Implementation

### Location
`app.py` - Added after Signal Analysis Results section (line ~1994)

### UI Components

1. **Configuration Panel**
   - ADX Threshold slider (15-40, default: 25)
   - Market Data Timeframe selector (1h, 4h, 1d)
   - Lookback Period input (30-730 days)

2. **Analysis Button**
   - Validates signal analysis results exist
   - Fetches market data for regime calculation
   - Runs 4-step pipeline
   - Displays progress indicators

3. **Results Display**
   - **Metrics Table**: Detailed performance by regime
   - **Heatmap**: Visual win rate matrix (Plotly)
   - **Key Insights**: Best/worst performing regimes
   - **Recommendations**: Actionable trading advice
   - **Export**: CSV download option

### Code Example

```python
# Run regime analysis
if st.button("🚀 Run Regime Analysis"):
    analyzer = SignalAnalyzer()
    
    # Step 1: Calculate indicators
    market_with_indicators = analyzer.calculate_regimes(market_df)
    
    # Step 2: Define regimes
    market_with_regimes = analyzer.define_regime(market_with_indicators, adx_threshold=25)
    
    # Step 3: Merge signals
    signals_with_regimes = analyzer.merge_signals_with_regimes(
        signals_df, market_with_regimes, timezone='UTC'
    )
    
    # Step 4: Calculate metrics
    regime_metrics = analyzer.calculate_metrics_by_regime(signals_with_regimes)
    
    # Display results
    st.dataframe(regime_metrics)
```

## Installation

### Requirements

Add to `requirements.txt`:
```
pandas-ta>=0.3.14b0
```

Install:
```bash
pip install pandas-ta
```

### Verification

Run the test suite:
```bash
python test_regime_analysis.py
```

Expected output:
```
✅ STEP 1 PASSED - Calculate Regimes
✅ STEP 2 PASSED - Define Regime
✅ STEP 3 PASSED - Merge Signals with Regimes
✅ STEP 4 PASSED - Calculate Metrics by Regime
✅ TIMEZONE TEST PASSED
🎉 ALL TESTS PASSED!
```

## Usage Guide

### Step-by-Step Workflow

1. **Run Signal Analysis First**
   - Navigate to "Signal Analysis Dashboard"
   - Select instrument and provider
   - Click "Run Analysis"
   - Wait for analysis results

2. **Configure Regime Parameters**
   - Adjust ADX Threshold (default: 25)
   - Select timeframe (1h recommended for intraday)
   - Set lookback period (365 days recommended)

3. **Run Regime Analysis**
   - Click "🚀 Run Regime Analysis"
   - Wait for 4-step pipeline to complete
   - Review results

4. **Interpret Results**
   - Check heatmap for visual patterns
   - Identify best/worst performing regimes
   - Read trading recommendations
   - Export data if needed

### Example Output

```
📊 Regime Analysis Results

Regime                    | Total_Trades | Win_Rate_% | Profit_Factor | Avg_PnL
--------------------------|--------------|------------|---------------|--------
Trending - High Vol       | 15           | 66.7       | 2.34          | 45.2
Trending - Low Vol        | 12           | 58.3       | 1.87          | 32.1
Ranging - High Vol        | 18           | 38.9       | 0.92          | -12.5
Ranging - Low Vol         | 10           | 40.0       | 1.05          | 5.3

💡 Key Insights:
- Best Performance: Trending - High Vol (66.7% win rate)
- Worst Performance: Ranging - High Vol (38.9% win rate)

🎯 Recommendations:
- Focus on Trending - High Vol conditions
- Avoid or reduce exposure during Ranging - High Vol
```

## Acceptance Criteria

### Definition of Done

- [x] App loads without error when market data is uploaded
- [x] ADX Threshold slider dynamically updates regime classification
- [x] User can see detailed metrics table by regime
- [x] Code is modular (logic separated from UI)
- [x] Timezone handling prevents look-ahead bias
- [x] Both Signal CSV and Market Data are converted to UTC before merge

## Performance Considerations

### Data Size Limits

- **Market Data**: Recommended max 2 years at 1-hour intervals (~17,520 candles)
- **Signals**: No hard limit, tested up to 10,000 signals
- **Processing Time**: ~2-5 seconds for typical dataset

### Optimization Tips

1. Use higher timeframes (4h, 1d) for faster processing
2. Reduce lookback period if data is slow to fetch
3. Cache market data in session state to avoid re-fetching

## Troubleshooting

### Common Issues

**Issue**: "No regime metrics calculated"
- **Cause**: Signals couldn't be matched to market data
- **Solution**: Check timezone alignment, ensure market data covers signal period

**Issue**: "pandas_ta import error"
- **Cause**: Library not installed
- **Solution**: Run `pip install pandas-ta`

**Issue**: "All regimes show 'Unknown'"
- **Cause**: Insufficient data for indicator calculation
- **Solution**: Increase lookback period or use lower timeframe

**Issue**: "Merge tolerance exceeded"
- **Cause**: Market data gaps larger than 1 hour
- **Solution**: Fetch more granular market data or increase tolerance

## Future Enhancements

### Planned Features

1. **Additional Regime Factors**
   - Volume profile (high/low volume)
   - Market session (Asian/European/US)
   - Day of week patterns

2. **Machine Learning Integration**
   - Automatic regime detection using clustering
   - Predictive regime classification

3. **Multi-Timeframe Analysis**
   - Compare regime performance across timeframes
   - Identify optimal timeframe for each provider

4. **Real-Time Regime Monitoring**
   - Live regime classification
   - Alerts when entering favorable regime

## References

### Technical Indicators

- **ADX**: Welles Wilder (1978) - "New Concepts in Technical Trading Systems"
- **ATR**: Welles Wilder (1978) - Volatility measurement
- **SMA**: Basic trend identification

### Best Practices

- Always use backward-looking merge to avoid look-ahead bias
- Ensure minimum 30 trades per regime for statistical significance
- Consider transaction costs when evaluating profit factors
- Validate results across multiple time periods

## Support

For issues or questions:
1. Check test suite: `python test_regime_analysis.py`
2. Review logs in Streamlit console
3. Verify timezone handling with `DEBUG_TIMEZONE=true`

## License

Part of Alpha Analyst Trading App - Phase 1 Implementation
Date: January 24, 2026

