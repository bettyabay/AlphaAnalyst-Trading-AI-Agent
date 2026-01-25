# Market Regime Segmentation - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (30 seconds)
```bash
pip install pandas-ta
```

### Step 2: Verify Installation (1 minute)
```bash
python test_regime_analysis.py
```

Expected output:
```
[PASS] STEP 1 PASSED
[PASS] STEP 2 PASSED
[PASS] STEP 3 PASSED
[PASS] STEP 4 PASSED
[PASS] TIMEZONE TEST PASSED
SUCCESS: ALL TESTS PASSED!
```

### Step 3: Launch App (30 seconds)
```bash
streamlit run app.py
```

### Step 4: Run Analysis (3 minutes)

#### 4a. Run Signal Analysis First
1. Navigate to "Signal Analysis Dashboard"
2. Select:
   - **Instrument**: e.g., C:XAUUSD (Gold)
   - **Provider**: e.g., PipXpert or "All Providers"
   - **Date Range**: e.g., "Last 90 Days"
3. Click **"Run Analysis"**
4. Wait for results to appear

#### 4b. Run Regime Analysis
1. Scroll down to "Market Regime Analysis" section
2. Configure parameters:
   - **ADX Threshold**: 25 (default, good for most cases)
   - **Timeframe**: 1h (recommended for intraday)
   - **Lookback Period**: 365 days (1 year of data)
3. Click **"🚀 Run Regime Analysis"**
4. Wait for 4-step pipeline to complete

### Step 5: Interpret Results

#### Heatmap
- **Green cells** = High win rate (good performance)
- **Red cells** = Low win rate (poor performance)
- **X-axis** = Volatility (High/Low)
- **Y-axis** = Trend (Trending/Ranging)

#### Metrics Table
- **Total_Trades**: Number of signals in this regime
- **Win_Rate_%**: Percentage of winning trades
- **Profit_Factor**: Gross Wins / Gross Losses (>1 is profitable)
- **Avg_PnL**: Average profit/loss per trade

#### Key Insights
- **Best Performance**: Focus trading on this regime
- **Worst Performance**: Avoid or reduce exposure

---

## 📊 Example Output

```
Regime Performance Metrics:

Regime                    | Total_Trades | Win_Rate_% | Profit_Factor | Avg_PnL
--------------------------|--------------|------------|---------------|--------
Trending - High Vol       | 15           | 66.7       | 2.34          | 45.2
Trending - Low Vol        | 12           | 58.3       | 1.87          | 32.1
Ranging - High Vol        | 18           | 38.9       | 0.92          | -12.5
Ranging - Low Vol         | 10           | 40.0       | 1.05          | 5.3
```

**Interpretation**:
- ✅ **Best**: Trending - High Vol (66.7% win rate, 2.34 profit factor)
- ❌ **Worst**: Ranging - High Vol (38.9% win rate, 0.92 profit factor)
- 💡 **Action**: Focus on trending markets with high volatility, avoid ranging markets with high volatility

---

## ⚙️ Parameter Tuning

### ADX Threshold
- **Lower (15-20)**: More markets classified as "Trending"
- **Default (25)**: Balanced classification
- **Higher (30-40)**: Only strong trends classified as "Trending"

### Timeframe
- **1h**: More granular, better for intraday signals
- **4h**: Smoother, better for swing trading
- **1d**: Very smooth, better for position trading

### Lookback Period
- **30-90 days**: Recent market conditions only
- **180-365 days**: Balanced view
- **365-730 days**: Long-term patterns

---

## 🔧 Troubleshooting

### Problem: "No signal analysis results found"
**Solution**: Run signal analysis first (Step 4a above)

### Problem: "No market data available"
**Solution**: Ensure market data has been ingested for the selected instrument

### Problem: "No regime metrics calculated"
**Solution**: 
- Check that signal dates overlap with market data period
- Increase lookback period
- Verify timezone alignment

### Problem: Test suite fails
**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade pandas numpy pandas-ta

# Run tests again
python test_regime_analysis.py
```

---

## 📚 Learn More

- **Full Documentation**: `docs/MARKET_REGIME_SEGMENTATION.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Technical Spec**: See original specification document

---

## 🎯 Pro Tips

1. **Sample Size**: Ensure at least 30 trades per regime for statistical significance
2. **Timeframe Matching**: Use 1h regime data for intraday signals, 4h/1d for swing signals
3. **Regular Updates**: Re-run regime analysis monthly to capture evolving market conditions
4. **Combine Insights**: Use regime analysis alongside other metrics (Sharpe ratio, max drawdown)
5. **Export Data**: Download CSV for further analysis in Excel/Python

---

## ✅ Success Checklist

- [ ] Installed pandas-ta
- [ ] Ran test suite (all tests passed)
- [ ] Started Streamlit app
- [ ] Ran signal analysis
- [ ] Ran regime analysis
- [ ] Interpreted results
- [ ] Identified best/worst performing regimes
- [ ] Adjusted trading strategy based on insights

---

**You're all set! Happy trading! 📈**

