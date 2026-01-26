# Indicator Calculations: Step-by-Step with Real Examples

This document explains **exactly how each indicator is calculated**, showing **where every number comes from** using **real formulas + intuitive examples**.

---

## 📊 Foundation: What Data We Start With

All indicators are calculated from **OHLC price data** on a chosen timeframe.

**For our examples, assume:**
- **Timeframe:** 4-Hour (4H)
- **Period:** 14 (for ADX, DI, ATR)
- **Market:** XAUUSD (Gold)
- **Current Candle:** 2025-12-12 08:00 → 12:00

**Each 4H candle has:**
```
Open, High, Low, Close
```

**We also need previous candles** (history) to calculate indicators.

---

## 1️⃣ True Range (TR) → Foundation of Everything

Before ADX, DI, or ATR, we must compute **True Range**.

### Formula (for each candle)

```
TR = max(
    High − Low,
    |High − Previous Close|,
    |Low − Previous Close|
)
```

### Why Three Options?

**True Range accounts for:**
1. **High - Low**: Normal intra-candle range
2. **|High - Previous Close|**: Gap up scenario
3. **|Low - Previous Close|**: Gap down scenario

We take the **maximum** because we want to know the **real price movement**, including gaps.

### Example: 4H Candle

**Previous Candle:**
```
Close = 2628.0
```

**Current Candle (2025-12-12 08:00-12:00):**
```
Open  = 2630.0
High  = 2660.0
Low   = 2635.0
Close = 2652.0
```

**Step-by-Step Calculation:**

```
Option 1: High − Low
  = 2660.0 − 2635.0
  = 25.0

Option 2: |High − Previous Close|
  = |2660.0 − 2628.0|
  = |32.0|
  = 32.0

Option 3: |Low − Previous Close|
  = |2635.0 − 2628.0|
  = |7.0|
  = 7.0
```

**Result:**
```
TR = max(25.0, 32.0, 7.0) = 32.0
```

### Interpretation

👉 **Price really moved 32 points** in this 4-hour period, accounting for the gap from previous close.

**Why this matters:**
- If we only used High - Low (25.0), we'd miss the gap
- True Range (32.0) captures the **actual price movement**

---

## 2️⃣ Directional Movement (+DM and −DM)

This measures **who is in control**: buyers or sellers.

### Formula

```
UpMove   = Current High − Previous High
DownMove = Previous Low − Current Low

+DM = UpMove   if (UpMove > DownMove) AND (UpMove > 0) else 0
-DM = DownMove if (DownMove > UpMove) AND (DownMove > 0) else 0
```

### Logic

- **+DM**: Upward directional movement (buyers pushing price up)
- **-DM**: Downward directional movement (sellers pushing price down)
- Only the **stronger** direction counts
- If neither is positive, both are 0

### Example: 4H Candle

**Previous Candle:**
```
High = 2642.0
Low  = 2615.0
```

**Current Candle:**
```
High = 2660.0
Low  = 2635.0
```

**Step-by-Step Calculation:**

```
Step 1: Calculate UpMove
  UpMove = Current High − Previous High
         = 2660.0 − 2642.0
         = 18.0

Step 2: Calculate DownMove
  DownMove = Previous Low − Current Low
           = 2615.0 − 2635.0
           = -20.0  (negative!)

Step 3: Determine +DM
  Condition 1: UpMove > DownMove? 18.0 > -20.0? ✅ Yes
  Condition 2: UpMove > 0? 18.0 > 0? ✅ Yes
  Result: +DM = 18.0

Step 4: Determine -DM
  Condition 1: DownMove > UpMove? -20.0 > 18.0? ❌ No
  Result: -DM = 0.0
```

**Result:**
```
+DM = 18.0
-DM = 0.0
```

### Interpretation

👉 **Buyers clearly dominated** this candle. Price moved up $18 more than it moved down.

**What this means:**
- Strong upward pressure
- Buyers in control
- Sellers couldn't push price down significantly

---

## 3️⃣ Wilder's Smoothing (VERY Important)

ADX does **NOT** use normal EMA or SMA. It uses **Wilder's Smoothing**.

### Formula

For period = 14:

```
Smoothed Today = Smoothed Yesterday − (Smoothed Yesterday / Period) + Today Value
```

**Or equivalently:**

```
Smoothed Today = (Smoothed Yesterday × (Period - 1) + Today Value) / Period
```

### Why Wilder's Method?

- **Slower to react** than EMA
- **More stable** (less noise)
- **Standard for ADX** (as designed by Wilder)

### Example: Smoothing TR

**After 14 candles, we have accumulated smoothed values:**

**Previous Smoothed TR:**
```
Smoothed TR (yesterday) = 410.0
```

**Current TR (today):**
```
TR (today) = 32.0
```

**Calculation:**

```
Smoothed TR (today) = (410.0 × 13 + 32.0) / 14
                    = (5330.0 + 32.0) / 14
                    = 5362.0 / 14
                    = 382.86
```

**But wait!** This is after many candles. Let's see the **first 14 candles**:

**First 14 Candles (Initial Period):**

```
Candle 1:  TR = 25.0  → Smoothed = 25.0 (first value)
Candle 2:  TR = 30.0  → Smoothed = (25.0 × 13 + 30.0) / 14 = 25.36
Candle 3:  TR = 28.0  → Smoothed = (25.36 × 13 + 28.0) / 14 = 25.48
...
Candle 14: TR = 32.0  → Smoothed = 28.5 (after full 14 periods)
```

**After Candle 14 (Steady State):**

```
Candle 15: TR = 35.0  → Smoothed = (28.5 × 13 + 35.0) / 14 = 28.89
Candle 16: TR = 30.0  → Smoothed = (28.89 × 13 + 30.0) / 14 = 28.82
```

### Key Point

These smoothed values are **not single candles** — they are **accumulated pressure over time**.

**Example Accumulated Values (after many candles):**

```
Smoothed TR     = 410.0  (average of ~28.5 per candle over 14 periods)
Smoothed +DM    = 171.0  (average of ~12.2 per candle over 14 periods)
Smoothed -DM    = 72.0   (average of ~5.1 per candle over 14 periods)
```

---

## 4️⃣ Directional Indicators (+DI and −DI)

Now we convert smoothed values into **percentages**.

### Formula

```
+DI = 100 × (Smoothed +DM / Smoothed TR)
-DI = 100 × (Smoothed -DM / Smoothed TR)
```

### Why Percentages?

- **Normalizes** across different price levels
- **Comparable** across different markets
- **Range: 0-100** (easy to interpret)

### Example Using Smoothed Values

**From previous calculations:**

```
Smoothed +DM = 171.0
Smoothed -DM = 72.0
Smoothed TR  = 410.0
```

**Step-by-Step Calculation:**

```
+DI = 100 × (171.0 / 410.0)
   = 100 × 0.417
   = 41.7

-DI = 100 × (72.0 / 410.0)
   = 100 × 0.176
   = 17.6
```

**Result:**
```
+DI(14) = 41.7
-DI(14) = 17.6
```

### Interpretation

👉 **Buyers have ~42% strength, sellers have ~18% strength**

**What this means:**
- **Buyers dominate** (41.7 > 17.6)
- **Bullish pressure** is strong
- **Sellers are weak** (only 17.6)

**Visual:**
```
Buyer Strength:  ████████████████████ 41.7%
Seller Strength: ████████ 17.6%
```

---

## 5️⃣ DX (Directional Index)

DX measures **how far apart buyers and sellers are**.

### Formula

```
DX = 100 × |+DI − −DI| / (+DI + −DI)
```

### Why This Formula?

- **|+DI - -DI|**: Distance between buyer and seller strength
- **+DI + -DI**: Total directional movement
- **Ratio**: How much one side dominates
- **× 100**: Convert to percentage

### Example

**From previous calculations:**

```
+DI = 41.7
-DI = 17.6
```

**Step-by-Step Calculation:**

```
Step 1: Calculate difference
  |+DI − −DI| = |41.7 − 17.6|
              = |24.1|
              = 24.1

Step 2: Calculate sum
  +DI + −DI = 41.7 + 17.6
            = 59.3

Step 3: Calculate ratio
  Ratio = 24.1 / 59.3
        = 0.406

Step 4: Convert to percentage
  DX = 100 × 0.406
     = 40.6
```

**Result:**
```
DX = 40.6
```

### Interpretation

👉 **This is raw trend strength for ONE candle**

**What this means:**
- **40.6% separation** between buyers and sellers
- **Moderate to strong** trend
- **Not extreme** (would be 80+ for very strong trends)

**Note:** DX changes every candle. ADX smooths this.

---

## 6️⃣ ADX(14): Trend Strength (Smoothed DX)

ADX is just **Wilder-smoothed DX**.

### Formula

```
ADX = Wilder-smoothed DX (14 periods)
```

### Example: Calculating ADX

**After 14 candles, we have accumulated DX values:**

**Previous ADX (from yesterday):**
```
ADX (yesterday) = 31.9
```

**Current DX (from today):**
```
DX (today) = 40.6
```

**Step-by-Step Calculation:**

```
ADX (today) = (ADX (yesterday) × 13 + DX (today)) / 14
            = (31.9 × 13 + 40.6) / 14
            = (414.7 + 40.6) / 14
            = 455.3 / 14
            = 32.5
```

**Result:**
```
ADX(14) = 32.5
```

### Interpretation

👉 **Strong trend (above 25), but not extreme (below 40)**

**What this means:**
- **Above 25**: Real trend exists (not just noise)
- **Below 40**: Healthy trend (not overextended)
- **Result**: **Healthy trending market**

**ADX Scale:**
- **0-20**: Very weak trend (ranging)
- **20-25**: Weak trend (borderline)
- **25-40**: Moderate to strong trend ✅ (our case)
- **40-50**: Very strong trend
- **50+**: Extreme trend (rare)

---

## 7️⃣ ATR(14): Volatility

ATR uses the **same TR**, just smoothed.

### Formula

```
ATR = Wilder-smoothed TR (14 periods)
```

### Example: Calculating ATR

**From our earlier TR calculation:**

```
Current TR = 32.0
```

**After smoothing over 14 periods:**

**Previous ATR:**
```
ATR (yesterday) = 28.2
```

**Current TR:**
```
TR (today) = 32.0
```

**Step-by-Step Calculation:**

```
ATR (today) = (ATR (yesterday) × 13 + TR (today)) / 14
            = (28.2 × 13 + 32.0) / 14
            = (366.6 + 32.0) / 14
            = 398.6 / 14
            = 28.5
```

**Result:**
```
ATR(14) = 28.5
```

### Interpretation

👉 **Gold is moving ~28.5 points per 4H candle on average**

**What this means:**
- **High volatility**: Prices moving $28.50 per 4 hours
- **Risk assessment**: Expect $28.50 price swings
- **Stop loss sizing**: Use ATR to set stops (e.g., 2× ATR = $57 stop)

**Real-World Context:**
- If Gold is at $2650, ATR of $28.50 = **1.08% volatility per 4H**
- This is **moderate to high** volatility for Gold

---

## 8️⃣ ATR_MA(50): Volatility Baseline

This is just a **moving average of ATR**.

### Formula

```
ATR_MA(50) = Average of last 50 ATR values
```

### Example: Calculating ATR_MA

**We need 50 ATR values:**

```
ATR values (last 50 candles):
  28.5, 28.2, 27.9, 28.1, 28.4, 27.8, 28.0, ...
  ... (50 values total)
```

**Step-by-Step Calculation:**

```
ATR_MA(50) = (28.5 + 28.2 + 27.9 + ... + 24.5) / 50
           = 1205.0 / 50
           = 24.1
```

**Result:**
```
ATR_MA(50) = 24.1
```

### Comparison

```
Current ATR  = 28.5
Baseline ATR = 24.1
```

**Interpretation:**

👉 **Volatility is 18% above normal** (28.5 / 24.1 = 1.18)

**What this means:**
- **Current volatility > Baseline**: High volatility period
- **Market is more volatile** than usual
- **Risk is elevated**

**Regime Classification:**
- If ATR (28.5) > ATR_MA (24.1) → **"High Vol"**
- If ATR (28.5) < ATR_MA (24.1) → **"Low Vol"**

In our case: **"High Vol"** ✅

---

## 9️⃣ SMA(200): Trend Alignment

Simple moving average of **200 closes**.

### Formula

```
SMA(200) = (Sum of last 200 closes) / 200
```

### Example: Calculating SMA(200)

**We need 200 close prices:**

```
Close prices (last 200 candles):
  2652.0, 2648.5, 2645.0, 2642.0, 2638.5, ...
  ... (200 values total)
```

**Step-by-Step Calculation:**

```
SMA(200) = (2652.0 + 2648.5 + 2645.0 + ... + 2500.0) / 200
         = 519000.0 / 200
         = 2595.0
```

**Result:**
```
SMA(200) = 2595.0
```

### Comparison

```
Current Close = 2652.0
SMA(200)      = 2595.0
```

**Interpretation:**

👉 **Price is $57 above long-term average (2.2% above)**

**What this means:**
- **Price > SMA(200)**: Bullish alignment
- **Long-term trend is up**
- **Market is in uptrend**

**Regime Classification:**
- Price > SMA(200) → **Bullish trend**
- Price < SMA(200) → **Bearish trend**

In our case: **Bullish** ✅

---

## 🔗 Putting All Numbers Together

### Complete Example: One 4H Candle

**Input Data:**
```
Previous Candle:
  High:  2642.0
  Low:   2615.0
  Close: 2628.0

Current Candle (2025-12-12 08:00-12:00):
  Open:  2630.0
  High:  2660.0
  Low:   2635.0
  Close: 2652.0
```

**Calculated Indicators:**

| Indicator | Value | Calculation |
|-----------|-------|-------------|
| **TR** | 32.0 | max(25.0, 32.0, 7.0) |
| **+DM** | 18.0 | UpMove (18.0) > DownMove (-20.0) |
| **-DM** | 0.0 | DownMove is negative |
| **Smoothed TR** | 410.0 | Wilder-smoothed over 14 periods |
| **Smoothed +DM** | 171.0 | Wilder-smoothed over 14 periods |
| **Smoothed -DM** | 72.0 | Wilder-smoothed over 14 periods |
| **+DI** | 41.7 | 100 × (171.0 / 410.0) |
| **-DI** | 17.6 | 100 × (72.0 / 410.0) |
| **DX** | 40.6 | 100 × |41.7 - 17.6| / (41.7 + 17.6) |
| **ADX** | 32.5 | Wilder-smoothed DX |
| **ATR** | 28.5 | Wilder-smoothed TR |
| **ATR_MA(50)** | 24.1 | Average of last 50 ATR values |
| **SMA(200)** | 2595.0 | Average of last 200 closes |

### Final Regime Classification

**Using our calculated values:**

```
Step 1: Trend Classification
  ADX = 32.5
  ADX > 25? ✅ Yes
  Result: "Trending"

Step 2: Volatility Classification
  ATR = 28.5
  ATR_MA = 24.1
  ATR > ATR_MA? ✅ Yes (28.5 > 24.1)
  Result: "High Vol"

Step 3: Combine
  Regime = "Trending - High Vol"
```

**Result:**
```
Regime: "Trending - High Vol"
```

---

## 📊 Summary Table: What Each Number Means

| Indicator | Value | Meaning |
|-----------|-------|---------|
| **ADX 32.5** | Above 25 | Strong trend exists |
| **+DI 41.7** | High | Buyers dominant |
| **-DI 17.6** | Low | Sellers weak |
| **ATR 28.5** | High | High volatility |
| **ATR_MA 24.1** | Baseline | Normal volatility level |
| **Price > SMA200** | $2652 > $2595 | Bullish alignment |

**Combined Interpretation:**
- **Strong uptrend** (ADX 32.5, +DI 41.7)
- **High volatility** (ATR 28.5 > ATR_MA 24.1)
- **Bullish alignment** (Price above SMA200)
- **Regime**: "Trending - High Vol" ✅

---

## 🧠 Why These Numbers Are Trustworthy

### 1. Uses Hundreds of Candles

**ADX calculation:**
- Needs **14 periods** for initial calculation
- Then **14 more periods** for smoothing
- **Total: 28+ candles** minimum

**ATR_MA calculation:**
- Needs **50 periods** of ATR
- Each ATR needs **14 periods** of TR
- **Total: 64+ candles** minimum

**SMA(200) calculation:**
- Needs **200 closes**
- **Total: 200 candles** minimum

### 2. Smoothed (Less Noise)

- **Wilder's smoothing** reduces false signals
- **Moving averages** filter out noise
- **Stable values** (don't jump around)

### 3. No Look-Ahead Bias

- **Only uses past data**
- **Backward matching** for signals
- **No future information** used

### 4. Industry Standard

- **Same formulas** as TradingView, MetaTrader, Bloomberg
- **Institutional-grade** calculations
- **Proven methodology** (since 1978)

---

## 🎯 Final Mental Shortcut (Remember This)

| Indicator | Question It Answers |
|-----------|-------------------|
| **+DI / -DI** | *Who is winning?* (Buyers or Sellers) |
| **ADX** | *How strong is the fight?* (Trend strength) |
| **ATR** | *How violent is the movement?* (Volatility) |
| **ATR_MA** | *Is volatility normal?* (Volatility baseline) |
| **SMA(200)** | *Which side is favored long-term?* (Trend direction) |

**Combined:**
- **ADX + ATR comparison** → Regime classification
- **+DI / -DI** → Trend direction (bullish/bearish)
- **SMA(200)** → Long-term trend confirmation

---

## 🔄 How Timeframe Affects Calculations

### Same Market, Different Timeframes

**Market:** Gold trending from $1950 to $2000 over 3 days

**1-Hour Bars:**
```
Each bar: $5-10 moves
+DM: ~$4.20 per hour
TR: ~$7.80 per hour
+DI: 53.85
ADX: 28.7 (moderate trend)
```

**4-Hour Bars:**
```
Each bar: $15-20 moves
+DM: ~$18.50 per 4 hours
TR: ~$22.30 per 4 hours
+DI: 82.96
ADX: 35.2 (stronger trend)
```

**1-Day Bars:**
```
Each bar: $30-50 moves
+DM: ~$35.20 per day
TR: ~$42.50 per day
+DI: 82.82
ADX: 42.5 (strongest trend)
```

**Key Insight:**
- **Higher timeframe = Higher ADX** (smoother, less noise)
- **Same trend, different confidence levels**

---

## 💡 Practical Example: Complete Calculation Flow

### Scenario: Gold 4H Candle Analysis

**Input: 15 consecutive 4H candles**

**Candle 1-14 (History):**
```
(Previous candles used for smoothing)
```

**Candle 15 (Current):**
```
Open:  2630.0
High:  2660.0
Low:   2635.0
Close: 2652.0
```

**Step 1: Calculate TR**
```
TR = max(25.0, 32.0, 7.0) = 32.0
```

**Step 2: Calculate +DM and -DM**
```
+DM = 18.0
-DM = 0.0
```

**Step 3: Smooth (using previous 14 candles)**
```
Smoothed TR  = 410.0
Smoothed +DM = 171.0
Smoothed -DM = 72.0
```

**Step 4: Calculate +DI and -DI**
```
+DI = 100 × (171.0 / 410.0) = 41.7
-DI = 100 × (72.0 / 410.0) = 17.6
```

**Step 5: Calculate DX**
```
DX = 100 × |41.7 - 17.6| / (41.7 + 17.6) = 40.6
```

**Step 6: Calculate ADX (smooth DX)**
```
ADX = (31.9 × 13 + 40.6) / 14 = 32.5
```

**Step 7: Calculate ATR (smooth TR)**
```
ATR = (28.2 × 13 + 32.0) / 14 = 28.5
```

**Step 8: Calculate ATR_MA (average of 50 ATRs)**
```
ATR_MA = Average of last 50 ATR values = 24.1
```

**Step 9: Calculate SMA(200) (average of 200 closes)**
```
SMA(200) = Average of last 200 closes = 2595.0
```

**Step 10: Classify Regime**
```
ADX (32.5) > 25? ✅ → "Trending"
ATR (28.5) > ATR_MA (24.1)? ✅ → "High Vol"
Result: "Trending - High Vol"
```

---

## ✅ Verification: Why These Numbers Make Sense

**Check 1: ADX Scale**
- ADX = 32.5 → Between 25-40 → Moderate to strong trend ✅
- Makes sense: Gold is trending up

**Check 2: DI Balance**
- +DI (41.7) > -DI (17.6) → Buyers dominate ✅
- Makes sense: Uptrend confirmed

**Check 3: Volatility**
- ATR (28.5) > ATR_MA (24.1) → High volatility ✅
- Makes sense: Trending markets often have higher volatility

**Check 4: Trend Alignment**
- Price ($2652) > SMA(200) ($2595) → Bullish ✅
- Makes sense: Price above long-term average

**All checks pass!** ✅

---

This is **institutional-grade math**, not heuristics. Every number has a **precise formula** and **clear meaning**.

