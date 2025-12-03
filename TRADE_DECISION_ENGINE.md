# Trade Decision Engine - Implementation Guide

## Overview

The **TradeDecisionEngine** has been successfully implemented to evaluate all conditions for a "TRADE YES" decision. This engine integrates with the QUANTUMTRADER system and enforces strict risk management rules.

## Location

**File**: `tradingagents/agents/utils/trading_engine.py`
**Class**: `TradeDecisionEngine`

## Features

### 1. Complete Condition Evaluation

The engine evaluates 5 core conditions before allowing a trade:

1. **Composite Score ≥ 6.5**
   - Checks QUANTUMTRADER composite score
   - Must meet or exceed 6.5 threshold (higher than standard PASS threshold of 6.0)

2. **All Phase 1 Gates Passed**
   - Volume spike detection
   - Trend confirmation (SMA50 > SMA200)
   - RSI momentum check
   - Liquidity verification

3. **R:R Ratio ≥ 1:2 Achievable**
   - Calculates ATR-based stop loss (1.5x ATR)
   - Sets targets at 2x and 2.5x stop distance
   - Verifies minimum 1:2 risk:reward is achievable

4. **Position Size Within $2,000 Exposure Limit**
   - Calculates position size based on:
     - Risk per share (ATR-based stop distance)
     - Maximum risk per trade ($100 = 1% of $10k account)
     - Exposure limit ($2,000 per trade)
   - Uses the most conservative of these constraints

5. **No Conflicting Daily Trend**
   - Analyzes 5-minute trend direction
   - Compares with daily timeframe trend
   - Rejects counter-trend setups

### 2. Direction Decision Logic

- **BUY**: 5-min trend UP and aligned with daily (UP or NEUTRAL)
- **SELL**: 5-min trend DOWN and aligned with daily (DOWN or NEUTRAL)  
- **NO TRADE**: Trend conflict detected

### 3. Risk Management Overlay

- **Max Daily Loss**: $400 (4% of $10k account)
- **Max Concurrent Trades**: 3 trades simultaneously
- **Auto-Close Time**: 4:00 PM ET (all positions)
- **Position Exposure**: $2,000 maximum per trade

## Usage

### Basic Usage

```python
from tradingagents.agents.utils.trading_engine import TradeDecisionEngine

# Initialize engine
engine = TradeDecisionEngine(
    max_exposure=2000.0,      # $2,000 per trade
    max_daily_loss=400.0      # $400 daily loss limit
)

# Evaluate trade decision
decision = engine.evaluate_trade_decision("AAPL")

# Check result
print(decision["trade_decision"])  # "TRADE YES" or "NO TRADE"
print(decision["direction"])       # "UP", "DOWN", or "CONFLICT"
print(decision["reason"])          # Detailed explanation
```

### Decision Output Structure

```python
{
    "symbol": "AAPL",
    "timestamp": "2025-12-03 12:48:34",
    "trade_decision": "TRADE YES" or "NO TRADE",
    "direction": "UP" / "DOWN" / "CONFLICT",
    "reason": "All conditions met" or detailed failure reasons,
    
    "conditions": {
        "composite_score": {
            "value": 6.8,
            "threshold": 6.5,
            "pass": True,
            "details": {...}
        },
        "phase1_gates": {
            "pass": True,
            "reason": "Passed all criteria",
            "metrics": {...}
        },
        "rr_ratio": {
            "achievable": True,
            "ratio": 2.0,
            "current_price": 269.79,
            "stop_loss": 268.50,
            "target1": 272.08,
            "target2": 273.10
        },
        "position_size": {
            "calculable": True,
            "exposure": 1800.00,
            "recommended_shares": 6,
            "risk_amount": 100.00,
            "reward_amount": 200.00
        },
        "trend_alignment": {
            "conflict": False,
            "m5_trend": "UP",
            "daily_trend": "UP"
        }
    },
    
    "risk_metrics": {
        "can_trade": True,
        "active_trades": 1,
        "max_concurrent": 3,
        "max_daily_loss": 400.0,
        "auto_close_time": "4:00 PM ET"
    },
    
    "recommendation": {  # Only if TRADE YES
        "action": "BUY",
        "entry_price": 269.79,
        "stop_loss": 268.50,
        "target1": 272.08,
        "target2": 273.10,
        "position_size_shares": 6,
        "exposure": 1800.00,
        "risk_amount": 100.00,
        "reward_amount": 200.00,
        "rr_ratio": 2.0
    }
}
```

## Integration with QUANTUMTRADER

The decision engine is integrated into the QUANTUMTRADER prompt system. The prompt now includes:

```
QUANTUMTRADER v0.1 - TRADE DECISION ENGINE
Command: EVALUATE_TRADE_DECISION [SYMBOL] [TIMESTAMP]

CONDITIONS FOR "TRADE YES":
1. Composite_Score ≥ 6.5
2. All Phase 1 gates passed ✓
3. R:R ratio achievable ≥ 1:2
4. Position size calculable within $2,000 exposure limit
5. No conflicting daily trend (avoid counter-trend)

DIRECTION DECISION:
- If 5-min trend = UP and aligned with higher timeframes → BUY
- If 5-min trend = DOWN and aligned with higher timeframes → SELL
- If conflicting → "NO TRADE - Trend conflict"

RISK MANAGEMENT OVERLAY:
- Max daily loss: $400 (4% of $10k)
- Max concurrent trades: 3
- Auto-close all positions at 4:00 PM ET
```

## Key Methods

### `evaluate_trade_decision(symbol, command_ts=None)`
Main method that evaluates all conditions and returns complete decision.

### `_calculate_rr_ratio(symbol, quantum_result)`
Calculates risk:reward ratio using ATR-based stop loss and targets.

### `_calculate_position_size(symbol, quantum_result, rr_result)`
Determines position size within exposure limits based on risk per share.

### `_check_trend_alignment(symbol, quantum_result)`
Analyzes 5-minute and daily trends to detect conflicts.

### `_determine_direction(trend_result)`
Determines trade direction (UP/DOWN/CONFLICT) based on trend alignment.

### `_check_risk_overlay(symbol)`
Verifies risk management constraints (concurrent trades, daily loss limits).

## Configuration

You can customize the engine parameters:

```python
engine = TradeDecisionEngine(
    max_exposure=2000.0,        # Change exposure limit
    max_daily_loss=400.0        # Change daily loss limit
)

# Change composite score threshold
engine.composite_score_threshold = 7.0  # More strict

# Change minimum R:R ratio
engine.min_rr_ratio = 2.5  # Require 1:2.5 instead of 1:2
```

## Example: Complete Workflow

```python
from tradingagents.agents.utils.trading_engine import TradeDecisionEngine

# Initialize
engine = TradeDecisionEngine()

# Evaluate
decision = engine.evaluate_trade_decision("AAPL")

# Process decision
if decision["trade_decision"] == "TRADE YES":
    rec = decision["recommendation"]
    print(f"✅ TRADE YES: {rec['action']} {decision['symbol']}")
    print(f"   Entry: ${rec['entry_price']:.2f}")
    print(f"   Stop: ${rec['stop_loss']:.2f}")
    print(f"   Target 1: ${rec['target1']:.2f}")
    print(f"   Target 2: ${rec['target2']:.2f}")
    print(f"   Shares: {rec['position_size_shares']}")
    print(f"   Exposure: ${rec['exposure']:.2f}")
    print(f"   Risk: ${rec['risk_amount']:.2f}")
    print(f"   Reward: ${rec['reward_amount']:.2f}")
    print(f"   R:R: 1:{rec['rr_ratio']:.2f}")
else:
    print(f"❌ NO TRADE: {decision['reason']}")
    
    # Show which conditions failed
    for cond_name, cond_data in decision["conditions"].items():
        if not cond_data.get("pass", False):
            print(f"   - {cond_name}: FAILED")

# Cleanup
engine.close()
```

## Dependencies

The engine requires:
- `FeatureLab` - For QUANTUMTRADER metrics
- `VolumeScreeningEngine` - For Phase 1 gate checks
- `TradeExecutionService` - For risk overlay checks
- `fetch_ohlcv` - For market data access

## Error Handling

The engine gracefully handles:
- Missing data (returns detailed error in condition)
- Calculation failures (marks condition as failed)
- Database connection issues (falls back gracefully)

## Next Steps

To use this in your trading workflow:

1. **Import the engine** in your trading script
2. **Evaluate decision** before placing any trade
3. **Check all conditions** before executing
4. **Use recommendation data** for precise entry/exit levels
5. **Monitor risk metrics** throughout the day

## Notes

- The composite score threshold is **6.5** (higher than standard PASS of 6.0)
- Position sizing is conservative (uses minimum of risk-based and exposure-based calculations)
- Trend analysis uses EMA for 5-minute and SMA for daily timeframes
- Auto-close at 4:00 PM ET is a policy that needs to be enforced separately (not automated in engine)

## Questions?

Refer to:
- `tradingagents/agents/utils/trading_engine.py` - Full implementation
- `tradingagents/dataflows/feature_lab.py` - QUANTUMTRADER metrics
- `TRADE_DECISION_ENGINE.md` - This documentation
