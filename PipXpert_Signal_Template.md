# PipXpert Signal Upload Template

## Required Excel Format

Your Excel file **must** have these columns (in any order):

### Required Columns:
1. **Date** (or DateTime, Timestamp)
   - Format: `YYYY-MM-DD HH:MM:SS` or `MM/DD/YYYY HH:MM:SS`
   - Example: `2025-06-06 14:16:37`

2. **Action**
   - Values: `buy` or `sell` (case-insensitive)

3. **Currency Pair** (or Symbol, Pair)
   - Examples: `EURUSD`, `GBPUSD`, `XAUUSD`, `BTCUSD`
   - **IMPORTANT**: This MUST be the actual trading symbol, NOT "TP1", "TP2", etc.

4. **Entry Price** (or Entry)
   - Numeric value
   - Example: `1.0850`

### Optional Columns:
- **Stop Loss**: Numeric value (example: `1.0900`)
- **Target 1**: Numeric value (example: `1.0800`)
- **Target 2**: Numeric value (example: `1.0750`)
- **Target 3**: Numeric value (example: `1.0700`)
- **Target 4**: Numeric value
- **Target 5**: Numeric value

---

## Example Correct Format

| Date | Action | **Currency Pair** | Entry Price | Stop Loss | Target 1 | Target 2 | Target 3 |
|----------------------|--------|-------------|-------------|-----------|----------|----------|----------|
| 2025-06-06 14:16:37 | sell | **EURUSD** | 1.0850 | 1.0900 | 1.0800 | 1.0750 | 1.0700 |
| 2025-06-07 10:30:00 | buy | **GBPUSD** | 1.2650 | 1.2600 | 1.2700 | 1.2750 | 1.2800 |
| 2025-06-08 08:15:00 | sell | **XAUUSD** | 3345.10 | 3333.10 | 3346.60 | 3348.10 | 3357.10 |

---

## Common Mistakes

### ❌ WRONG - Missing Currency Pair Column
| Date | Action | Entry Price | Stop Loss | Target 1 | Target 2 |
|----------------------|--------|-------------|-----------|----------|----------|
| 2025-06-06 14:16:37 | sell | 1.0850 | 1.0900 | 1.0800 | 1.0750 |

**Problem**: System reads "Entry Price" as "Currency Pair" and "Stop Loss" as "Entry Price", etc.

### ❌ WRONG - TP1/TP2 in Currency Pair Column
| Date | Action | Currency Pair | Entry Price | Stop Loss | Target 1 |
|----------------------|--------|---------------|-------------|-----------|----------|
| 2025-06-06 14:16:37 | sell | **TP1** | 1.0850 | 1.0900 | 1.0800 |

**Problem**: "TP1" is NOT a valid trading symbol. It should be "EURUSD", "GBPUSD", etc.

---

## How to Fix Your Current File

1. Open your Excel file
2. Check if you have a "Currency Pair" (or "Symbol") column
3. If missing, insert a new column after "Action"
4. Name it "Currency Pair"
5. Fill in the correct trading symbols for each row
6. Save and re-upload

---

## Supported Symbol Examples

### Forex Pairs:
- EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD
- EURGBP, EURJPY, GBPJPY, AUDJPY

### Precious Metals:
- XAUUSD (Gold), XAGUSD (Silver)

### Crypto:
- BTCUSD, ETHUSD, BNBUSD

### Indices:
- NAS100, US30, SPX, DJI

