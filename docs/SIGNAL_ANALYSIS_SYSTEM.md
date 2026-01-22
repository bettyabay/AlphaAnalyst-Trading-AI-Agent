# Signal Analysis & Validation System

## Overview

This system provides comprehensive analysis, backtesting, and validation capabilities for signal provider signals. It ensures accuracy by comparing automated analysis with manual Excel analysis.

## Components

### 1. Database Tables

**Location**: `sql/create_analysis_tables.sql`

- `analysis_results` - Stores automated analysis outputs
- `validation_reports` - Stores comparison results between manual and automated
- `backtest_results` - Stores backtesting simulation results
- `daily_progress_log` - Tracks daily progress for catch-up
- `provider_performance_summary` - Aggregated metrics per provider

### 2. Core Engines

#### Signal Analyzer (`signal_analyzer.py`)
- Analyzes signals to determine TP/SL hits
- Calculates performance metrics
- Generates provider-level statistics

**Usage**:
```python
from tradingagents.dataflows.signal_analyzer import SignalAnalyzer

analyzer = SignalAnalyzer()
result = analyzer.analyze_signal(signal_dict)
analyzer.save_analysis_result(result)

# Calculate provider metrics
metrics = analyzer.calculate_provider_metrics("Provider Name")
```

#### Signal Backtester (`signal_backtester.py`)
- Simulates historical trades
- Calculates P&L and performance metrics
- Supports partial position exits (TP1, TP2, TP3)

**Usage**:
```python
from tradingagents.dataflows.signal_backtester import SignalBacktester

backtester = SignalBacktester()
result = backtester.backtest_signal(
    signal=signal_dict,
    initial_capital=10000.0,
    position_size_percent=10.0
)
backtester.save_backtest_result(result)
```

#### Validation Engine (`validation_engine.py`)
- Compares automated vs manual analysis
- Detects discrepancies
- Generates validation reports

**Usage**:
```python
from tradingagents.dataflows.validation_engine import ValidationEngine

validator = ValidationEngine()
validation = validator.validate_signal(
    signal_id=123,
    automated_result=auto_result,
    manual_result=manual_result
)
validator.save_validation_report(validation)

# Validate from Excel
validations = validator.validate_from_excel("path/to/excel.xlsx")
```

#### Daily Reporter (`daily_reporter.py`)
- Tracks daily progress
- Generates progress reports
- Monitors completion status

**Usage**:
```python
from tradingagents.dataflows.daily_reporter import DailyReporter

reporter = DailyReporter()
report = reporter.generate_daily_report()
reporter.save_daily_report(report)
```

### 3. Batch Processing

**Location**: `run_signal_analysis.py`

Provides batch processing functions:
- `run_analysis_for_all_signals()` - Analyze all signals
- `run_backtest_for_all_signals()` - Backtest all signals
- `generate_daily_report_and_save()` - Generate daily report
- `calculate_provider_metrics()` - Calculate provider metrics

## Workflow

### Step 1: Run Automated Analysis
```python
from tradingagents.dataflows.run_signal_analysis import run_analysis_for_all_signals

# Analyze all signals
summary = run_analysis_for_all_signals()
print(f"Analyzed: {summary['analyzed']}/{summary['total_signals']}")
```

### Step 2: Run Backtesting
```python
from tradingagents.dataflows.run_signal_analysis import run_backtest_for_all_signals

# Backtest all signals
summary = run_backtest_for_all_signals(
    initial_capital=10000.0,
    position_size_percent=10.0
)
print(f"Total P&L: ${summary['total_pnl']}")
```

### Step 3: Manual Analysis (Excel)
1. Export signals to Excel using existing export functionality
2. Manually mark TP/SL hits in Excel
3. Save completed Excel file

### Step 4: Validate (Cross-Check)
```python
from tradingagents.dataflows.validation_engine import ValidationEngine

validator = ValidationEngine()
validations = validator.validate_from_excel("path/to/manual_analysis.xlsx")

# Get summary
summary = validator.get_validation_summary()
print(f"Match Rate: {summary['match_rate']}%")
```

### Step 5: Generate Daily Report
```python
from tradingagents.dataflows.run_signal_analysis import generate_daily_report_and_save

report = generate_daily_report_and_save()
print(f"Currencies: {report['currencies_ingested']}/28")
print(f"Signals Analyzed: {report['signals_analyzed']}")
```

## Database Setup

Run the SQL script to create tables:
```sql
-- Execute: sql/create_analysis_tables.sql
```

## Key Metrics

### Provider Performance Metrics
- Total Signals
- TP1/TP2/TP3 Success Rates
- Win Rate
- Risk/Reward Ratio
- Average Hold Time
- Total P&L
- Sharpe Ratio

### Validation Metrics
- Match Rate (Automated vs Manual)
- Discrepancy Types (TP_MISMATCH, SL_MISMATCH, TIMESTAMP_MISMATCH)
- Discrepancy Severity (CRITICAL, WARNING, MINOR)

### Daily Progress Metrics
- Currencies: X/28 ingested
- Indices: Started (Yes/No)
- Signals: Total, New Today, Parsing Success Rate
- Analysis: Analyzed, Backtesting Complete %
- Validation: Match Rate
- Issues: Symbol Mismatches, Data Gaps, Parsing Errors

## Timezone

All calculations use **GMT+4 (Asia/Dubai)** timezone as per requirements.

## Notes

- All timestamps are stored in ISO format with timezone
- Market data is fetched from Supabase (not external APIs)
- Analysis uses 1-minute interval data for precision
- Backtesting supports partial position exits (33% TP1, 33% TP2, 34% TP3)
- Validation allows ±5 minute tolerance for timestamp comparison

