"""
Batch Signal Analysis Runner
Runs automated analysis and generates reports for all signals.
Supports instrument-specific analysis with market data validation.
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pytz

from tradingagents.database.config import get_supabase
from tradingagents.dataflows.signal_analyzer import SignalAnalyzer
from tradingagents.dataflows.signal_backtester import SignalBacktester
from tradingagents.dataflows.daily_reporter import DailyReporter


def get_available_instruments() -> List[str]:
    """
    Get list of unique instruments (symbols) from signal_provider_signals table.
    
    Returns:
        List of unique instrument symbols
    """
    supabase = get_supabase()
    if not supabase:
        return []
    
    try:
        # Get distinct symbols
        result = supabase.table('signal_provider_signals').select('symbol').execute()
        if result.data:
            symbols = list(set([row['symbol'] for row in result.data]))
            symbols.sort()
            return symbols
        return []
    except Exception as e:
        print(f"Error fetching instruments: {e}")
        return []


def check_market_data_availability(symbol: str) -> Dict:
    """
    Check if market data exists for the given instrument.
    Tries multiple symbol format variants to handle different storage formats.
    
    Args:
        symbol: Instrument symbol to check
        
    Returns:
        Dictionary with availability status and details
    """
    from tradingagents.dataflows.ingestion_pipeline import get_1min_table_name_for_symbol
    
    supabase = get_supabase()
    if not supabase:
        return {"available": False, "error": "Database not available"}
    
    try:
        # Determine which table to check
        table_name = get_1min_table_name_for_symbol(symbol)
        symbol_upper = symbol.upper()
        
        # Build list of symbol variants to try (similar to market_data_service.py)
        symbols_to_try = [symbol_upper]
        
        # Handle Gold/XAUUSD variants
        if symbol_upper == "XAUUSD" or "XAU" in symbol_upper:
            symbols_to_try = ["C:XAUUSD", "XAUUSD", "^XAUUSD", "GOLD"]
        elif symbol_upper == "C:XAUUSD":
            symbols_to_try = ["C:XAUUSD", "XAUUSD", "^XAUUSD", "GOLD"]
        elif symbol_upper == "^XAUUSD":
            symbols_to_try = ["^XAUUSD", "C:XAUUSD", "XAUUSD", "GOLD"]
        elif symbol_upper == "GOLD":
            symbols_to_try = ["GOLD", "C:XAUUSD", "^XAUUSD", "XAUUSD"]
        # Handle currency pairs - try with and without C: prefix
        elif len(symbol_upper) >= 6 and len(symbol_upper) <= 7 and not symbol_upper.startswith("C:") and not symbol_upper.startswith("^") and not symbol_upper.startswith("I:"):
            # Likely a currency pair without prefix
            symbols_to_try = [symbol_upper, f"C:{symbol_upper}"]
        elif symbol_upper.startswith("C:") and len(symbol_upper) >= 8:
            # Has C: prefix, also try without
            base_symbol = symbol_upper[2:]  # Remove "C:" prefix
            symbols_to_try = [symbol_upper, base_symbol]
        
        # Try each symbol variant
        found_symbol = None
        for try_symbol in symbols_to_try:
            try:
                result = supabase.table(table_name).select('timestamp').eq('symbol', try_symbol).limit(1).execute()
                if result.data and len(result.data) > 0:
                    found_symbol = try_symbol
                    break
            except Exception:
                continue
        
        if found_symbol:
            # Get date range using the found symbol
            date_result = supabase.table(table_name).select('timestamp').eq('symbol', found_symbol).order('timestamp', desc=False).limit(1).execute()
            latest_result = supabase.table(table_name).select('timestamp').eq('symbol', found_symbol).order('timestamp', desc=True).limit(1).execute()
            
            earliest = date_result.data[0]['timestamp'] if date_result.data and len(date_result.data) > 0 else None
            latest = latest_result.data[0]['timestamp'] if latest_result.data and len(latest_result.data) > 0 else None
            
            return {
                "available": True,
                "table": table_name,
                "symbol_found": found_symbol,
                "earliest_date": earliest,
                "latest_date": latest
            }
        else:
            # Try to get distinct symbols from the table to help with debugging
            try:
                distinct_result = supabase.table(table_name).select('symbol').limit(100).execute()
                distinct_symbols = list(set([row.get('symbol', '') for row in (distinct_result.data or [])]))
                distinct_symbols.sort()
                sample_symbols = distinct_symbols[:10]  # Show first 10
                return {
                    "available": False,
                    "table": table_name,
                    "symbols_tried": symbols_to_try,
                    "sample_symbols_in_table": sample_symbols,
                    "error": f"No market data found for {symbol} (tried: {', '.join(symbols_to_try)}) in {table_name}. Sample symbols in table: {', '.join(sample_symbols) if sample_symbols else 'none'}"
                }
            except Exception:
                return {
                    "available": False,
                    "table": table_name,
                    "symbols_tried": symbols_to_try,
                    "error": f"No market data found for {symbol} (tried: {', '.join(symbols_to_try)}) in {table_name}"
                }
    except Exception as e:
        return {
            "available": False,
            "error": f"Error checking market data: {str(e)}"
        }


def run_analysis_for_all_signals(
    provider_name: Optional[str] = None,
    symbol: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    save_results: bool = True
) -> Dict:
    """
    Run analysis for all signals.
    
    Args:
        provider_name: Optional provider name filter
        symbol: Optional instrument/symbol filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        save_results: Whether to save results to database
        
    Returns:
        Dictionary with analysis summary
    """
    supabase = get_supabase()
    if not supabase:
        return {"error": "Database not available"}
    
    analyzer = SignalAnalyzer()
    
    # Fetch signals
    query = supabase.table('signal_provider_signals').select('*')
    
    if provider_name:
        query = query.eq('provider_name', provider_name)
    if symbol:
        query = query.eq('symbol', symbol.upper())
    if start_date:
        query = query.gte('signal_date', start_date.isoformat())
    if end_date:
        query = query.lte('signal_date', end_date.isoformat())
    
    result = query.execute()
    signals = result.data if result.data else []
    
    if not signals:
        return {"error": "No signals found"}
    
    print(f"Analyzing {len(signals)} signals...")
    
    analyzed_count = 0
    error_count = 0
    
    for i, signal in enumerate(signals, 1):
        try:
            symbol = signal.get('symbol', 'N/A')
            signal_date = signal.get('signal_date', 'N/A')
            print(f"Analyzing signal {i}/{len(signals)}: {symbol} @ {signal_date}")
            
            analysis_result = analyzer.analyze_signal(signal)
            
            if 'error' not in analysis_result:
                if save_results:
                    analyzer.save_analysis_result(analysis_result)
                analyzed_count += 1
            else:
                error_msg = str(analysis_result.get('error', 'Unknown error'))
                # Remove emojis and special characters that can't be encoded
                error_msg = error_msg.encode('ascii', 'ignore').decode('ascii')
                print(f"  Error: {error_msg}")
                error_count += 1
                
        except Exception as e:
            error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            print(f"  Exception: {error_msg}")
            error_count += 1
    
    return {
        'total_signals': len(signals),
        'analyzed': analyzed_count,
        'errors': error_count,
        'success_rate': (analyzed_count / len(signals) * 100) if signals else 0
    }


def run_backtest_for_all_signals(
    provider_name: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    initial_capital: float = 10000.0,
    position_size_percent: float = 10.0,
    save_results: bool = True
) -> Dict:
    """
    Run backtesting for all signals.
    
    Args:
        provider_name: Optional provider name filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        initial_capital: Starting capital
        position_size_percent: Position size percentage
        save_results: Whether to save results to database
        
    Returns:
        Dictionary with backtest summary
    """
    supabase = get_supabase()
    if not supabase:
        return {"error": "Database not available"}
    
    backtester = SignalBacktester()
    
    # Fetch signals
    query = supabase.table('signal_provider_signals').select('*')
    
    if provider_name:
        query = query.eq('provider_name', provider_name)
    if start_date:
        query = query.gte('signal_date', start_date.isoformat())
    if end_date:
        query = query.lte('signal_date', end_date.isoformat())
    
    result = query.execute()
    signals = result.data if result.data else []
    
    if not signals:
        return {"error": "No signals found"}
    
    print(f"Backtesting {len(signals)} signals...")
    
    backtested_count = 0
    error_count = 0
    total_pnl = 0.0
    
    for i, signal in enumerate(signals, 1):
        try:
            symbol = signal.get('symbol', 'N/A')
            signal_date = signal.get('signal_date', 'N/A')
            print(f"Backtesting signal {i}/{len(signals)}: {symbol} @ {signal_date}")
            
            backtest_result = backtester.backtest_signal(
                signal=signal,
                initial_capital=initial_capital,
                position_size_percent=position_size_percent
            )
            
            if 'error' not in backtest_result:
                if save_results:
                    backtester.save_backtest_result(backtest_result)
                backtested_count += 1
                total_pnl += backtest_result.get('net_profit_loss', 0)
            else:
                error_msg = str(backtest_result.get('error', 'Unknown error'))
                # Remove emojis and special characters that can't be encoded
                error_msg = error_msg.encode('ascii', 'ignore').decode('ascii')
                print(f"  Error: {error_msg}")
                error_count += 1
                
        except Exception as e:
            error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            print(f"  Exception: {error_msg}")
            error_count += 1
    
    return {
        'total_signals': len(signals),
        'backtested': backtested_count,
        'errors': error_count,
        'success_rate': (backtested_count / len(signals) * 100) if signals else 0,
        'total_pnl': round(total_pnl, 2)
    }


def generate_daily_report_and_save() -> Dict:
    """
    Generate and save daily progress report.
    
    Returns:
        Daily report dictionary
    """
    reporter = DailyReporter()
    report = reporter.generate_daily_report()
    
    if 'error' not in report:
        reporter.save_daily_report(report)
        print("Daily report generated and saved successfully")
    else:
        print(f"Error generating report: {report.get('error')}")
    
    return report


def calculate_provider_metrics(provider_name: str) -> Dict:
    """
    Calculate and save provider performance metrics.
    
    Args:
        provider_name: Name of the signal provider
        
    Returns:
        Provider metrics dictionary
    """
    analyzer = SignalAnalyzer()
    metrics = analyzer.calculate_provider_metrics(provider_name)
    
    if 'error' not in metrics and get_supabase():
        try:
            # Save to provider_performance_summary table
            get_supabase().table('provider_performance_summary').upsert(metrics).execute()
            print(f"Provider metrics calculated and saved for: {provider_name}")
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    return metrics


if __name__ == "__main__":
    """
    Example usage:
    
    # Run analysis for all signals
    analysis_summary = run_analysis_for_all_signals()
    print(analysis_summary)
    
    # Run backtesting
    backtest_summary = run_backtest_for_all_signals()
    print(backtest_summary)
    
    # Generate daily report
    daily_report = generate_daily_report_and_save()
    print(daily_report)
    
    # Calculate provider metrics
    metrics = calculate_provider_metrics("Provider Name")
    print(metrics)
    """
    print("Signal Analysis Batch Runner")
    print("=" * 50)
    
    # Generate daily report
    print("\n1. Generating daily report...")
    daily_report = generate_daily_report_and_save()
    print(f"   Report generated: {daily_report.get('log_date', 'N/A')}")
    
    # Run analysis (commented out to avoid long execution)
    # print("\n2. Running signal analysis...")
    # analysis_summary = run_analysis_for_all_signals()
    # print(f"   Analyzed: {analysis_summary.get('analyzed', 0)}/{analysis_summary.get('total_signals', 0)}")
    
    print("\nDone!")

