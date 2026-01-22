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


def get_available_providers() -> List[str]:
    """
    Get list of unique provider names from signal_provider_signals table.
    
    Returns:
        List of unique provider names (sorted)
    """
    supabase = get_supabase()
    if not supabase:
        return []
    
    try:
        # Get distinct provider names
        result = supabase.table('signal_provider_signals').select('provider_name').execute()
        if result.data:
            providers = list(set([row['provider_name'] for row in result.data if row.get('provider_name')]))
            # Return sorted list
            return sorted(providers)
        return []
    except Exception as e:
        print(f"Error fetching providers: {e}")
        return []


def get_available_instruments() -> List[str]:
    """
    Get list of unique instruments (symbols) from signal_provider_signals table.
    Normalizes symbols to avoid duplicates (e.g., EURUSD and C:EURUSD).
    
    Returns:
        List of unique instrument symbols (normalized)
    """
    supabase = get_supabase()
    if not supabase:
        return []
    
    try:
        # Get distinct symbols
        result = supabase.table('signal_provider_signals').select('symbol').execute()
        if result.data:
            raw_symbols = list(set([row['symbol'] for row in result.data]))
            
            # Normalize symbols to avoid duplicates
            # Strategy: Prefer format without prefix, but keep C: prefix for currencies if that's the only format
            normalized_symbols = {}
            
            for symbol in raw_symbols:
                symbol_upper = symbol.upper().strip()
                
                # Skip empty symbols
                if not symbol_upper:
                    continue
                
                # Handle C: prefix (currencies)
                if symbol_upper.startswith("C:") and len(symbol_upper) > 2:
                    base = symbol_upper[2:]  # Remove "C:" prefix
                    # If we already have the base symbol, prefer that
                    if base not in normalized_symbols:
                        normalized_symbols[base] = base
                    # Otherwise, keep the C: version if base doesn't exist
                    elif base not in [s for s in normalized_symbols.values()]:
                        normalized_symbols[symbol_upper] = symbol_upper
                # Handle ^ prefix (indices)
                elif symbol_upper.startswith("^"):
                    # Keep as-is for indices
                    normalized_symbols[symbol_upper] = symbol_upper
                # Handle I: prefix (indices)
                elif symbol_upper.startswith("I:"):
                    base = symbol_upper[2:]
                    # Prefer without prefix if both exist
                    if base not in normalized_symbols:
                        normalized_symbols[base] = base
                    else:
                        normalized_symbols[symbol_upper] = symbol_upper
                else:
                    # No prefix - use as-is
                    normalized_symbols[symbol_upper] = symbol_upper
            
            # Return sorted list of normalized symbols
            symbols = sorted(normalized_symbols.values())
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
        elif symbol_upper.startswith("C:") and len(symbol_upper) >= 8:
            # Has C: prefix, also try without
            base_symbol = symbol_upper[2:]  # Remove "C:" prefix
            symbols_to_try = [symbol_upper, base_symbol]
        elif len(symbol_upper) >= 6 and len(symbol_upper) <= 7 and not symbol_upper.startswith("C:") and not symbol_upper.startswith("^") and not symbol_upper.startswith("I:"):
            # Likely a currency pair without prefix
            symbols_to_try = [symbol_upper, f"C:{symbol_upper}"]
        # Handle single character or very short symbols (edge case)
        elif len(symbol_upper) < 3:
            # Don't try variants for very short symbols
            symbols_to_try = [symbol_upper]
        
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
    
    # Fetch signals - try multiple symbol formats if symbol provided
    query = supabase.table('signal_provider_signals').select('*')
    
    if provider_name:
        query = query.eq('provider_name', provider_name)
    
    if symbol:
        symbol_upper = symbol.upper().strip()
        
        # Validate symbol is not too short (edge case protection)
        if len(symbol_upper) < 2:
            return {"error": f"Invalid symbol: '{symbol}'. Symbol must be at least 2 characters."}
        
        # Build symbol variants to try (similar to market data check)
        # Start with the exact symbol as entered (most likely to match)
        symbol_variants = [symbol_upper]
        
        # Handle XAUUSD/Gold variants
        if "XAU" in symbol_upper or symbol_upper == "GOLD":
            # Add common variants, but keep original first
            additional_variants = ["C:XAUUSD", "^XAUUSD", "GOLD"]
            for v in additional_variants:
                if v not in symbol_variants:
                    symbol_variants.append(v)
        # Handle currency pairs with C: prefix
        elif symbol_upper.startswith("C:") and len(symbol_upper) >= 8:
            # Has C: prefix, also try without
            base_symbol = symbol_upper[2:]
            if base_symbol not in symbol_variants:
                symbol_variants.append(base_symbol)
        # Handle currency pairs without prefix (6-7 chars, no special prefixes)
        elif len(symbol_upper) >= 6 and len(symbol_upper) <= 7 and not symbol_upper.startswith("C:") and not symbol_upper.startswith("^") and not symbol_upper.startswith("I:"):
            # Likely currency pair without prefix, try with C: prefix
            variant_with_prefix = f"C:{symbol_upper}"
            if variant_with_prefix not in symbol_variants:
                symbol_variants.append(variant_with_prefix)
        
        # Try to fetch signals with any of the symbol variants
        # Use fallback approach: try each variant separately for better compatibility and error messages
        all_signals = []
        variants_to_try = []
        
        # First, try the exact symbol as entered (both uppercase and lowercase) - prioritize this
        variants_to_try.append(symbol_upper)
        symbol_lower = symbol_upper.lower()
        if symbol_lower != symbol_upper:
            variants_to_try.append(symbol_lower)
        
        # Add other variants (both uppercase and lowercase)
        for variant in symbol_variants:
            # Ensure variant is a string and not empty, and not already added
            if isinstance(variant, str) and len(variant) > 1 and variant not in variants_to_try:
                variants_to_try.append(variant)
                # Also add lowercase version if different
                variant_lower = variant.lower()
                if variant_lower != variant and variant_lower not in variants_to_try and len(variant_lower) > 1:
                    variants_to_try.append(variant_lower)
        
        # Debug: print what variants we're trying
        print(f"DEBUG: Trying {len(variants_to_try)} symbol variants: {variants_to_try}")
        
        # Also try a test query to see what symbols actually exist in the database
        try:
            # Test query without any filters to see what's actually there
            test_query = supabase.table('signal_provider_signals').select('symbol, provider_name').limit(20).execute()
            if test_query.data:
                sample_symbols = list(set([row.get('symbol', '') for row in test_query.data]))
                sample_providers = list(set([row.get('provider_name', '') for row in test_query.data]))
                print(f"DEBUG: Sample symbols in database: {sample_symbols}")
                print(f"DEBUG: Sample providers in database: {sample_providers}")
                
                # Check if XAUUSD exists
                xauusd_signals = [row for row in test_query.data if row.get('symbol', '').upper() == 'XAUUSD']
                if xauusd_signals:
                    print(f"DEBUG: Found {len(xauusd_signals)} XAUUSD signals in sample (provider filter: {provider_name})")
                    for sig in xauusd_signals[:3]:
                        print(f"  - Symbol: '{sig.get('symbol')}', Provider: '{sig.get('provider_name')}'")
        except Exception as e:
            print(f"DEBUG: Could not fetch sample symbols: {e}")
        
        # Filter out any invalid variants before querying
        valid_variants = [v for v in variants_to_try if v and isinstance(v, str) and len(v) > 1]
        print(f"DEBUG: Filtered to {len(valid_variants)} valid variants: {valid_variants}")
        
        for variant in valid_variants:
            try:
                # First try with all filters
                variant_query = supabase.table('signal_provider_signals').select('*')
                if provider_name:
                    variant_query = variant_query.eq('provider_name', provider_name)
                variant_query = variant_query.eq('symbol', variant)
                if start_date:
                    variant_query = variant_query.gte('signal_date', start_date.isoformat())
                if end_date:
                    variant_query = variant_query.lte('signal_date', end_date.isoformat())
                
                print(f"DEBUG: Querying variant '{variant}' with provider='{provider_name}', start_date={start_date}, end_date={end_date}")
                variant_result = variant_query.execute()
                
                if variant_result.data:
                    all_signals.extend(variant_result.data)
                    print(f"✅ Found {len(variant_result.data)} signals for symbol variant: '{variant}'")
                else:
                    # If no results with filters, try without date filters to see if date range is the issue
                    print(f"⚠️ No signals found for variant '{variant}' with current filters")
                    test_query = supabase.table('signal_provider_signals').select('*')
                    if provider_name:
                        test_query = test_query.eq('provider_name', provider_name)
                    test_query = test_query.eq('symbol', variant)
                    test_result = test_query.execute()
                    if test_result.data:
                        print(f"  ℹ️ But found {len(test_result.data)} signals without date filters - date range might be excluding signals")
                        # Use the unfiltered results if date filters are too restrictive
                        all_signals.extend(test_result.data)
                        print(f"✅ Using {len(test_result.data)} signals without date filters for variant: '{variant}'")
            except Exception as e:
                print(f"⚠️ Error querying variant '{variant}': {e}")
                continue
        
        # Remove duplicates based on signal id
        seen_ids = set()
        signals = []
        for sig in all_signals:
            sig_id = sig.get('id')
            if sig_id and sig_id not in seen_ids:
                seen_ids.add(sig_id)
                signals.append(sig)
        
        if signals:
            # Found signals, continue with analysis
            print(f"Total unique signals found: {len(signals)}")
        else:
            # No signals found with any variant
            # Filter out any invalid variants (single characters, empty strings)
            valid_variants = [v for v in variants_to_try if v and isinstance(v, str) and len(v) > 1]
            
            # Show first few variants in error message (to avoid too long messages)
            variants_display = ', '.join(valid_variants[:8])
            if len(valid_variants) > 8:
                variants_display += f" (and {len(valid_variants) - 8} more variants)"
            
            symbol_info = f" for symbol '{symbol}' (tried variants: {variants_display})" if symbol else ""
            return {"error": f"No signals found{symbol_info}. Check if signals exist in signal_provider_signals table. Note: Signals might be stored as lowercase (e.g., 'xauusd')."}
        
        # Skip the query execution below and use signals directly
        analyzed_count = 0
        error_count = 0
        analysis_results_list = []  # Store results for return (without saving to DB)
        
        for i, signal in enumerate(signals, 1):
            try:
                signal_symbol = signal.get('symbol', 'N/A')
                signal_date = signal.get('signal_date', 'N/A')
                print(f"Analyzing signal {i}/{len(signals)}: {signal_symbol} @ {signal_date}")
                
                analysis_result = analyzer.analyze_signal(signal)
                
                if 'error' not in analysis_result:
                    if save_results:
                        analyzer.save_analysis_result(analysis_result)
                    else:
                        # Store result in list for display (testing mode)
                        analysis_results_list.append(analysis_result)
                    analyzed_count += 1
                else:
                    error_msg = str(analysis_result.get('error', 'Unknown error'))
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
            'success_rate': (analyzed_count / len(signals) * 100) if signals else 0,
            'analysis_results': analysis_results_list  # Include results for display
        }
    
    # If no symbol filter, continue with normal query
    # If we didn't use fallback, continue with normal query execution
    if start_date:
        query = query.gte('signal_date', start_date.isoformat())
    if end_date:
        query = query.lte('signal_date', end_date.isoformat())
    
    result = query.execute()
    signals = result.data if result.data else []
    
    if not signals:
        # Build better error message with symbol variants tried
        if symbol:
            symbol_upper = symbol.upper().strip()
            symbol_variants = [symbol_upper]
            
            # Rebuild variants for error message
            if "XAU" in symbol_upper or symbol_upper == "GOLD":
                symbol_variants = ["C:XAUUSD", "XAUUSD", "^XAUUSD", "GOLD"]
                symbol_variants = list(dict.fromkeys(symbol_variants))
            elif symbol_upper.startswith("C:") and len(symbol_upper) >= 8:
                base_symbol = symbol_upper[2:]
                symbol_variants = [symbol_upper, base_symbol]
            elif len(symbol_upper) >= 6 and len(symbol_upper) <= 7 and not symbol_upper.startswith("C:") and not symbol_upper.startswith("^") and not symbol_upper.startswith("I:"):
                symbol_variants = [symbol_upper, f"C:{symbol_upper}"]
            
            symbol_info = f" for symbol '{symbol}' (tried variants: {', '.join(symbol_variants)})"
        else:
            symbol_info = ""
        return {"error": f"No signals found{symbol_info}. Check if signals exist in signal_provider_signals table."}
    
    print(f"Analyzing {len(signals)} signals...")
    
    analyzed_count = 0
    error_count = 0
    analysis_results_list = []  # Store results for return (without saving to DB)
    
    for i, signal in enumerate(signals, 1):
        try:
            symbol = signal.get('symbol', 'N/A')
            signal_date = signal.get('signal_date', 'N/A')
            print(f"Analyzing signal {i}/{len(signals)}: {symbol} @ {signal_date}")
            
            analysis_result = analyzer.analyze_signal(signal)
            
            if 'error' not in analysis_result:
                if save_results:
                    analyzer.save_analysis_result(analysis_result)
                else:
                    # Store result in list for display (testing mode)
                    analysis_results_list.append(analysis_result)
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
        'success_rate': (analyzed_count / len(signals) * 100) if signals else 0,
        'analysis_results': analysis_results_list  # Include results for display
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

