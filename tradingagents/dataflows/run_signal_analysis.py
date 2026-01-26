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
    Uses a more reliable query to ensure we get all distinct providers.
    
    Returns:
        List of unique provider names (sorted)
    """
    supabase = get_supabase()
    if not supabase:
        return []
    
    try:
        # Get all provider names - fetch in batches to handle large datasets
        # Supabase has a default limit of 1000, so we need to fetch all records
        all_providers = []
        page_size = 1000
        offset = 0
        max_iterations = 100  # Safety limit to prevent infinite loops
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            try:
                result = supabase.table('signal_provider_signals')\
                    .select('provider_name')\
                    .range(offset, offset + page_size - 1)\
                    .execute()
                
                if not result.data or len(result.data) == 0:
                    break
                    
                all_providers.extend(result.data)
                
                # If we got fewer records than page_size, we've reached the end
                if len(result.data) < page_size:
                    break
                    
                offset += page_size
            except Exception as e:
                print(f"Error fetching providers batch at offset {offset}: {e}")
                break
        
        if all_providers:
            # Extract unique provider names, filtering out None/empty values
            providers = []
            seen = set()
            for row in all_providers:
                provider_name = row.get('provider_name')
                if provider_name and isinstance(provider_name, str):
                    provider_name = provider_name.strip()
                    if provider_name and provider_name not in seen:
                        providers.append(provider_name)
                        seen.add(provider_name)
            
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
        # Get ALL distinct symbols from signal_provider_signals table
        # Use pagination to fetch ALL records (Supabase default limit is 1000)
        # This ensures we get every symbol, even if there are thousands of signals
        all_symbols = []
        offset = 0
        page_size = 1000  # Supabase max limit per request
        
        while True:
            result = supabase.table('signal_provider_signals')\
                .select('symbol')\
                .range(offset, offset + page_size - 1)\
                .execute()
            
            if not result.data or len(result.data) == 0:
                break
            
            # Extract symbols from this page
            page_symbols = [row.get('symbol', '').strip() for row in result.data if row.get('symbol')]
            all_symbols.extend(page_symbols)
            
            # If we got less than page_size, we've reached the end
            if len(result.data) < page_size:
                break
            
            offset += page_size
        
        if all_symbols:
            # Get unique set of all symbols
            raw_symbols = list(set(all_symbols))
            
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
            print(f"✅ Fetched {len(symbols)} distinct instruments from {len(all_symbols)} total signal records")
            return symbols
        return []
    except Exception as e:
        print(f"❌ Error fetching instruments: {e}")
        import traceback
        traceback.print_exc()
        return []


def check_market_data_availability(symbol: str) -> Dict:
    """
    Check if market data exists for the given instrument.
    Tries multiple symbol format variants and multiple tables to handle different storage formats.
    
    Args:
        symbol: Instrument symbol to check
        
    Returns:
        Dictionary with availability status and details
    """
    supabase = get_supabase()
    if not supabase:
        return {"available": False, "error": "Database not available"}
    
    try:
        symbol_upper = symbol.upper()
        
        # Build list of symbol variants to try (similar to market_data_service.py)
        symbols_to_try = [symbol_upper]
        
        # Determine which tables to try based on symbol pattern
        # IMPORTANT: Check for Gold/Silver FIRST, even if it has C: prefix
        tables_to_try = []
        if "XAU" in symbol_upper or "XAG" in symbol_upper or "*" in symbol_upper or symbol_upper == "GOLD":
            # Gold/Silver - try commodities table first
            tables_to_try = ["market_data_commodities_1min"]
            if symbol_upper == "XAUUSD" or "XAU" in symbol_upper:
                symbols_to_try = ["C:XAUUSD", "XAUUSD", "^XAUUSD", "GOLD"]
            elif symbol_upper == "C:XAUUSD":
                symbols_to_try = ["C:XAUUSD", "XAUUSD", "^XAUUSD", "GOLD"]
            elif symbol_upper == "^XAUUSD":
                symbols_to_try = ["^XAUUSD", "C:XAUUSD", "XAUUSD", "GOLD"]
            elif symbol_upper == "GOLD":
                symbols_to_try = ["GOLD", "C:XAUUSD", "^XAUUSD", "XAUUSD"]
        elif symbol_upper.startswith("I:") or symbol_upper.startswith("^"):
            # Indices
            tables_to_try = ["market_data_indices_1min"]
        elif symbol_upper.startswith("C:"):
            # Currency with C: prefix
            tables_to_try = ["market_data_currencies_1min"]
            base_symbol = symbol_upper[2:]  # Remove "C:" prefix
            symbols_to_try = [symbol_upper, base_symbol]
        elif len(symbol_upper) >= 6 and len(symbol_upper) <= 7 and not symbol_upper.startswith("^") and not symbol_upper.startswith("I:"):
            # Likely a currency pair without prefix (EURUSD, GBPUSD, etc.)
            # Try currencies table first, then stocks as fallback
            tables_to_try = ["market_data_currencies_1min", "market_data_stocks_1min"]
            symbols_to_try = [symbol_upper, f"C:{symbol_upper}"]
        else:
            # Default to stocks, but also try currencies if it's a 6-7 char symbol
            if len(symbol_upper) >= 6 and len(symbol_upper) <= 7:
                tables_to_try = ["market_data_stocks_1min", "market_data_currencies_1min"]
                symbols_to_try = [symbol_upper, f"C:{symbol_upper}"]
            else:
                tables_to_try = ["market_data_stocks_1min"]
        
        # Try each table and symbol combination
        found_symbol = None
        found_table = None
        for table_name in tables_to_try:
            for try_symbol in symbols_to_try:
                try:
                    result = supabase.table(table_name).select('timestamp').eq('symbol', try_symbol).limit(1).execute()
                    if result.data and len(result.data) > 0:
                        found_symbol = try_symbol
                        found_table = table_name
                        break
                except Exception:
                    continue
            if found_symbol:
                break
        
        if found_symbol and found_table:
            # Get date range using the found symbol
            date_result = supabase.table(found_table).select('timestamp').eq('symbol', found_symbol).order('timestamp', desc=False).limit(1).execute()
            latest_result = supabase.table(found_table).select('timestamp').eq('symbol', found_symbol).order('timestamp', desc=True).limit(1).execute()
            
            earliest = date_result.data[0]['timestamp'] if date_result.data and len(date_result.data) > 0 else None
            latest = latest_result.data[0]['timestamp'] if latest_result.data and len(latest_result.data) > 0 else None
            
            return {
                "available": True,
                "table": found_table,
                "symbol_found": found_symbol,
                "earliest_date": earliest,
                "latest_date": latest
            }
        else:
            # Try to get distinct symbols from the tables to help with debugging
            all_sample_symbols = []
            for table_name in tables_to_try:
                try:
                    distinct_result = supabase.table(table_name).select('symbol').limit(100).execute()
                    distinct_symbols = list(set([row.get('symbol', '') for row in (distinct_result.data or [])]))
                    all_sample_symbols.extend(distinct_symbols)
                except Exception:
                    continue
            
            all_sample_symbols = sorted(list(set(all_sample_symbols)))[:10]  # Show first 10 unique
            tables_tried_str = ", ".join(tables_to_try)
            return {
                "available": False,
                "tables_tried": tables_to_try,
                "symbols_tried": symbols_to_try,
                "sample_symbols_in_tables": all_sample_symbols,
                "error": f"No market data found for {symbol} (tried: {', '.join(symbols_to_try)}) in tables: {tables_tried_str}. Sample symbols in tables: {', '.join(all_sample_symbols) if all_sample_symbols else 'none'}"
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
        
        # Diagnostic: Check total signals for this provider (without filters)
        total_signals = 0  # Initialize for later use
        try:
            diagnostic_query = supabase.table('signal_provider_signals').select('id, symbol, signal_date, provider_name')
            if provider_name:
                diagnostic_query = diagnostic_query.eq('provider_name', provider_name)
            diagnostic_result = diagnostic_query.execute()
            
            if diagnostic_result.data:
                total_signals = len(diagnostic_result.data)
                print(f"🔍 DIAGNOSTIC: Total signals for provider '{provider_name}': {total_signals}")
                
                # Count by symbol
                symbol_counts = {}
                for sig in diagnostic_result.data:
                    sym = sig.get('symbol', 'N/A')
                    symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
                print(f"🔍 DIAGNOSTIC: Signals by symbol: {symbol_counts}")
                
                # Show date range
                if diagnostic_result.data:
                    dates = [sig.get('signal_date') for sig in diagnostic_result.data if sig.get('signal_date')]
                    if dates:
                        try:
                            parsed_dates = [datetime.fromisoformat(d.replace('Z', '+00:00')) if isinstance(d, str) else d for d in dates]
                            min_date = min(parsed_dates)
                            max_date = max(parsed_dates)
                            print(f"🔍 DIAGNOSTIC: Date range in DB: {min_date.date()} to {max_date.date()}")
                        except Exception as e:
                            print(f"🔍 DIAGNOSTIC: Date range parsing failed: {e}")
                
                # If date filters are provided, show how many are excluded
                if start_date or end_date:
                    filtered_count = 0
                    for sig in diagnostic_result.data:
                        sig_date_str = sig.get('signal_date')
                        if sig_date_str:
                            try:
                                sig_date = datetime.fromisoformat(sig_date_str.replace('Z', '+00:00')) if isinstance(sig_date_str, str) else sig_date_str
                                if start_date and sig_date < start_date:
                                    continue
                                if end_date and sig_date > end_date:
                                    continue
                                filtered_count += 1
                            except:
                                pass
                    excluded = total_signals - filtered_count
                    if excluded > 0:
                        print(f"⚠️ DIAGNOSTIC: {excluded} signals excluded by date filter (outside range {start_date.date() if start_date else 'N/A'} to {end_date.date() if end_date else 'N/A'})")
        except Exception as e:
            print(f"DEBUG: Could not run diagnostic query: {e}")
        
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
                    # If no results with filters, check if date filters were provided
                    print(f"⚠️ No signals found for variant '{variant}' with current filters")
                    
                    # Only use fallback if date filters were NOT explicitly provided
                    # If date filters were provided and no signals found, respect that (don't fallback)
                    if start_date or end_date:
                        # Date filters were provided - don't fallback, just log the issue
                        date_range_info = ""
                        if start_date and end_date:
                            date_range_info = f" in date range {start_date.date()} to {end_date.date()}"
                        elif start_date:
                            date_range_info = f" from {start_date.date()}"
                        elif end_date:
                            date_range_info = f" until {end_date.date()}"
                        print(f"  ℹ️ No signals found for variant '{variant}'{date_range_info} - date filters are respected, no fallback")
                    else:
                        # No date filters provided - try without any filters to see if signals exist
                        test_query = supabase.table('signal_provider_signals').select('*')
                        if provider_name:
                            test_query = test_query.eq('provider_name', provider_name)
                        test_query = test_query.eq('symbol', variant)
                        test_result = test_query.execute()
                        if test_result.data:
                            print(f"  ℹ️ But found {len(test_result.data)} signals without date filters - using all available signals")
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
        
        # Final diagnostic: If we found fewer signals than expected, show what we're missing
        if provider_name:
            try:
                # Query all signals for this provider with same filters (no symbol filter) to see what we're missing
                all_provider_query = supabase.table('signal_provider_signals').select('id, symbol, signal_date, action')
                all_provider_query = all_provider_query.eq('provider_name', provider_name)
                if start_date:
                    all_provider_query = all_provider_query.gte('signal_date', start_date.isoformat())
                if end_date:
                    all_provider_query = all_provider_query.lte('signal_date', end_date.isoformat())
                all_provider_result = all_provider_query.execute()
                
                if all_provider_result.data:
                    expected_count = len(all_provider_result.data)
                    found_ids = {sig.get('id') for sig in signals if sig.get('id')}
                    missing_signals = [sig for sig in all_provider_result.data if sig.get('id') not in found_ids]
                    
                    if missing_signals:
                        print(f"⚠️ DIAGNOSTIC: Found {len(signals)} signals with symbol variants, but {expected_count} signals exist for provider '{provider_name}' (with date filters). {len(missing_signals)} signals weren't matched:")
                        for missing in missing_signals[:10]:  # Show first 10 missing
                            print(f"  - ID: {missing.get('id')}, Symbol: '{missing.get('symbol')}', Date: {missing.get('signal_date')}, Action: {missing.get('action')}")
                        if len(missing_signals) > 10:
                            print(f"  ... and {len(missing_signals) - 10} more")
                    elif len(signals) < expected_count:
                        print(f"ℹ️ DIAGNOSTIC: Found {len(signals)} unique signals after deduplication, {expected_count} total signals match filters")
            except Exception as e:
                print(f"DEBUG: Could not check for missing signals: {e}")
        
        # If date filters were provided and no signals found, return error
        if not signals and (start_date or end_date):
            date_range_info = ""
            if start_date and end_date:
                date_range_info = f" for date range {start_date.date()} to {end_date.date()}"
            elif start_date:
                date_range_info = f" from {start_date.date()}"
            elif end_date:
                date_range_info = f" until {end_date.date()}"
            
            symbol_info = f" for symbol '{symbol}'" if symbol else ""
            provider_info = f" from provider '{provider_name}'" if provider_name else ""
            
            return {
                "error": f"No signals found{symbol_info}{provider_info}{date_range_info}. Please select a different date range or check if signals exist for the selected criteria."
            }
        
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
            
            # Build comprehensive error message
            symbol_info = f" for symbol '{symbol}'" if symbol else ""
            provider_info = f" from provider '{provider_name}'" if provider_name else ""
            variants_info = f" (tried variants: {variants_display})" if symbol and valid_variants else ""
            
            # Check if provider has signals for other symbols (helpful diagnostic)
            other_symbols_info = ""
            if provider_name and symbol:
                try:
                    # Check what symbols this provider actually has
                    provider_symbols_query = supabase.table('signal_provider_signals')\
                        .select('symbol')\
                        .eq('provider_name', provider_name)\
                        .execute()
                    
                    if provider_symbols_query.data:
                        provider_symbols = list(set([row.get('symbol', '') for row in provider_symbols_query.data if row.get('symbol')]))
                        if provider_symbols:
                            # Show first 10 symbols this provider has
                            sample_symbols = sorted(provider_symbols)[:10]
                            other_symbols_info = f"\n\n💡 Provider '{provider_name}' has signals for these symbols: {', '.join(sample_symbols)}"
                            if len(provider_symbols) > 10:
                                other_symbols_info += f" (and {len(provider_symbols) - 10} more)"
                            other_symbols_info += f"\n💡 Try selecting one of these symbols, or select 'All Providers' to see signals from all providers."
                except Exception as e:
                    print(f"DEBUG: Could not check provider symbols: {e}")
            
            error_msg = f"No signals found{symbol_info}{provider_info}{variants_info}."
            if not provider_name:
                error_msg += " Check if signals exist in signal_provider_signals table."
            else:
                error_msg += f"\n\n💡 Suggestions:"
                error_msg += f"\n   • Try selecting 'All Providers' to see if other providers have signals for '{symbol}'"
                error_msg += f"\n   • Check if the symbol name matches exactly (case-sensitive)"
                error_msg += f"\n   • Verify that provider '{provider_name}' has signals for this symbol"
            
            error_msg += other_symbols_info
            
            return {"error": error_msg}
        
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
                    
                    # Include error results in display so users can see signals that couldn't be analyzed
                    error_result = {
                        'signal_id': signal.get('id'),
                        'provider_name': signal.get('provider_name'),
                        'symbol': signal_symbol,
                        'signal_date': signal_date,
                        'action': signal.get('action', '').upper(),
                        'entry_price': signal.get('entry_price'),
                        'target_1': signal.get('target_1'),
                        'target_2': signal.get('target_2'),
                        'target_3': signal.get('target_3'),
                        'stop_loss': signal.get('stop_loss'),
                        'final_status': 'NO_DATA',
                        'error': error_msg,
                        'tp1_hit': False,
                        'tp2_hit': False,
                        'tp3_hit': False,
                        'sl_hit': False,
                        'max_profit': 0.0,
                        'max_drawdown': 0.0,
                        'hold_time_hours': 0.0,
                        'pips_made': 0
                    }
                    if not save_results:
                        # Store error result in list for display (testing mode)
                        analysis_results_list.append(error_result)
                    
            except Exception as e:
                error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
                print(f"  Exception: {error_msg}")
                error_count += 1
                
                # Include exception results in display
                error_result = {
                    'signal_id': signal.get('id'),
                    'provider_name': signal.get('provider_name'),
                    'symbol': signal.get('symbol', 'N/A'),
                    'signal_date': signal.get('signal_date', 'N/A'),
                    'action': signal.get('action', '').upper(),
                    'entry_price': signal.get('entry_price'),
                    'target_1': signal.get('target_1'),
                    'target_2': signal.get('target_2'),
                    'target_3': signal.get('target_3'),
                    'stop_loss': signal.get('stop_loss'),
                    'final_status': 'ERROR',
                    'error': error_msg,
                    'tp1_hit': False,
                    'tp2_hit': False,
                    'tp3_hit': False,
                    'sl_hit': False,
                    'max_profit': 0.0,
                    'max_drawdown': 0.0,
                    'hold_time_hours': 0.0,
                    'pips_made': 0
                }
                if not save_results:
                    # Store error result in list for display (testing mode)
                    analysis_results_list.append(error_result)
        
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
                
                # Include error results in display so users can see signals that couldn't be analyzed
                error_result = {
                    'signal_id': signal.get('id'),
                    'provider_name': signal.get('provider_name'),
                    'symbol': symbol,
                    'signal_date': signal_date,
                    'action': signal.get('action', '').upper(),
                    'entry_price': signal.get('entry_price'),
                    'target_1': signal.get('target_1'),
                    'target_2': signal.get('target_2'),
                    'target_3': signal.get('target_3'),
                    'stop_loss': signal.get('stop_loss'),
                    'final_status': 'NO_DATA',
                    'error': error_msg,
                    'tp1_hit': False,
                    'tp2_hit': False,
                    'tp3_hit': False,
                    'sl_hit': False,
                    'max_profit': 0.0,
                    'max_drawdown': 0.0,
                    'hold_time_hours': 0.0,
                    'pips_made': 0
                }
                if not save_results:
                    # Store error result in list for display (testing mode)
                    analysis_results_list.append(error_result)
                
        except Exception as e:
            error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            print(f"  Exception: {error_msg}")
            error_count += 1
            
            # Include exception results in display
            error_result = {
                'signal_id': signal.get('id'),
                'provider_name': signal.get('provider_name'),
                'symbol': symbol,
                'signal_date': signal_date,
                'action': signal.get('action', '').upper(),
                'entry_price': signal.get('entry_price'),
                'target_1': signal.get('target_1'),
                'target_2': signal.get('target_2'),
                'target_3': signal.get('target_3'),
                'stop_loss': signal.get('stop_loss'),
                'final_status': 'ERROR',
                'error': error_msg,
                'tp1_hit': False,
                'tp2_hit': False,
                'tp3_hit': False,
                'sl_hit': False,
                'max_profit': 0.0,
                'max_drawdown': 0.0,
                'hold_time_hours': 0.0,
                'pips_made': 0
            }
            if not save_results:
                # Store error result in list for display (testing mode)
                analysis_results_list.append(error_result)
    
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


def run_analysis_for_all_providers_and_instruments(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    save_results: bool = True
) -> Dict:
    """
    Run analysis for ALL signal providers and ALL instruments in the database.
    This applies the same logic that works for XAUUSD and pipxpert to every instrument and provider.
    
    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
        save_results: Whether to save results to database
        
    Returns:
        Dictionary with analysis summary across all providers and instruments
    """
    supabase = get_supabase()
    if not supabase:
        return {"error": "Database not available"}
    
    # Get all unique providers
    providers = get_available_providers()
    if not providers:
        return {"error": "No signal providers found in database"}
    
    # Get all unique instruments
    instruments = get_available_instruments()
    if not instruments:
        return {"error": "No instruments found in database"}
    
    print(f"=" * 80)
    print(f"Running analysis for ALL providers and ALL instruments")
    print(f"=" * 80)
    print(f"Providers: {len(providers)}")
    print(f"Instruments: {len(instruments)}")
    print(f"Date range: {start_date.date() if start_date else 'All'} to {end_date.date() if end_date else 'All'}")
    print(f"=" * 80)
    
    total_analyzed = 0
    total_errors = 0
    provider_summaries = {}
    
    # Run analysis for each provider-instrument combination
    for provider_idx, provider_name in enumerate(providers, 1):
        print(f"\n[{provider_idx}/{len(providers)}] Processing provider: {provider_name}")
        print(f"-" * 80)
        
        provider_analyzed = 0
        provider_errors = 0
        
        for inst_idx, instrument in enumerate(instruments, 1):
            print(f"  [{inst_idx}/{len(instruments)}] Analyzing {instrument} for {provider_name}...")
            
            try:
                # Check market data availability first
                market_data_status = check_market_data_availability(instrument)
                if not market_data_status.get("available"):
                    error_msg = market_data_status.get('error', 'Market data not available')
                    print(f"    ⚠️  Skipping {instrument}: {error_msg}")
                    provider_errors += 1
                    total_errors += 1
                    continue
                
                # Run analysis for this provider-instrument combination
                result = run_analysis_for_all_signals(
                    provider_name=provider_name,
                    symbol=instrument,
                    start_date=start_date,
                    end_date=end_date,
                    save_results=save_results
                )
                
                if 'error' in result:
                    print(f"    ❌ Error: {result['error']}")
                    provider_errors += 1
                    total_errors += 1
                else:
                    analyzed = result.get('analyzed', 0)
                    errors = result.get('errors', 0)
                    provider_analyzed += analyzed
                    provider_errors += errors
                    total_analyzed += analyzed
                    total_errors += errors
                    
                    if analyzed > 0:
                        print(f"    ✅ Analyzed {analyzed} signals (errors: {errors})")
                    else:
                        print(f"    ℹ️  No signals found for this combination")
                
            except Exception as e:
                error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
                print(f"    ❌ Exception: {error_msg}")
                provider_errors += 1
                total_errors += 1
        
        provider_summaries[provider_name] = {
            'analyzed': provider_analyzed,
            'errors': provider_errors
        }
        
        print(f"  Provider summary: {provider_analyzed} analyzed, {provider_errors} errors")
    
    print(f"\n{'=' * 80}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total signals analyzed: {total_analyzed}")
    print(f"Total errors: {total_errors}")
    print(f"Success rate: {(total_analyzed / (total_analyzed + total_errors) * 100) if (total_analyzed + total_errors) > 0 else 0:.2f}%")
    print(f"{'=' * 80}")
    
    return {
        'total_providers': len(providers),
        'total_instruments': len(instruments),
        'total_analyzed': total_analyzed,
        'total_errors': total_errors,
        'success_rate': (total_analyzed / (total_analyzed + total_errors) * 100) if (total_analyzed + total_errors) > 0 else 0,
        'provider_summaries': provider_summaries
    }


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

