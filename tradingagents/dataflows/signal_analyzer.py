"""
Signal Analysis Engine
Calculates performance metrics, success rates, and analysis results for signal providers.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz

from tradingagents.database.config import get_supabase
from tradingagents.dataflows.market_data_service import fetch_ohlcv


class SignalAnalyzer:
    """
    Analyzes signal provider signals and calculates performance metrics.
    
    TIMEZONE HANDLING:
    - All signals are stored in GMT+4 (Asia/Dubai) in the database
    - Market data timestamps are stored in GMT+4 (Asia/Dubai) in the database
    - All internal processing uses GMT+4 to ensure consistency
    - Timestamps are normalized to GMT+4 before any comparisons
    """
    
    def __init__(self, timezone: str = 'Asia/Dubai'):
        """
        Initialize analyzer.
        
        Args:
            timezone: Timezone for all calculations (default: GMT+4)
        """
        self.tz = pytz.timezone(timezone)
        self.utc_tz = pytz.timezone('UTC')
        self.supabase = get_supabase()
    
    def analyze_signal(
        self,
        signal: Dict,
        max_hold_days: int = 30,  # Not used anymore, kept for backward compatibility
        price_tolerance: float = 0.0001
    ) -> Dict:
        """
        Analyze a single signal to determine TP/SL hits.
        
        Args:
            signal: Signal dictionary
            max_hold_days: Deprecated - analysis is now limited to 72 hours
            price_tolerance: Price tolerance for hit detection (0.01% default)
            
        Returns:
            Dictionary with analysis results
            
        Note:
            Market data is fetched for 72 hours from signal date (not 30 days)
        """
        try:
            symbol = signal.get('symbol', '').upper()
            signal_date = signal.get('signal_date')
            action = signal.get('action', '').lower()
            entry_price = signal.get('entry_price')
            target_1 = signal.get('target_1')
            target_2 = signal.get('target_2')
            target_3 = signal.get('target_3')
            stop_loss = signal.get('stop_loss')
            
            # Validate required fields
            if not all([symbol, signal_date, action, entry_price]):
                return {"error": "Missing required signal fields"}
            
            # Parse signal_date
            if isinstance(signal_date, str):
                try:
                    signal_date = datetime.fromisoformat(signal_date.replace('Z', '+00:00'))
                except:
                    signal_date = datetime.strptime(signal_date, '%Y-%m-%d %H:%M:%S')
                    signal_date = self.tz.localize(signal_date)
            
            if signal_date.tzinfo is None:
                signal_date = self.tz.localize(signal_date)
            else:
                signal_date = signal_date.astimezone(self.tz)
            
            # Fetch market data
            # Determine asset_class based on symbol prefix
            asset_class = None
            if symbol.startswith("C:"):
                asset_class = "Currencies"
            elif symbol.startswith("I:") or symbol.startswith("^"):
                asset_class = "Indices"
            elif "XAU" in symbol or "XAG" in symbol or "*" in symbol:
                asset_class = "Commodities"
            else:
                # Try to infer from symbol length/pattern
                if len(symbol) >= 6 and len(symbol) <= 7 and not symbol.startswith("C:"):
                    # Likely currency pair without prefix - try with C: prefix
                    symbol_with_prefix = f"C:{symbol}"
                    asset_class = "Currencies"
                else:
                    asset_class = "Stocks"
            
            # Limit to 72 hours from signal date (instead of 30 days)
            end_date = signal_date + timedelta(hours=72)
            
            # CRITICAL: Query using GMT+4 (matching database storage format)
            # Database stores timestamps in GMT+4, so query should use GMT+4
            market_data = fetch_ohlcv(
                symbol=symbol,
                interval='1min',
                start=signal_date,  # Already in GMT+4, use directly
                end=end_date,       # Already in GMT+4, use directly (72 hours later)
                asset_class=asset_class
            )
            
            # If no data found and symbol doesn't have C: prefix, try with C: prefix
            if (market_data is None or market_data.empty) and not symbol.startswith("C:") and len(symbol) >= 6 and len(symbol) <= 7:
                symbol_with_prefix = f"C:{symbol}"
                market_data = fetch_ohlcv(
                    symbol=symbol_with_prefix,
                    interval='1min',
                    start=signal_date,  # Already in GMT+4
                    end=end_date,       # Already in GMT+4 (72 hours later)
                    asset_class="Currencies"
                )
                if market_data is not None and not market_data.empty:
                    symbol = symbol_with_prefix  # Update symbol for consistency
            
            if market_data is None or market_data.empty:
                return {"error": f"No market data available for {symbol}"}
            
            # Ensure timestamp column
            if 'timestamp' in market_data.columns:
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
                market_data = market_data.set_index('timestamp')
            
            # CRITICAL: Normalize timezones before comparison
            # Ensure signal_date is in GMT+4 first (should already be, but double-check)
            signal_date = signal_date.astimezone(self.tz)
            
            # Ensure both signal_date and market_data.index are in GMT+4
            if market_data.index.tz is None:
                # If timezone-naive, assume GMT+4 (based on database storage format)
                market_data.index = market_data.index.tz_localize(self.tz)
            elif market_data.index.tz != signal_date.tzinfo:
                # If different timezone, convert to GMT+4 to match signal_date
                # market_data.index.tz is pandas timezone, signal_date.tzinfo is datetime timezone
                market_data.index = market_data.index.tz_convert(self.tz)
            
            # Debug logging for timezone validation
            import os
            if os.getenv("DEBUG_TIMEZONE", "false").lower() == "true":
                print(f"🔍 [TZ] Signal date: {signal_date} (tz: {signal_date.tzinfo})")
                if len(market_data) > 0:
                    print(f"🔍 [TZ] Market data first: {market_data.index[0]} (tz: {market_data.index[0].tzinfo})")
                    print(f"🔍 [TZ] Market data last: {market_data.index[-1]} (tz: {market_data.index[-1].tzinfo})")
                    print(f"🔍 [TZ] Timezone match: {market_data.index[0].tzinfo == signal_date.tzinfo}")
            
            # Filter from signal_date onwards (now both are in same timezone)
            market_data = market_data[market_data.index >= signal_date]
            
            if market_data.empty:
                return {"error": f"No market data after signal date {signal_date.isoformat()} (GMT+4). Check if market data exists for this time period."}
            
            # Analyze price movements
            result = self._analyze_price_movements(
                market_data=market_data,
                action=action,
                entry_price=entry_price,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                stop_loss=stop_loss,
                price_tolerance=price_tolerance
            )
            
            # Add metadata and signal data for display
            result['signal_id'] = signal.get('id')
            result['provider_name'] = signal.get('provider_name')
            result['symbol'] = symbol
            result['signal_date'] = signal_date.isoformat()
            result['analysis_date'] = datetime.now(self.tz).isoformat()
            
            # Add signal data for table display
            result['action'] = action.upper()  # BUY or SELL
            result['entry_price'] = entry_price
            result['target_1'] = target_1
            result['target_2'] = target_2
            result['target_3'] = target_3
            result['stop_loss'] = stop_loss
            
            # Calculate "Pips Made" based on final_status
            # For GOLD/XAUUSD, pips are typically price points (not forex pips)
            pips_made = 0
            is_buy = (action.lower() == 'buy')
            
            if result['final_status'] == 'TP1' and target_1:
                pips_made = (target_1 - entry_price) if is_buy else (entry_price - target_1)
            elif result['final_status'] == 'TP2' and target_2:
                pips_made = (target_2 - entry_price) if is_buy else (entry_price - target_2)
            elif result['final_status'] == 'TP3' and target_3:
                pips_made = (target_3 - entry_price) if is_buy else (entry_price - target_3)
            elif result['final_status'] == 'SL' and stop_loss:
                pips_made = (stop_loss - entry_price) if is_buy else (entry_price - stop_loss)
            elif result['final_status'] in ['EXPIRED', 'OPEN']:
                # Use max_profit percentage converted to price points
                # max_profit is in percentage (e.g., 0.43% = 0.0043)
                if result.get('max_profit'):
                    pips_made = (result['max_profit'] / 100) * entry_price
                    if not is_buy:
                        pips_made = -pips_made
            
            # Round to nearest integer (as shown in image)
            result['pips_made'] = round(pips_made)
            
            return result
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _analyze_price_movements(
        self,
        market_data: pd.DataFrame,
        action: str,
        entry_price: float,
        target_1: Optional[float],
        target_2: Optional[float],
        target_3: Optional[float],
        stop_loss: Optional[float],
        price_tolerance: float
    ) -> Dict:
        """
        Analyze price movements to detect TP/SL hits.
        
        Returns:
            Dictionary with analysis results
        """
        is_buy = (action == 'buy')
        
        tp1_hit = False
        tp1_hit_datetime = None
        tp2_hit = False
        tp2_hit_datetime = None
        tp3_hit = False
        tp3_hit_datetime = None
        sl_hit = False
        sl_hit_datetime = None
        
        max_profit = 0.0
        max_drawdown = 0.0
        final_status = 'OPEN'
        
        # Track if entry was reached
        entry_reached = False
        entry_datetime = None
        
        # Debug logging - always enabled for signal analysis
        import os
        debug_mode = True  # Always show detailed logging for signal analysis
        
        if debug_mode:
            print(f"\n{'='*80}")
            print(f"🔍 [SIGNAL ANALYSIS] Starting analysis")
            print(f"{'='*80}")
            print(f"   Entry: {entry_price:.2f}")
            tp1_str = f"{target_1:.2f}" if target_1 else "N/A"
            tp2_str = f"{target_2:.2f}" if target_2 else "N/A"
            tp3_str = f"{target_3:.2f}" if target_3 else "N/A"
            sl_str = f"{stop_loss:.2f}" if stop_loss else "N/A"
            print(f"   TP1: {tp1_str}, TP2: {tp2_str}, TP3: {tp3_str}")
            print(f"   SL: {sl_str}")
            print(f"   Action: {action.upper()}")
            print(f"   Total candles available: {len(market_data)}")
            if len(market_data) > 0:
                first_candle = market_data.iloc[0]
                last_candle = market_data.iloc[-1]
                print(f"   📅 First candle: {market_data.index[0]} | H:{first_candle['High']:.2f}, L:{first_candle['Low']:.2f}, C:{first_candle['Close']:.2f}")
                print(f"   📅 Last candle: {market_data.index[-1]} | H:{last_candle['High']:.2f}, L:{last_candle['Low']:.2f}, C:{last_candle['Close']:.2f}")
                hours_covered = (market_data.index[-1] - market_data.index[0]).total_seconds() / 3600.0
                print(f"   ⏱️  Time coverage: {hours_covered:.2f} hours")
            print(f"{'='*80}")
        
        # Check if we have any market data
        if market_data.empty:
            return {
                'tp1_hit': False,
                'tp2_hit': False,
                'tp3_hit': False,
                'sl_hit': False,
                'max_profit': 0.0,
                'max_drawdown': 0.0,
                'final_status': 'NO_DATA',
                'hold_time_hours': 0.0
            }
        
        # More lenient entry detection: If first candle is close to entry, assume entry reached
        first_candle = market_data.iloc[0]
        first_high = float(first_candle.get('High', 0))
        first_low = float(first_candle.get('Low', 0))
        first_close = float(first_candle.get('Close', 0))
        
        # If price is very close to entry in first candle, assume entry reached
        entry_tolerance = 0.01  # 0.01% tolerance
        price_diff_pct = abs(first_close - entry_price) / entry_price if entry_price > 0 else 1.0
        
        if price_diff_pct <= entry_tolerance:
            entry_reached = True
            entry_datetime = market_data.index[0]
            if debug_mode:
                print(f"✅ [DEBUG] Entry assumed reached (first candle close {first_close:.2f} close to entry {entry_price:.2f})")
        
        for timestamp, row in market_data.iterrows():
            high = float(row.get('High', 0))
            low = float(row.get('Low', 0))
            close = float(row.get('Close', 0))
            
            # Check entry (if not already reached)
            if not entry_reached:
                if is_buy and low <= entry_price <= high:
                    entry_reached = True
                    entry_datetime = timestamp
                    if debug_mode:
                        print(f"\n🎯 [ENTRY] Entry reached at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Condition: Low ({low:.2f}) <= Entry ({entry_price:.2f}) <= High ({high:.2f})")
                elif not is_buy and low <= entry_price <= high:
                    entry_reached = True
                    entry_datetime = timestamp
                    if debug_mode:
                        print(f"\n🎯 [ENTRY] Entry reached at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Condition: Low ({low:.2f}) <= Entry ({entry_price:.2f}) <= High ({high:.2f})")
                # If price moved past entry without touching it, assume entry at first opportunity
                elif is_buy and low > entry_price:
                    # Price already above entry - assume we entered at entry price
                    entry_reached = True
                    entry_datetime = timestamp
                    if debug_mode:
                        print(f"\n🎯 [ENTRY] Entry assumed at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Reason: Price ({low:.2f}) already above entry ({entry_price:.2f})")
                elif not is_buy and high < entry_price:
                    # Price already below entry - assume we entered at entry price
                    entry_reached = True
                    entry_datetime = timestamp
                    if debug_mode:
                        print(f"\n🎯 [ENTRY] Entry assumed at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Reason: Price ({high:.2f}) already below entry ({entry_price:.2f})")
            
            if not entry_reached:
                continue
            
            # Calculate current P&L
            if is_buy:
                current_profit = (close - entry_price) / entry_price
            else:
                current_profit = (entry_price - close) / entry_price
            
            # Track max profit and drawdown
            if current_profit > max_profit:
                max_profit = current_profit
            if current_profit < max_drawdown:
                max_drawdown = current_profit
            
            # CORE PRINCIPLE: Whichever price is touched FIRST in time wins - TP or SL
            # Use worst-case assumption: Check SL first in each candle (risk management)
            # However, if TP1/TP2 are already hit, continue tracking but SL can still override
            
            if is_buy:
                # BUY trade: SL is below entry, TPs are above entry
                # Check if SL (low) is hit in this candle
                if stop_loss and low <= stop_loss:
                    sl_hit = True
                    sl_hit_datetime = timestamp
                    final_status = 'SL'
                    if debug_mode:
                        print(f"\n🛑 [HIT] SL HIT at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Condition: Low ({low:.2f}) <= SL ({stop_loss:.2f})")
                        print(f"   Previous TPs: TP1={tp1_hit}, TP2={tp2_hit}, TP3={tp3_hit}")
                        print(f"   Final Status: SL")
                    break  # SL closes entire position immediately (even if TPs were hit before)
                
                # Then check TPs (high) - price moving up
                # Track TPs sequentially, but SL can override later
                if target_1 and not tp1_hit and high >= target_1:
                    tp1_hit = True
                    tp1_hit_datetime = timestamp
                    final_status = 'TP1'  # Update status, but continue (SL can override)
                    if debug_mode:
                        print(f"\n✅ [HIT] TP1 HIT at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Condition: High ({high:.2f}) >= TP1 ({target_1:.2f})")
                        print(f"   Status: TP1 (continuing to check TP2/TP3/SL)")
                
                if target_2 and tp1_hit and not tp2_hit and high >= target_2:
                    tp2_hit = True
                    tp2_hit_datetime = timestamp
                    final_status = 'TP2'  # Update status, but continue (SL can override)
                    if debug_mode:
                        print(f"\n✅ [HIT] TP2 HIT at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Condition: High ({high:.2f}) >= TP2 ({target_2:.2f})")
                        print(f"   Status: TP2 (continuing to check TP3/SL)")
                
                if target_3 and tp2_hit and not tp3_hit and high >= target_3:
                    tp3_hit = True
                    tp3_hit_datetime = timestamp
                    final_status = 'TP3'
                    if debug_mode:
                        print(f"\n✅ [HIT] TP3 HIT at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Condition: High ({high:.2f}) >= TP3 ({target_3:.2f})")
                        print(f"   Status: TP3 (position closed, all profit taken)")
                    break  # TP3 closes position - all profit taken, no SL possible after
                    
            else:
                # SELL trade: SL is above entry, TPs are below entry
                # Check if SL (high) is hit in this candle
                if stop_loss and high >= stop_loss:
                    sl_hit = True
                    sl_hit_datetime = timestamp
                    final_status = 'SL'
                    if debug_mode:
                        print(f"\n🛑 [HIT] SL HIT at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Condition: High ({high:.2f}) >= SL ({stop_loss:.2f})")
                        print(f"   Previous TPs: TP1={tp1_hit}, TP2={tp2_hit}, TP3={tp3_hit}")
                        print(f"   Final Status: SL")
                    break  # SL closes entire position immediately (even if TPs were hit before)
                
                # Then check TPs (low) - price moving down
                # Track TPs sequentially, but SL can override later
                if target_1 and not tp1_hit and low <= target_1:
                    tp1_hit = True
                    tp1_hit_datetime = timestamp
                    final_status = 'TP1'  # Update status, but continue (SL can override)
                    if debug_mode:
                        print(f"\n✅ [HIT] TP1 HIT at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Condition: Low ({low:.2f}) <= TP1 ({target_1:.2f})")
                        print(f"   Status: TP1 (continuing to check TP2/TP3/SL)")
                
                if target_2 and tp1_hit and not tp2_hit and low <= target_2:
                    tp2_hit = True
                    tp2_hit_datetime = timestamp
                    final_status = 'TP2'  # Update status, but continue (SL can override)
                    if debug_mode:
                        print(f"\n✅ [HIT] TP2 HIT at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Condition: Low ({low:.2f}) <= TP2 ({target_2:.2f})")
                        print(f"   Status: TP2 (continuing to check TP3/SL)")
                
                if target_3 and tp2_hit and not tp3_hit and low <= target_3:
                    tp3_hit = True
                    tp3_hit_datetime = timestamp
                    final_status = 'TP3'
                    if debug_mode:
                        print(f"\n✅ [HIT] TP3 HIT at {timestamp}")
                        print(f"   Candle: H={high:.2f}, L={low:.2f}, C={close:.2f}")
                        print(f"   Condition: Low ({low:.2f}) <= TP3 ({target_3:.2f})")
                        print(f"   Status: TP3 (position closed, all profit taken)")
                    break  # TP3 closes position - all profit taken, no SL possible after
        
        # Determine final status if still open
        if not sl_hit and not tp3_hit:
            if tp2_hit:
                final_status = 'TP2'
            elif tp1_hit:
                final_status = 'TP1'
            else:
                # Check if expired (beyond 72 hour hold period)
                if market_data.index[-1] - market_data.index[0] >= timedelta(hours=72):
                    final_status = 'EXPIRED'
                else:
                    final_status = 'OPEN'
        
        # Final summary logging
        if debug_mode:
            print(f"\n{'='*80}")
            print(f"📊 [ANALYSIS SUMMARY]")
            print(f"{'='*80}")
            print(f"   Entry Reached: {entry_reached} ({entry_datetime})")
            print(f"   TP1 Hit: {tp1_hit} ({tp1_hit_datetime})")
            print(f"   TP2 Hit: {tp2_hit} ({tp2_hit_datetime})")
            print(f"   TP3 Hit: {tp3_hit} ({tp3_hit_datetime})")
            print(f"   SL Hit: {sl_hit} ({sl_hit_datetime})")
            print(f"   Final Status: {final_status}")
            print(f"   Max Profit: {max_profit*100:.4f}%, Max Drawdown: {max_drawdown*100:.4f}%")
            print(f"{'='*80}\n")
        
        # Calculate hold time
        if tp1_hit_datetime:
            hold_time = (tp1_hit_datetime - market_data.index[0]).total_seconds() / 3600.0
        elif tp2_hit_datetime:
            hold_time = (tp2_hit_datetime - market_data.index[0]).total_seconds() / 3600.0
        elif tp3_hit_datetime:
            hold_time = (tp3_hit_datetime - market_data.index[0]).total_seconds() / 3600.0
        elif sl_hit_datetime:
            hold_time = (sl_hit_datetime - market_data.index[0]).total_seconds() / 3600.0
        else:
            hold_time = (market_data.index[-1] - market_data.index[0]).total_seconds() / 3600.0
        
        return {
            'tp1_hit': tp1_hit,
            'tp1_hit_datetime': tp1_hit_datetime.isoformat() if tp1_hit_datetime else None,
            'tp2_hit': tp2_hit,
            'tp2_hit_datetime': tp2_hit_datetime.isoformat() if tp2_hit_datetime else None,
            'tp3_hit': tp3_hit,
            'tp3_hit_datetime': tp3_hit_datetime.isoformat() if tp3_hit_datetime else None,
            'sl_hit': sl_hit,
            'sl_hit_datetime': sl_hit_datetime.isoformat() if sl_hit_datetime else None,
            'max_profit': round(max_profit * 100, 4),  # As percentage
            'max_drawdown': round(max_drawdown * 100, 4),  # As percentage
            'final_status': final_status,
            'hold_time_hours': round(hold_time, 2)
        }
    
    def calculate_provider_metrics(
        self,
        provider_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Calculate aggregated metrics for a signal provider.
        
        Args:
            provider_name: Name of the signal provider
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            
        Returns:
            Dictionary with provider metrics
        """
        if not self.supabase:
            return {"error": "Database not available"}
        
        try:
            # Fetch signals
            query = self.supabase.table('signal_provider_signals').select('*')
            query = query.eq('provider_name', provider_name)
            
            if start_date:
                query = query.gte('signal_date', start_date.isoformat())
            if end_date:
                query = query.lte('signal_date', end_date.isoformat())
            
            result = query.execute()
            signals = result.data if result.data else []
            
            if not signals:
                return {"error": f"No signals found for provider: {provider_name}"}
            
            # Fetch analysis results
            analysis_query = self.supabase.table('analysis_results').select('*')
            analysis_query = analysis_query.eq('provider_name', provider_name)
            analysis_query = analysis_query.eq('analysis_method', 'automated')
            
            if start_date:
                analysis_query = analysis_query.gte('signal_date', start_date.isoformat())
            if end_date:
                analysis_query = analysis_query.lte('signal_date', end_date.isoformat())
            
            analysis_result = analysis_query.execute()
            analyses = analysis_result.data if analysis_result.data else []
            
            # Create analysis lookup
            analysis_lookup = {a.get('signal_id'): a for a in analyses}
            
            # Calculate metrics
            total_signals = len(signals)
            analyzed_signals = len(analyses)
            
            tp1_success = sum(1 for a in analyses if a.get('tp1_hit', False))
            tp2_success = sum(1 for a in analyses if a.get('tp2_hit', False))
            tp3_success = sum(1 for a in analyses if a.get('tp3_hit', False))
            sl_hit_count = sum(1 for a in analyses if a.get('sl_hit', False))
            
            tp1_success_rate = (tp1_success / analyzed_signals * 100) if analyzed_signals > 0 else 0
            tp2_success_rate = (tp2_success / analyzed_signals * 100) if analyzed_signals > 0 else 0
            tp3_success_rate = (tp3_success / analyzed_signals * 100) if analyzed_signals > 0 else 0
            sl_hit_rate = (sl_hit_count / analyzed_signals * 100) if analyzed_signals > 0 else 0
            
            win_rate = ((tp1_success + tp2_success + tp3_success) / analyzed_signals * 100) if analyzed_signals > 0 else 0
            
            # Calculate average hold time
            hold_times = [a.get('hold_time_hours', 0) for a in analyses if a.get('hold_time_hours')]
            avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
            
            # Calculate risk/reward ratio
            # Average TP1 distance / Average SL distance
            tp1_distances = []
            sl_distances = []
            
            for signal in signals:
                entry = signal.get('entry_price')
                tp1 = signal.get('target_1')
                sl = signal.get('stop_loss')
                
                if entry and tp1:
                    if signal.get('action', '').lower() == 'buy':
                        tp1_distances.append((tp1 - entry) / entry)
                    else:
                        tp1_distances.append((entry - tp1) / entry)
                
                if entry and sl:
                    if signal.get('action', '').lower() == 'buy':
                        sl_distances.append((entry - sl) / entry)
                    else:
                        sl_distances.append((sl - entry) / entry)
            
            avg_tp1_distance = sum(tp1_distances) / len(tp1_distances) if tp1_distances else 0
            avg_sl_distance = sum(sl_distances) / len(sl_distances) if sl_distances else 0
            risk_reward_ratio = avg_tp1_distance / avg_sl_distance if avg_sl_distance > 0 else 0
            
            # Calculate backtest metrics if available
            backtest_query = self.supabase.table('backtest_results').select('*')
            backtest_query = backtest_query.eq('provider_name', provider_name)
            backtest_result = backtest_query.execute()
            backtests = backtest_result.data if backtest_result.data else []
            
            total_pnl = sum(float(b.get('net_profit_loss', 0)) for b in backtests)
            winning_trades = sum(1 for b in backtests if float(b.get('net_profit_loss', 0)) > 0)
            losing_trades = sum(1 for b in backtests if float(b.get('net_profit_loss', 0)) < 0)
            
            wins = [float(b.get('net_profit_loss', 0)) for b in backtests if float(b.get('net_profit_loss', 0)) > 0]
            losses = [float(b.get('net_profit_loss', 0)) for b in backtests if float(b.get('net_profit_loss', 0)) < 0]
            
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            return {
                'provider_name': provider_name,
                'calculation_date': datetime.now(self.tz).date().isoformat(),
                'total_signals': total_signals,
                'analyzed_signals': analyzed_signals,
                'tp1_success_count': tp1_success,
                'tp1_success_rate': round(tp1_success_rate, 2),
                'tp2_success_count': tp2_success,
                'tp2_success_rate': round(tp2_success_rate, 2),
                'tp3_success_count': tp3_success,
                'tp3_success_rate': round(tp3_success_rate, 2),
                'sl_hit_count': sl_hit_count,
                'sl_hit_rate': round(sl_hit_rate, 2),
                'win_rate': round(win_rate, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 4),
                'average_hold_time_hours': round(avg_hold_time, 2),
                'total_pnl': round(total_pnl, 2),
                'total_trades': len(backtests),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'average_win': round(avg_win, 2),
                'average_loss': round(avg_loss, 2)
            }
            
        except Exception as e:
            return {"error": f"Failed to calculate metrics: {str(e)}"}
    
    def save_analysis_result(self, result: Dict) -> bool:
        """
        Save analysis result to database.
        
        Args:
            result: Analysis result dictionary
            
        Returns:
            True if saved successfully
        """
        if not self.supabase or 'error' in result:
            return False
        
        try:
            record = {
                'signal_id': result.get('signal_id'),
                'provider_name': result.get('provider_name'),
                'symbol': result.get('symbol'),
                'signal_date': result.get('signal_date'),
                'tp1_hit': result.get('tp1_hit', False),
                'tp1_hit_datetime': result.get('tp1_hit_datetime'),
                'tp2_hit': result.get('tp2_hit', False),
                'tp2_hit_datetime': result.get('tp2_hit_datetime'),
                'tp3_hit': result.get('tp3_hit', False),
                'tp3_hit_datetime': result.get('tp3_hit_datetime'),
                'sl_hit': result.get('sl_hit', False),
                'sl_hit_datetime': result.get('sl_hit_datetime'),
                'max_profit': result.get('max_profit'),
                'max_drawdown': result.get('max_drawdown'),
                'final_status': result.get('final_status'),
                'hold_time_hours': result.get('hold_time_hours'),
                'analysis_method': 'automated',
                'timezone_offset': '+04:00',
                'analysis_date': result.get('analysis_date')
            }
            
            # Remove None values
            record = {k: v for k, v in record.items() if v is not None}
            
            self.supabase.table('analysis_results').upsert(record).execute()
            return True
            
        except Exception as e:
            print(f"Error saving analysis result: {e}")
            return False

