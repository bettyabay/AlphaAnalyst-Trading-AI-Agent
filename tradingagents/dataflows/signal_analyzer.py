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
        max_hold_days: int = 30,
        price_tolerance: float = 0.0001
    ) -> Dict:
        """
        Analyze a single signal to determine TP/SL hits.
        
        Args:
            signal: Signal dictionary
            max_hold_days: Maximum days to analyze
            price_tolerance: Price tolerance for hit detection (0.01% default)
            
        Returns:
            Dictionary with analysis results
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
            
            end_date = signal_date + timedelta(days=max_hold_days)
            market_data = fetch_ohlcv(
                symbol=symbol,
                interval='1min',
                start=signal_date.astimezone(self.utc_tz),
                end=end_date.astimezone(self.utc_tz),
                asset_class=asset_class
            )
            
            # If no data found and symbol doesn't have C: prefix, try with C: prefix
            if (market_data is None or market_data.empty) and not symbol.startswith("C:") and len(symbol) >= 6 and len(symbol) <= 7:
                symbol_with_prefix = f"C:{symbol}"
                market_data = fetch_ohlcv(
                    symbol=symbol_with_prefix,
                    interval='1min',
                    start=signal_date.astimezone(self.utc_tz),
                    end=end_date.astimezone(self.utc_tz),
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
            
            # Filter from signal_date onwards
            market_data = market_data[market_data.index >= signal_date]
            
            if market_data.empty:
                return {"error": "No market data after signal date"}
            
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
            
            # Add metadata
            result['signal_id'] = signal.get('id')
            result['provider_name'] = signal.get('provider_name')
            result['symbol'] = symbol
            result['signal_date'] = signal_date.isoformat()
            result['analysis_date'] = datetime.now(self.tz).isoformat()
            
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
        
        for timestamp, row in market_data.iterrows():
            high = float(row.get('High', 0))
            low = float(row.get('Low', 0))
            close = float(row.get('Close', 0))
            
            # Check entry
            if not entry_reached:
                if is_buy and low <= entry_price <= high:
                    entry_reached = True
                elif not is_buy and low <= entry_price <= high:
                    entry_reached = True
            
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
            
            # Check TP1
            if target_1 and not tp1_hit:
                if (is_buy and high >= target_1) or (not is_buy and low <= target_1):
                    tp1_hit = True
                    tp1_hit_datetime = timestamp
                    final_status = 'TP1'
            
            # Check TP2 (only if TP1 hit)
            if target_2 and tp1_hit and not tp2_hit:
                if (is_buy and high >= target_2) or (not is_buy and low <= target_2):
                    tp2_hit = True
                    tp2_hit_datetime = timestamp
                    final_status = 'TP2'
            
            # Check TP3 (only if TP2 hit)
            if target_3 and tp2_hit and not tp3_hit:
                if (is_buy and high >= target_3) or (not is_buy and low <= target_3):
                    tp3_hit = True
                    tp3_hit_datetime = timestamp
                    final_status = 'TP3'
            
            # Check Stop Loss (takes priority)
            if stop_loss:
                if (is_buy and low <= stop_loss) or (not is_buy and high >= stop_loss):
                    sl_hit = True
                    sl_hit_datetime = timestamp
                    final_status = 'SL'
                    break  # SL closes position immediately
        
        # Determine final status if still open
        if not sl_hit and not tp3_hit:
            if tp2_hit:
                final_status = 'TP2'
            elif tp1_hit:
                final_status = 'TP1'
            else:
                # Check if expired (beyond max hold period)
                if market_data.index[-1] - market_data.index[0] >= timedelta(days=30):
                    final_status = 'EXPIRED'
                else:
                    final_status = 'OPEN'
        
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

