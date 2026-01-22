"""
Signal Backtesting Engine
Simulates historical trades based on signal provider signals and calculates performance metrics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz

from tradingagents.database.config import get_supabase
from tradingagents.dataflows.market_data_service import fetch_ohlcv


class SignalBacktester:
    """
    Backtesting engine for signal provider signals.
    Simulates trades and calculates P&L, success rates, and performance metrics.
    """
    
    def __init__(self, timezone: str = 'Asia/Dubai'):
        """
        Initialize backtester.
        
        Args:
            timezone: Timezone for all calculations (default: GMT+4)
        """
        self.tz = pytz.timezone(timezone)
        self.utc_tz = pytz.timezone('UTC')
        self.supabase = get_supabase()
    
    def backtest_signal(
        self,
        signal: Dict,
        initial_capital: float = 10000.0,
        position_size_percent: float = 10.0,
        commission: float = 0.0,
        max_hold_days: int = 30
    ) -> Dict:
        """
        Backtest a single signal.
        
        Args:
            signal: Signal dictionary with fields:
                - symbol, signal_date, action, entry_price, target_1, target_2, target_3, stop_loss
            initial_capital: Starting capital
            position_size_percent: Percentage of capital to risk per trade
            commission: Commission per trade
            max_hold_days: Maximum days to hold before considering expired
            
        Returns:
            Dictionary with backtest results
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
            
            if action not in ['buy', 'sell']:
                return {"error": f"Invalid action: {action}"}
            
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
            
            # Calculate position size
            position_size = initial_capital * (position_size_percent / 100.0)
            
            # Fetch market data from signal_date onwards
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
                return {"error": f"No market data available for {symbol} from {signal_date}"}
            
            # Ensure timestamp column exists and is datetime
            if 'timestamp' in market_data.columns:
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
                market_data = market_data.set_index('timestamp')
            elif market_data.index.name == 'timestamp' or isinstance(market_data.index, pd.DatetimeIndex):
                pass
            else:
                return {"error": "Market data missing timestamp column"}
            
            # Filter data from signal_date onwards
            market_data = market_data[market_data.index >= signal_date]
            
            if market_data.empty:
                return {"error": "No market data after signal date"}
            
            # Simulate trade
            result = self._simulate_trade(
                market_data=market_data,
                action=action,
                entry_price=entry_price,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3,
                stop_loss=stop_loss,
                position_size=position_size,
                commission=commission,
                signal_date=signal_date
            )
            
            # Add metadata
            result['signal_id'] = signal.get('id')
            result['provider_name'] = signal.get('provider_name')
            result['symbol'] = symbol
            result['backtest_start_date'] = signal_date.isoformat()
            result['initial_capital'] = initial_capital
            result['position_size_percent'] = position_size_percent
            result['position_size'] = position_size
            
            return result
            
        except Exception as e:
            return {"error": f"Backtest failed: {str(e)}"}
    
    def _simulate_trade(
        self,
        market_data: pd.DataFrame,
        action: str,
        entry_price: float,
        target_1: Optional[float],
        target_2: Optional[float],
        target_3: Optional[float],
        stop_loss: Optional[float],
        position_size: float,
        commission: float,
        signal_date: datetime
    ) -> Dict:
        """
        Simulate a trade execution.
        
        Returns:
            Dictionary with trade results
        """
        # Initialize tracking
        entry_datetime = None
        exit_datetime = None
        exit_price = None
        exit_reason = 'EXPIRED'
        
        # Track partial exits for multiple TPs
        remaining_position = 1.0  # 100% of position
        tp1_exit_percent = 0.33
        tp2_exit_percent = 0.33
        tp3_exit_percent = 0.34
        
        total_pnl = 0.0
        max_profit = 0.0
        max_drawdown = 0.0
        
        tp1_hit = False
        tp2_hit = False
        tp3_hit = False
        sl_hit = False
        
        # Determine price direction based on action
        is_buy = (action == 'buy')
        
        # Check each minute for exit conditions
        for timestamp, row in market_data.iterrows():
            high = float(row.get('High', 0))
            low = float(row.get('Low', 0))
            close = float(row.get('Close', 0))
            
            # Entry: Check if price reached entry_price
            if entry_datetime is None:
                if is_buy and low <= entry_price <= high:
                    entry_datetime = timestamp
                elif not is_buy and low <= entry_price <= high:
                    entry_datetime = timestamp
            
            if entry_datetime is None:
                continue  # Haven't entered yet
            
            # After entry, check exit conditions
            current_price = close
            
            # Calculate current P&L
            if is_buy:
                current_pnl = (current_price - entry_price) / entry_price * position_size
            else:
                current_pnl = (entry_price - current_price) / entry_price * position_size
            
            # Track max profit and drawdown
            if current_pnl > max_profit:
                max_profit = current_pnl
            if current_pnl < max_drawdown:
                max_drawdown = current_pnl
            
            # Check TP1
            if target_1 and not tp1_hit and remaining_position > 0:
                if (is_buy and high >= target_1) or (not is_buy and low <= target_1):
                    tp1_hit = True
                    exit_price_partial = target_1
                    pnl_partial = (exit_price_partial - entry_price) / entry_price * position_size * tp1_exit_percent if is_buy else (entry_price - exit_price_partial) / entry_price * position_size * tp1_exit_percent
                    total_pnl += pnl_partial - (commission * tp1_exit_percent)
                    remaining_position -= tp1_exit_percent
                    if remaining_position <= 0.01:  # All position closed
                        exit_datetime = timestamp
                        exit_price = exit_price_partial
                        exit_reason = 'TP1'
                        break
            
            # Check TP2
            if target_2 and tp1_hit and not tp2_hit and remaining_position > 0:
                if (is_buy and high >= target_2) or (not is_buy and low <= target_2):
                    tp2_hit = True
                    exit_price_partial = target_2
                    pnl_partial = (exit_price_partial - entry_price) / entry_price * position_size * tp2_exit_percent if is_buy else (entry_price - exit_price_partial) / entry_price * position_size * tp2_exit_percent
                    total_pnl += pnl_partial - (commission * tp2_exit_percent)
                    remaining_position -= tp2_exit_percent
                    if remaining_position <= 0.01:  # All position closed
                        exit_datetime = timestamp
                        exit_price = exit_price_partial
                        exit_reason = 'TP2'
                        break
            
            # Check TP3
            if target_3 and tp2_hit and not tp3_hit and remaining_position > 0:
                if (is_buy and high >= target_3) or (not is_buy and low <= target_3):
                    tp3_hit = True
                    exit_price_partial = target_3
                    pnl_partial = (exit_price_partial - entry_price) / entry_price * position_size * tp3_exit_percent if is_buy else (entry_price - exit_price_partial) / entry_price * position_size * tp3_exit_percent
                    total_pnl += pnl_partial - (commission * tp3_exit_percent)
                    remaining_position -= tp3_exit_percent
                    exit_datetime = timestamp
                    exit_price = exit_price_partial
                    exit_reason = 'TP3'
                    break
            
            # Check Stop Loss (takes priority, closes entire position)
            if stop_loss:
                if (is_buy and low <= stop_loss) or (not is_buy and high >= stop_loss):
                    sl_hit = True
                    exit_datetime = timestamp
                    exit_price = stop_loss
                    exit_reason = 'SL'
                    # Calculate P&L for remaining position
                    pnl_remaining = (exit_price - entry_price) / entry_price * position_size * remaining_position if is_buy else (entry_price - exit_price) / entry_price * position_size * remaining_position
                    total_pnl += pnl_remaining - (commission * remaining_position)
                    break
        
        # If still open at end, calculate final P&L
        if exit_datetime is None and remaining_position > 0:
            final_price = float(market_data.iloc[-1]['Close'])
            pnl_remaining = (final_price - entry_price) / entry_price * position_size * remaining_position if is_buy else (entry_price - final_price) / entry_price * position_size * remaining_position
            total_pnl += pnl_remaining - (commission * remaining_position)
            exit_datetime = market_data.index[-1]
            exit_price = final_price
        
        # Calculate metrics
        if entry_datetime and exit_datetime:
            hold_time = (exit_datetime - entry_datetime).total_seconds() / 3600.0  # hours
        else:
            hold_time = 0.0
        
        profit_loss_percent = (total_pnl / position_size) * 100 if position_size > 0 else 0.0
        net_profit_loss = total_pnl
        
        return {
            'entry_datetime': entry_datetime.isoformat() if entry_datetime else None,
            'exit_datetime': exit_datetime.isoformat() if exit_datetime else None,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'tp1_hit': tp1_hit,
            'tp2_hit': tp2_hit,
            'tp3_hit': tp3_hit,
            'sl_hit': sl_hit,
            'profit_loss': round(total_pnl, 2),
            'profit_loss_percent': round(profit_loss_percent, 4),
            'commission': commission,
            'net_profit_loss': round(net_profit_loss, 2),
            'max_profit': round(max_profit, 2),
            'max_drawdown': round(max_drawdown, 2),
            'hold_time_hours': round(hold_time, 2),
            'tp1_exit_percent': tp1_exit_percent * 100,
            'tp2_exit_percent': tp2_exit_percent * 100,
            'tp3_exit_percent': tp3_exit_percent * 100
        }
    
    def backtest_multiple_signals(
        self,
        signals: List[Dict],
        initial_capital: float = 10000.0,
        position_size_percent: float = 10.0,
        commission: float = 0.0,
        max_hold_days: int = 30
    ) -> List[Dict]:
        """
        Backtest multiple signals.
        
        Returns:
            List of backtest result dictionaries
        """
        results = []
        for signal in signals:
            result = self.backtest_signal(
                signal=signal,
                initial_capital=initial_capital,
                position_size_percent=position_size_percent,
                commission=commission,
                max_hold_days=max_hold_days
            )
            results.append(result)
        return results
    
    def save_backtest_result(self, result: Dict) -> bool:
        """
        Save backtest result to database.
        
        Args:
            result: Backtest result dictionary
            
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
                'backtest_start_date': result.get('backtest_start_date'),
                'backtest_end_date': result.get('exit_datetime'),
                'initial_capital': result.get('initial_capital'),
                'position_size_percent': result.get('position_size_percent'),
                'entry_datetime': result.get('entry_datetime'),
                'exit_datetime': result.get('exit_datetime'),
                'entry_price': result.get('entry_price'),
                'exit_price': result.get('exit_price'),
                'position_size': result.get('position_size'),
                'profit_loss': result.get('profit_loss'),
                'profit_loss_percent': result.get('profit_loss_percent'),
                'commission': result.get('commission'),
                'net_profit_loss': result.get('net_profit_loss'),
                'exit_reason': result.get('exit_reason'),
                'tp1_exit_percent': result.get('tp1_exit_percent'),
                'tp2_exit_percent': result.get('tp2_exit_percent'),
                'tp3_exit_percent': result.get('tp3_exit_percent'),
                'max_profit': result.get('max_profit'),
                'max_drawdown': result.get('max_drawdown'),
                'hold_time_hours': result.get('hold_time_hours'),
                'timezone_offset': '+04:00',
                'backtest_date': datetime.now(self.tz).isoformat()
            }
            
            # Remove None values
            record = {k: v for k, v in record.items() if v is not None}
            
            self.supabase.table('backtest_results').insert(record).execute()
            return True
            
        except Exception as e:
            print(f"Error saving backtest result: {e}")
            return False

