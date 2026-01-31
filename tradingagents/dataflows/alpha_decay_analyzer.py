"""
Alpha Decay & Robustness Analyzer
Detects performance decay and measures alpha against benchmarks.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pytz

from tradingagents.database.db_service import get_backtest_results_with_efficiency
from tradingagents.dataflows.market_data_service import fetch_ohlcv


class AlphaDecayAnalyzer:
    """
    Analyzes signal provider performance for decay and alpha generation.
    Measures performance against benchmarks and tracks rolling statistics.
    """
    
    def __init__(self, timezone: str = 'Asia/Dubai'):
        """
        Initialize analyzer.
        
        Args:
            timezone: Timezone for all calculations (default: GMT+4)
        """
        self.tz = pytz.timezone(timezone)
        self.utc_tz = pytz.timezone('UTC')
    
    def generate_benchmark(
        self,
        market_df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Generate Buy & Hold benchmark returns from market data.
        
        Args:
            market_df: DataFrame with OHLCV data, indexed by timestamp
            start_date: Optional start date to filter
            end_date: Optional end date to filter
            
        Returns:
            Series of cumulative % returns indexed by date
        """
        if market_df.empty:
            return pd.Series(dtype=float)
        
        # Ensure we have a datetime index
        if not isinstance(market_df.index, pd.DatetimeIndex):
            if 'timestamp' in market_df.columns:
                market_df = market_df.set_index('timestamp')
            elif 'date' in market_df.columns:
                market_df = market_df.set_index('date')
            else:
                raise ValueError("Market data must have datetime index or timestamp/date column")
        
        # Filter by date range if provided
        if start_date:
            market_df = market_df[market_df.index >= start_date]
        if end_date:
            market_df = market_df[market_df.index <= end_date]
        
        if market_df.empty:
            return pd.Series(dtype=float)
        
        # Ensure we have Close prices
        if 'Close' not in market_df.columns:
            if 'close' in market_df.columns:
                market_df['Close'] = market_df['close']
            else:
                raise ValueError("Market data must have Close/close column")
        
        # Sort by date
        market_df = market_df.sort_index()
        
        # Calculate daily returns (percentage change)
        daily_returns = market_df['Close'].pct_change().fillna(0)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        
        # Convert to percentage
        cumulative_returns = cumulative_returns * 100
        
        return cumulative_returns
    
    def generate_naive_strategy_benchmark(
        self,
        market_df: pd.DataFrame,
        sma_period: int = 50,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Generate a naive strategy benchmark (Long when Price > SMA).
        
        Args:
            market_df: DataFrame with OHLCV data
            sma_period: Period for Simple Moving Average
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Series of cumulative % returns
        """
        if market_df.empty:
            return pd.Series(dtype=float)
        
        # Ensure datetime index
        if not isinstance(market_df.index, pd.DatetimeIndex):
            if 'timestamp' in market_df.columns:
                market_df = market_df.set_index('timestamp')
            elif 'date' in market_df.columns:
                market_df = market_df.set_index('date')
        
        # Filter by date range
        if start_date:
            market_df = market_df[market_df.index >= start_date]
        if end_date:
            market_df = market_df[market_df.index <= end_date]
        
        if market_df.empty:
            return pd.Series(dtype=float)
        
        market_df = market_df.sort_index()
        
        # Ensure Close column
        if 'Close' not in market_df.columns:
            if 'close' in market_df.columns:
                market_df['Close'] = market_df['close']
            else:
                raise ValueError("Market data must have Close/close column")
        
        # Calculate SMA
        market_df['SMA'] = market_df['Close'].rolling(window=sma_period).mean()
        
        # Generate signals: Long when Close > SMA
        market_df['Signal'] = (market_df['Close'] > market_df['SMA']).astype(int)
        
        # Calculate returns: If signal is 1 (long), take the daily return, else 0
        daily_returns = market_df['Close'].pct_change().fillna(0)
        strategy_returns = daily_returns * market_df['Signal'].shift(1).fillna(0)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod() - 1
        cumulative_returns = cumulative_returns * 100
        
        return cumulative_returns
    
    def resample_trades_to_daily(
        self,
        signals_df: pd.DataFrame,
        pnl_column: str = 'net_profit_loss'
    ) -> pd.DataFrame:
        """
        Resample trade PnL to daily equity curve.
        Handles data frequency mismatch between trades and benchmark.
        
        Args:
            signals_df: DataFrame with trade results (must have exit_datetime and PnL)
            pnl_column: Column name for PnL values
            
        Returns:
            DataFrame with daily aggregated PnL and cumulative equity
        """
        if signals_df.empty:
            return pd.DataFrame()
        
        # Ensure exit_datetime is datetime
        if 'exit_datetime' not in signals_df.columns:
            raise ValueError("signals_df must have 'exit_datetime' column")
        
        if pnl_column not in signals_df.columns:
            raise ValueError(f"signals_df must have '{pnl_column}' column")
        
        # Convert to datetime if needed
        signals_df = signals_df.copy()
        signals_df['exit_datetime'] = pd.to_datetime(signals_df['exit_datetime'], errors='coerce')
        
        # Remove rows with invalid dates
        signals_df = signals_df.dropna(subset=['exit_datetime'])
        
        if signals_df.empty:
            return pd.DataFrame()
        
        # Set exit_datetime as index
        signals_df = signals_df.set_index('exit_datetime')
        
        # Ensure PnL is numeric
        signals_df[pnl_column] = pd.to_numeric(signals_df[pnl_column], errors='coerce').fillna(0)
        
        # Resample to daily: sum all PnL for each day
        daily_pnl = signals_df[pnl_column].resample('D').sum()
        
        # Create DataFrame with daily data
        daily_df = pd.DataFrame({
            'daily_pnl': daily_pnl,
            'trade_count': signals_df[pnl_column].resample('D').count()
        })
        
        # Calculate cumulative PnL
        daily_df['cumulative_pnl'] = daily_df['daily_pnl'].cumsum()
        
        # Calculate cumulative return % (assuming initial capital of 10000)
        initial_capital = 10000.0
        daily_df['cumulative_return_pct'] = (daily_df['cumulative_pnl'] / initial_capital) * 100
        
        return daily_df
    
    def calculate_rolling_stats(
        self,
        signals_df: pd.DataFrame,
        window_size: int = 50,
        pnl_column: str = 'net_profit_loss',
        time_column: str = 'exit_datetime'
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics (win rate, average PnL).
        
        Args:
            signals_df: DataFrame with trade results
            window_size: Number of trades for rolling window
            pnl_column: Column name for PnL values
            time_column: Column name for time (to sort by)
            
        Returns:
            DataFrame with rolling statistics added
        """
        if signals_df.empty:
            return pd.DataFrame()
        
        df = signals_df.copy()
        
        # Ensure time column is datetime and sort
        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
            df = df.sort_values(time_column).reset_index(drop=True)
        else:
            df = df.sort_index().reset_index(drop=True)
        
        # Ensure PnL is numeric
        if pnl_column not in df.columns:
            raise ValueError(f"Column '{pnl_column}' not found in signals_df")
        
        df[pnl_column] = pd.to_numeric(df[pnl_column], errors='coerce').fillna(0)
        
        # Convert PnL to binary win/loss
        df['Win_Binary'] = (df[pnl_column] > 0).astype(int)
        
        # Calculate rolling win rate
        df['Rolling_Win_Rate'] = df['Win_Binary'].rolling(window=window_size, min_periods=1).mean() * 100
        
        # Calculate rolling average PnL
        df['Rolling_Avg_PnL'] = df[pnl_column].rolling(window=window_size, min_periods=1).mean()
        
        # Calculate rolling expectancy (average PnL per trade)
        df['Rolling_Expectancy'] = df['Rolling_Avg_PnL']
        
        return df
    
    def calculate_drawdown(
        self,
        equity_curve: pd.Series,
        return_pct: bool = True
    ) -> pd.DataFrame:
        """
        Calculate drawdown duration and depth.
        
        Args:
            equity_curve: Series of cumulative PnL or returns (indexed by time)
            return_pct: If True, equity_curve is in percentage; if False, it's absolute PnL
            
        Returns:
            DataFrame with drawdown metrics
        """
        if equity_curve.empty:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame({'cumulative': equity_curve})
        df = df.sort_index()
        
        # Calculate High Water Mark (HWM)
        df['hwm'] = df['cumulative'].cummax()
        
        # Calculate drawdown (current - HWM)
        df['drawdown'] = df['cumulative'] - df['hwm']
        
        # Convert to percentage if needed
        if not return_pct:
            # If absolute PnL, convert to percentage assuming initial capital
            initial_capital = 10000.0
            df['drawdown_pct'] = (df['drawdown'] / initial_capital) * 100
        else:
            df['drawdown_pct'] = df['drawdown']
        
        # Calculate drawdown duration
        df['is_drawdown'] = df['drawdown'] < 0
        df['drawdown_start'] = (df['is_drawdown'] & ~df['is_drawdown'].shift(1).fillna(False)).astype(int)
        df['drawdown_id'] = df['drawdown_start'].cumsum()
        
        # Calculate duration for each drawdown period
        drawdown_durations = []
        for drawdown_id in df[df['is_drawdown']]['drawdown_id'].unique():
            drawdown_period = df[df['drawdown_id'] == drawdown_id]
            if not drawdown_period.empty:
                duration = (drawdown_period.index[-1] - drawdown_period.index[0]).days
                max_dd = drawdown_period['drawdown_pct'].min()
                drawdown_durations.append({
                    'start_date': drawdown_period.index[0],
                    'end_date': drawdown_period.index[-1],
                    'duration_days': duration,
                    'max_drawdown_pct': max_dd
                })
        
        return df
    
    def detect_decay(
        self,
        rolling_stats_df: pd.DataFrame,
        win_rate_column: str = 'Rolling_Win_Rate',
        threshold_pct: float = 10.0
    ) -> Dict:
        """
        Detect if performance is decaying.
        
        Args:
            rolling_stats_df: DataFrame with rolling statistics
            win_rate_column: Column name for rolling win rate
            threshold_pct: Percentage drop threshold to flag decay
            
        Returns:
            Dictionary with decay detection results
        """
        if rolling_stats_df.empty or win_rate_column not in rolling_stats_df.columns:
            return {
                'has_decay': False,
                'decay_pct': 0.0,
                'first_window_win_rate': 0.0,
                'last_window_win_rate': 0.0,
                'message': 'Insufficient data'
            }
        
        # Get first and last valid win rates (use a window of values for stability)
        valid_win_rates = rolling_stats_df[win_rate_column].dropna()
        
        if len(valid_win_rates) < 2:
            return {
                'has_decay': False,
                'decay_pct': 0.0,
                'first_window_win_rate': float(valid_win_rates.iloc[0]) if len(valid_win_rates) > 0 else 0.0,
                'last_window_win_rate': float(valid_win_rates.iloc[-1]) if len(valid_win_rates) > 0 else 0.0,
                'message': 'Insufficient data points'
            }
        
        # Use average of first 10% and last 10% of data for more stable comparison
        window_size = max(1, len(valid_win_rates) // 10)
        first_window = valid_win_rates.iloc[:window_size]
        last_window = valid_win_rates.iloc[-window_size:]
        
        first_win_rate = float(first_window.mean())
        last_win_rate = float(last_window.mean())
        
        # Calculate decay percentage
        if first_win_rate > 0:
            decay_pct = ((first_win_rate - last_win_rate) / first_win_rate) * 100
        else:
            decay_pct = 0.0
        
        has_decay = decay_pct > threshold_pct
        
        return {
            'has_decay': has_decay,
            'decay_pct': round(decay_pct, 2),
            'first_window_win_rate': round(first_win_rate, 2),
            'last_window_win_rate': round(last_win_rate, 2),
            'message': f'Performance decayed by {decay_pct:.2f}%' if has_decay else 'No significant decay detected'
        }
    
    def analyze_provider_alpha(
        self,
        provider_name: str,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        window_size: int = 50,
        benchmark_type: str = 'buy_hold'  # 'buy_hold' or 'naive_strategy'
    ) -> Dict:
        """
        Complete alpha decay analysis for a provider.
        
        Args:
            provider_name: Provider name
            symbol: Optional symbol filter
            start_date: Optional start date
            end_date: Optional end date
            window_size: Rolling window size
            benchmark_type: Type of benchmark ('buy_hold' or 'naive_strategy')
            
        Returns:
            Dictionary with all analysis results
        """
        # Fetch backtest results
        backtest_df = get_backtest_results_with_efficiency(
            provider_name=provider_name,
            symbol=symbol,
            limit=10000
        )
        
        if backtest_df.empty:
            return {
                'error': f'No backtest results found for provider: {provider_name}'
            }
        
        # Filter by date range if provided
        if start_date:
            if 'exit_datetime' in backtest_df.columns:
                backtest_df = backtest_df[pd.to_datetime(backtest_df['exit_datetime']) >= start_date]
        if end_date:
            if 'exit_datetime' in backtest_df.columns:
                backtest_df = backtest_df[pd.to_datetime(backtest_df['exit_datetime']) <= end_date]
        
        if backtest_df.empty:
            return {
                'error': 'No backtest results in specified date range'
            }
        
        # Get symbol for benchmark (use first symbol if not specified)
        if not symbol:
            symbol = backtest_df['symbol'].iloc[0] if 'symbol' in backtest_df.columns else None
        
        if not symbol:
            return {
                'error': 'Could not determine symbol for benchmark'
            }
        
        # Resample trades to daily equity curve
        daily_equity = self.resample_trades_to_daily(backtest_df, pnl_column='net_profit_loss')
        
        if daily_equity.empty:
            return {
                'error': 'Could not generate daily equity curve'
            }
        
        # Fetch market data for benchmark
        # Determine asset class
        asset_class = None
        if symbol.startswith("C:"):
            asset_class = "Currencies"
        elif symbol.startswith("I:") or symbol.startswith("^"):
            asset_class = "Indices"
        elif "XAU" in symbol or "XAG" in symbol or "*" in symbol:
            asset_class = "Commodities"
        else:
            if len(symbol) >= 6 and len(symbol) <= 7:
                asset_class = "Currencies"
            else:
                asset_class = "Stocks"
        
        # Get date range from trades
        trade_start = pd.to_datetime(backtest_df['exit_datetime']).min() if 'exit_datetime' in backtest_df.columns else None
        trade_end = pd.to_datetime(backtest_df['exit_datetime']).max() if 'exit_datetime' in backtest_df.columns else None
        
        # Fetch market data
        market_data = fetch_ohlcv(
            symbol=symbol,
            interval='1d',
            start=trade_start.astimezone(self.utc_tz) if trade_start else None,
            end=trade_end.astimezone(self.utc_tz) if trade_end else None,
            asset_class=asset_class
        )
        
        if market_data is None or market_data.empty:
            return {
                'error': f'Could not fetch market data for {symbol}'
            }
        
        # Generate benchmark
        if benchmark_type == 'buy_hold':
            benchmark_returns = self.generate_benchmark(
                market_data,
                start_date=trade_start,
                end_date=trade_end
            )
        else:
            benchmark_returns = self.generate_naive_strategy_benchmark(
                market_data,
                start_date=trade_start,
                end_date=trade_end
            )
        
        # Align benchmark with daily equity curve dates
        # Reindex benchmark to match daily equity dates
        aligned_benchmark = benchmark_returns.reindex(daily_equity.index, method='ffill').fillna(0)
        
        # Calculate rolling statistics
        rolling_stats = self.calculate_rolling_stats(
            backtest_df,
            window_size=window_size,
            pnl_column='net_profit_loss',
            time_column='exit_datetime'
        )
        
        # Detect decay
        decay_detection = self.detect_decay(rolling_stats)
        
        # Calculate drawdown
        drawdown_df = self.calculate_drawdown(
            daily_equity['cumulative_return_pct'],
            return_pct=True
        )
        
        # Find longest drawdown
        max_drawdown_duration = 0
        max_drawdown_pct = 0
        if not drawdown_df.empty:
            max_drawdown_pct = drawdown_df['drawdown_pct'].min()
            # Calculate duration of max drawdown
            max_dd_period = drawdown_df[drawdown_df['drawdown_pct'] == max_drawdown_pct]
            if not max_dd_period.empty:
                max_drawdown_duration = (max_dd_period.index[-1] - max_dd_period.index[0]).days
        
        return {
            'provider_name': provider_name,
            'symbol': symbol,
            'daily_equity': daily_equity,
            'benchmark_returns': aligned_benchmark,
            'rolling_stats': rolling_stats,
            'decay_detection': decay_detection,
            'drawdown_df': drawdown_df,
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'max_drawdown_duration_days': max_drawdown_duration,
            'total_trades': len(backtest_df),
            'date_range': {
                'start': trade_start.isoformat() if trade_start else None,
                'end': trade_end.isoformat() if trade_end else None
            }
        }

