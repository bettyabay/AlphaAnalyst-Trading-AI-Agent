"""
Trade Efficiency Analysis Module
Calculates MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion)
for historical trades to analyze trade execution efficiency.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz

from tradingagents.database.config import get_supabase
from tradingagents.dataflows.market_data_service import fetch_ohlcv


class TradeEfficiencyAnalyzer:
    """
    Analyzes trade efficiency using MFE/MAE metrics.
    """
    
    def __init__(self, timezone: str = 'Asia/Dubai'):
        """
        Initialize efficiency analyzer.
        
        Args:
            timezone: Timezone for all calculations (default: GMT+4)
        """
        self.tz = pytz.timezone(timezone)
        self.utc_tz = pytz.timezone('UTC')
        self.supabase = get_supabase()
    
    def calculate_excursions(
        self,
        trade: Dict,
        market_data: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Calculate MFE and MAE for a single trade.
        
        Args:
            trade: Dictionary with keys:
                - entry_datetime: Entry timestamp
                - exit_datetime: Exit timestamp
                - entry_price: Entry price
                - direction: 'BUY' or 'SELL' (defaults to 'BUY' if not provided)
                - symbol: Symbol for pip calculation
            market_data: DataFrame with OHLC data indexed by timestamp
        
        Returns:
            Dictionary with 'mfe', 'mae', 'mfe_pips', 'mae_pips', 'confidence'
        """
        try:
            # Parse entry/exit times
            entry_time = self._parse_datetime(trade.get('entry_datetime'))
            exit_time = self._parse_datetime(trade.get('exit_datetime'))
            entry_price = float(trade.get('entry_price', 0))
            
            if entry_price == 0:
                return {
                    'mfe': 0.0,
                    'mae': 0.0,
                    'mfe_pips': 0.0,
                    'mae_pips': 0.0,
                    'confidence': 'ERROR',
                    'error': 'Invalid entry price'
                }
            
            # Determine direction (default to BUY if not specified)
            direction = trade.get('direction', 'BUY').upper()
            if direction not in ['BUY', 'SELL']:
                # Try to infer from action field
                action = trade.get('action', '').upper()
                if action in ['BUY', 'LONG']:
                    direction = 'BUY'
                elif action in ['SELL', 'SHORT']:
                    direction = 'SELL'
                else:
                    direction = 'BUY'  # Default
            
            # Slice market data to trade lifespan
            trade_lifespan = market_data[
                (market_data.index >= entry_time) & 
                (market_data.index <= exit_time)
            ].copy()
            
            if trade_lifespan.empty:
                return {
                    'mfe': 0.0,
                    'mae': 0.0,
                    'mfe_pips': 0.0,
                    'mae_pips': 0.0,
                    'confidence': 'LOW',
                    'error': 'No market data in trade lifespan'
                }
            
            # Find extremes
            max_price_reached = trade_lifespan['High'].max()
            min_price_reached = trade_lifespan['Low'].min()
            
            # Check if entry/exit are on same bar (low confidence)
            same_bar = len(trade_lifespan) == 1
            confidence = 'LOW' if same_bar else 'HIGH'
            
            # Calculate MFE and MAE based on direction
            if direction == 'BUY':
                mfe = max_price_reached - entry_price
                mae = entry_price - min_price_reached
            else:  # SELL
                mfe = entry_price - min_price_reached
                mae = max_price_reached - entry_price
            
            # Ensure non-negative values
            mfe = max(0.0, float(mfe))
            mae = max(0.0, float(mae))
            
            # Calculate pips
            symbol = trade.get('symbol', '')
            pip_value = self._calculate_pip_value(entry_price, symbol)
            mfe_pips = mfe / pip_value if pip_value > 0 else 0.0
            mae_pips = mae / pip_value if pip_value > 0 else 0.0
            
            return {
                'mfe': mfe,
                'mae': mae,
                'mfe_pips': float(mfe_pips),
                'mae_pips': float(mae_pips),
                'confidence': confidence,
                'max_price': float(max_price_reached),
                'min_price': float(min_price_reached),
                'bars_in_trade': len(trade_lifespan)
            }
            
        except Exception as e:
            return {
                'mfe': 0.0,
                'mae': 0.0,
                'mfe_pips': 0.0,
                'mae_pips': 0.0,
                'confidence': 'ERROR',
                'error': str(e)
            }
    
    def calculate_excursions_batch(
        self,
        trades_df: pd.DataFrame,
        symbol: str,
        interval: str = '1min',
        asset_class: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate MFE/MAE for multiple trades efficiently.
        
        Args:
            trades_df: DataFrame with columns:
                - entry_datetime, exit_datetime, entry_price, direction/action, stop_loss
            symbol: Symbol for market data fetching
            interval: Data interval ('1min', '5min', '1d')
            asset_class: Optional asset class for market data routing
        
        Returns:
            DataFrame with added MFE/MAE columns
        """
        if trades_df.empty:
            return trades_df
        
        results = []
        
        # Determine asset class if not provided
        if not asset_class:
            if symbol.startswith("C:") or '/' in symbol:
                asset_class = "Currencies"
            elif symbol.startswith("I:") or symbol.startswith("^"):
                asset_class = "Indices"
            elif "*" in symbol or "XAU" in symbol or "XAG" in symbol:
                asset_class = "Commodities"
            else:
                asset_class = "Stocks"
        
        # Pre-load market data for all trades (optimization)
        min_entry = trades_df['entry_datetime'].min()
        max_exit = trades_df['exit_datetime'].max()
        
        # Add buffer for safety
        start_buffer = pd.to_datetime(min_entry) - timedelta(hours=1)
        end_buffer = pd.to_datetime(max_exit) + timedelta(hours=1)
        
        # Fetch market data once
        market_data = fetch_ohlcv(
            symbol=symbol,
            interval=interval,
            start=start_buffer.astimezone(self.utc_tz) if hasattr(start_buffer, 'astimezone') else start_buffer,
            end=end_buffer.astimezone(self.utc_tz) if hasattr(end_buffer, 'astimezone') else end_buffer,
            asset_class=asset_class
        )
        
        if market_data.empty:
            # Try fallback interval
            if interval == '1min':
                market_data = fetch_ohlcv(
                    symbol=symbol,
                    interval='5min',
                    start=start_buffer.astimezone(self.utc_tz) if hasattr(start_buffer, 'astimezone') else start_buffer,
                    end=end_buffer.astimezone(self.utc_tz) if hasattr(end_buffer, 'astimezone') else end_buffer,
                    asset_class=asset_class
                )
        
        # CRITICAL: Ensure market_data has DatetimeIndex with GMT+4 timezone
        if not market_data.empty:
            # Check if index is a DatetimeIndex
            if not isinstance(market_data.index, pd.DatetimeIndex):
                # If it's a RangeIndex or other type, check if there's a timestamp column
                if 'Timestamp' in market_data.columns or 'Date' in market_data.columns:
                    time_col = 'Timestamp' if 'Timestamp' in market_data.columns else 'Date'
                    market_data = market_data.set_index(time_col)
                    market_data.index = pd.to_datetime(market_data.index)
                elif market_data.index.name in ['Timestamp', 'Date', 'timestamp', 'date']:
                    market_data.index = pd.to_datetime(market_data.index)
                else:
                    # Try to convert index directly
                    try:
                        market_data.index = pd.to_datetime(market_data.index)
                    except:
                        # If that fails, reset and use a default
                        market_data = market_data.reset_index()
                        if 'timestamp' in market_data.columns or 'date' in market_data.columns:
                            time_col = 'timestamp' if 'timestamp' in market_data.columns else 'date'
                            market_data = market_data.set_index(time_col)
                            market_data.index = pd.to_datetime(market_data.index)
            
            # Ensure it's a DatetimeIndex now
            if not isinstance(market_data.index, pd.DatetimeIndex):
                raise ValueError(f"Could not convert market_data index to DatetimeIndex. Index type: {type(market_data.index)}, Index name: {market_data.index.name}")
            
            # Ensure timezone is GMT+4
            if market_data.index.tz is None:
                market_data.index = market_data.index.tz_localize(self.tz)
            elif market_data.index.tz != self.tz:
                market_data.index = market_data.index.tz_convert(self.tz)
        
        # Process each trade
        for idx, trade in trades_df.iterrows():
            trade_dict = trade.to_dict()
            trade_dict['symbol'] = symbol  # Ensure symbol is in trade dict
            
            # Slice market data for this trade
            entry_time = self._parse_datetime(trade['entry_datetime'])
            exit_time = self._parse_datetime(trade['exit_datetime'])
            
            # Ensure entry_time and exit_time are in GMT+4
            if entry_time.tzinfo is None:
                entry_time = self.tz.localize(entry_time)
            elif entry_time.tzinfo != self.tz:
                entry_time = entry_time.astimezone(self.tz)
            
            if exit_time.tzinfo is None:
                exit_time = self.tz.localize(exit_time)
            elif exit_time.tzinfo != self.tz:
                exit_time = exit_time.astimezone(self.tz)
            
            # CRITICAL: Double-check market_data has DatetimeIndex before comparison
            # This check must happen INSIDE the loop because market_data might be modified
            if market_data.empty:
                trade_market_data = pd.DataFrame()
            else:
                # Ensure index is DatetimeIndex - check every time in case it was reset
                if not isinstance(market_data.index, pd.DatetimeIndex):
                    # Try to fix it - check if there's a timestamp column
                    if 'Timestamp' in market_data.columns:
                        market_data = market_data.set_index('Timestamp')
                        market_data.index = pd.to_datetime(market_data.index)
                    elif 'Date' in market_data.columns:
                        market_data = market_data.set_index('Date')
                        market_data.index = pd.to_datetime(market_data.index)
                    else:
                        # Try converting index directly
                        try:
                            market_data.index = pd.to_datetime(market_data.index)
                        except Exception as e:
                            print(f"⚠️ Error converting market_data index to DatetimeIndex for trade {idx}: {e}")
                            print(f"   Index type: {type(market_data.index)}, Index name: {market_data.index.name}")
                            print(f"   Market data shape: {market_data.shape}, Columns: {list(market_data.columns)}")
                            trade_market_data = pd.DataFrame()
                            continue
                    
                    # Ensure timezone after conversion
                    if isinstance(market_data.index, pd.DatetimeIndex):
                        if market_data.index.tz is None:
                            market_data.index = market_data.index.tz_localize(self.tz)
                        elif market_data.index.tz != self.tz:
                            market_data.index = market_data.index.tz_convert(self.tz)
                
                # Now safe to do comparison (only if we have DatetimeIndex)
                if isinstance(market_data.index, pd.DatetimeIndex):
                    try:
                        trade_market_data = market_data[
                            (market_data.index >= entry_time) & 
                            (market_data.index <= exit_time)
                        ]
                    except TypeError as e:
                        print(f"⚠️ TypeError during market_data slicing for trade {idx}: {e}")
                        print(f"   Entry time: {entry_time} (type: {type(entry_time)}, tz: {entry_time.tzinfo})")
                        print(f"   Exit time: {exit_time} (type: {type(exit_time)}, tz: {exit_time.tzinfo})")
                        print(f"   Market data index type: {type(market_data.index)}, first index: {market_data.index[0] if len(market_data) > 0 else 'N/A'}")
                        trade_market_data = pd.DataFrame()
                else:
                    print(f"⚠️ market_data index is still not DatetimeIndex after conversion attempt for trade {idx}")
                    trade_market_data = pd.DataFrame()
            
            # If still empty, try fetching individual trade data
            if trade_market_data.empty:
                trade_market_data = fetch_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    start=entry_time.astimezone(self.utc_tz),
                    end=exit_time.astimezone(self.utc_tz),
                    asset_class=asset_class
                )
                
                # Ensure fetched data also has proper DatetimeIndex with GMT+4
                if not trade_market_data.empty:
                    if not isinstance(trade_market_data.index, pd.DatetimeIndex):
                        # Try to find timestamp column
                        if 'Timestamp' in trade_market_data.columns or 'Date' in trade_market_data.columns:
                            time_col = 'Timestamp' if 'Timestamp' in trade_market_data.columns else 'Date'
                            trade_market_data = trade_market_data.set_index(time_col)
                        trade_market_data.index = pd.to_datetime(trade_market_data.index)
                    if trade_market_data.index.tz is None:
                        trade_market_data.index = trade_market_data.index.tz_localize(self.tz)
                    elif trade_market_data.index.tz != self.tz:
                        trade_market_data.index = trade_market_data.index.tz_convert(self.tz)
            
            # Calculate excursions
            excursion_result = self.calculate_excursions(
                trade_dict,
                trade_market_data
            )
            
            # Add to results
            result_row = trade_dict.copy()
            result_row.update(excursion_result)
            results.append(result_row)
        
        result_df = pd.DataFrame(results)
        
        # Calculate R-multiples if stop_loss exists
        if 'stop_loss' in result_df.columns:
            result_df = self.calculate_r_multiples(result_df)
        
        return result_df
    
    def calculate_r_multiples(
        self,
        trades_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate R-multiple normalized MFE/MAE.
        
        R-Multiple = Excursion / Stop Loss Distance
        
        Args:
            trades_df: DataFrame with MFE/MAE and stop_loss columns
        
        Returns:
            DataFrame with added MAE_R and MFE_R columns
        """
        df = trades_df.copy()
        
        # Calculate stop loss distance
        def calc_sl_distance(row):
            entry = row.get('entry_price')
            sl = row.get('stop_loss')
            if pd.isna(entry) or pd.isna(sl) or entry == 0:
                return None
            return abs(float(entry) - float(sl))
        
        df['stop_loss_distance'] = df.apply(calc_sl_distance, axis=1)
        
        # Calculate R-multiples
        def calc_mae_r(row):
            mae = row.get('mae', 0)
            sl_dist = row.get('stop_loss_distance')
            if pd.isna(sl_dist) or sl_dist is None or sl_dist == 0:
                return None
            return float(mae) / float(sl_dist)
        
        def calc_mfe_r(row):
            mfe = row.get('mfe', 0)
            sl_dist = row.get('stop_loss_distance')
            if pd.isna(sl_dist) or sl_dist is None or sl_dist == 0:
                return None
            return float(mfe) / float(sl_dist)
        
        df['mae_r'] = df.apply(calc_mae_r, axis=1)
        df['mfe_r'] = df.apply(calc_mfe_r, axis=1)
        
        return df
    
    def simulate_stop_loss_optimization(
        self,
        trades_df: pd.DataFrame,
        proposed_sl_pips: float
    ) -> Dict:
        """
        Simulate PnL with a tighter stop loss.
        
        Args:
            trades_df: DataFrame with MFE/MAE and original PnL
            proposed_sl_pips: Proposed stop loss in pips
        
        Returns:
            Dictionary with projected metrics
        """
        df = trades_df.copy()
        
        # Determine if trade would have been stopped out
        df['would_stop_out'] = df['mae_pips'] > proposed_sl_pips
        
        # Calculate new PnL
        # If stopped out: PnL = -1R (assuming 1R = stop loss distance)
        def calc_projected_pnl(row):
            if row['would_stop_out']:
                entry = row.get('entry_price', 0)
                sl = row.get('stop_loss')
                if pd.notna(sl) and entry > 0:
                    # Calculate -1R loss
                    sl_distance = abs(float(entry) - float(sl))
                    # Determine direction
                    direction = row.get('direction', 'BUY').upper()
                    if direction not in ['BUY', 'SELL']:
                        action = row.get('action', '').upper()
                        direction = 'BUY' if action in ['BUY', 'LONG'] else 'SELL'
                    
                    # For BUY: loss if exit < entry, for SELL: loss if exit > entry
                    # Since we're stopped out, we exit at stop loss
                    if direction == 'BUY':
                        return -sl_distance  # Exit at SL, which is below entry
                    else:
                        return -sl_distance  # Exit at SL, which is above entry
                else:
                    # Fallback: use original PnL if we can't calculate
                    return row.get('profit_loss', 0)
            else:
                # Trade would not be stopped out, keep original PnL
                return row.get('profit_loss', 0)
        
        df['projected_pnl'] = df.apply(calc_projected_pnl, axis=1)
        
        # Calculate metrics
        stopped_out_count = df['would_stop_out'].sum()
        original_total_pnl = df.get('profit_loss', pd.Series([0] * len(df))).sum()
        if original_total_pnl is None or pd.isna(original_total_pnl):
            original_total_pnl = 0.0
        projected_total_pnl = df['projected_pnl'].sum()
        if projected_total_pnl is None or pd.isna(projected_total_pnl):
            projected_total_pnl = 0.0
        
        # Calculate win rates
        original_wins = (df.get('profit_loss', pd.Series([0] * len(df))) > 0).sum()
        original_win_rate = (original_wins / len(df) * 100) if len(df) > 0 else 0.0
        
        projected_wins = (df['projected_pnl'] > 0).sum()
        projected_win_rate = (projected_wins / len(df) * 100) if len(df) > 0 else 0.0
        
        return {
            'proposed_sl_pips': float(proposed_sl_pips),
            'stopped_out_trades': int(stopped_out_count),
            'total_trades': len(df),
            'original_total_pnl': float(original_total_pnl),
            'projected_total_pnl': float(projected_total_pnl),
            'pnl_difference': float(projected_total_pnl - original_total_pnl),
            'win_rate_original': float(original_win_rate),
            'win_rate_projected': float(projected_win_rate)
        }
    
    def _parse_datetime(self, dt) -> pd.Timestamp:
        """Parse datetime to timezone-aware Timestamp."""
        if dt is None:
            raise ValueError("Datetime cannot be None")
        
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        elif isinstance(dt, datetime):
            dt = pd.Timestamp(dt)
        
        if isinstance(dt, pd.Timestamp):
            if dt.tzinfo is None:
                dt = self.tz.localize(dt)
            else:
                dt = dt.astimezone(self.tz)
        
        return pd.Timestamp(dt)
    
    def _calculate_pip_value(self, price: float, symbol: str) -> float:
        """
        Calculate pip value based on symbol type.
        For forex: 0.0001, for stocks: 0.01 (cents)
        """
        if not symbol:
            # Default to forex pip value
            return 0.0001
        
        symbol_upper = symbol.upper()
        
        # Forex pairs (C: prefix or / in symbol)
        if symbol_upper.startswith('C:') or '/' in symbol_upper:
            return 0.0001
        
        # Indices (^ or I: prefix)
        if symbol_upper.startswith('^') or symbol_upper.startswith('I:'):
            return 0.01
        
        # Commodities (XAU, XAG, or * in symbol)
        if 'XAU' in symbol_upper or 'XAG' in symbol_upper or '*' in symbol_upper:
            return 0.01
        
        # Stocks - use 0.01 for most
        return 0.01

