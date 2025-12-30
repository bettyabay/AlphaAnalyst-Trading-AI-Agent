"""
KPI Calculator for trading indicators
Supports ATR, Volume, VAMP, EMA, SMA calculations for any financial instrument including Gold
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
from datetime import datetime


def calculate_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR)
    
    Args:
        df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
        period: Period for ATR calculation (default 14)
        
    Returns:
        Latest ATR value or None if calculation fails
    """
    try:
        if df.empty or len(df) < period:
            return None
        
        high = pd.to_numeric(df['High'], errors='coerce')
        low = pd.to_numeric(df['Low'], errors='coerce')
        close = pd.to_numeric(df['Close'], errors='coerce')
        
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if not atr.empty and pd.notna(atr.iloc[-1]) else None
    except Exception as e:
        print(f"Error calculating ATR: {e}")
        return None


def calculate_volume(df: pd.DataFrame, period: int = 20) -> Optional[Dict[str, float]]:
    """
    Calculate Volume metrics
    
    Args:
        df: DataFrame with Volume column
        period: Period for moving average (default 20)
        
    Returns:
        Dictionary with current_volume, avg_volume, volume_ratio
    """
    try:
        if df.empty:
            return None
        
        volume = pd.to_numeric(df['Volume'], errors='coerce')
        current_volume = float(volume.iloc[-1]) if not volume.empty and pd.notna(volume.iloc[-1]) else None
        
        if current_volume is None:
            return None
        
        if len(volume) >= period:
            avg_volume = float(volume.tail(period).mean())
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else None
        else:
            avg_volume = float(volume.mean()) if len(volume) > 0 else None
            volume_ratio = current_volume / avg_volume if avg_volume and avg_volume > 0 else None
        
        return {
            "current_volume": current_volume,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio
        }
    except Exception as e:
        print(f"Error calculating Volume: {e}")
        return None


def calculate_vamp(df: pd.DataFrame, period: int = 20) -> Optional[float]:
    """
    Calculate Volume Adjusted Moving Price (VAMP) - Volume Weighted Average Price
    
    Args:
        df: DataFrame with OHLCV data
        period: Period for VWAP calculation (default 20)
        
    Returns:
        Latest VWAP value or None if calculation fails
    """
    try:
        if df.empty or len(df) < period:
            return None
        
        close = pd.to_numeric(df['Close'], errors='coerce')
        volume = pd.to_numeric(df['Volume'], errors='coerce')
        
        # Calculate VWAP for the last N periods
        tail_df = df.tail(period)
        close_tail = pd.to_numeric(tail_df['Close'], errors='coerce')
        volume_tail = pd.to_numeric(tail_df['Volume'], errors='coerce')
        
        if close_tail.empty or volume_tail.empty:
            return None
        
        # VWAP = sum(Price * Volume) / sum(Volume)
        typical_price = close_tail  # Using Close price
        price_volume = typical_price * volume_tail
        sum_price_volume = price_volume.sum()
        sum_volume = volume_tail.sum()
        
        if sum_volume == 0:
            return None
        
        vwap = sum_price_volume / sum_volume
        return float(vwap) if pd.notna(vwap) else None
    except Exception as e:
        print(f"Error calculating VAMP/VWAP: {e}")
        return None


def calculate_ema(df: pd.DataFrame, period: int = 15, column: str = 'Close') -> Optional[float]:
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        df: DataFrame with price data
        period: Period for EMA calculation (default 15)
        column: Column to use for calculation (default 'Close')
        
    Returns:
        Latest EMA value or None if calculation fails
    """
    try:
        if df.empty or len(df) < period:
            return None
        
        if column not in df.columns:
            return None
        
        prices = pd.to_numeric(df[column], errors='coerce')
        if prices.empty:
            return None
        
        # Calculate EMA using pandas ewm (exponential weighted moving)
        ema = prices.ewm(span=period, adjust=False).mean()
        latest_ema = ema.iloc[-1]
        
        return float(latest_ema) if pd.notna(latest_ema) else None
    except Exception as e:
        print(f"Error calculating EMA: {e}")
        return None


def calculate_sma(df: pd.DataFrame, period: int = 50, column: str = 'Close') -> Optional[float]:
    """
    Calculate Simple Moving Average (SMA)
    
    Args:
        df: DataFrame with price data
        period: Period for SMA calculation (default 50)
        column: Column to use for calculation (default 'Close')
        
    Returns:
        Latest SMA value or None if calculation fails
    """
    try:
        if df.empty or len(df) < period:
            return None
        
        if column not in df.columns:
            return None
        
        prices = pd.to_numeric(df[column], errors='coerce')
        if prices.empty:
            return None
        
        sma = prices.rolling(window=period).mean()
        latest_sma = sma.iloc[-1]
        
        return float(latest_sma) if pd.notna(latest_sma) else None
    except Exception as e:
        print(f"Error calculating SMA: {e}")
        return None


def calculate_kpi(kpi_name: str, df: pd.DataFrame, **kwargs) -> Optional[Union[float, Dict]]:
    """
    Calculate a KPI by name
    
    Args:
        kpi_name: Name of KPI (ATR, Volume, VAMP, EMA, SMA)
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters for KPI calculation
        
    Returns:
        KPI value(s) or None if calculation fails
    """
    kpi_name = kpi_name.upper().strip()
    
    if kpi_name == "ATR":
        period = kwargs.get('period', 14)
        return calculate_atr(df, period)
    elif kpi_name == "VOLUME":
        period = kwargs.get('period', 20)
        return calculate_volume(df, period)
    elif kpi_name == "VAMP" or kpi_name == "VWAP":
        period = kwargs.get('period', 20)
        return calculate_vamp(df, period)
    elif kpi_name == "EMA":
        period = kwargs.get('period', 15)
        column = kwargs.get('column', 'Close')
        return calculate_ema(df, period, column)
    elif kpi_name == "SMA":
        period = kwargs.get('period', 50)
        column = kwargs.get('column', 'Close')
        return calculate_sma(df, period, column)
    else:
        print(f"Unknown KPI: {kpi_name}")
        return None

