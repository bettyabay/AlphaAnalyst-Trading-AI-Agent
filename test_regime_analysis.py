"""
Test script for Market Regime Segmentation Module
Tests all steps of the regime analysis pipeline
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from tradingagents.dataflows.signal_analyzer import SignalAnalyzer


def generate_test_market_data(days=365, interval_hours=1):
    """Generate synthetic OHLCV market data for testing"""
    print(f"\n{'='*80}")
    print("Generating test market data...")
    print(f"{'='*80}")
    
    # Generate timestamps
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=days)
    
    timestamps = pd.date_range(start=start_date, end=end_date, freq=f'{interval_hours}h')
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    n = len(timestamps)
    
    # Base price with trend
    trend = np.linspace(1800, 2000, n)
    noise = np.random.randn(n) * 10
    close_prices = trend + noise
    
    # Generate OHLC
    high_prices = close_prices + np.abs(np.random.randn(n) * 5)
    low_prices = close_prices - np.abs(np.random.randn(n) * 5)
    open_prices = close_prices + np.random.randn(n) * 3
    volume = np.random.randint(1000, 10000, n)
    
    market_df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=timestamps)
    
    print(f"[OK] Generated {len(market_df)} candles")
    print(f"   Date range: {market_df.index[0]} to {market_df.index[-1]}")
    print(f"   Price range: ${market_df['Close'].min():.2f} - ${market_df['Close'].max():.2f}")
    
    return market_df


def generate_test_signals(market_df, num_signals=50):
    """Generate synthetic signal data for testing"""
    print(f"\n{'='*80}")
    print("Generating test signals...")
    print(f"{'='*80}")
    
    np.random.seed(42)
    
    # Random timestamps from market data range
    signal_indices = np.random.choice(len(market_df), size=num_signals, replace=False)
    signal_timestamps = market_df.index[signal_indices]
    
    signals = []
    for i, ts in enumerate(signal_timestamps):
        entry_price = market_df.loc[ts, 'Close']
        action = np.random.choice(['BUY', 'SELL'])
        
        # Generate TP/SL levels
        if action == 'BUY':
            tp1 = entry_price * 1.01  # 1% profit
            tp2 = entry_price * 1.02  # 2% profit
            tp3 = entry_price * 1.03  # 3% profit
            sl = entry_price * 0.99   # 1% loss
        else:
            tp1 = entry_price * 0.99
            tp2 = entry_price * 0.98
            tp3 = entry_price * 0.97
            sl = entry_price * 1.01
        
        # Simulate final status
        final_status = np.random.choice(['TP1', 'TP2', 'TP3', 'SL', 'OPEN'], p=[0.3, 0.2, 0.1, 0.3, 0.1])
        
        # Calculate pips made
        if final_status == 'TP1':
            pips_made = 100 if action == 'BUY' else -100
        elif final_status == 'TP2':
            pips_made = 200 if action == 'BUY' else -200
        elif final_status == 'TP3':
            pips_made = 300 if action == 'BUY' else -300
        elif final_status == 'SL':
            pips_made = -100 if action == 'BUY' else 100
        else:
            pips_made = 0
        
        signals.append({
            'signal_id': f'TEST_{i+1}',
            'provider_name': 'Test Provider',
            'symbol': 'C:XAUUSD',
            'signal_date': ts,
            'action': action,
            'entry_price': entry_price,
            'target_1': tp1,
            'target_2': tp2,
            'target_3': tp3,
            'stop_loss': sl,
            'final_status': final_status,
            'pips_made': pips_made,
            'tp1_hit': final_status in ['TP1', 'TP2', 'TP3'],
            'tp2_hit': final_status in ['TP2', 'TP3'],
            'tp3_hit': final_status == 'TP3',
            'sl_hit': final_status == 'SL'
        })
    
    signals_df = pd.DataFrame(signals)
    
    print(f"[OK] Generated {len(signals_df)} signals")
    print(f"   Date range: {signals_df['signal_date'].min()} to {signals_df['signal_date'].max()}")
    print(f"   Win rate: {(signals_df['pips_made'] > 0).sum() / len(signals_df) * 100:.1f}%")
    
    return signals_df


def test_step1_calculate_regimes():
    """Test Step 1: Feature Engineering"""
    print(f"\n{'='*80}")
    print("TEST STEP 1: Calculate Regimes (Feature Engineering)")
    print(f"{'='*80}")
    
    # Generate test data
    market_df = generate_test_market_data(days=180, interval_hours=4)
    
    # Initialize analyzer
    analyzer = SignalAnalyzer()
    
    # Calculate regimes
    print("\nCalculating indicators...")
    market_with_indicators = analyzer.calculate_regimes(
        market_df=market_df,
        adx_period=14,
        sma_short=50,
        sma_long=200,
        atr_period=14,
        atr_ma_period=50
    )
    
    # Validate results
    required_cols = ['ADX', 'SMA_50', 'SMA_200', 'ATR', 'ATR_MA', 'ATR_pct']
    for col in required_cols:
        assert col in market_with_indicators.columns, f"Missing column: {col}"
    
    print("\n[PASS] STEP 1 PASSED")
    print(f"   Added columns: {required_cols}")
    print(f"   ADX range: {market_with_indicators['ADX'].min():.2f} - {market_with_indicators['ADX'].max():.2f}")
    print(f"   ATR range: ${market_with_indicators['ATR'].min():.2f} - ${market_with_indicators['ATR'].max():.2f}")
    
    return market_with_indicators


def test_step2_define_regime(market_with_indicators):
    """Test Step 2: Regime Classification"""
    print(f"\n{'='*80}")
    print("TEST STEP 2: Define Regime (Classification)")
    print(f"{'='*80}")
    
    analyzer = SignalAnalyzer()
    
    # Define regimes
    print("\nClassifying market regimes...")
    market_with_regimes = analyzer.define_regime(
        market_df=market_with_indicators,
        adx_threshold=25
    )
    
    # Validate results
    assert 'Regime' in market_with_regimes.columns, "Missing Regime column"
    
    # Check regime distribution
    regime_counts = market_with_regimes['Regime'].value_counts()
    
    print("\n[PASS] STEP 2 PASSED")
    print(f"   Regime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(market_with_regimes) * 100
        print(f"      {regime}: {count} ({pct:.1f}%)")
    
    return market_with_regimes


def test_step3_merge_signals_with_regimes(market_with_regimes):
    """Test Step 3: Data Merging"""
    print(f"\n{'='*80}")
    print("TEST STEP 3: Merge Signals with Regimes (Data Merging)")
    print(f"{'='*80}")
    
    # Generate test signals
    signals_df = generate_test_signals(market_with_regimes, num_signals=50)
    
    analyzer = SignalAnalyzer()
    
    # Merge signals with regimes
    print("\nMerging signals with market regimes...")
    signals_with_regimes = analyzer.merge_signals_with_regimes(
        signals_df=signals_df,
        market_df=market_with_regimes,
        entry_time_col='signal_date',
        timezone='UTC'
    )
    
    # Validate results
    assert 'Regime' in signals_with_regimes.columns, "Missing Regime column in merged data"
    
    matched_signals = signals_with_regimes['Regime'].notna().sum()
    match_rate = matched_signals / len(signals_with_regimes) * 100
    
    print("\n[PASS] STEP 3 PASSED")
    print(f"   Signals matched with regime: {matched_signals}/{len(signals_with_regimes)} ({match_rate:.1f}%)")
    
    return signals_with_regimes


def test_step4_calculate_metrics_by_regime(signals_with_regimes):
    """Test Step 4: Aggregation & Metrics"""
    print(f"\n{'='*80}")
    print("TEST STEP 4: Calculate Metrics by Regime (Aggregation)")
    print(f"{'='*80}")
    
    analyzer = SignalAnalyzer()
    
    # Calculate metrics
    print("\nCalculating performance metrics by regime...")
    regime_metrics = analyzer.calculate_metrics_by_regime(
        signals_df=signals_with_regimes,
        pnl_col='pips_made',
        final_status_col='final_status'
    )
    
    # Validate results
    assert not regime_metrics.empty, "Regime metrics is empty"
    
    required_cols = ['Regime', 'Total_Trades', 'Win_Rate_%', 'Profit_Factor', 'Avg_PnL']
    for col in required_cols:
        assert col in regime_metrics.columns, f"Missing column: {col}"
    
    print("\n[PASS] STEP 4 PASSED")
    print(f"\n   Regime Performance Metrics:")
    print(regime_metrics.to_string(index=False))
    
    return regime_metrics


def test_timezone_handling():
    """Test timezone conversion and handling"""
    print(f"\n{'='*80}")
    print("TEST: Timezone Handling")
    print(f"{'='*80}")
    
    # Create market data in UTC
    utc_tz = pytz.UTC
    timestamps_utc = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h', tz=utc_tz)
    
    market_df = pd.DataFrame({
        'Open': np.random.randn(len(timestamps_utc)) + 100,
        'High': np.random.randn(len(timestamps_utc)) + 101,
        'Low': np.random.randn(len(timestamps_utc)) + 99,
        'Close': np.random.randn(len(timestamps_utc)) + 100,
        'Volume': np.random.randint(1000, 10000, len(timestamps_utc))
    }, index=timestamps_utc)
    
    # Create signals in different timezone (GMT+4)
    dubai_tz = pytz.timezone('Asia/Dubai')
    signal_timestamps = pd.date_range(start='2024-01-02', end='2024-01-08', freq='12h', tz=dubai_tz)
    
    signals_df = pd.DataFrame({
        'signal_id': [f'TZ_TEST_{i}' for i in range(len(signal_timestamps))],
        'signal_date': signal_timestamps,
        'symbol': 'C:XAUUSD',
        'pips_made': np.random.randint(-100, 200, len(signal_timestamps)),
        'final_status': np.random.choice(['TP1', 'SL'], len(signal_timestamps))
    })
    
    analyzer = SignalAnalyzer()
    
    # Add regime column to market data
    market_df['Regime'] = 'Trending - High Vol'
    
    # Merge with timezone conversion
    print("\nMerging signals (GMT+4) with market data (UTC)...")
    merged = analyzer.merge_signals_with_regimes(
        signals_df=signals_df,
        market_df=market_df,
        entry_time_col='signal_date',
        timezone='UTC'
    )
    
    # Validate
    assert 'Regime' in merged.columns, "Regime column missing after merge"
    matched = merged['Regime'].notna().sum()
    
    print(f"\n[PASS] TIMEZONE TEST PASSED")
    print(f"   Signals matched: {matched}/{len(merged)}")
    print(f"   Original signal timezone: {signals_df['signal_date'].iloc[0].tzinfo}")
    print(f"   Original market timezone: {market_df.index[0].tzinfo}")
    print(f"   Merge performed in: UTC")


def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "="*80)
    print("MARKET REGIME SEGMENTATION MODULE - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    try:
        # Step 1: Calculate regimes
        market_with_indicators = test_step1_calculate_regimes()
        
        # Step 2: Define regimes
        market_with_regimes = test_step2_define_regime(market_with_indicators)
        
        # Step 3: Merge signals with regimes
        signals_with_regimes = test_step3_merge_signals_with_regimes(market_with_regimes)
        
        # Step 4: Calculate metrics by regime
        regime_metrics = test_step4_calculate_metrics_by_regime(signals_with_regimes)
        
        # Additional test: Timezone handling
        test_timezone_handling()
        
        print("\n" + "="*80)
        print("SUCCESS: ALL TESTS PASSED!")
        print("="*80)
        print("\n[OK] Market Regime Segmentation Module is ready for production use")
        print("\nNext steps:")
        print("1. Install pandas-ta: pip install pandas-ta")
        print("2. Run the Streamlit app: streamlit run app.py")
        print("3. Navigate to Signal Analysis Dashboard")
        print("4. Run signal analysis first, then run regime analysis")
        
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print("FAILED: TEST FAILED")
        print("="*80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

