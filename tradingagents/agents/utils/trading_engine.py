"""
Trading Engine Core for Phase 3
Implements volume screening, fire-testing, and AI-enhanced analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf

from ...dataflows.ai_analysis import AIResearchAnalyzer
from ...dataflows.document_manager import DocumentManager
from ...dataflows.market_data_service import fetch_ohlcv, period_to_days
from ...dataflows.feature_lab import FeatureLab
from ...agents.utils.session_manager import TradingSessionManager, TradeExecutionService


class VolumeScreeningEngine:
    """Phase 1 Volume Screening Engine"""
    
    def __init__(self):
        self.min_volume_spike = 0.75  # Lowered from 1.3 to 0.75 to allow more candidates during testing
        self.min_liquidity = 1_000_000  # Minimum average daily volume
        self.rsi_min = 30
        self.rsi_max = 70
    
    def _calc_sma(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=window).mean()
    
    def _calc_rsi(self, close_prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _fetch_history(self, symbol: str, period: str = "3mo", interval: str = "1d", asset_class: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch historical data"""
        lookback = period_to_days(period, default=180)
        data = fetch_ohlcv(symbol, interval=interval, lookback_days=lookback, asset_class=asset_class)
        if data is None or data.empty:
            return None
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                return None
        return data
    
    def screen_symbol(self, symbol: str, asset_class: Optional[str] = None) -> Dict:
        """Screen a single symbol for volume spike criteria"""
        # Fetch 1 year of data to ensure we have enough for SMA200 calculation
        hist = self._fetch_history(symbol, period="1y", asset_class=asset_class)
        
        if hist is None or hist.empty or len(hist) < 50:
            return {
                "symbol": symbol,
                "pass": False,
                "reason": "Insufficient data (need 50+ days)",
                "metrics": {}
            }
        
        # Calculate metrics
        avg_vol_20 = hist['Volume'].tail(20).mean()
        today_vol = hist['Volume'].iloc[-1]
        vol_spike = today_vol / avg_vol_20 if avg_vol_20 > 0 else 0
        
        # Calculate SMAs
        sma50 = self._calc_sma(hist['Close'], 50)
        sma200 = self._calc_sma(hist['Close'], 200)
        sma20 = self._calc_sma(hist['Close'], 20)  # For alternative trend check
        
        # Calculate RSI
        rsi = self._calc_rsi(hist['Close'])
        
        # Get latest values
        sma50_last = sma50.iloc[-1] if len(sma50) > 0 and not pd.isna(sma50.iloc[-1]) else None
        sma200_last = sma200.iloc[-1] if len(sma200) > 0 and not pd.isna(sma200.iloc[-1]) else None
        sma20_last = sma20.iloc[-1] if len(sma20) > 0 and not pd.isna(sma20.iloc[-1]) else None
        rsi_last = rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else None
        
        # Calculate SMA spread (prefer SMA200, fallback to SMA20 > SMA50)
        has_sma200 = sma50_last is not None and sma200_last is not None
        if has_sma200:
            sma_spread = sma50_last - sma200_last
            trend_pass = sma_spread > 0
        elif sma20_last is not None and sma50_last is not None:
            sma_spread = None  # Can't calculate SMA50-SMA200 spread
            trend_pass = sma20_last > sma50_last  # Use alternative: SMA20 > SMA50
        else:
            sma_spread = None
            trend_pass = False
        
        # Check criteria
        volume_pass = vol_spike >= self.min_volume_spike
        rsi_pass = rsi_last is not None and self.rsi_min < rsi_last < self.rsi_max
        liquidity_pass = avg_vol_20 >= self.min_liquidity
        
        overall_pass = volume_pass and trend_pass and rsi_pass and liquidity_pass
        
        return {
            "symbol": symbol,
            "pass": overall_pass,
            "reason": self._get_failure_reason(volume_pass, trend_pass, rsi_pass, liquidity_pass),
            "metrics": {
                "avg_vol_20": int(avg_vol_20) if not pd.isna(avg_vol_20) else None,
                "today_vol": int(today_vol) if not pd.isna(today_vol) else None,
                "vol_spike": round(vol_spike, 2) if not pd.isna(vol_spike) else None,
                "sma50": round(sma50_last, 2) if sma50_last else None,
                "sma200": round(sma200_last, 2) if sma200_last else None,
                "sma_spread": round(sma_spread, 2) if sma_spread else None,
                "rsi": round(float(rsi_last), 1) if rsi_last else None,
                "current_price": round(float(hist['Close'].iloc[-1]), 2) if len(hist) > 0 else None
            }
        }
    
    def _get_failure_reason(self, vol: bool, trend: bool, rsi: bool, liquidity: bool) -> str:
        """Get reason for failure"""
        if not liquidity:
            return "Insufficient liquidity"
        if not vol:
            return "Volume spike insufficient"
        if not trend:
            return "No uptrend (SMA50 < SMA200)"
        if not rsi:
            return "RSI out of range"
        return "Passed all criteria"
    
    def screen_watchlist(self, symbols: List[str]) -> pd.DataFrame:
        """Screen entire watchlist"""
        results = []
        for symbol in symbols:
            result = self.screen_symbol(symbol)
            result_row = {
                "Symbol": symbol,
                "Pass": "✅" if result["pass"] else "❌",
                "Avg Vol (20d)": result["metrics"].get("avg_vol_20", "N/A"),
                "Today Vol": result["metrics"].get("today_vol", "N/A"),
                "Vol Spike": result["metrics"].get("vol_spike", "N/A"),
                "SMA50": result["metrics"].get("sma50", "N/A"),
                "SMA200": result["metrics"].get("sma200", "N/A"),
                "SMA Spread": result["metrics"].get("sma_spread", "N/A"),
                "RSI": result["metrics"].get("rsi", "N/A"),
                "Price": result["metrics"].get("current_price", "N/A"),
                "Reason": result["reason"]
            }
            results.append(result_row)
        
        return pd.DataFrame(results)


class FireTestingEngine:
    """7-Stage Fire Testing System"""
    
    def __init__(self):
        self.ai_analyzer = AIResearchAnalyzer()
    
    def _calc_sma(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=window).mean()
    
    def _calc_rsi(self, close_prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _fetch_history(self, symbol: str, period: str = "6mo", interval: str = "1d", asset_class: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch historical data"""
        lookback = period_to_days(period, default=180)
        data = fetch_ohlcv(symbol, interval=interval, lookback_days=lookback, asset_class=asset_class)
        if data is None or data.empty:
            return None
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                return None
        return data
    
    def _get_col_series(self, df: pd.DataFrame, col_name: str) -> Optional[pd.Series]:
        """Safely extract column series from DataFrame"""
        if not isinstance(df, pd.DataFrame):
            return None
        
        try:
            if col_name in df.columns:
                s = df[col_name]
            elif isinstance(df.columns, pd.MultiIndex):
                try:
                    s = df.xs(col_name, axis=1, level=0)
                except Exception:
                    try:
                        s = df.xs(col_name, axis=1, level=1)
                    except Exception:
                        return None
            else:
                return None
            
            if isinstance(s, pd.DataFrame):
                if s.shape[1] >= 1:
                    s = s.iloc[:, 0]
                else:
                    return None
            
            if not isinstance(s, pd.Series):
                try:
                    arr = np.ravel(s)
                    s = pd.Series(arr, index=df.index[:len(arr)])
                except Exception:
                    return None
            
            return s
        except Exception:
            return None
    
    def run_fire_test(self, symbol: str) -> Dict:
        """Run complete 7-stage fire test"""
        result = {
            "symbol": symbol,
            "stages": [],
            "score": 0,
            "max_score": 7,
            "timestamp": datetime.now().isoformat()
        }
        
        hist = self._fetch_history(symbol, period="6mo")
        if hist is None or hist.empty:
            result["error"] = "No price data available"
            return result
        
        # Extract OHLCV series
        open_s = self._get_col_series(hist, 'Open')
        high_s = self._get_col_series(hist, 'High')
        low_s = self._get_col_series(hist, 'Low')
        close_s = self._get_col_series(hist, 'Close')
        vol_s = self._get_col_series(hist, 'Volume')
        
        if close_s is None or high_s is None or low_s is None or vol_s is None:
            result["error"] = "Insufficient OHLCV data"
            return result
        
        # Stage 1: Liquidity (Volume Adequacy)
        vol_tail = pd.to_numeric(vol_s.tail(20), errors='coerce')
        avg20 = float(vol_tail.mean()) if len(vol_tail) else np.nan
        pass1 = pd.notna(avg20) and avg20 >= 1_000_000
        result["stages"].append({
            "name": "Liquidity",
            "pass": pass1,
            "detail": f"Avg20Vol={int(avg20):,}" if pd.notna(avg20) else "Avg20Vol=NA",
            "description": "Average 20-day volume must be >= 1M shares"
        })
        result["score"] += 1 if pass1 else 0
        
        # Stage 2: Volatility (ATR relative to price)
        price = close_s.iloc[-1]
        atr_df = pd.DataFrame({
            'High': pd.to_numeric(high_s, errors='coerce'),
            'Low': pd.to_numeric(low_s, errors='coerce'),
            'Close': pd.to_numeric(close_s, errors='coerce'),
        })
        atr = self._calc_atr(atr_df)
        atr_last = atr.iloc[-1] if len(atr) else np.nan
        atr_pct = (atr_last / price) * 100 if pd.notna(atr_last) and pd.notna(price) and price != 0 else np.nan
        pass2 = pd.notna(atr_pct) and 1 <= atr_pct <= 8
        result["stages"].append({
            "name": "Volatility",
            "pass": pass2,
            "detail": f"ATR%={atr_pct:.2f}%" if pd.notna(atr_pct) else "ATR%=NA",
            "description": "ATR should be 1-8% of price for manageable risk"
        })
        result["score"] += 1 if pass2 else 0
        
        # Stage 3: Trend (SMA50 > SMA200)
        sma50 = self._calc_sma(close_s, 50)
        sma200 = self._calc_sma(close_s, 200)
        sma50_last = sma50.iloc[-1] if len(sma50) else np.nan
        sma200_last = sma200.iloc[-1] if len(sma200) else np.nan
        pass3 = pd.notna(sma50_last) and pd.notna(sma200_last) and sma50_last > sma200_last
        detail3 = f"SMA50-SMA200=${(sma50_last - sma200_last):.2f}" if pd.notna(sma50_last) and pd.notna(sma200_last) else "SMA50-SMA200=NA"
        result["stages"].append({
            "name": "Trend",
            "pass": pass3,
            "detail": detail3,
            "description": "SMA50 must be above SMA200 (uptrend)"
        })
        result["score"] += 1 if pass3 else 0
        
        # Stage 4: Momentum (RSI between 40 and 65)
        rsi_series = self._calc_rsi(close_s)
        rsi = rsi_series.iloc[-1] if len(rsi_series) else np.nan
        pass4 = pd.notna(rsi) and 40 <= rsi <= 65
        result["stages"].append({
            "name": "Momentum",
            "pass": pass4,
            "detail": f"RSI={rsi:.1f}" if pd.notna(rsi) else "RSI=NA",
            "description": "RSI should be between 40-65 (not overbought/oversold)"
        })
        result["score"] += 1 if pass4 else 0
        
        # Stage 5: Breakout (close near recent high)
        recent_high = pd.to_numeric(high_s.tail(20), errors='coerce').max()
        pass5 = pd.notna(price) and pd.notna(recent_high) and price > recent_high * 0.995
        detail5 = f"Px=${price:.2f} vs 20dHigh=${recent_high:.2f}" if pd.notna(price) and pd.notna(recent_high) else "Px/High=NA"
        result["stages"].append({
            "name": "Breakout",
            "pass": pass5,
            "detail": detail5,
            "description": "Price should be near 20-day high (breakout potential)"
        })
        result["score"] += 1 if pass5 else 0
        
        # Stage 6: Risk (support proximity using SMA20)
        sma20 = self._calc_sma(close_s, 20)
        support = sma20.iloc[-1] if len(sma20) and pd.notna(sma20.iloc[-1]) else price
        risk_pct = ((price - support) / price) * 100 if pd.notna(price) and price != 0 and pd.notna(support) else np.nan
        pass6 = pd.notna(risk_pct) and 0 <= risk_pct <= 5
        result["stages"].append({
            "name": "Risk",
            "pass": pass6,
            "detail": f"Risk%={risk_pct:.2f}% (Support=${support:.2f})" if pd.notna(risk_pct) else "Risk%=NA",
            "description": "Distance to support should be <= 5% for manageable stop-loss"
        })
        result["score"] += 1 if pass6 else 0
        
        # Stage 7: AI Sentiment Integration
        ai_pass, ai_detail = self._get_ai_sentiment(symbol)
        result["stages"].append({
            "name": "AI Sentiment",
            "pass": ai_pass,
            "detail": ai_detail,
            "description": "AI analysis from research documents should be Bullish/Neutral with confidence >= 5"
        })
        result["score"] += 1 if ai_pass else 0
        
        return result
    
    def _get_ai_sentiment(self, symbol: str) -> tuple:
        """Get AI sentiment from research documents"""
        try:
            profile = self.ai_analyzer.analyze_instrument_profile(symbol)
            ai = profile.get("ai_analysis", {}) if isinstance(profile, dict) else {}
            
            if "error" in ai:
                return False, "AI analysis unavailable"
            
            sentiment = ai.get("overall_sentiment", "Neutral")
            confidence = ai.get("confidence", 5)
            
            # Check if bullish/neutral and confidence >= 5
            ai_pass = sentiment in ["Bullish", "Neutral"] and float(confidence) >= 5
            ai_detail = f"Sent={sentiment}, Conf={confidence}/10"
            
            return ai_pass, ai_detail
        except Exception as e:
            return False, f"AI error: {str(e)[:50]}"
    
    def close(self):
        """Close connections"""
        self.ai_analyzer.close()


class AIEnhancedScoringEngine:
    """AI-Enhanced Analysis Scoring System"""
    
    def __init__(self):
        self.ai_analyzer = AIResearchAnalyzer()
        self.fire_tester = FireTestingEngine()
    
    def calculate_enhanced_score(self, symbol: str) -> Dict:
        """Calculate AI-enhanced confidence score combining all factors"""
        try:
            # Run fire test for technical score
            fire_test = self.fire_tester.run_fire_test(symbol)
            fire_score_ratio = fire_test["score"] / fire_test["max_score"] if fire_test.get("max_score", 7) > 0 else 0
            technical_score = fire_score_ratio * 10  # Convert to 0-10 scale
            
            # Get AI insights score
            ai_profile = self.ai_analyzer.analyze_instrument_profile(symbol)
            ai_analysis = ai_profile.get("ai_analysis", {}) if isinstance(ai_profile, dict) else {}
            
            if "error" in ai_analysis:
                ai_score = 5.0  # Default if AI unavailable
                ai_sentiment = "Neutral"
                ai_confidence = 5.0
            else:
                ai_sentiment = ai_analysis.get("overall_sentiment", "Neutral")
                ai_confidence = float(ai_analysis.get("confidence", 5))
                
                # Convert sentiment to score (Bullish=7-10, Neutral=4-6, Bearish=1-3)
                if ai_sentiment == "Bullish":
                    ai_score = min(ai_confidence + 2, 10)
                elif ai_sentiment == "Bearish":
                    ai_score = max(ai_confidence - 2, 1)
                else:
                    ai_score = ai_confidence
            
            # Combine scores with weights (70% technical, 30% AI)
            enhanced_score = (0.7 * technical_score) + (0.3 * ai_score)
            
            # Determine recommendation
            if enhanced_score >= 8.5:
                recommendation = "Strong Buy"
            elif enhanced_score >= 7.0:
                recommendation = "Buy"
            elif enhanced_score >= 5.0:
                recommendation = "Hold"
            elif enhanced_score >= 3.0:
                recommendation = "Sell"
            else:
                recommendation = "Strong Sell"
            
            return {
                "symbol": symbol,
                "enhanced_score": round(enhanced_score, 2),
                "technical_score": round(technical_score, 2),
                "ai_score": round(ai_score, 2),
                "recommendation": recommendation,
                "fire_test": fire_test,
                "ai_analysis": ai_analysis,
                "ai_sentiment": ai_sentiment,
                "ai_confidence": ai_confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "symbol": symbol,
                "error": str(e),
                "enhanced_score": 5.0,
                "recommendation": "Hold"
            }
    
    def close(self):
        """Close connections"""
        self.ai_analyzer.close()
        self.fire_tester.close()


class TradeDecisionEngine:
    """
    TRADE YES Decision Engine
    Evaluates all conditions for trade execution based on QUANTUMTRADER metrics
    """
    
    def __init__(self, max_exposure: float = 2000.0, max_daily_loss: float = 400.0):
        """
        Initialize Trade Decision Engine
        
        Args:
            max_exposure: Maximum position exposure limit ($2,000 default)
            max_daily_loss: Maximum daily loss limit ($400 = 4% of $10k default)
        """
        self.max_exposure = max_exposure
        self.max_daily_loss = max_daily_loss
        self.composite_score_threshold = 6.5
        self.min_rr_ratio = 2.0  # Minimum 1:2 risk:reward
        self.feature_lab = FeatureLab()
        self.volume_screener = VolumeScreeningEngine()
        self.trade_execution_service = TradeExecutionService()
    
    def evaluate_trade_decision(self, symbol: str, command_ts: Optional[str] = None, asset_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate all conditions for "TRADE YES" decision
        
        Returns comprehensive decision with all condition checks
        """
        symbol = (symbol or "").upper()
        if not symbol:
            raise ValueError("Symbol is required")
        
        decision = {
            "symbol": symbol,
            "timestamp": command_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "conditions": {},
            "trade_decision": "NO TRADE",
            "direction": None,
            "reason": "",
            "risk_metrics": {},
            "position_size": None,
            "recommendation": {}
        }
        
        # Condition 1: Composite Score >= 6.5
        try:
            quantum_result = self.feature_lab.run_quantum_screen(symbol, command_ts, asset_class=asset_class)
            composite_score = quantum_result["metrics"].get("composite_score", 0.0)
            condition1_pass = composite_score >= self.composite_score_threshold
            decision["conditions"]["composite_score"] = {
                "value": composite_score,
                "threshold": self.composite_score_threshold,
                "pass": condition1_pass,
                "details": quantum_result["metrics"]
            }
        except Exception as e:
            condition1_pass = False
            decision["conditions"]["composite_score"] = {
                "value": None,
                "threshold": self.composite_score_threshold,
                "pass": False,
                "error": str(e)
            }
        
        # Condition 2: All Phase 1 gates passed
        phase1_result = self.volume_screener.screen_symbol(symbol, asset_class=asset_class)
        condition2_pass = phase1_result.get("pass", False)
        decision["conditions"]["phase1_gates"] = {
            "pass": condition2_pass,
            "reason": phase1_result.get("reason", "Unknown"),
            "metrics": phase1_result.get("metrics", {})
        }
        
        # Condition 3: R:R ratio achievable >= 1:2
        rr_result = self._calculate_rr_ratio(symbol, quantum_result if condition1_pass else None, asset_class=asset_class or "stocks")
        condition3_pass = rr_result["achievable"] and rr_result["ratio"] >= self.min_rr_ratio
        decision["conditions"]["rr_ratio"] = rr_result
        
        # Condition 4: Position size calculable within exposure limit
        position_result = self._calculate_position_size(
            symbol, 
            quantum_result if condition1_pass else None,
            rr_result
        )
        condition4_pass = position_result["calculable"] and position_result["exposure"] <= self.max_exposure
        decision["conditions"]["position_size"] = position_result
        decision["position_size"] = position_result.get("recommended_shares")
        
        # Condition 5: No conflicting daily trend
        trend_result = self._check_trend_alignment(symbol, quantum_result if condition1_pass else None)
        condition5_pass = not trend_result["conflict"]
        decision["conditions"]["trend_alignment"] = trend_result
        
        # Check risk management overlay
        risk_check = self._check_risk_overlay(symbol)
        decision["risk_metrics"] = risk_check
        
        # Direction Decision
        direction = self._determine_direction(trend_result)
        decision["direction"] = direction
        
        # Final Trade Decision
        all_conditions_pass = (
            condition1_pass and 
            condition2_pass and 
            condition3_pass and 
            condition4_pass and 
            condition5_pass and
            risk_check["can_trade"]
        )
        
        if all_conditions_pass:
            if direction == "CONFLICT":
                decision["trade_decision"] = "NO TRADE"
                decision["reason"] = "Trend conflict - 5-min trend contradicts higher timeframes"
            else:
                decision["trade_decision"] = "TRADE YES"
                decision["reason"] = "All conditions met"
                decision["recommendation"] = {
                    "action": "BUY" if direction == "UP" else "SELL",
                    "entry_price": position_result.get("entry_price"),
                    "stop_loss": position_result.get("stop_loss"),
                    "target1": position_result.get("target1"),
                    "target2": position_result.get("target2"),
                    "position_size_shares": position_result.get("recommended_shares"),
                    "exposure": position_result.get("exposure"),
                    "risk_amount": position_result.get("risk_amount"),
                    "reward_amount": position_result.get("reward_amount"),
                    "rr_ratio": rr_result.get("ratio")
                }
        else:
            failed_conditions = []
            if not condition1_pass:
                failed_conditions.append(f"Composite Score {decision['conditions']['composite_score']['value']:.2f} < {self.composite_score_threshold}")
            if not condition2_pass:
                failed_conditions.append(f"Phase 1 Gates: {phase1_result.get('reason', 'Failed')}")
            if not condition3_pass:
                failed_conditions.append(f"R:R Ratio {rr_result.get('ratio', 0):.2f} < {self.min_rr_ratio}")
            if not condition4_pass:
                failed_conditions.append(f"Position size exceeds ${self.max_exposure} limit")
            if not condition5_pass:
                failed_conditions.append(f"Trend conflict: {trend_result.get('conflict_reason', 'Unknown')}")
            if not risk_check["can_trade"]:
                failed_conditions.append(risk_check.get("reason", "Risk check failed"))
            
            decision["trade_decision"] = "NO TRADE"
            decision["reason"] = " | ".join(failed_conditions)
        
        return decision
    
    def _calculate_rr_ratio(self, symbol: str, quantum_result: Optional[Dict] = None, asset_class: str = "stocks") -> Dict[str, Any]:
        """Calculate Risk:Reward ratio and check if 1:2 is achievable"""
        try:
            # Get current price and ATR for stop loss calculation
            hist = self.volume_screener._fetch_history(symbol, period="3mo", asset_class=asset_class)
            if hist is None or hist.empty:
                return {"achievable": False, "ratio": 0.0, "error": "No price data"}
            
            current_price = float(hist['Close'].iloc[-1])
            
            # Calculate ATR for stop loss distance
            atr_df = pd.DataFrame({
                'High': pd.to_numeric(hist['High'], errors='coerce'),
                'Low': pd.to_numeric(hist['Low'], errors='coerce'),
                'Close': pd.to_numeric(hist['Close'], errors='coerce'),
            })
            
            fire_tester = FireTestingEngine()
            atr_series = fire_tester._calc_atr(atr_df, 14)
            atr_value = float(atr_series.iloc[-1]) if len(atr_series) > 0 else None
            fire_tester.close()
            
            if not atr_value or atr_value <= 0:
                return {"achievable": False, "ratio": 0.0, "error": "ATR calculation failed"}
            
            # Stop loss at 1.5x ATR
            stop_distance = atr_value * 1.5
            stop_loss = current_price - stop_distance  # For long position
            
            # Target 1: 2x stop distance (1:2 R:R)
            target1 = current_price + (stop_distance * 2)
            
            # Target 2: 2.5x stop distance (1:2.5 R:R)
            target2 = current_price + (stop_distance * 2.5)
            
            # Calculate R:R ratio
            risk = stop_distance
            reward = stop_distance * 2
            rr_ratio = reward / risk if risk > 0 else 0.0
            
            return {
                "achievable": rr_ratio >= self.min_rr_ratio,
                "ratio": round(rr_ratio, 2),
                "current_price": round(current_price, 2),
                "stop_loss": round(stop_loss, 2),
                "target1": round(target1, 2),
                "target2": round(target2, 2),
                "stop_distance": round(stop_distance, 2),
                "atr": round(atr_value, 2)
            }
        except Exception as e:
            return {"achievable": False, "ratio": 0.0, "error": str(e)}
    
    def _calculate_position_size(
        self, 
        symbol: str, 
        quantum_result: Optional[Dict] = None,
        rr_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Calculate position size within $2,000 exposure limit"""
        try:
            if not rr_result or not rr_result.get("achievable"):
                return {
                    "calculable": False,
                    "exposure": 0.0,
                    "error": "R:R ratio not achievable"
                }
            
            current_price = rr_result.get("current_price")
            stop_distance = rr_result.get("stop_distance", 0)
            
            if not current_price or stop_distance <= 0:
                return {
                    "calculable": False,
                    "exposure": 0.0,
                    "error": "Invalid price or stop distance"
                }
            
            # Risk per share
            risk_per_share = stop_distance
            
            # Maximum risk per trade (typically 1-2% of account, but we use a fixed $ amount)
            # For $10k account, 1% = $100 risk per trade
            max_risk_per_trade = 100.0  # Conservative: $100 risk per trade
            
            # Calculate position size based on risk
            max_shares_by_risk = int(max_risk_per_trade / risk_per_share) if risk_per_share > 0 else 0
            
            # Calculate position size based on exposure limit
            max_shares_by_exposure = int(self.max_exposure / current_price) if current_price > 0 else 0
            
            # Use the smaller of the two (more conservative)
            recommended_shares = min(max_shares_by_risk, max_shares_by_exposure)
            
            # Calculate actual exposure
            exposure = recommended_shares * current_price
            
            # Calculate actual risk
            risk_amount = recommended_shares * risk_per_share
            reward_amount = recommended_shares * (stop_distance * 2)
            
            return {
                "calculable": exposure <= self.max_exposure and recommended_shares > 0,
                "exposure": round(exposure, 2),
                "recommended_shares": recommended_shares,
                "entry_price": current_price,
                "stop_loss": rr_result.get("stop_loss"),
                "target1": rr_result.get("target1"),
                "target2": rr_result.get("target2"),
                "risk_amount": round(risk_amount, 2),
                "reward_amount": round(reward_amount, 2),
                "max_risk_per_trade": max_risk_per_trade,
                "max_exposure_limit": self.max_exposure
            }
        except Exception as e:
            return {
                "calculable": False,
                "exposure": 0.0,
                "error": str(e)
            }
    
    def _check_trend_alignment(self, symbol: str, quantum_result: Optional[Dict] = None, asset_class: Optional[str] = None) -> Dict[str, Any]:
        """Check for trend conflicts between 5-min and higher timeframes"""
        try:
            # Get 5-minute and daily data
            m5 = self.feature_lab._fetch_df(symbol, "5min", lookback_days=7, asset_class=asset_class)
            daily = self.feature_lab._fetch_df(symbol, "1d", lookback_days=60, asset_class=asset_class)
            
            if m5.empty or daily.empty:
                return {
                    "conflict": True,
                    "conflict_reason": "Insufficient data for trend analysis",
                    "m5_trend": None,
                    "daily_trend": None
                }
            
            # Determine 5-minute trend (using EMA or SMA)
            m5_close = pd.to_numeric(m5["Close"], errors="coerce")
            if len(m5_close) < 20:
                m5_ema = m5_close.ewm(span=10, adjust=False).mean()
            else:
                m5_ema = m5_close.ewm(span=20, adjust=False).mean()
            
            current_m5_price = float(m5_close.iloc[-1])
            m5_ema_current = float(m5_ema.iloc[-1])
            m5_trend = "UP" if current_m5_price > m5_ema_current else "DOWN"
            
            # Get daily trend from quantum metrics if available
            daily_trend = "NEUTRAL"
            if quantum_result:
                daily_metrics = quantum_result.get("metrics", {})
                daily_trend = daily_metrics.get("daily_trend", "NEUTRAL")
            else:
                # Calculate daily trend manually
                daily_close = pd.to_numeric(daily["Close"], errors="coerce")
                if len(daily_close) >= 50:
                    sma20 = daily_close.tail(20).mean()
                    sma50 = daily_close.tail(50).mean()
                    current_daily_price = float(daily_close.iloc[-1])
                    
                    if current_daily_price > sma20 > sma50:
                        daily_trend = "BULLISH"
                    elif current_daily_price < sma20 < sma50:
                        daily_trend = "BEARISH"
                    else:
                        daily_trend = "NEUTRAL"
            
            # Normalize daily trend to UP/DOWN
            daily_direction = "UP" if daily_trend in ["BULLISH", "UP"] else ("DOWN" if daily_trend in ["BEARISH", "DOWN"] else "NEUTRAL")
            
            # Check for conflict
            conflict = False
            conflict_reason = None
            
            if daily_direction == "NEUTRAL":
                conflict = False  # Neutral daily trend doesn't conflict
            elif m5_trend == "UP" and daily_direction == "DOWN":
                conflict = True
                conflict_reason = "5-min UP but daily DOWN (counter-trend)"
            elif m5_trend == "DOWN" and daily_direction == "UP":
                conflict = True
                conflict_reason = "5-min DOWN but daily UP (counter-trend)"
            
            return {
                "conflict": conflict,
                "conflict_reason": conflict_reason,
                "m5_trend": m5_trend,
                "daily_trend": daily_direction,
                "daily_trend_full": daily_trend
            }
        except Exception as e:
            return {
                "conflict": True,
                "conflict_reason": f"Error analyzing trends: {str(e)}",
                "m5_trend": None,
                "daily_trend": None
            }
    
    def _determine_direction(self, trend_result: Dict[str, Any]) -> str:
        """Determine trade direction based on trend alignment"""
        if trend_result.get("conflict"):
            return "CONFLICT"
        
        m5_trend = trend_result.get("m5_trend")
        daily_trend = trend_result.get("daily_trend")
        
        # If 5-min and daily are aligned UP → BUY
        if m5_trend == "UP" and daily_trend in ["UP", "NEUTRAL"]:
            return "UP"
        
        # If 5-min and daily are aligned DOWN → SELL
        if m5_trend == "DOWN" and daily_trend in ["DOWN", "NEUTRAL"]:
            return "DOWN"
        
        return "CONFLICT"
    
    def _check_risk_overlay(self, symbol: str) -> Dict[str, Any]:
        """Check risk management overlay conditions"""
        try:
            # Check max concurrent trades
            can_open, message = self.trade_execution_service.can_open_trade()
            active_trades = self.trade_execution_service.get_active_trades()
            active_trades_count = len([t for t in active_trades 
                                     if t.get("status") not in ("closed", "stopped_out")])
            
            # Check if we can trade (within 3 concurrent limit)
            can_trade = can_open and active_trades_count < TradingSessionManager.MAX_CONCURRENT_TRADES
            
            return {
                "can_trade": can_trade,
                "active_trades": active_trades_count,
                "max_concurrent": TradingSessionManager.MAX_CONCURRENT_TRADES,
                "max_daily_loss": self.max_daily_loss,
                "max_exposure_per_trade": self.max_exposure,
                "auto_close_time": "4:00 PM ET",
                "reason": message if not can_trade else "Risk checks passed"
            }
        except Exception as e:
            return {
                "can_trade": False,
                "reason": f"Error checking risk overlay: {str(e)}",
                "max_daily_loss": self.max_daily_loss,
                "max_exposure_per_trade": self.max_exposure
            }
    
    def close(self):
        """Close connections"""
        self.feature_lab = None
        self.volume_screener = None
        self.trade_execution_service = None

