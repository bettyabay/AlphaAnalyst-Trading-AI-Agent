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


class VolumeScreeningEngine:
    """Phase 1 Volume Screening Engine"""
    
    def __init__(self):
        self.min_volume_spike = 1.3  # 130% of average volume (adjusted for more realistic screening)
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
    
    def _fetch_history(self, symbol: str, period: str = "3mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch historical data"""
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            if data is None or data.empty:
                return None
            
            # Handle MultiIndex columns (yfinance returns MultiIndex with symbol name as second level)
            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(1, axis=1)
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in data.columns:
                    return None
            
            return data
        except Exception:
            return None
    
    def screen_symbol(self, symbol: str) -> Dict:
        """Screen a single symbol for volume spike criteria"""
        # Fetch 1 year of data to ensure we have enough for SMA200 calculation
        hist = self._fetch_history(symbol, period="1y")
        
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
    
    def _fetch_history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch historical data"""
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            if data is None or data.empty:
                return None
            
            # Handle MultiIndex columns (yfinance returns MultiIndex with symbol name as second level)
            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(1, axis=1)
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in data.columns:
                    return None
            
            return data
        except Exception:
            return None
    
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

