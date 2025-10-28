"""
AI Analysis module for AlphaAnalyst Trading AI Agent
Phase 2: Enhanced AI integration for research analysis
"""
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from phi.model.xai import xAI
from phi.agent.agent import Agent
from .document_manager import DocumentManager
from .polygon_integration import PolygonDataClient
from groq import Groq


class AIResearchAnalyzer:
    """AI-powered research analysis for trading insights"""
    
    def __init__(self):
        xai_key = os.getenv("XAI_API_KEY", "")
        if not xai_key:
            print("Warning: XAI_API_KEY not found. xAI provider disabled; will fallback to Groq if configured.")
            self.xai_model = None
        else:
            self.xai_model = xAI(
                model="grok-beta",
                api_key=xai_key
            )
        self.document_manager = DocumentManager()
        self.polygon_client = PolygonDataClient()
    
    def analyze_instrument_profile(self, symbol: str) -> Dict:
        """Create comprehensive AI analysis of an instrument"""
        try:
            # Get market data
            market_data = self._get_market_data(symbol)
            
            # Get document insights
            doc_insights = self.document_manager.get_document_insights(symbol)
            
            # Get news sentiment
            news_sentiment = self._analyze_news_sentiment(symbol)
            
            # Generate AI analysis
            ai_analysis = self._generate_comprehensive_analysis(
                symbol, market_data, doc_insights, news_sentiment
            )
            
            return {
                "symbol": symbol,
                "market_data": market_data,
                "document_insights": doc_insights,
                "news_sentiment": news_sentiment,
                "ai_analysis": ai_analysis,
                "analysis_timestamp": datetime.now().isoformat(),
                "confidence_score": self._calculate_confidence_score(doc_insights, news_sentiment)
            }
            
        except Exception as e:
            return {"error": str(e), "symbol": symbol}
    
    def _get_market_data(self, symbol: str) -> Dict:
        """Get recent market data for analysis"""
        try:
            # Get recent price data
            recent_data = self.polygon_client.get_recent_data(symbol, days=30)
            
            if recent_data and len(recent_data) > 0:
                latest = recent_data.iloc[-1]
                first = recent_data.iloc[0]
                
                price_change = latest['close'] - first['close']
                price_change_pct = (price_change / first['close']) * 100
                
                return {
                    "current_price": latest['close'],
                    "price_change": price_change,
                    "price_change_pct": price_change_pct,
                    "volume": latest['volume'],
                    "high_30d": recent_data['high'].max(),
                    "low_30d": recent_data['low'].min(),
                    "volatility": recent_data['close'].std()
                }
            else:
                return {"error": "No market data available"}
                
        except Exception as e:
            return {"error": f"Market data error: {str(e)}"}
    
    def _analyze_news_sentiment(self, symbol: str) -> Dict:
        """Analyze news sentiment for the symbol"""
        try:
            # This would integrate with news APIs in a real implementation
            # For now, return a placeholder structure
            return {
                "sentiment_score": 0.0,  # -1 to 1 scale
                "news_count": 0,
                "positive_news": 0,
                "negative_news": 0,
                "neutral_news": 0,
                "key_headlines": []
            }
        except Exception as e:
            return {"error": f"News analysis error: {str(e)}"}
    
    def _generate_comprehensive_analysis(self, symbol: str, market_data: Dict, 
                                       doc_insights: List[Dict], news_sentiment: Dict) -> Dict:
        """Generate comprehensive AI analysis"""
        try:
            if not self.xai_model:
                # Fallback to Groq if available
                groq_key = os.getenv("GROQ_API_KEY", "")
                if not groq_key:
                    return {"error": "AI analysis disabled - XAI_API_KEY or GROQ_API_KEY not configured"}
            
            # Prepare context for AI analysis
            context = f"""
            Symbol: {symbol}
            
            Market Data:
            - Current Price: ${market_data.get('current_price', 'N/A')}
            - 30-day Change: {market_data.get('price_change_pct', 'N/A')}%
            - Volume: {market_data.get('volume', 'N/A')}
            - 30-day High: ${market_data.get('high_30d', 'N/A')}
            - 30-day Low: ${market_data.get('low_30d', 'N/A')}
            - Volatility: {market_data.get('volatility', 'N/A')}
            
            Document Insights:
            {self._format_document_insights(doc_insights)}
            
            News Sentiment:
            - Sentiment Score: {news_sentiment.get('sentiment_score', 'N/A')}
            - News Count: {news_sentiment.get('news_count', 'N/A')}
            """
            
            analysis_prompt = f"""
            As a professional financial analyst, provide a comprehensive analysis of {symbol} based on the following data:
            
            {context}
            
            Please provide:
            1. Overall Assessment (Bullish/Bearish/Neutral)
            2. Key Strengths
            3. Key Risks
            4. Technical Analysis Summary
            5. Fundamental Analysis Summary
            6. Investment Recommendation (BUY/SELL/HOLD)
            7. Price Target (if applicable)
            8. Confidence Level (1-10)
            9. Key Catalysts to Watch
            10. Risk Factors
            
            Format your response as a structured analysis suitable for investment decision making.
            """
            
            # Try xAI via Agent first
            response = None
            if self.xai_model:
                try:
                    agent = Agent(model=self.xai_model)
                    response = agent.run(analysis_prompt)
                except Exception as e:
                    # On xAI failure, fallback to Groq
                    response = None
            if response is None:
                groq_key = os.getenv("GROQ_API_KEY", "")
                if not groq_key:
                    return {"error": "AI analysis error: No working provider. Set GROQ_API_KEY or XAI_API_KEY"}
                client = Groq(api_key=groq_key)
                chat = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a professional financial analyst."},
                        {"role": "user", "content": analysis_prompt},
                    ],
                    temperature=0.2,
                )
                response = chat.choices[0].message.content if chat.choices else ""
            
            return {
                "analysis_text": response,
                "overall_sentiment": self._extract_sentiment_from_analysis(response),
                "recommendation": self._extract_recommendation(response),
                "confidence": self._extract_confidence(response)
            }
            
        except Exception as e:
            msg = str(e)
            if "403" in msg or "permission" in msg.lower() or "credits" in msg.lower():
                return {"error": "AI analysis error: Permission/credits issue with provider. Set XAI_API_KEY or GROQ_API_KEY and ensure access.", "details": msg}
            return {"error": f"AI analysis error: {msg}"}
    
    def _format_document_insights(self, doc_insights: List[Dict]) -> str:
        """Format document insights for AI analysis"""
        if not doc_insights:
            return "No research documents available"
        
        formatted = []
        for insight in doc_insights:
            signals = insight.get("signals", {})
            if signals.get("success"):
                formatted.append(f"""
                Document: {insight.get('filename', 'Unknown')}
                - Sentiment: {signals.get('overall_sentiment', 'Unknown')}
                - Bullish Signals: {len(signals.get('bullish_signals', []))}
                - Bearish Signals: {len(signals.get('bearish_signals', []))}
                - Confidence: {signals.get('confidence', 'N/A')}
                """)
        
        return "\n".join(formatted) if formatted else "No document insights available"
    
    def _extract_sentiment_from_analysis(self, analysis_text: str) -> str:
        """Extract overall sentiment from AI analysis"""
        analysis_lower = analysis_text.lower()
        if "bullish" in analysis_lower:
            return "Bullish"
        elif "bearish" in analysis_lower:
            return "Bearish"
        else:
            return "Neutral"
    
    def _extract_recommendation(self, analysis_text: str) -> str:
        """Extract investment recommendation from AI analysis"""
        analysis_lower = analysis_text.lower()
        if "buy" in analysis_lower and "sell" not in analysis_lower:
            return "BUY"
        elif "sell" in analysis_lower:
            return "SELL"
        else:
            return "HOLD"
    
    def _extract_confidence(self, analysis_text: str) -> int:
        """Extract confidence level from AI analysis"""
        # Simple extraction - look for confidence numbers
        import re
        confidence_match = re.search(r'confidence[:\s]*(\d+)', analysis_text.lower())
        if confidence_match:
            return int(confidence_match.group(1))
        return 5  # Default confidence
    
    def _calculate_confidence_score(self, doc_insights: List[Dict], news_sentiment: Dict) -> float:
        """Calculate overall confidence score"""
        try:
            # Base confidence on available data
            confidence_factors = []
            
            # Document insights factor
            if doc_insights:
                avg_doc_confidence = np.mean([
                    insight.get("signals", {}).get("confidence", 5) 
                    for insight in doc_insights 
                    if insight.get("signals", {}).get("success")
                ])
                confidence_factors.append(avg_doc_confidence)
            
            # News sentiment factor
            news_count = news_sentiment.get("news_count", 0)
            if news_count > 0:
                confidence_factors.append(min(news_count / 10, 10))  # Scale news count
            
            # Calculate weighted average
            if confidence_factors:
                return np.mean(confidence_factors)
            else:
                return 5.0  # Default confidence
                
        except Exception:
            return 5.0
    
    def get_master_data_summary(self, symbols: List[str] = None) -> Dict:
        """Get master data summary for all instruments or specified symbols"""
        try:
            if not symbols:
                from ..config.watchlist import get_watchlist_symbols
                symbols = get_watchlist_symbols()
            
            summary = {
                "total_instruments": len(symbols),
                "analysis_timestamp": datetime.now().isoformat(),
                "instruments": []
            }
            
            for symbol in symbols:
                instrument_profile = self.analyze_instrument_profile(symbol)
                summary["instruments"].append(instrument_profile)
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}
    
    def close(self):
        """Close connections"""
        self.document_manager.close()
