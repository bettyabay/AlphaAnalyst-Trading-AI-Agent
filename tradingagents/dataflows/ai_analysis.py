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

from .document_manager import DocumentManager
from .market_data_service import fetch_ohlcv
from groq import Groq


class AIResearchAnalyzer:
    """AI-powered research analysis for trading insights"""
    
    def __init__(self):
        # Groq-only initialization
        self.groq_key = os.getenv("GROQ_API_KEY", "") or os.getenv("GROK_API_KEY", "")
        self.document_manager = DocumentManager()
    
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
            recent_data = fetch_ohlcv(symbol, interval="1d", lookback_days=40)
            if recent_data is None or recent_data.empty:
                return {"error": "No stored market data in Supabase. Please ingest data first."}

            latest = recent_data.iloc[-1]
            first = recent_data.iloc[0]

            price_change = float(latest['Close']) - float(first['Close'])
            price_change_pct = (price_change / float(first['Close'])) * 100 if float(first['Close']) else 0.0

            return {
                "current_price": float(latest['Close']),
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "volume": int(latest['Volume']) if 'Volume' in recent_data.columns else None,
                "high_30d": float(recent_data['High'].max()),
                "low_30d": float(recent_data['Low'].min()),
                "volatility": float(recent_data['Close'].std()),
                "source": "supabase_market_data"
            }
        except Exception as e:
            return {"error": f"Market data error: {str(e)}"}
    
    def _analyze_news_sentiment(self, symbol: str) -> Dict:
        """Analyze news sentiment for the symbol"""
        try:
            # Integrate with AlphaVantage for real news sentiment
            from .alpha_vantage_news import get_news
            import json
            from datetime import timedelta
            
            # Calculate date range (last 7 days for relevant news)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Fetch news from AlphaVantage
            news_response = get_news(symbol, start_date, end_date)
            
            # Parse response if needed
            if isinstance(news_response, str):
                try:
                    data = json.loads(news_response)
                except json.JSONDecodeError:
                    return {"error": f"Failed to parse news response: {news_response[:100]}"}
            else:
                data = news_response
                
            # Handle API errors or empty data
            if "feed" not in data:
                # Fallback to neutral if no data, but preserve error info if present
                error_msg = data.get("Information", data.get("Note", "No news feed returned"))
                if "limit" in str(error_msg).lower():
                     return {"error": f"API Limit: {error_msg}"}
                
                return {
                    "sentiment_score": 0.0,
                    "news_count": 0,
                    "positive_news": 0,
                    "negative_news": 0,
                    "neutral_news": 0,
                    "key_headlines": []
                }

            feed = data["feed"]
            news_count = len(feed)
            
            if news_count == 0:
                 return {
                    "sentiment_score": 0.0,
                    "news_count": 0,
                    "positive_news": 0,
                    "negative_news": 0,
                    "neutral_news": 0,
                    "key_headlines": []
                }

            # Calculate sentiment metrics
            total_sentiment = 0.0
            positive = 0
            negative = 0
            neutral = 0
            headlines = []
            
            for article in feed:
                # AlphaVantage gives "overall_sentiment_score" (-1 to 1)
                sentiment_score = float(article.get("overall_sentiment_score", 0))
                total_sentiment += sentiment_score
                
                # Count sentiment categories (0.15 threshold)
                if sentiment_score > 0.15:
                    positive += 1
                elif sentiment_score < -0.15:
                    negative += 1
                else:
                    neutral += 1
                    
                # Collect top 5 headlines
                if len(headlines) < 5:
                    headlines.append({
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "sentiment": article.get("overall_sentiment_label", "Neutral"),
                        "score": sentiment_score,
                        "time": article.get("time_published", "")
                    })
            
            avg_sentiment = total_sentiment / news_count
            
            return {
                "sentiment_score": round(avg_sentiment, 2),
                "news_count": news_count,
                "positive_news": positive,
                "negative_news": negative,
                "neutral_news": neutral,
                "key_headlines": headlines
            }
            
        except Exception as e:
            return {"error": f"News analysis error: {str(e)}"}
    
    def _generate_comprehensive_analysis(self, symbol: str, market_data: Dict, 
                                       doc_insights: List[Dict], news_sentiment: Dict) -> Dict:
        """Generate comprehensive AI analysis"""
        try:
            # Require Groq API key
            if not self.groq_key:
                return {"error": "AI analysis disabled - GROQ_API_KEY not configured"}

            # Prepare context for AI analysis - handle market_data errors gracefully
            market_info = ""
            if "error" not in market_data:
                market_info = f"""
            Market Data:
            - Current Price: ${market_data.get('current_price', 'N/A')}
            - 30-day Change: {market_data.get('price_change_pct', 'N/A')}%
            - Volume: {market_data.get('volume', 'N/A')}
            - 30-day High: ${market_data.get('high_30d', 'N/A')}
            - 30-day Low: ${market_data.get('low_30d', 'N/A')}
            - Volatility: {market_data.get('volatility', 'N/A')}
            """
            else:
                market_info = f"Market Data: {market_data.get('error', 'Not available')}"
            
            context = f"""
            Symbol: {symbol}
            
            {market_info}
            
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
            
            # Groq generation
            # Validate API key format
            if not self.groq_key:
                return {
                    "error": "AI analysis disabled - GROQ_API_KEY not configured",
                    "details": "Please set GROQ_API_KEY in your .env file"
                }
            
            if not self.groq_key.startswith('gsk_'):
                return {
                    "error": "Invalid API Key Format",
                    "details": "GROQ_API_KEY should start with 'gsk_'. Please check your .env file and ensure you have a valid Groq API key from https://console.groq.com/"
                }
            
            try:
                client = Groq(api_key=self.groq_key)
                chat = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
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
            except Exception as api_error:
                error_str = str(api_error)
                if "401" in error_str or "invalid_api_key" in error_str.lower() or "Invalid API Key" in error_str:
                    masked_key = f"{self.groq_key[:8]}...{self.groq_key[-4:]}" if len(self.groq_key) > 12 else "***"
                    return {
                        "error": "Invalid API Key",
                        "details": f"401 Unauthorized - The GROQ_API_KEY appears to be invalid. Key format: {masked_key}. Please:\n1. Get a valid API key from https://console.groq.com/\n2. Update your .env file: GROQ_API_KEY=gsk_your_actual_key_here\n3. Restart the application"
                    }
                elif "403" in error_str or "permission" in error_str.lower() or "credits" in error_str.lower():
                    return {
                        "error": "AI analysis error: Permission/credits issue with provider. Ensure GROQ_API_KEY has access.",
                        "details": error_str
                    }
                else:
                    return {"error": f"AI analysis error: {error_str}"}
            
        except Exception as e:
            msg = str(e)
            if "401" in msg or "invalid_api_key" in msg.lower():
                return {
                    "error": "Invalid API Key",
                    "details": f"401 Unauthorized - Please verify your GROQ_API_KEY in the .env file. Get a key from https://console.groq.com/"
                }
            elif "403" in msg or "permission" in msg.lower() or "credits" in msg.lower():
                return {"error": "AI analysis error: Permission/credits issue with provider. Ensure GROQ_API_KEY has access.", "details": msg}
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
    
    def save_master_data_summary_with_rag(self, summary: Dict) -> Dict:
        """
        Save master data summary to database with embeddings for RAG retrieval.
        Saves each instrument as a separate row in the master_data table.
        
        Args:
            summary: Master data summary dictionary from get_master_data_summary()
        
        Returns:
            Dictionary with success status and details
        """
        try:
            from ..database.db_service import save_master_data_with_rag
            from datetime import datetime
            
            if "error" in summary:
                return {"success": False, "error": summary.get("error")}
            
            instruments = summary.get("instruments", [])
            analysis_timestamp = summary.get("analysis_timestamp", datetime.now().isoformat())
            
            saved_count = 0
            failed_count = 0
            errors = []
            
            for instrument in instruments:
                try:
                    symbol = instrument.get("symbol")
                    if not symbol:
                        failed_count += 1
                        errors.append(f"Missing symbol in instrument profile")
                        continue
                    
                    # Convert to text for embedding
                    content_text = self._master_data_to_text(instrument)
                    
                    # Generate embedding using document manager
                    embedding_vector = self.document_manager._generate_embedding(content_text)
                    
                    # Save to database
                    result = save_master_data_with_rag(
                        symbol=symbol,
                        content_text=content_text,
                        embedding_vector=embedding_vector,
                        full_data=instrument,
                        analysis_timestamp=analysis_timestamp
                    )
                    
                    if result:
                        saved_count += 1
                    else:
                        failed_count += 1
                        errors.append(f"Failed to save {symbol}: No result returned")
                        
                except Exception as e:
                    failed_count += 1
                    error_msg = str(e)
                    errors.append(f"Failed to save {instrument.get('symbol', 'Unknown')}: {error_msg}")
                    # Continue with other instruments even if one fails
                    continue
            
            return {
                "success": saved_count > 0,
                "saved_count": saved_count,
                "failed_count": failed_count,
                "total_instruments": len(instruments),
                "errors": errors[:10] if errors else [],  # Limit error messages
                "analysis_timestamp": analysis_timestamp
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _master_data_to_text(self, instrument_profile: Dict) -> str:
        """Convert instrument profile to text format for embedding generation"""
        try:
            symbol = instrument_profile.get("symbol", "Unknown")
            ai_analysis = instrument_profile.get("ai_analysis", {})
            market_data = instrument_profile.get("market_data", {})
            doc_insights = instrument_profile.get("document_insights", [])
            news_sentiment = instrument_profile.get("news_sentiment", {})
            
            # Build comprehensive text representation
            text_parts = [
                f"Master Data Analysis for {symbol}",
                f"Analysis Timestamp: {instrument_profile.get('analysis_timestamp', 'N/A')}",
                f"Confidence Score: {instrument_profile.get('confidence_score', 'N/A')}/10",
                ""
            ]
            
            # Market Data Section
            if "error" not in market_data:
                text_parts.extend([
                    "Market Data:",
                    f"  Current Price: ${market_data.get('current_price', 'N/A')}",
                    f"  30-day Price Change: {market_data.get('price_change_pct', 'N/A')}%",
                    f"  Volume: {market_data.get('volume', 'N/A')}",
                    f"  30-day High: ${market_data.get('high_30d', 'N/A')}",
                    f"  30-day Low: ${market_data.get('low_30d', 'N/A')}",
                    f"  Volatility: {market_data.get('volatility', 'N/A')}",
                    f"  Data Source: {market_data.get('source', 'N/A')}",
                    ""
                ])
            else:
                text_parts.append(f"Market Data: {market_data.get('error', 'Not available')}\n")
            
            # AI Analysis Section
            if "error" not in ai_analysis:
                text_parts.extend([
                    "AI Analysis:",
                    f"  Overall Sentiment: {ai_analysis.get('overall_sentiment', 'N/A')}",
                    f"  Recommendation: {ai_analysis.get('recommendation', 'N/A')}",
                    f"  Confidence: {ai_analysis.get('confidence', 'N/A')}/10",
                    ""
                ])
                if ai_analysis.get("analysis_text"):
                    # Truncate analysis text to avoid embedding size limits
                    analysis_text = ai_analysis.get('analysis_text', '')
                    if len(analysis_text) > 1000:
                        analysis_text = analysis_text[:1000] + "..."
                    text_parts.append(f"  Analysis Text: {analysis_text}")
                    text_parts.append("")
            else:
                text_parts.append(f"AI Analysis: {ai_analysis.get('error', 'Not available')}\n")
            
            # Document Insights Section
            if doc_insights:
                text_parts.append("Document Insights:")
                for idx, insight in enumerate(doc_insights, 1):
                    signals = insight.get("signals", {})
                    if signals.get("success"):
                        text_parts.append(
                            f"  Document {idx} ({insight.get('filename', 'Unknown')}): "
                            f"Sentiment: {signals.get('overall_sentiment', 'N/A')}, "
                            f"Confidence: {signals.get('confidence', 'N/A')}/10, "
                            f"Bullish Signals: {len(signals.get('bullish_signals', []))}, "
                            f"Bearish Signals: {len(signals.get('bearish_signals', []))}"
                        )
                text_parts.append("")
            else:
                text_parts.append("Document Insights: No documents available\n")
            
            # News Sentiment Section
            if news_sentiment.get("news_count", 0) > 0:
                text_parts.extend([
                    "News Sentiment:",
                    f"  Sentiment Score: {news_sentiment.get('sentiment_score', 'N/A')}",
                    f"  News Count: {news_sentiment.get('news_count', 0)}",
                    f"  Positive News: {news_sentiment.get('positive_news', 0)}",
                    f"  Negative News: {news_sentiment.get('negative_news', 0)}",
                    f"  Neutral News: {news_sentiment.get('neutral_news', 0)}",
                    ""
                ])
            
            return "\n".join(text_parts)
        except Exception as e:
            # Fallback: return JSON string representation
            import json
            return json.dumps(instrument_profile, indent=2, default=str)
    
    def close(self):
        """Close connections"""
        self.document_manager.close()
