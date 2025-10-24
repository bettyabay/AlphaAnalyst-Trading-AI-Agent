"""
Master Data Visualization module for AlphaAnalyst Trading AI Agent
Phase 2: Enhanced visualization and comprehensive instrument profiles
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import streamlit as st

from .polygon_integration import PolygonDataClient
from .ai_analysis import AIResearchAnalyzer


class MasterDataVisualizer:
    """Comprehensive visualization for master data and instrument profiles"""
    
    def __init__(self):
        self.polygon_client = PolygonDataClient()
        self.ai_analyzer = AIResearchAnalyzer()
    
    def create_instrument_profile_chart(self, symbol: str, days: int = 90) -> go.Figure:
        """Create comprehensive instrument profile chart"""
        try:
            # Get market data
            market_data = self.polygon_client.get_recent_data(symbol, days=days)
            
            if market_data is None or market_data.empty:
                return self._create_error_chart(f"No data available for {symbol}")
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=[
                    f"{symbol} Price Chart ({days} days)",
                    "Volume",
                    "Technical Indicators"
                ],
                vertical_spacing=0.08,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Price chart with candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=market_data.index,
                    open=market_data['open'],
                    high=market_data['high'],
                    low=market_data['low'],
                    close=market_data['close'],
                    name="Price",
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            market_data['MA20'] = market_data['close'].rolling(window=20).mean()
            market_data['MA50'] = market_data['close'].rolling(window=50).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=market_data.index,
                    y=market_data['MA20'],
                    name='MA20',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=market_data.index,
                    y=market_data['MA50'],
                    name='MA50',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Volume chart
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(market_data['close'], market_data['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=market_data.index,
                    y=market_data['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Technical indicators
            # RSI
            market_data['RSI'] = self._calculate_rsi(market_data['close'])
            
            fig.add_trace(
                go.Scatter(
                    x=market_data.index,
                    y=market_data['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Overbought (70)", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Oversold (30)", row=3, col=1)
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} Comprehensive Analysis",
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                template="plotly_white"
            )
            
            # Update x-axis labels
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating chart: {str(e)}")
    
    def create_portfolio_overview_chart(self, symbols: List[str]) -> go.Figure:
        """Create portfolio overview visualization"""
        try:
            # Get data for all symbols
            portfolio_data = []
            
            for symbol in symbols:
                try:
                    data = self.polygon_client.get_recent_data(symbol, days=30)
                    if data is not None and not data.empty:
                        latest = data.iloc[-1]
                        first = data.iloc[0]
                        
                        price_change_pct = ((latest['close'] - first['close']) / first['close']) * 100
                        
                        portfolio_data.append({
                            'Symbol': symbol,
                            'Current_Price': latest['close'],
                            'Price_Change_Pct': price_change_pct,
                            'Volume': latest['volume']
                        })
                except Exception as e:
                    print(f"Error getting data for {symbol}: {e}")
                    continue
            
            if not portfolio_data:
                return self._create_error_chart("No portfolio data available")
            
            df = pd.DataFrame(portfolio_data)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Price Change Distribution",
                    "Current Prices",
                    "Volume Distribution",
                    "Portfolio Performance"
                ],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Price change distribution
            fig.add_trace(
                go.Bar(
                    x=df['Symbol'],
                    y=df['Price_Change_Pct'],
                    name='Price Change %',
                    marker_color=['green' if x > 0 else 'red' for x in df['Price_Change_Pct']]
                ),
                row=1, col=1
            )
            
            # Current prices
            fig.add_trace(
                go.Bar(
                    x=df['Symbol'],
                    y=df['Current_Price'],
                    name='Current Price',
                    marker_color='blue'
                ),
                row=1, col=2
            )
            
            # Volume distribution
            fig.add_trace(
                go.Bar(
                    x=df['Symbol'],
                    y=df['Volume'],
                    name='Volume',
                    marker_color='orange'
                ),
                row=2, col=1
            )
            
            # Portfolio performance scatter
            fig.add_trace(
                go.Scatter(
                    x=df['Current_Price'],
                    y=df['Price_Change_Pct'],
                    mode='markers+text',
                    text=df['Symbol'],
                    textposition="top center",
                    name='Performance',
                    marker=dict(
                        size=10,
                        color=df['Price_Change_Pct'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Price Change %")
                    )
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Portfolio Overview Dashboard",
                height=800,
                showlegend=False,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating portfolio chart: {str(e)}")
    
    def create_sentiment_analysis_chart(self, analysis_data: List[Dict]) -> go.Figure:
        """Create sentiment analysis visualization"""
        try:
            if not analysis_data:
                return self._create_error_chart("No sentiment data available")
            
            # Extract sentiment data
            symbols = []
            sentiments = []
            confidence_scores = []
            recommendations = []
            
            for data in analysis_data:
                symbols.append(data.get('symbol', 'Unknown'))
                ai_analysis = data.get('ai_analysis', {})
                sentiments.append(ai_analysis.get('overall_sentiment', 'Neutral'))
                confidence_scores.append(ai_analysis.get('confidence', 5))
                recommendations.append(ai_analysis.get('recommendation', 'HOLD'))
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Sentiment Distribution",
                    "Confidence Scores",
                    "Recommendation Distribution",
                    "Sentiment vs Confidence"
                ],
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "pie"}, {"type": "scatter"}]]
            )
            
            # Sentiment distribution pie chart
            sentiment_counts = pd.Series(sentiments).value_counts()
            fig.add_trace(
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    name="Sentiment"
                ),
                row=1, col=1
            )
            
            # Confidence scores bar chart
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=confidence_scores,
                    name='Confidence',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
            
            # Recommendation distribution pie chart
            rec_counts = pd.Series(recommendations).value_counts()
            fig.add_trace(
                go.Pie(
                    labels=rec_counts.index,
                    values=rec_counts.values,
                    name="Recommendations"
                ),
                row=2, col=1
            )
            
            # Sentiment vs Confidence scatter
            sentiment_numeric = [1 if s == 'Bullish' else -1 if s == 'Bearish' else 0 for s in sentiments]
            fig.add_trace(
                go.Scatter(
                    x=sentiment_numeric,
                    y=confidence_scores,
                    mode='markers+text',
                    text=symbols,
                    textposition="top center",
                    name='Sentiment vs Confidence',
                    marker=dict(
                        size=10,
                        color=confidence_scores,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Confidence")
                    )
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="AI Sentiment Analysis Dashboard",
                height=800,
                showlegend=False,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating sentiment chart: {str(e)}")
    
    def create_document_insights_chart(self, document_insights: List[Dict]) -> go.Figure:
        """Create document insights visualization"""
        try:
            if not document_insights:
                return self._create_error_chart("No document insights available")
            
            # Extract document data
            symbols = []
            bullish_counts = []
            bearish_counts = []
            sentiment_scores = []
            confidence_scores = []
            
            for insight in document_insights:
                symbols.append(insight.get('symbol', 'Unknown'))
                signals = insight.get('signals', {})
                if signals.get('success'):
                    bullish_counts.append(len(signals.get('bullish_signals', [])))
                    bearish_counts.append(len(signals.get('bearish_signals', [])))
                    sentiment_scores.append(signals.get('sentiment_score', 0))
                    confidence_scores.append(signals.get('confidence', 5))
                else:
                    bullish_counts.append(0)
                    bearish_counts.append(0)
                    sentiment_scores.append(0)
                    confidence_scores.append(5)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Bullish vs Bearish Signals",
                    "Sentiment Scores",
                    "Confidence Distribution",
                    "Signal Strength Analysis"
                ]
            )
            
            # Bullish vs Bearish signals
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=bullish_counts,
                    name='Bullish Signals',
                    marker_color='green'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=bearish_counts,
                    name='Bearish Signals',
                    marker_color='red'
                ),
                row=1, col=1
            )
            
            # Sentiment scores
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=sentiment_scores,
                    name='Sentiment Score',
                    marker_color=['green' if x > 0 else 'red' if x < 0 else 'gray' for x in sentiment_scores]
                ),
                row=1, col=2
            )
            
            # Confidence distribution
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=confidence_scores,
                    name='Confidence',
                    marker_color='blue'
                ),
                row=2, col=1
            )
            
            # Signal strength analysis
            total_signals = [b + bear for b, bear in zip(bullish_counts, bearish_counts)]
            fig.add_trace(
                go.Scatter(
                    x=total_signals,
                    y=confidence_scores,
                    mode='markers+text',
                    text=symbols,
                    textposition="top center",
                    name='Signal Strength',
                    marker=dict(
                        size=10,
                        color=sentiment_scores,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Sentiment Score")
                    )
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Document Insights Analysis",
                height=800,
                showlegend=True,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            return self._create_error_chart(f"Error creating document insights chart: {str(e)}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart when data is unavailable"""
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    def close(self):
        """Close connections"""
        self.ai_analyzer.close()
