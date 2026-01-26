import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import io
from uuid import uuid4
from groq import Groq

# Load environment variables early
load_dotenv()

from phi.agent.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch

# Phase 1 imports
from tradingagents.database.config import get_supabase
from tradingagents.dataflows.ingestion_pipeline import DataIngestionPipeline, convert_instrument_to_polygon_symbol
from tradingagents.dataflows.universal_ingestion import ingest_market_data, ingest_from_polygon_api
from tradingagents.dataflows.signal_provider_ingestion import ingest_signal_provider_data, validate_signal_provider_data
from tradingagents.dataflows.kpi_calculator import calculate_kpi
from tradingagents.dataflows.kpi_calculator import calculate_kpi
from tradingagents.dataflows.document_manager import DocumentManager
from tradingagents.dataflows.signal_analyzer import SignalAnalyzer
from tradingagents.dataflows.signal_backtester import SignalBacktester
from tradingagents.dataflows.validation_engine import ValidationEngine
from tradingagents.dataflows.daily_reporter import DailyReporter
from tradingagents.dataflows.run_signal_analysis import (
    run_analysis_for_all_signals,
    run_analysis_for_all_providers_and_instruments,
    run_backtest_for_all_signals,
    calculate_provider_metrics,
    get_available_instruments,
    get_available_providers,
    check_market_data_availability
)
from tradingagents.dataflows.trade_efficiency import TradeEfficiencyAnalyzer
from tradingagents.database.db_service import get_backtest_results_with_efficiency, update_backtest_efficiency
from tradingagents.config.watchlist import WATCHLIST_STOCKS, get_watchlist_symbols
from tradingagents.dataflows.polygon_integration import PolygonDataClient
from tradingagents.dataflows.market_data_service import (
    fetch_ohlcv,
    get_latest_price,
    period_to_days,
    fetch_latest_bar
)
from tradingagents.dataflows.data_guardrails import DataCoverageService
from tradingagents.dataflows.feature_lab import FeatureLab

# Phase 2 imports
from tradingagents.dataflows.ai_analysis import AIResearchAnalyzer

# Phase 2 raw data / non-financial tools (vendor-routed)
# For the app UI we call the routing layer directly (not the LangChain tool wrappers)
from tradingagents.dataflows.interface import route_to_vendor

# Analyst agents imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tradingagents.agents import (
    create_market_analyst,
    create_news_analyst,
    create_fundamentals_analyst,
    create_social_media_analyst
)

# Phase 3 imports
from tradingagents.agents.utils.trading_engine import (
    VolumeScreeningEngine,
    FireTestingEngine,
    AIEnhancedScoringEngine,
    TradeDecisionEngine
)
from tradingagents.agents.utils.historical_datascan import HistoricalDataScanEngine

# Phase 4 imports
from tradingagents.agents.utils.session_manager import (
    TradingSessionManager,
    TradeExecutionService
)

# Load GROQ_API_KEY from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "") or os.getenv("GROK_API_KEY", "") 

COMMON_STOCKS = {
    'NVIDIA': 'NVDA', 'APPLE': 'AAPL', 'GOOGLE': 'GOOGL', 'MICROSOFT': 'MSFT',
    'TESLA': 'TSLA', 'AMAZON': 'AMZN', 'META': 'META', 'NETFLIX': 'NFLX',
    'TCS': 'TCS.NS', 'RELIANCE': 'RELIANCE.NS', 'INFOSYS': 'INFY.NS',
    'WIPRO': 'WIPRO.NS', 'HDFC': 'HDFCBANK.NS', 'TATAMOTORS': 'TATAMOTORS.NS',
    'ICICIBANK': 'ICICIBANK.NS', 'SBIN': 'SBIN.NS'
}

st.set_page_config(page_title="Stocks Analysis AI Agents", page_icon="", layout="wide")

st.markdown("""
    <style>
    /* Main Layout */
    .main { 
        padding: 2rem; 
        background: #ffffff;
        color: #000000;
        min-height: 100vh;
    }
    
    .stApp { 
        max-width: 1400px; 
        margin: 0 auto; 
        background: transparent;
    }
    
    /* Phase Navigation */
    .phase-nav {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #f6f8fa 0%, #ffffff 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 20px 20px 0 0;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin: 2rem 0 1.5rem 0;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
        border: 2px solid rgba(255, 255, 255, 0.1);
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        letter-spacing: 0.5px;
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        color: white !important;
    }
    
    /* Small download buttons */
    button[data-testid*="baseButton-secondary"] {
        padding: 0.2rem 0.4rem !important;
        font-size: 0.75rem !important;
        min-width: 2rem !important;
        height: 1.8rem !important;
        line-height: 1 !important;
    }
    
    /* Specifically target download buttons */
    .stDownloadButton > button {
        padding: 0.2rem 0.4rem !important;
        font-size: 0.75rem !important;
        min-width: 2rem !important;
        height: 1.8rem !important;
        line-height: 1 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 15px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #667eea;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* DataFrames */
    .dataframe {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .stError {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
    }
    
    .stInfo {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
    }
    
    /* Phase Selector */
    .phase-selector {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .phase-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        cursor: pointer;
    }
    
    .phase-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Active Phase Display */
    .active-phase {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #495057;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 1rem 0;
        border: 2px solid #dee2e6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        letter-spacing: 0.3px;
    }
    
    /* Phase Title Display */
    .phase-title {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: #212529;
        padding: 1.2rem 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1.5rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        letter-spacing: 0.5px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main { padding: 1rem; }
        .feature-card { padding: 1rem; }
        .metric-value { font-size: 1.5rem; }
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_agents():
    # Agents disabled for Groq-only setup; keep UI functional without xAI
    if not st.session_state.get('agents_initialized', False):
        st.session_state.agents_initialized = True
        return True


def _summarize_with_groq(raw_text: str, title: str) -> str:
    """
    Use Groq LLM to turn raw vendor output (JSON/text) into a human-readable summary.
    Falls back to the original text if Groq is not configured or fails.
    """
    if not raw_text:
        return "No data available to summarize."

    groq_key = os.getenv("GROQ_API_KEY", "") or os.getenv("GROK_API_KEY", "")
    if not groq_key or not groq_key.startswith("gsk_"):
        # Groq not configured properly – return truncated raw text
        return raw_text if len(raw_text) < 4000 else raw_text[:4000] + "\n\n...[truncated]..."

    try:
        client = Groq(api_key=groq_key)
        prompt = f"""
You are an expert macro and equity analyst.

TITLE: {title}

Below is raw vendor output (it may be JSON, CSV-like text, or markdown) containing news or fundamentals.
Your job:
- Extract the key points.
- Group them into 3–7 concise bullet points.
- Call out overall sentiment (Bullish / Bearish / Neutral) and why.
- Mention any important risks, catalysts, or macro context.
- If the text is company-specific, clearly state the company/ticker in your answer.

RAW DATA (do NOT echo verbatim, just use it as source material):
{raw_text[:6000]}
"""
        chat = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You turn messy raw data into clean, trader-friendly summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = chat.choices[0].message.content if chat.choices else ""
        return content.strip() if content else raw_text
    except Exception:
        # On any Groq/API error, fall back to raw text
        return raw_text if len(raw_text) < 4000 else raw_text[:4000] + "\n\n...[truncated]..."

def _get_llm_for_analyst():
    """Get LLM instance for analyst agents (supports Groq via OpenAI-compatible API)"""
    groq_key = os.getenv("GROQ_API_KEY", "") or os.getenv("GROK_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    
    # Prefer Groq if available (via OpenAI-compatible endpoint)
    if groq_key and groq_key.startswith("gsk_"):
        return ChatOpenAI(
            model="llama-3.1-8b-instant",
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.2
        )
    elif openai_key:
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=0.2
        )
    else:
        return None

def _run_analyst_report(analyst_type: str, symbol: str) -> dict:
    """
    Run an analyst agent and return the report.
    
    Args:
        analyst_type: One of "market", "news", "fundamentals", "social_media"
        symbol: Stock ticker symbol
    
    Returns:
        dict with "success", "report", and "error" keys
    """
    llm = _get_llm_for_analyst()
    if not llm:
        return {
            "success": False,
            "report": None,
            "error": "No LLM configured. Please set GROQ_API_KEY or OPENAI_API_KEY in your .env file."
        }
    
    try:
        # Create the appropriate analyst
        if analyst_type == "market":
            analyst_node = create_market_analyst(llm)
        elif analyst_type == "news":
            analyst_node = create_news_analyst(llm)
        elif analyst_type == "fundamentals":
            analyst_node = create_fundamentals_analyst(llm)
        elif analyst_type == "social_media":
            analyst_node = create_social_media_analyst(llm)
        else:
            return {
                "success": False,
                "report": None,
                "error": f"Unknown analyst type: {analyst_type}"
            }
        
        # Prepare state for the analyst
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state = {
            "trade_date": current_date,
            "company_of_interest": symbol.upper(),
            "messages": [HumanMessage(content=f"Analyze {symbol.upper()} and provide a comprehensive report.")]
        }
        
        # Run the analyst
        result = analyst_node(state)
        
        # Extract report based on analyst type
        report_key = {
            "market": "market_report",
            "news": "news_report",
            "fundamentals": "fundamentals_report",
            "social_media": "sentiment_report"
        }.get(analyst_type, "report")
        
        report = result.get(report_key, "")
        
        # If no report in result, try to get from messages
        if not report and result.get("messages"):
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                report = last_message.content
        
        return {
            "success": True,
            "report": report or "Report generated but content is empty.",
            "error": None
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "report": None,
            "error": f"Error running {analyst_type} analyst: {str(e)}\n\n{traceback.format_exc()}"
        }

def get_stock_data(symbol):
    try:
        hist = fetch_ohlcv(symbol, interval="1d", lookback_days=365)
        if hist is None or hist.empty:
            st.error(f"No stored historical data for {symbol}. Please run the ingestion pipeline.")
            return None, None
        
        latest_close = float(hist["Close"].iloc[-1])
        first_close = float(hist["Close"].iloc[0])
        price_change_pct = ((latest_close - first_close) / first_close) * 100 if first_close else 0.0
        avg_volume = float(hist["Volume"].tail(20).mean()) if "Volume" in hist.columns else 0.0
        
        info = {
            "symbol": symbol.upper(),
            "currentPrice": latest_close,
            "priceChangePct": price_change_pct,
            "recommendationKey": "Neutral",
            "forwardPE": None,
            "averageVolume": avg_volume,
            "fiftyTwoWeekHigh": float(hist["Close"].max()),
            "fiftyTwoWeekLow": float(hist["Close"].min()),
        }
        return info, hist
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None

def create_price_chart(hist_data, symbol, signals=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index, open=hist_data['Open'],
        high=hist_data['High'], low=hist_data['Low'],
        close=hist_data['Close'], name='OHLC'
    ))
    
    if signals:
        # Separate buy and sell signals
        buy_signals = []
        sell_signals = []
        
        for s in signals:
            try:
                # Ensure date format compatibility
                s_date = pd.to_datetime(s.get('signal_date'))
                # If timezone aware, convert to naive or match hist_data
                if s_date.tzinfo is not None:
                    s_date = s_date.tz_localize(None)
                
                price = s.get('entry_price')
                if price is None:
                    continue
                    
                if s.get('action', '').lower() == 'buy':
                    buy_signals.append((s_date, price))
                elif s.get('action', '').lower() == 'sell':
                    sell_signals.append((s_date, price))
            except Exception:
                continue
        
        if buy_signals:
            fig.add_trace(go.Scatter(
                x=[x[0] for x in buy_signals],
                y=[x[1] for x in buy_signals],
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1, color='darkgreen'))
            ))
            
        if sell_signals:
            fig.add_trace(go.Scatter(
                x=[x[0] for x in sell_signals],
                y=[x[1] for x in sell_signals],
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='darkred'))
            ))

    fig.update_layout(
        title=f'{symbol} Price Movement',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=500
    )
    return fig

def initialize_database():
    """Initialize database tables"""
    try:
        supabase = get_supabase()
        if supabase:
            st.success("Supabase connection verified!")
            return True
        else:
            st.error("Supabase not configured. Please set SUPABASE_URL and SUPABASE_KEY in your .env file.")
            return False
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        return False

def test_polygon_connection():
    """Test Polygon API connection"""
    try:
        from tradingagents.dataflows.polygon_integration import PolygonDataClient
        client = PolygonDataClient()
        
        # Test with a simple stock
        data = client.get_historical_data("AAPL", "2024-01-01", "2024-01-05")
        
        if data.empty:
            st.error("❌ Polygon API: No data returned (check API key)")
            return False
        else:
            st.success(f"✅ Polygon API: Working! Got {len(data)} records for AAPL")
            return True
            
    except Exception as e:
        st.error(f"❌ Polygon API Error: {str(e)}")
        return False

def test_database_connection():
    """Test database connection"""
    try:
        from tradingagents.database.config import get_supabase
        sb = get_supabase()
        
        if not sb:
            st.error("❌ Database: Not configured (check SUPABASE_URL and SUPABASE_KEY)")
            return False
        
        # Test basic connection
        try:
            # Try to access market_data table first
            result = sb.table("market_data").select("id").limit(1).execute()
            st.success("✅ Database: Connected successfully")
            st.info(f"📊 Found {len(result.data)} records in market_data table")
            return True
        except Exception as table_error:
            # If market_data fails, try research_documents table to verify connection
            try:
                result = sb.table("research_documents").select("id").limit(1).execute()
                st.warning("⚠️ Database connected, but market_data table has permission issues")
                st.info(f"📊 Found {len(result.data)} records in research_documents table")
                st.info("💡 You need to disable RLS or create a policy for market_data table")
                return False
            except Exception as doc_error:
                st.error(f"❌ Both tables failed: market_data={str(table_error)}, research_documents={str(doc_error)}")
            
            # Try to get table list
            try:
                # This might work to list tables
                st.info("🔍 Attempting to diagnose table issue...")
                st.code(f"Error details: {str(table_error)}")
                
                # Check if it's a permissions issue
                if "permission" in str(table_error).lower():
                    st.warning("⚠️ This might be a permissions issue. Check your Supabase RLS policies.")
                elif "does not exist" in str(table_error).lower():
                    st.warning("⚠️ Table doesn't exist. Make sure you created 'market_data' table (not 'market-data' or 'marketdata')")
                
            except Exception as diag_error:
                st.error(f"❌ Diagnosis failed: {str(diag_error)}")
            
            return False
            
    except Exception as e:
        st.error(f"❌ Database Connection Error: {str(e)}")
        return False

def test_historical_ingestion_single():
    """Test historical data ingestion for a single stock"""
    try:
        from tradingagents.dataflows.ingestion_pipeline import DataIngestionPipeline
        
        st.info("🧪 Testing single stock ingestion...")
        pipeline = DataIngestionPipeline()
        
        # Test with a stock that's likely not in the database yet
        test_symbols = ["MSFT", "GOOGL", "AMZN"]  # Try different stocks
        
        for symbol in test_symbols:
            st.info(f"Testing with {symbol}...")
            result = pipeline.ingest_historical_data(symbol, days_back=7)  # Just 7 days to avoid duplicates
            
            if result:
                st.success(f"✅ Single stock ingestion test PASSED with {symbol}")
                pipeline.close()
                return True
            else:
                st.warning(f"⚠️ Failed with {symbol}, trying next...")
        
        pipeline.close()
        st.error("❌ Single stock ingestion test FAILED with all test symbols")
        st.info("💡 This might be due to:")
        st.info("- Database insertion error")
        st.info("- Polygon API returning no data")
        st.info("- All test symbols already exist in database")
        return False
            
    except Exception as e:
        st.error(f"❌ Single stock test error: {str(e)}")
        st.info("💡 Error details:")
        st.code(str(e))
        return False

def test_polygon_api_detailed():
    """Test Polygon API with detailed error reporting"""
    try:
        from tradingagents.dataflows.polygon_integration import PolygonDataClient
        client = PolygonDataClient()
        
        st.info("🧪 Testing Polygon API with detailed diagnostics...")
        
        # Test 1: Check API key
        if not client.api_key:
            st.error("❌ POLYGON_API_KEY not found in environment variables")
            return False
        else:
            st.success(f"✅ API Key found: {client.api_key[:8]}...")
        
        # Test 2: Try to get data for AAPL
        st.info("📊 Testing data fetch for AAPL...")
        data = client.get_historical_data("AAPL", "2024-01-01", "2024-01-05")
        
        if data.empty:
            st.error("❌ No data returned from Polygon API")
            st.info("💡 Possible causes:")
            st.info("- Invalid API key")
            st.info("- API rate limit exceeded")
            st.info("- Network connectivity issues")
            return False
        else:
            st.success(f"✅ Polygon API working! Got {len(data)} records")
            st.dataframe(data.head())
            return True
            
    except Exception as e:
        st.error(f"❌ Polygon API Error: {str(e)}")
        return False

def check_environment_variables():
    """Check environment variables configuration"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    st.info("🔍 Checking environment variables...")
    # Support GROK_API_KEY alias -> GROQ_API_KEY for AI features
    groq_key = os.getenv("GROQ_API_KEY")
    grok_key = os.getenv("GROK_API_KEY")
    if not groq_key and grok_key:
        os.environ["GROQ_API_KEY"] = grok_key
        groq_key = grok_key
    
    env_vars = {
        "POLYGON_API_KEY": os.getenv("POLYGON_API_KEY"),
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
        "GROQ_API_KEY": groq_key or ""
    }
    
    all_good = True
    for var_name, var_value in env_vars.items():
        if var_value:
            st.success(f"✅ {var_name}: Found")
        else:
            st.error(f"❌ {var_name}: Missing")
            all_good = False
    
    if all_good:
        st.success("🎉 All environment variables are configured!")
    else:
        st.error("⚠️ Some environment variables are missing. Please check your .env file.")
        st.info("💡 Create a .env file in your project root with:")
        st.code("""
POLYGON_API_KEY=your_polygon_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
GROQ_API_KEY=your_groq_or_grok_api_key_here  # GROK_API_KEY also supported
        """)
    
    return all_good

def run_full_diagnostic():
    """Run comprehensive diagnostic tests"""
    st.info("🔬 Running full diagnostic...")
    
    results = {}
    
    # Test 1: Environment variables
    st.subheader("1. Environment Variables")
    results["env_vars"] = check_environment_variables()
    
    # Test 2: Database connection
    st.subheader("2. Database Connection")
    results["database"] = test_database_connection()
    
    # Test 3: Polygon API
    st.subheader("3. Polygon API")
    results["polygon"] = test_polygon_api_detailed()
    
    # Test 4: Single ingestion
    if results["database"] and results["polygon"]:
        st.subheader("4. Single Stock Ingestion")
        results["single_ingestion"] = test_historical_ingestion_single()
    else:
        st.warning("⚠️ Skipping single ingestion test due to previous failures")
        results["single_ingestion"] = False
    
    # Summary
    st.subheader("📊 Diagnostic Summary")
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        st.success(f"🎉 All tests passed! ({passed}/{total})")
        st.info("✅ Your system is ready for historical data ingestion!")
    else:
        st.error(f"❌ {total - passed} test(s) failed ({passed}/{total})")
        st.info("🔧 Please fix the failed tests before proceeding with data ingestion.")
    
    return results

# Removed create_database_tables function as requested

def list_database_tables():
    """List all tables in the database"""
    try:
        from tradingagents.database.config import get_supabase
        sb = get_supabase()
        
        if not sb:
            st.error("❌ Database: Not configured")
            return
        
        # Try to query information_schema to list tables
        try:
            # This query should work to list tables
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
            """
            
            result = sb.rpc('exec_sql', {'sql': query}).execute()
            if result.data:
                st.success("✅ Database Tables Found:")
                for table in result.data:
                    st.write(f"📋 {table['table_name']}")
            else:
                st.warning("⚠️ No tables found in public schema")
                
        except Exception as e:
            st.warning(f"⚠️ Could not list tables automatically: {str(e)}")
            st.info("💡 Please check your Supabase dashboard → Table Editor to see what tables exist")
            
            # Try a different approach - test common table names
            common_tables = ['market_data_stocks_1min', 'market_data_commodities_1min', 'market_data_indices_1min', 'market_data_currencies_1min', 'research_documents', 'users', 'positions', 'instrument_master_data', 'portfolio', 'system_logs', 'trade_signals']
            st.info("🔍 Testing common table names:")
            
            for table_name in common_tables:
                try:
                    result = sb.table(table_name).select("id").limit(1).execute()
                    st.success(f"✅ {table_name} - EXISTS")
                except:
                    st.error(f"❌ {table_name} - NOT FOUND")
                    
    except Exception as e:
        st.error(f"❌ Error listing tables: {str(e)}")

def phase1_foundation_data():
    """Phase 1: Foundation & Data Infrastructure"""
    
    
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Selection")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sb = get_supabase()
        def _read_df(file):
            file.seek(0)
            name = file.name.lower()
            df = None
            try:
                if name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    # Try Excel first
                    try:
                        df = pd.read_excel(file)
                    except Exception as e:
                        # Fallback to CSV
                        file.seek(0)
                        try:
                            df = pd.read_csv(file)
                        except:
                            raise e
            except Exception as e:
                st.error(f"Error reading file {name}: {str(e)}")
                file.seek(0)
                return None
            
            file.seek(0)
            return df
        # 1. Ensure Catalog is seeded if empty
        if sb:
            try:
                res = sb.table("instrument_catalog").select("symbol", count="exact", head=True).execute()
                if res.count == 0:
                     default_map = {
                        "Commodities": ["GOLD"],
                        "Indices": ["S&P 500"],
                        "Currencies": ["EUR/USD"],
                        "Stocks": list(WATCHLIST_STOCKS.keys())
                     }
                     seed_rows = []
                     for cat, items in default_map.items():
                        for itm in items:
                            seed_rows.append({
                                "symbol": str(itm).upper(),
                                "name": WATCHLIST_STOCKS.get(itm, itm),
                                "category": cat,
                                "sector": "",
                                "exchange": "",
                                "source": "default_seed",
                                "created_at": datetime.utcnow().isoformat(),
                                "updated_at": datetime.utcnow().isoformat()
                            })
                     if seed_rows:
                        sb.table("instrument_catalog").upsert(seed_rows).execute()
            except Exception:
                pass

        # 2. Fetch instruments from catalog to populate UI
        catalog_data = []
        if sb:
            try:
                res = sb.table("instrument_catalog").select("symbol,category").execute()
                catalog_data = res.data
            except Exception:
                pass
        
        # Build dynamic map
        instruments_map = {}
        # Ensure base categories exist
        for base_cat in ["Commodities", "Indices", "Currencies", "Stocks"]:
            instruments_map[base_cat] = []

        for item in catalog_data:
            c = item.get("category", "Other")
            s = item.get("symbol")
            if c not in instruments_map:
                instruments_map[c] = []
            if s and s not in instruments_map[c]:
                instruments_map[c].append(s)
                
        # Sort and add "Add..."
        for c in instruments_map:
            instruments_map[c].sort()
            if "Add..." not in instruments_map[c]:
                instruments_map[c].append("Add...")

        categories = sorted(list(instruments_map.keys()))
        if "Add..." not in categories:
            categories.append("Add...")

        selected_category = st.selectbox(
            "Financial Instrument Category",
            options=categories,
            index=categories.index("Commodities") if "Commodities" in categories else 0,
            key="select_financial_category"
        )

        if selected_category == "Add...":
            with st.expander("Add New Instrument", expanded=False):
                with st.form("add_new_instrument_main"):
                    st.write("Add a new instrument to the catalog.")
                    i_symbol = st.text_input("Symbol", help="e.g. AAPL").upper().strip()
                    i_name = st.text_input("Name", help="e.g. Apple Inc.")
                    
                    cat_opts = ["Commodities", "Indices", "Currencies", "Stocks", "Other"]
                    i_cat_select = st.selectbox("Category", cat_opts)
                    i_cat = i_cat_select
                    if i_cat_select == "Other":
                        i_cat = st.text_input("Custom Category Name")
                    
                    i_sector = st.text_input("Sector (Optional)")
                    i_exchange = st.text_input("Exchange (Optional)")
                    
                    if st.form_submit_button("Save to Catalog"):
                        if i_symbol and i_name and i_cat:
                            if not sb:
                                st.error("Database not configured")
                            else:
                                row = {
                                    "symbol": i_symbol,
                                    "name": i_name,
                                    "category": i_cat,
                                    "sector": i_sector,
                                    "exchange": i_exchange,
                                    "source": "manual_input",
                                    "created_at": datetime.utcnow().isoformat(),
                                    "updated_at": datetime.utcnow().isoformat()
                                }
                                try:
                                    sb.table("instrument_catalog").upsert([row]).execute()
                                    st.success(f"Saved {i_symbol}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.error("Symbol, Name, and Category are required.")

        current_instruments = instruments_map.get(selected_category, ["Add..."])
        
        # Default to GOLD if Commodities is selected, otherwise first item
        default_instrument_index = 0
        if selected_category == "Commodities" and "GOLD" in current_instruments:
            default_instrument_index = current_instruments.index("GOLD")
        
        selected_instrument_item = st.selectbox(
            "Financial Instrument",
            options=current_instruments,
            index=default_instrument_index,
            key="select_financial_instrument_item"
        )
        
        if selected_instrument_item == "Add...":
            with st.expander(f"Add Instrument to {selected_category}", expanded=False):
                with st.form("add_instrument_sub"):
                    st.write(f"Add to {selected_category}")
                    s_symbol = st.text_input("Symbol", key="sub_sym").upper().strip()
                    s_name = st.text_input("Name", key="sub_name")
                    s_sector = st.text_input("Sector (Optional)", key="sub_sec")
                    s_exchange = st.text_input("Exchange (Optional)", key="sub_exch")
                    
                    if st.form_submit_button("Save Instrument"):
                        if s_symbol and s_name:
                            if not sb:
                                st.error("Database not configured")
                            else:
                                row = {
                                    "symbol": s_symbol,
                                    "name": s_name,
                                    "category": selected_category,
                                    "sector": s_sector,
                                    "exchange": s_exchange,
                                    "source": "manual_input",
                                    "created_at": datetime.utcnow().isoformat(),
                                    "updated_at": datetime.utcnow().isoformat()
                                }
                                try:
                                    sb.table("instrument_catalog").upsert([row]).execute()
                                    st.success(f"Saved {s_symbol}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.error("Missing symbol or name")
        
        # Market Data Completeness Check Section
        if selected_instrument_item != "Add...":
            st.markdown("---")
            with st.expander("🔍 Coverage Audit & Auto Backfill", expanded=False):
                from tradingagents.dataflows.ingestion_pipeline import get_1min_table_name_for_symbol, DataIngestionPipeline
                from tradingagents.dataflows.data_guardrails import DataCoverageService
                
                # Determine the symbol to check (may need conversion)
                check_symbol = selected_instrument_item
                if selected_instrument_item == "GOLD":
                    check_symbol = "C:XAUUSD"
                elif selected_instrument_item == "S&P 500":
                    check_symbol = "^SPX"
                
                # Try to find the symbol in the database (may be stored with different format)
                sb = get_supabase()
                symbol_to_check = None
                
                if sb:
                    # Get table name
                    table_name = get_1min_table_name_for_symbol(check_symbol)
                    
                    # Try multiple symbol formats to find the actual symbol in database
                    symbol_variants = [check_symbol]
                    if selected_instrument_item == "GOLD":
                        symbol_variants = ["C:XAUUSD", "^XAUUSD", "GOLD", "XAUUSD"]
                    elif selected_instrument_item == "S&P 500":
                        symbol_variants = ["^SPX", "SPX", "I:SPX", "S&P 500", "SPY"]  # SPY is used for minute data
                    
                    # Try to find actual symbol in database
                    try:
                        # Get all distinct symbols from the table
                        result = sb.table(table_name).select("symbol").execute()
                        if result.data:
                            db_symbols = list(set([row["symbol"] for row in result.data]))
                            # Try to match our symbol variants
                            for variant in symbol_variants:
                                for db_sym in db_symbols:
                                    if variant.upper() == db_sym.upper() or \
                                       variant.upper() in db_sym.upper() or \
                                       db_sym.upper() in variant.upper():
                                        symbol_to_check = db_sym
                                        break
                                if symbol_to_check:
                                    break
                        
                        # If no match found, use the first variant
                        if not symbol_to_check:
                            symbol_to_check = symbol_variants[0]
                        
                        if symbol_to_check:
                            # Determine start date based on asset class for display
                            if selected_category == "Indices":
                                start_date_display = "Jan 13, 2025"
                            elif selected_category == "Stocks":
                                start_date_display = "Jan 1, 2024"
                            elif selected_category in ["Commodities", "Currencies"]:
                                start_date_display = "Jan 10, 2024"
                            else:
                                start_date_display = "Jan 1, 2024"
                            
                            # Coverage audit description
                            st.info(f"""
                            **Coverage Audit** checks if your 1-minute data meets the required historical range:
                            - **1-minute**: {start_date_display} → current time minus 15 minutes
                            
                            **Auto Backfill** will automatically ingest missing data to fill gaps.
                            """)
                            
                            # Coverage audit and backfill buttons
                            audit_col1, audit_col2 = st.columns(2)
                            
                            with audit_col1:
                                if st.button("🔍 Run Coverage Audit", key=f"coverage_audit_{selected_category}_{selected_instrument_item}"):
                                    with st.spinner(f"Checking coverage for {symbol_to_check}..."):
                                        try:
                                            coverage_service = DataCoverageService(asset_class=selected_category)
                                            report_records = coverage_service.build_symbol_report(symbol_to_check)
                                            
                                            # Store in session state with unique key
                                            session_key = f"coverage_records_{selected_category}_{selected_instrument_item}"
                                            st.session_state[session_key] = report_records
                                            st.session_state[f"coverage_rows_{selected_category}_{selected_instrument_item}"] = [rec.to_dict() for rec in report_records]
                                            st.success("✅ Coverage audit completed")
                                        except Exception as e:
                                            st.error(f"❌ Coverage audit failed: {str(e)}")
                            
                            with audit_col2:
                                session_key = f"coverage_records_{selected_category}_{selected_instrument_item}"
                                records = st.session_state.get(session_key) or []
                                missing_records = [rec for rec in records if rec.needs_backfill()]
                                
                                if st.button("🛠️ Auto Backfill Missing Data", key=f"coverage_backfill_{selected_category}_{selected_instrument_item}"):
                                    if not records:
                                        st.warning("⚠️ Run the coverage audit first to identify gaps.")
                                    elif not missing_records:
                                        st.success("✅ No gaps detected. Data coverage is complete.")
                                    else:
                                        # Check API key before attempting backfill
                                        import os
                                        polygon_key = os.getenv("POLYGON_API_KEY")
                                        if not polygon_key:
                                            st.error("❌ POLYGON_API_KEY not found in environment variables. Cannot backfill data.")
                                            st.info("💡 Please add POLYGON_API_KEY to your .env file. Get a key from: https://polygon.io/dashboard/api-keys")
                                        else:
                                            with st.spinner("Running targeted ingestion jobs..."):
                                                try:
                                                    pipeline = DataIngestionPipeline()
                                                    coverage_service = DataCoverageService()
                                                    logs = coverage_service.backfill_missing(pipeline, missing_records)
                                                    pipeline.close()
                                                    
                                                    log_key = f"coverage_backfill_logs_{selected_category}_{selected_instrument_item}"
                                                    st.session_state[log_key] = logs
                                                    
                                                    # Check for 401 errors in logs
                                                    has_401_error = False
                                                    for log in logs:
                                                        if isinstance(log, dict) and "error" in str(log.get("message", "")).lower():
                                                            if "401" in str(log.get("message", "")) or "unauthorized" in str(log.get("message", "")).lower():
                                                                has_401_error = True
                                                                break
                                                    
                                                    if has_401_error:
                                                        st.error("❌ Backfill failed: Polygon API authentication error (401 Unauthorized)")
                                                        st.warning("💡 Your POLYGON_API_KEY appears to be invalid or expired. Please check your API key.")
                                                    else:
                                                        success_count = sum(1 for log in logs if log.get("success", False))
                                                        total_count = len(logs)
                                                        if success_count > 0:
                                                            st.success(f"✅ Backfill completed: {success_count}/{total_count} tasks succeeded")
                                                        else:
                                                            st.warning(f"⚠️ Backfill completed but no tasks succeeded. Check logs below for details.")
                                                except ValueError as e:
                                                    error_msg = str(e)
                                                    if "401" in error_msg or "unauthorized" in error_msg.lower():
                                                        st.error("❌ Polygon API Authentication Failed")
                                                        st.warning("💡 Your POLYGON_API_KEY is invalid or expired.")
                                                    else:
                                                        st.error(f"❌ Backfill error: {error_msg}")
                                                except Exception as e:
                                                    error_msg = str(e)
                                                    if "401" in error_msg or "unauthorized" in error_msg.lower():
                                                        st.error("❌ Polygon API Authentication Failed (401 Unauthorized)")
                                                        st.warning("💡 Please check your POLYGON_API_KEY in the .env file")
                                                    else:
                                                        st.error(f"❌ Backfill failed: {error_msg}")
                            
                            # Display coverage results
                            rows_key = f"coverage_rows_{selected_category}_{selected_instrument_item}"
                            coverage_rows = st.session_state.get(rows_key)
                            if coverage_rows:
                                coverage_df = pd.DataFrame(coverage_rows)
                                st.markdown("#### 📊 Coverage Report")
                                st.dataframe(coverage_df, use_container_width=True, hide_index=True)
                                
                                # Status summary
                                if "status" in coverage_df.columns:
                                    statuses = coverage_df["status"].value_counts().to_dict()
                                    status_display = ", ".join([f"{k}: {v}" for k, v in statuses.items()])
                                    st.caption(f"**Status Summary**: {status_display}")
                                
                                # Show gaps information
                                missing_count = sum(1 for row in coverage_rows if row.get("status") != "ready")
                                if missing_count > 0:
                                    st.warning(f"⚠️ {missing_count} interval(s) need backfilling. Use the 'Auto Backfill' button above.")
                            
                            # Display backfill logs
                            log_key = f"coverage_backfill_logs_{selected_category}_{selected_instrument_item}"
                            backfill_logs = st.session_state.get(log_key)
                            if backfill_logs:
                                log_df = pd.DataFrame(backfill_logs)
                                st.markdown("#### 📋 Backfill Activity")
                                st.dataframe(log_df, use_container_width=True, hide_index=True)
                        else:
                            st.info(f"Symbol '{check_symbol}' not found in database. Ingest data first to run coverage audit.")
                    except Exception as e:
                        st.info(f"Error checking database: {str(e)}. Ingest data first to run coverage audit.")
                else:
                    st.warning("Database not configured. Cannot run coverage audit.")
        
        if selected_instrument_item != "Add...":
            # 1. Polygon API Ingestion (Primary)
            with st.expander(f"Data Management: {selected_instrument_item} (Polygon API)", expanded=False):
                st.info(f"Ingest 1-minute data for {selected_instrument_item} directly from Polygon API. Automatically resumes from latest timestamp to now.")
                
                # Default API Symbol Logic - use conversion function for all instruments
                default_api_symbol = convert_instrument_to_polygon_symbol(selected_category, selected_instrument_item)
                
                # Add helpful note for indices about SPY
                help_text = "e.g. C:XAUUSD, I:SPX, C:EURUSD, AAPL"
                if selected_category == "Indices" and "SPX" in default_api_symbol.upper():
                    help_text += "\n\n⚠️ Note: For 1-minute data, Polygon doesn't support I:SPX. The system will auto-convert to SPY (ETF)."
                
                # Sanitize key to avoid issues with special characters like "/" in instrument names
                safe_key = f"api_sym_{selected_category}_{selected_instrument_item}".replace("/", "_").replace("\\", "_").replace(" ", "_")
                api_symbol = st.text_input("Polygon Symbol", value=default_api_symbol, help=help_text, key=safe_key)
                
                # Use same sanitized key pattern for button
                safe_btn_key = f"btn_api_{selected_category}_{selected_instrument_item}".replace("/", "_").replace("\\", "_").replace(" ", "_")
                if st.button("Fetch & Ingest from API", key=safe_btn_key):
                    with st.spinner("Fetching data from Polygon..."):
                        # CRITICAL: Clean the symbol immediately - remove all whitespace including newlines
                        api_symbol_cleaned = api_symbol.strip().replace('\n', '').replace('\r', '').replace('\t', '') if api_symbol else ""
                        
                        # CRITICAL FIX: If the cleaned symbol still contains "/" (like "EUR/USD"), 
                        # it means the text input returned the original instrument name instead of the converted symbol.
                        # Re-convert it using the conversion function.
                        if "/" in api_symbol_cleaned and selected_category == "Currencies":
                            original_for_display = api_symbol_cleaned
                            api_symbol_cleaned = convert_instrument_to_polygon_symbol(selected_category, api_symbol_cleaned)
                            st.info(f"ℹ️ Detected currency pair with '/', converted '{original_for_display}' → '{api_symbol_cleaned}'")
                        
                        # Validate symbol is not empty or just a single character
                        if not api_symbol_cleaned or len(api_symbol_cleaned) <= 1:
                            st.error(f"❌ Invalid symbol: '{api_symbol}'. Please enter a complete symbol (e.g., C:EURUSD for currencies).")
                            st.stop()
                        
                        # Validate symbol is not just a prefix (C, I, C:, I:)
                        if api_symbol_cleaned in ["C", "I", "C:", "I:"]:
                            st.error(f"❌ Symbol '{api_symbol_cleaned}' is incomplete. For currencies, use format like 'C:EURUSD'. For indices, use format like 'I:SPX'.")
                            st.stop()
                        
                        # Handle specific symbol mappings for DB storage
                        effective_db_symbol = selected_instrument_item
                        if selected_instrument_item == "GOLD":
                            effective_db_symbol = "C:XAUUSD"
                        elif selected_instrument_item == "S&P 500":
                            effective_db_symbol = "^SPX"

                        # Use indices-specific ingestion for indices (1 year, 1-day chunks, UTC→GMT+4)
                        try:
                            if selected_category == "Indices":
                                from ingest_indices_polygon import ingest_indices_from_polygon
                                result = ingest_indices_from_polygon(
                                    api_symbol=api_symbol_cleaned,  # Use cleaned symbol
                                    interval="1min",
                                    years=1,  # Polygon free plan: 1 year for indices
                                    db_symbol=effective_db_symbol
                                )
                            else:
                                # Use universal ingestion for other asset classes
                                # Always ingests up to current UTC time (when button is clicked)
                                result = ingest_from_polygon_api(
                                    api_symbol=api_symbol_cleaned,  # Use cleaned symbol
                                    asset_class=selected_category,
                                    db_symbol=effective_db_symbol,
                                    auto_resume=True
                                )
                            
                            # Always display the result message
                            if result and isinstance(result, dict):
                                if result.get("success"):
                                    st.success(result.get("message", "✅ Ingestion completed successfully"))
                                else:
                                    st.error(f"Failed: {result.get('message', 'Unknown error occurred')}")
                            else:
                                st.error(f"Unexpected result format: {result}")
                                st.write(f"Debug - Result type: {type(result)}, Value: {result}")
                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            st.error(f"❌ Exception during ingestion: {str(e)}")
                            st.exception(e)  # Show full traceback in Streamlit
                            print(f"❌ Exception during ingestion: {error_details}")
            
            # 2. File Upload (Secondary)
            with st.expander(f"Data Management: {selected_instrument_item} (File Upload)", expanded=False):
                st.info(f"Ingest 1-minute data for {selected_instrument_item} ({selected_category}) via CSV/Excel.")
                
                # Timezone Selection for File Upload
                tz_options = ["America/New_York", "UTC", "Asia/Dubai", "Europe/London"]
                source_tz = st.selectbox(
                    "Source Timezone (of the file)", 
                    options=tz_options, 
                    index=0, 
                    key=f"tz_{selected_category}_{selected_instrument_item}",
                    help="Data will be converted to GMT+4 (Dubai) automatically."
                )
                
                upload_key = f"upload_{selected_category}_{selected_instrument_item}"
                uploaded_file = st.file_uploader(f"Upload Data for {selected_instrument_item}", type=["csv", "xls", "xlsx"], key=upload_key)
                
                if uploaded_file:
                    if st.button(f"Ingest File", key=f"btn_ingest_file_{selected_category}_{selected_instrument_item}"):
                        with st.spinner("Ingesting file..."):
                            # Handle specific symbol mappings if needed
                            effective_symbol = selected_instrument_item
                            if selected_instrument_item == "GOLD":
                                effective_symbol = "C:XAUUSD"
                            elif selected_instrument_item == "S&P 500":
                                effective_symbol = "^SPX"
                                
                            result = ingest_market_data(
                                uploaded_file, 
                                asset_class=selected_category, 
                                default_symbol=effective_symbol,
                                source_timezone=source_tz
                            )
                            
                            if result.get("success"):
                                st.success(result.get("message"))
                            else:
                                st.error(f"Failed: {result.get('message')}")
    with col_b:
        # Fetch providers from database
        db_providers = get_available_providers()
        
        # Base options
        signal_provider_options = ["PipXpert", "Add..."]
        
        # Add providers from database (excluding PipXpert if already in base options)
        db_providers_filtered = [p for p in db_providers if p not in signal_provider_options]
        
        # Merge: base options + database providers (sorted)
        merged_providers = signal_provider_options + sorted(db_providers_filtered)
        
        selected_provider = st.selectbox(
            "Signal Provider",
            options=merged_providers,
            index=0,
            key="select_signal_provider"
        )
        
        # Show upload UI for any selected provider
        if selected_provider and selected_provider != "Add...":
            # Handle PipXpert or any other provider
            if selected_provider == "PipXpert":
                with st.expander("PipXpert Signal Data", expanded=False):
                    st.info("Upload PipXpert signal data from Excel file.")
                    provider_name = "PipXpert"
                    # symbol input removed as it is now read from the file
                    
                    # Date range selection (optional)
                    col_date1, col_date2 = st.columns(2)
                    with col_date1:
                        start_date = st.date_input("Start Date (optional)", value=None, key="pipxpert_start_date")
                    with col_date2:
                        end_date = st.date_input("End Date (optional)", value=None, key="pipxpert_end_date")
                    
                    # All signals from Excel are assumed to be in UTC
                    source_timezone = "UTC"
                    
                    signal_file = st.file_uploader("Upload Signal Data (Excel)", type=["xls", "xlsx", "csv"], key="pipxpert_upload")
                    
                    if signal_file:
                        # Preview and validate
                        try:
                            df_preview = _read_df(signal_file)
                            if df_preview is not None and not df_preview.empty:
                                st.dataframe(df_preview.head(10), use_container_width=True)
                                
                                # Validate data (timezone_offset parameter kept for backward compatibility, but actual conversion happens in ingestion)
                                validation = validate_signal_provider_data(
                                    df_preview, 
                                    provider_name, 
                                    "+04:00"  # Default, actual conversion uses source_timezone
                                )
                                
                                if validation["valid"]:
                                    st.success("✓ Data validation passed")
                                    if validation.get("warnings"):
                                        for warning in validation["warnings"]:
                                            st.warning(f"⚠ {warning}")
                                    
                                    # Show data summary
                                    summary = validation.get("data_summary", {})
                                    if summary:
                                        st.info(f"**Summary:** {summary.get('total_rows', 0)} rows | "
                                               f"Actions: {summary.get('action_counts', {})} | "
                                               f"Date range: {summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}")
                                    
                                    # Confirmation and submit button
                                    if st.button("✅ Ingest Signal Data", key="ingest_pipxpert_btn", type="primary"):
                                        # Create progress container
                                        progress_container = st.container()
                                        progress_placeholder = progress_container.empty()
                                        
                                        # Progress callback function
                                        progress_messages = []
                                        def update_progress(msg, level="info"):
                                            progress_messages.append((msg, level))
                                            # Update UI with last few messages
                                            with progress_placeholder:
                                                for pm, pl in progress_messages[-5:]:
                                                    if pl == "success":
                                                        st.success(pm)
                                                    elif pl == "warning":
                                                        st.warning(pm)
                                                    elif pl == "error":
                                                        st.error(pm)
                                                    else:
                                                        st.info(pm)
                                        
                                        result = ingest_signal_provider_data(
                                            signal_file,
                                            provider_name,
                                            source_timezone=source_timezone,
                                            progress_callback=update_progress
                                        )
                                        
                                        # Clear progress container and show final result
                                        progress_placeholder.empty()
                                        
                                        if result.get("success"):
                                            st.success(result.get("message"))
                                            
                                            # Show details if available
                                            if result.get("details"):
                                                details = result["details"]
                                                st.markdown("#### 📊 Ingestion Details")
                                                col1, col2, col3, col4 = st.columns(4)
                                                with col1:
                                                    st.metric("Total Processed", details.get("total_processed", 0))
                                                with col2:
                                                    st.metric("Inserted", details.get("inserted", 0))
                                                with col3:
                                                    if details.get("skipped_validation", 0) > 0:
                                                        st.metric("Skipped (Validation)", details.get("skipped_validation", 0))
                                                with col4:
                                                    if details.get("skipped_duplicates", 0) > 0:
                                                        st.metric("Skipped (Duplicates)", details.get("skipped_duplicates", 0))
                                                if details.get("skip_reasons"):
                                                    st.write("**Skip Reasons:**")
                                                    for reason, count in details["skip_reasons"].items():
                                                        st.write(f"- {reason}: {count}")
                                            
                                            # Mark instruments list for refresh
                                            st.session_state.instruments_need_refresh = True
                                            
                                            # Force refresh to update dropdown with new provider
                                            # Small delay to ensure database commit is complete
                                            import time
                                            time.sleep(0.5)
                                            st.rerun()
                                        else:
                                            st.error(f"Failed: {result.get('message')}")
                                else:
                                    st.error(f"Validation failed: {validation.get('message')}")
                                    if validation.get("warnings"):
                                        for warning in validation["warnings"]:
                                            st.warning(f"⚠ {warning}")
                            else:
                                st.error("Could not read file. Please ensure it is a valid Excel or CSV file.")
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
            
            else:
                # Handle other providers (not PipXpert)
                provider_name = selected_provider
                with st.expander(f"{provider_name} Signal Data", expanded=False):
                    st.info(f"Upload {provider_name} signal data from Excel file.")
                    
                    # Date range selection (optional)
                    col_date1, col_date2 = st.columns(2)
                    with col_date1:
                        start_date = st.date_input("Start Date (optional)", value=None, key=f"{provider_name}_start_date")
                    with col_date2:
                        end_date = st.date_input("End Date (optional)", value=None, key=f"{provider_name}_end_date")
                    
                    # All signals from Excel are assumed to be in UTC
                    source_timezone = "UTC"
                    
                    signal_file = st.file_uploader("Upload Signal Data (Excel)", type=["xls", "xlsx", "csv"], key=f"{provider_name}_upload")
                    
                    if signal_file:
                        # Preview and validate
                        try:
                            df_preview = _read_df(signal_file)
                            if df_preview is not None and not df_preview.empty:
                                st.dataframe(df_preview.head(10), use_container_width=True)
                                
                                # Validate data
                                validation = validate_signal_provider_data(
                                    df_preview, 
                                    provider_name, 
                                    "+04:00"
                                )
                                
                                if validation["valid"]:
                                    st.success("✓ Data validation passed")
                                    if validation.get("warnings"):
                                        for warning in validation["warnings"]:
                                            st.warning(f"⚠ {warning}")
                                    
                                    # Show data summary
                                    summary = validation.get("data_summary", {})
                                    if summary:
                                        st.info(f"**Summary:** {summary.get('total_rows', 0)} rows | "
                                               f"Actions: {summary.get('action_counts', {})} | "
                                               f"Date range: {summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}")
                                    
                                    # Confirmation and submit button
                                    if st.button("✅ Ingest Signal Data", key=f"ingest_{provider_name}_btn", type="primary"):
                                        # Create progress container
                                        progress_container = st.container()
                                        progress_placeholder = progress_container.empty()
                                        
                                        # Progress callback function
                                        progress_messages = []
                                        def update_progress(msg, level="info"):
                                            progress_messages.append((msg, level))
                                            # Update UI with last few messages
                                            with progress_placeholder:
                                                for pm, pl in progress_messages[-5:]:
                                                    if pl == "success":
                                                        st.success(pm)
                                                    elif pl == "warning":
                                                        st.warning(pm)
                                                    elif pl == "error":
                                                        st.error(pm)
                                                    else:
                                                        st.info(pm)
                                        
                                        result = ingest_signal_provider_data(
                                            signal_file,
                                            provider_name,
                                            source_timezone=source_timezone,
                                            progress_callback=update_progress
                                        )
                                        
                                        # Clear progress container and show final result
                                        progress_placeholder.empty()
                                        
                                        if result.get("success"):
                                            st.success(result.get("message"))
                                            
                                            # Show details if available
                                            if result.get("details"):
                                                details = result["details"]
                                                st.markdown("#### 📊 Ingestion Details")
                                                col1, col2, col3, col4 = st.columns(4)
                                                with col1:
                                                    st.metric("Total Processed", details.get("total_processed", 0))
                                                with col2:
                                                    st.metric("Inserted", details.get("inserted", 0))
                                                with col3:
                                                    if details.get("skipped_validation", 0) > 0:
                                                        st.metric("Skipped (Validation)", details.get("skipped_validation", 0))
                                                with col4:
                                                    if details.get("skipped_duplicates", 0) > 0:
                                                        st.metric("Skipped (Duplicates)", details.get("skipped_duplicates", 0))
                                                if details.get("skip_reasons"):
                                                    st.write("**Skip Reasons:**")
                                                    for reason, count in details["skip_reasons"].items():
                                                        st.write(f"- {reason}: {count}")
                                            
                                            # Mark instruments list for refresh
                                            st.session_state.instruments_need_refresh = True
                                            
                                            # Force refresh to update dropdown with new provider
                                            # Small delay to ensure database commit is complete
                                            import time
                                            time.sleep(0.5)
                                            st.rerun()
                                        else:
                                            st.error(f"Failed: {result.get('message')}")
                                else:
                                    st.error(f"Validation failed: {validation.get('message')}")
                                    if validation.get("warnings"):
                                        for warning in validation["warnings"]:
                                            st.warning(f"⚠ {warning}")
                            else:
                                st.error("Could not read file. Please ensure it is a valid Excel or CSV file.")
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
        
        # Handle "Add..." option separately
        if selected_provider == "Add...":
                with st.expander("Add New Signal Provider", expanded=False):
                    st.write("Enter signal provider details and upload data file.")
                    
                    provider_name = st.text_input("Provider Name", key="new_provider_name", help="Enter the name of the signal provider")
                    # symbol input removed as it is now read from the file
                    
                    # Date range selection (optional)
                    col_date1, col_date2 = st.columns(2)
                    with col_date1:
                        start_date = st.date_input("Start Date (optional)", value=None, key="new_provider_start_date")
                    with col_date2:
                        end_date = st.date_input("End Date (optional)", value=None, key="new_provider_end_date")
                    
                    # Timezone Selection (source timezone of the file)
                    source_timezone_options = ["America/New_York", "UTC", "Asia/Dubai", "Europe/London", "Asia/Kolkata", "America/Chicago", "America/Los_Angeles"]
                    source_timezone = st.selectbox(
                        "Source Timezone (of the Excel file)",
                        options=source_timezone_options,
                        index=1,  # Default to UTC
                        key="new_provider_timezone",
                        help="Timezone of the dates in the Excel file. Data will be converted to GMT+4 (Asia/Dubai) automatically."
                    )
                    
                    signal_file = st.file_uploader("Upload Signal Data (Excel/CSV)", type=["xls", "xlsx", "csv"], key="new_provider_upload")
                    
                    if signal_file and provider_name:
                        # Preview and validate
                        try:
                            df_preview = _read_df(signal_file)
                            if df_preview is not None and not df_preview.empty:
                                st.dataframe(df_preview.head(10), use_container_width=True)
                                
                                # Validate data (timezone_offset parameter kept for backward compatibility, but actual conversion happens in ingestion)
                                validation = validate_signal_provider_data(
                                    df_preview, 
                                    provider_name, 
                                    "+04:00"  # Default, actual conversion uses source_timezone
                                )
                                
                                if validation["valid"]:
                                    st.success("✓ Data validation passed")
                                    if validation.get("warnings"):
                                        for warning in validation["warnings"]:
                                            st.warning(f"⚠ {warning}")
                                    
                                    # Show data summary
                                    summary = validation.get("data_summary", {})
                                    if summary:
                                        st.info(f"**Summary:** {summary.get('total_rows', 0)} rows | "
                                               f"Actions: {summary.get('action_counts', {})} | "
                                               f"Date range: {summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}")
                                    
                                    # Confirmation and submit button
                                    if st.button("✅ Submit Signal Data", key="ingest_new_provider_btn", type="primary"):
                                        # Create progress container
                                        progress_container = st.container()
                                        progress_placeholder = progress_container.empty()
                                        
                                        # Progress callback function
                                        progress_messages = []
                                        def update_progress(msg, level="info"):
                                            progress_messages.append((msg, level))
                                            # Update UI with last few messages
                                            with progress_placeholder:
                                                for pm, pl in progress_messages[-5:]:
                                                    if pl == "success":
                                                        st.success(pm)
                                                    elif pl == "warning":
                                                        st.warning(pm)
                                                    elif pl == "error":
                                                        st.error(pm)
                                                    else:
                                                        st.info(pm)
                                        
                                        result = ingest_signal_provider_data(
                                            signal_file,
                                            provider_name,
                                            source_timezone=source_timezone,
                                            progress_callback=update_progress
                                        )
                                        
                                        # Clear progress container and show final result
                                        progress_placeholder.empty()
                                        
                                        if result.get("success"):
                                            st.success(result.get("message"))
                                            # Show details if available
                                            if result.get("details"):
                                                details = result["details"]
                                                st.markdown("#### 📊 Ingestion Details")
                                                col1, col2, col3, col4 = st.columns(4)
                                                with col1:
                                                    st.metric("Total Processed", details.get("total_processed", 0))
                                                with col2:
                                                    st.metric("Inserted", details.get("inserted", 0))
                                                with col3:
                                                    if details.get("skipped_validation", 0) > 0:
                                                        st.metric("Skipped (Validation)", details.get("skipped_validation", 0))
                                                with col4:
                                                    if details.get("skipped_duplicates", 0) > 0:
                                                        st.metric("Skipped (Duplicates)", details.get("skipped_duplicates", 0))
                                                if details.get("skip_reasons"):
                                                    st.write("**Skip Reasons:**")
                                                    for reason, count in details["skip_reasons"].items():
                                                        st.write(f"- {reason}: {count}")
                                            # Add provider to session state immediately
                                            if provider_name and provider_name not in (st.session_state.get("signal_providers", []) or []):
                                                if "signal_providers" not in st.session_state:
                                                    st.session_state.signal_providers = []
                                                st.session_state.signal_providers.append(provider_name)
                                            
                                            # Mark instruments list for refresh
                                            st.session_state.instruments_need_refresh = True
                                            
                                            # Force refresh to update dropdown with new provider
                                            # Small delay to ensure database commit is complete
                                            import time
                                            time.sleep(0.5)
                                            st.rerun()
                                        else:
                                            st.error(f"Failed: {result.get('message')}")
                                else:
                                    st.error(f"Validation failed: {validation.get('message')}")
                                    if validation.get("warnings"):
                                        for warning in validation["warnings"]:
                                            st.warning(f"⚠ {warning}")
                            else:
                                st.error("Could not read file. Please ensure it is a valid Excel or CSV file.")
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
                    elif signal_file:
                        st.warning("Please fill in provider name before uploading")
        
        # Fetch and Save Telegram Signals
        st.markdown("---")
        with st.expander("📥 Fetch & Save Telegram Signals", expanded=False):
            st.markdown("### Fetch and Save Signal Messages from Telegram Channel")
            st.info("💡 Fetch all signal-related messages from a Telegram channel, save them to the database, and export to Excel.")
            
            # Check if telethon is available
            try:
                import telethon
                telethon_available = True
            except ImportError:
                telethon_available = False
                st.warning("⚠️ `telethon` not installed. Install with: `pip install telethon`")
                st.code("pip install telethon", language="bash")
            
            if telethon_available:
                # Check for Telegram API credentials
                import os
                telegram_api_id = os.getenv("TELEGRAM_API_ID")
                telegram_api_hash = os.getenv("TELEGRAM_API_HASH")
                
                if not telegram_api_id or not telegram_api_hash:
                    st.warning("⚠️ Telegram API credentials not configured.")
                    st.info("""
                    **To set up Telegram integration:**
                    1. Go to https://my.telegram.org/apps
                    2. Create a new application
                    3. Get your `api_id` and `api_hash`
                    4. Add them to your `.env` file:
                       ```
                       TELEGRAM_API_ID=your_api_id
                       TELEGRAM_API_HASH=your_api_hash
                       ```
                    """)
                else:
                    st.success("✅ Telegram API credentials configured")
                    
                    from tradingagents.dataflows.telegram_signal_service import TelegramSignalService
                    from tradingagents.dataflows.signal_export import export_telegram_messages_to_excel
                    import asyncio
                    
                    fetch_channel = st.text_input(
                        "Channel Username",
                        placeholder="@signal_provider",
                        key="fetch_channel_username",
                        help="Enter the Telegram channel username to fetch signal messages from. Will fetch all messages from channel creation."
                    )
                    
                    st.info("💡 This will fetch all signal-related messages from the channel's creation date and prepare them for download. You can then upload the downloaded file through the Signal Provider section above.")
                    
                    if st.button("📥 Fetch Signal Messages", key="fetch_telegram_messages"):
                        if fetch_channel:
                            with st.spinner(f"Fetching signal messages from {fetch_channel} (this may take a while for channels with many messages)..."):
                                try:
                                    # Create service instance
                                    service = TelegramSignalService()
                                    
                                    # Connect and fetch messages (all from channel creation, signals only)
                                    async def fetch_messages():
                                        try:
                                            await service.connect()
                                            messages = await service.fetch_all_channel_messages(
                                                fetch_channel,
                                                limit=None,  # Fetch all messages from channel creation
                                                filter_signals_only=True  # Only signal-related messages
                                            )
                                            return messages
                                        finally:
                                            # Always try to disconnect, even if there's an error
                                            try:
                                                await service.disconnect()
                                            except Exception as disconnect_error:
                                                # Ignore database lock errors during disconnect
                                                error_str = str(disconnect_error).lower()
                                                if "database is locked" not in error_str and "locked" not in error_str:
                                                    # Only log non-lock errors
                                                    print(f"Warning during disconnect: {disconnect_error}")
                                    
                                    messages = asyncio.run(fetch_messages())
                                    
                                    if messages:
                                        st.success(f"✅ Fetched {len(messages)} signal messages from {fetch_channel}")
                                        
                                        # Count how many messages have parsed signals
                                        signals_with_data = sum(1 for msg in messages if msg.get('parsed_signal'))
                                        st.info(f"📊 {signals_with_data} out of {len(messages)} messages contain parsed signal data")
                                        
                                        # Export to Excel in signal provider format (Date, Action, Currency Pair, Entry Price, etc.)
                                        try:
                                            excel_file = export_telegram_messages_to_excel(
                                                messages, 
                                                fetch_channel,
                                                format_as_signal_provider=True
                                            )
                                            
                                            # Check if file has data
                                            excel_file.seek(0)
                                            file_size = len(excel_file.getvalue())
                                            excel_file.seek(0)
                                            
                                            if file_size > 0:
                                                # Download button
                                                channel_clean = fetch_channel.lstrip('@').replace('/', '_')
                                                filename = f"telegram_signals_{channel_clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                                                
                                                st.download_button(
                                                    label="📥 Download Signals as Excel",
                                                    data=excel_file.getvalue(),
                                                    file_name=filename,
                                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                    key="download_telegram_messages"
                                                )
                                                
                                                st.info("💡 After downloading, upload this file through the Signal Provider section above (e.g., PipXpert or Add New Signal Provider) to save signals to the database.")
                                            else:
                                                st.warning("⚠️ Excel file is empty. No valid signals could be exported.")
                                        except Exception as export_error:
                                            st.error(f"❌ Error exporting to Excel: {str(export_error)}")
                                            import traceback
                                            st.code(traceback.format_exc())
                                        
                                        # Show preview
                                        st.markdown("#### Signal Preview (First 10)")
                                        # Convert to DataFrame for preview
                                        preview_data = []
                                        for msg in messages[:10]:
                                            signal = msg.get('parsed_signal', {})
                                            if signal:
                                                preview_data.append({
                                                    'symbol': signal.get('symbol', ''),
                                                    'action': signal.get('action', ''),
                                                    'entry_price': signal.get('entry_price', ''),
                                                    'stop_loss': signal.get('stop_loss', ''),
                                                    'target_1': signal.get('target_1', ''),
                                                    'date': msg.get('date', '')
                                                })
                                        
                                        if preview_data:
                                            df_preview = pd.DataFrame(preview_data)
                                            st.dataframe(df_preview, use_container_width=True)
                                    else:
                                        st.warning(f"No signal messages found in {fetch_channel}")
                                except Exception as e:
                                    st.error(f"❌ Error fetching messages: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                        else:
                            st.warning("⚠️ Please enter a channel username")
        
        # View Stored Signals for Selected Provider
        if selected_provider and selected_provider != "Add...":
            st.markdown("---")
            with st.expander(f"📋 View Stored Signals: {selected_provider}", expanded=False):
                # Import here locally to avoid circular imports or long top-level imports
                from tradingagents.database.db_service import get_provider_signals
                
                # Fetch signals for this provider (all symbols)
                stored_signals = get_provider_signals(provider=selected_provider, limit=100)
                
                if stored_signals:
                    df_signals = pd.DataFrame(stored_signals)
                    # Reorder columns if possible
                    cols = ['signal_date', 'symbol', 'action', 'entry_price', 'stop_loss', 'target_1', 'target_2', 'target_3']
                    # Add new columns if they exist in df
                    for c in ['entry_price_max', 'target_4', 'target_5']:
                        if c in df_signals.columns:
                            cols.append(c)
                            
                    # Filter existing columns
                    display_cols = [c for c in cols if c in df_signals.columns]
                    # Add any other columns not in display_cols
                    other_cols = [c for c in df_signals.columns if c not in display_cols and c not in ['id', 'created_at', 'provider_name', 'timezone_offset']]
                    display_cols.extend(other_cols)
                    
                    st.dataframe(df_signals[display_cols], use_container_width=True)
                    st.caption(f"Showing last {len(df_signals)} signals.")
                else:
                    st.info(f"No stored signals found for {selected_provider}.")
    
    # Signal Analysis Dashboard - Full Width Section
    st.markdown("---")
    st.markdown("### 📊 Signal Analysis Dashboard")
    st.info("Run automated analysis for signal providers. Select an instrument to analyze its signals against market data.")
    
    st.markdown("#### Automated Signal Analysis")
    st.write("Analyze signals to determine TP/SL hits and calculate performance metrics. Select an instrument to analyze its signals against market data.")
    
    # Get available instruments and providers from database
    # Always fetch fresh data on every page load/refresh - no caching to ensure latest symbols appear
    # Clear cached instruments if signals were just uploaded
    if st.session_state.get("instruments_need_refresh", False):
        # Clear any cached instrument list
        if "cached_instruments" in st.session_state:
            del st.session_state.cached_instruments
        st.session_state.instruments_need_refresh = False
    
    # Always fetch fresh instruments from database on every page load (no caching)
    # This ensures new symbols appear automatically when signals are added
    available_instruments = get_available_instruments()
    available_providers = get_available_providers()
    
    # Show count of available instruments
    if available_instruments:
        st.caption(f"📊 {len(available_instruments)} instrument(s) available with signals")
    
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        analysis_instrument = st.selectbox(
            "Instrument (Required)",
            options=["Select Instrument"] + (available_instruments or []),
            key="analysis_instrument",
            help="Select the instrument to analyze. All distinct symbols from signal_provider_signals table are shown here."
        )
    with col_a2:
        analysis_provider = st.selectbox(
            "Provider (Optional)",
            options=["All Providers"] + (available_providers or []),
            key="analysis_provider",
            help="Select a specific provider to filter signals. 'All Providers' will analyze signals from all providers."
        )
    with col_a3:
        analysis_date_range = st.selectbox(
            "Date Range",
            options=["All Time", "Last 30 Days", "Last 60 Days", "Last 90 Days", "Custom"],
            key="analysis_date_range"
        )
    
    if analysis_date_range == "Custom":
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            analysis_start_date = st.date_input("Start Date", key="analysis_start")
        with col_d2:
            analysis_end_date = st.date_input("End Date", key="analysis_end")
    else:
        analysis_start_date = None
        analysis_end_date = None
        if analysis_date_range != "All Time":
            days = int(analysis_date_range.split()[1])
            analysis_start_date = datetime.now().date() - timedelta(days=days)
            analysis_end_date = datetime.now().date()
    
    # Check market data availability when instrument is selected
    if analysis_instrument and analysis_instrument != "Select Instrument":
        with st.spinner("Checking market data availability..."):
            market_data_status = check_market_data_availability(analysis_instrument)
            
            if market_data_status.get("available"):
                st.success(f"✅ Market data available for {analysis_instrument}")
            else:
                st.warning(f"⚠️ {market_data_status.get('error', 'Market data not available for this instrument')}")
                st.info("💡 Please ensure market data has been ingested for this instrument before running analysis.")
        
        # Show informational message when specific provider is selected
        if analysis_provider and analysis_provider != "All Providers":
            st.caption(f"ℹ️ Filtering by provider: **{analysis_provider}** for symbol **{analysis_instrument}**. If no signals are found, try selecting 'All Providers' to see signals from all providers.")
    
    if st.button("Run Analysis", key="run_analysis_btn", type="primary"):
        if analysis_instrument == "Select Instrument":
            st.error("Please select an instrument to analyze.")
        else:
            with st.spinner("Running automated analysis..."):
                try:
                    # Check market data availability first
                    market_data_status = check_market_data_availability(analysis_instrument)
                    if not market_data_status.get("available"):
                        st.error(f"Cannot run analysis: {market_data_status.get('error', 'Market data not available')}")
                        st.stop()
                    
                    provider_filter = None if analysis_provider == "All Providers" else analysis_provider
                    start_dt = datetime.combine(analysis_start_date, datetime.min.time()) if analysis_start_date else None
                    end_dt = datetime.combine(analysis_end_date, datetime.max.time()) if analysis_end_date else None
                    
                    result = run_analysis_for_all_signals(
                        provider_name=provider_filter,
                        symbol=analysis_instrument,
                        start_date=start_dt,
                        end_date=end_dt,
                        save_results=False  # Don't save to DB during testing
                    )
                    
                    if 'error' not in result:
                        st.success(f"Analysis Complete!")
                        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                        with col_r1:
                            st.metric("Total Signals", result.get('total_signals', 0))
                        with col_r2:
                            st.metric("Analyzed", result.get('analyzed', 0))
                        with col_r3:
                            st.metric("Errors", result.get('errors', 0))
                        with col_r4:
                            st.metric("Success Rate", f"{result.get('success_rate', 0):.1f}%")
                        
                        # Store results in session state for display (without saving to DB)
                        st.session_state['latest_analysis_results'] = result.get('analysis_results', [])
                    else:
                        # Display informational message instead of error
                        error_msg = result.get('error', 'Unknown error')
                        
                        # Use informational message instead of error
                        if '💡' in error_msg or 'Suggestions:' in error_msg:
                            # Split message into main message and suggestions
                            lines = error_msg.split('\n')
                            main_message = lines[0]
                            suggestions = '\n'.join(lines[1:]) if len(lines) > 1 else ''
                            
                            st.info(f"ℹ️ {main_message}")
                            if suggestions:
                                st.info(suggestions)
                        else:
                            st.info(f"ℹ️ {error_msg}")
                        
                        if 'latest_analysis_results' in st.session_state:
                            del st.session_state['latest_analysis_results']
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Show analysis results (from session state during testing, not from DB)
    st.markdown("#### Analysis Results")
    
    # Check if we have results in session state (testing mode - not saved to DB)
    if 'latest_analysis_results' in st.session_state and st.session_state['latest_analysis_results']:
        try:
            results_list = st.session_state['latest_analysis_results']
            if results_list:
                df_results = pd.DataFrame(results_list)
                
                # Calculate summary statistics
                total_signals = len(df_results)
                
                # Count TP1, TP2, TP3, SL hits (convert boolean to int for proper counting)
                tp1_count = int(df_results['tp1_hit'].fillna(False).astype(int).sum()) if 'tp1_hit' in df_results.columns else 0
                tp2_count = int(df_results['tp2_hit'].fillna(False).astype(int).sum()) if 'tp2_hit' in df_results.columns else 0
                tp3_count = int(df_results['tp3_hit'].fillna(False).astype(int).sum()) if 'tp3_hit' in df_results.columns else 0
                sl_count = int(df_results['sl_hit'].fillna(False).astype(int).sum()) if 'sl_hit' in df_results.columns else 0
                
                # Calculate percentages
                tp1_pct = (tp1_count / total_signals * 100) if total_signals > 0 else 0
                tp2_pct = (tp2_count / total_signals * 100) if total_signals > 0 else 0
                tp3_pct = (tp3_count / total_signals * 100) if total_signals > 0 else 0
                sl_pct = (sl_count / total_signals * 100) if total_signals > 0 else 0
                
                # Sum of pips made (handle NaN values)
                total_pips = df_results['pips_made'].fillna(0).sum() if 'pips_made' in df_results.columns else 0
                
                # Display summary statistics
                st.markdown("##### 📊 Summary Statistics")
                col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
                
                with col_s1:
                    st.metric("TP1 Hits", f"{tp1_count}", f"{tp1_pct:.1f}%")
                with col_s2:
                    st.metric("TP2 Hits", f"{tp2_count}", f"{tp2_pct:.1f}%")
                with col_s3:
                    st.metric("TP3 Hits", f"{tp3_count}", f"{tp3_pct:.1f}%")
                with col_s4:
                    st.metric("SL Hits", f"{sl_count}", f"{sl_pct:.1f}%")
                with col_s5:
                    st.metric("Total Pips", f"{int(total_pips)}", "Sum")
                
                st.markdown("---")
                
                # Format table to match the image format
                # Columns: Date, Time, Asset, Direction, Entry, TP1, TP2, TP3, SL, Pips Made
                formatted_data = []
                
                def calculate_pips_made(row):
                    """Calculate pips made for a trade result row."""
                    # Get values
                    entry_price = row.get('entry_price')
                    action = row.get('action', '').upper()
                    target_1 = row.get('target_1')
                    target_2 = row.get('target_2')
                    target_3 = row.get('target_3')
                    target_4 = row.get('target_4')
                    target_5 = row.get('target_5')
                    stop_loss = row.get('stop_loss')
                    symbol = str(row.get('symbol', '')).upper()
                    
                    # Check if forex pair (for pip calculation)
                    # Remove "C:" prefix for length check if present
                    symbol_for_check = symbol.replace("C:", "") if symbol.startswith("C:") else symbol
                    is_forex_pair = (
                        (len(symbol_for_check) >= 6 and len(symbol_for_check) <= 7) and
                        (symbol.startswith("C:") or (not symbol.startswith("^") and not symbol.startswith("I:"))) and
                        ("XAU" not in symbol and "XAG" not in symbol)
                    )
                    
                    if not entry_price or pd.isna(entry_price):
                        return 0
                    
                    is_buy = (action == 'BUY')
                    total_price_diff = 0
                    
                    # Calculate cumulative pips: add TP1 if hit, add TP2 if hit, etc.
                    if row.get('tp1_hit') and target_1:
                        tp1_diff = (target_1 - entry_price) if is_buy else (entry_price - target_1)
                        total_price_diff += tp1_diff
                    
                    if row.get('tp2_hit') and target_2:
                        tp2_diff = (target_2 - entry_price) if is_buy else (entry_price - target_2)
                        total_price_diff += tp2_diff
                    
                    if row.get('tp3_hit') and target_3:
                        tp3_diff = (target_3 - entry_price) if is_buy else (entry_price - target_3)
                        total_price_diff += tp3_diff
                    
                    if row.get('tp4_hit') and target_4:
                        tp4_diff = (target_4 - entry_price) if is_buy else (entry_price - target_4)
                        total_price_diff += tp4_diff
                    
                    if row.get('tp5_hit') and target_5:
                        tp5_diff = (target_5 - entry_price) if is_buy else (entry_price - target_5)
                        total_price_diff += tp5_diff
                    
                    # If SL is hit (and no TPs were hit), use SL pips (negative)
                    if row.get('final_status') == 'SL' and stop_loss:
                        any_tp_hit = (row.get('tp1_hit') or row.get('tp2_hit') or 
                                     row.get('tp3_hit') or row.get('tp4_hit') or row.get('tp5_hit'))
                        if not any_tp_hit:
                            total_price_diff = (stop_loss - entry_price) if is_buy else (entry_price - stop_loss)
                    
                    # If EXPIRED/OPEN, use max_profit percentage converted to price points
                    if row.get('final_status') in ['EXPIRED', 'OPEN']:
                        max_profit = row.get('max_profit')
                        if max_profit and not pd.isna(max_profit):
                            total_price_diff = (max_profit / 100) * entry_price
                            if not is_buy:
                                total_price_diff = -total_price_diff
                    
                    # Convert to pips based on instrument type
                    if is_forex_pair:
                        pips_made = total_price_diff * 10000
                    else:
                        pips_made = total_price_diff
                    
                    # Round to nearest integer
                    if pd.isna(pips_made) or pips_made is None:
                        return 0
                    return int(round(pips_made))
                
                for _, row in df_results.iterrows():
                    # Parse signal_date
                    signal_date_str = row.get('signal_date', '')
                    if signal_date_str:
                        try:
                            # datetime is already imported at module level
                            if 'T' in signal_date_str:
                                dt = datetime.fromisoformat(signal_date_str.replace('Z', '+00:00'))
                            else:
                                dt = datetime.fromisoformat(signal_date_str)
                            date_str = dt.strftime('%d/%m/%Y')
                            time_str = dt.strftime('%H:%M')
                        except:
                            date_str = signal_date_str[:10] if len(signal_date_str) >= 10 else signal_date_str
                            time_str = signal_date_str[11:16] if len(signal_date_str) >= 16 else ''
                    else:
                        date_str = 'N/A'
                        time_str = 'N/A'
                    
                    # Format TP/SL with checkmarks if hit
                    tp1_value = row.get('target_1', 'N/A')
                    tp1_hit = row.get('tp1_hit', False)
                    if tp1_value != 'N/A' and tp1_hit:
                        tp1_value = f"{tp1_value} ✓"
                    
                    tp2_value = row.get('target_2', 'N/A')
                    tp2_hit = row.get('tp2_hit', False)
                    if tp2_value != 'N/A' and tp2_hit:
                        tp2_value = f"{tp2_value} ✓"
                    
                    tp3_value = row.get('target_3', 'N/A')
                    tp3_hit = row.get('tp3_hit', False)
                    if tp3_value != 'N/A' and tp3_hit:
                        tp3_value = f"{tp3_value} ✓"
                    
                    # Get stop_loss from the analysis results (should match the original signal)
                    sl_value = row.get('stop_loss', 'N/A')
                    sl_hit = row.get('sl_hit', False)
                    
                    # Format stop_loss value
                    if sl_value != 'N/A' and sl_value is not None:
                        try:
                            # Convert to float and format
                            sl_value = float(sl_value)
                            if sl_hit:
                                sl_value = f"{sl_value} ✓"
                            else:
                                sl_value = f"{sl_value}"
                        except (ValueError, TypeError):
                            # If conversion fails, use as-is
                            if sl_hit:
                                sl_value = f"{sl_value} ✓"
                    else:
                        sl_value = 'N/A'
                    
                    # Get status - show final_status, don't show error messages
                    status = row.get('final_status', 'N/A')
                    
                    # If status is NO_DATA or ERROR, show a user-friendly message instead
                    if status in ['NO_DATA', 'ERROR']:
                        status = 'NO_DATA'
                    elif not status or status == 'N/A':
                        status = 'N/A'
                    
                    # Calculate pips_made if missing or recalculate to ensure accuracy
                    pips_made = row.get('pips_made', 0)
                    if pips_made == 0 or pd.isna(pips_made):
                        # Recalculate if missing or 0 (might be incorrectly calculated)
                        pips_made = calculate_pips_made(row)
                    else:
                        # Use existing value but ensure it's valid
                        try:
                            pips_made = int(round(float(pips_made)))
                        except:
                            pips_made = calculate_pips_made(row)
                    
                    formatted_data.append({
                        'Date': date_str,
                        'Time': time_str,
                        'Asset': row.get('symbol', 'N/A'),
                        'Direction': row.get('action', 'N/A'),
                        'Entry': row.get('entry_price', 'N/A'),
                        'TP1': tp1_value,
                        'TP2': tp2_value,
                        'TP3': tp3_value,
                        'SL': sl_value,
                        'Pips Made': pips_made,
                        'Status': status
                    })
                
                if formatted_data:
                    df_display = pd.DataFrame(formatted_data)
                    # Format numeric columns
                    # Entry should be decimal
                    if 'Entry' in df_display.columns:
                        def format_entry(x):
                            if x == 'N/A' or x is None:
                                return 'N/A'
                            try:
                                if isinstance(x, (int, float)) and not pd.isna(x):
                                    return float(x)
                                return x
                            except:
                                return x
                        df_display['Entry'] = df_display['Entry'].apply(format_entry)
                    
                    # TP1, TP2, TP3, SL may have checkmarks, so format carefully
                    tp_cols = ['TP1', 'TP2', 'TP3', 'SL']
                    for col in tp_cols:
                        if col in df_display.columns:
                            def format_tp_sl(x):
                                if x == 'N/A' or x is None:
                                    return 'N/A'
                                # Check if it already has a checkmark (string format)
                                if isinstance(x, str) and '✓' in x:
                                    # Extract the number part and format it
                                    try:
                                        num_part = float(x.split('✓')[0].strip())
                                        return f"{num_part} ✓"
                                    except:
                                        return x
                                # Otherwise, format as number
                                try:
                                    if isinstance(x, (int, float)) and not pd.isna(x):
                                        return float(x)
                                    return x
                                except:
                                    return x
                            df_display[col] = df_display[col].apply(format_tp_sl)
                    
                    # Pips Made should be integer (can be negative)
                    if 'Pips Made' in df_display.columns:
                        def format_pips(x):
                            if x is None:
                                return 0
                            try:
                                if isinstance(x, (int, float)) and not pd.isna(x):
                                    return int(x)
                                return 0
                            except:
                                return 0
                        df_display['Pips Made'] = df_display['Pips Made'].apply(format_pips)
                    
                    # Only display the columns we want (exclude max_profit, max_drawdown, etc.)
                    display_columns = ['Date', 'Time', 'Asset', 'Direction', 'Entry', 'TP1', 'TP2', 'TP3', 'SL', 'Pips Made', 'Status']
                    df_display = df_display[[col for col in display_columns if col in df_display.columns]]
                    
                    st.dataframe(df_display, use_container_width=True)
                    st.caption(f"📊 Showing {len(df_display)} analysis results (Testing Mode - Not saved to database)")
                else:
                    st.info("No analysis results to display. Run analysis first.")
            else:
                st.info("No analysis results to display. Run analysis first.")
        except Exception as e:
            st.warning(f"Could not display analysis results: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("No analysis results found. Run analysis first. (Testing Mode - Results are not saved to database)")
    
    # =============================================================================
    # MARKET REGIME SEGMENTATION MODULE - NEW SECTION
    # =============================================================================
    st.markdown("---")
    st.markdown("### 🌐 Market Regime Analysis (Context-Based Performance)")
    st.info("Analyze signal performance across different market conditions (Trending vs. Ranging, High Volatility vs. Low Volatility)")
    
    with st.expander("📖 About Market Regime Analysis", expanded=False):
        st.markdown("""
        **Market Regime Segmentation** helps answer the critical question:
        
        > *"Does this signal provider perform better in trending markets or ranging markets?"*
        
        This analysis segments your signal performance based on:
        - **Trend Strength**: Trending vs Ranging (using ADX)
        - **Volatility**: High Vol vs Low Vol (using ATR)
        
        **How it works:**
        1. We calculate technical indicators (ADX, SMA, ATR) on historical market data
        2. We classify each time period into a "regime" (e.g., "Trending - High Vol")
        3. We match your signals to the market regime at the moment of entry
        4. We calculate performance metrics for each regime
        """)
    
    # Sidebar controls for regime parameters
    st.markdown("#### ⚙️ Regime Configuration")
    regime_col1, regime_col2, regime_col3 = st.columns(3)
    
    with regime_col1:
        adx_threshold = st.slider(
            "ADX Threshold (Trending Cutoff)",
            min_value=15,
            max_value=40,
            value=25,
            step=1,
            help="ADX above this value = Trending market, below = Ranging market"
        )
    
    with regime_col2:
        regime_timeframe = st.selectbox(
            "Market Data Timeframe",
            options=["1h", "4h", "1d"],
            index=0,
            help="Timeframe for regime calculation (higher = smoother, lower = more granular)"
        )
    
    with regime_col3:
        regime_lookback_days = st.number_input(
            "Lookback Period (Days)",
            min_value=30,
            max_value=730,
            value=365,
            step=30,
            help="How many days of historical data to fetch for regime analysis"
        )
    
    # Run Regime Analysis button
    if st.button("🚀 Run Regime Analysis", key="run_regime_analysis_btn", type="primary"):
        # Check if we have analysis results to work with
        if 'latest_analysis_results' not in st.session_state or not st.session_state['latest_analysis_results']:
            st.error("⚠️ No signal analysis results found. Please run signal analysis first (above).")
        else:
            with st.spinner("Running market regime analysis..."):
                try:
                    from tradingagents.dataflows.signal_analyzer import SignalAnalyzer
                    from tradingagents.dataflows.market_data_service import fetch_ohlcv
                    
                    # Get analyzed signals from session state
                    signals_data = st.session_state['latest_analysis_results']
                    signals_df = pd.DataFrame(signals_data)
                    
                    # Determine the symbol from the signals
                    if 'symbol' not in signals_df.columns or signals_df['symbol'].isna().all():
                        st.error("⚠️ Signal data doesn't contain symbol information.")
                    else:
                        # Get the most common symbol (or first symbol if multiple)
                        symbol = signals_df['symbol'].mode()[0] if len(signals_df['symbol'].mode()) > 0 else signals_df['symbol'].iloc[0]
                        
                        # Validate symbol
                        if not symbol or len(symbol) < 2:
                            st.error(f"⚠️ Invalid symbol: '{symbol}'. Please ensure signals have valid symbol data.")
                            symbol = None
                    
                    if symbol:
                        
                        # Determine asset class for the symbol
                        asset_class = None
                        if "XAU" in symbol or "XAG" in symbol:
                            asset_class = "Commodities"
                        elif symbol.startswith("C:"):
                            asset_class = "Currencies"
                        elif symbol.startswith("I:") or symbol.startswith("^"):
                            asset_class = "Indices"
                        else:
                            asset_class = "Currencies"  # Default
                        
                        st.info(f"Analyzing regime for: **{symbol}** ({asset_class})")
                        
                        # Fetch market data for regime analysis
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=regime_lookback_days)
                        
                        # Fetch 1min data from database and resample to desired timeframe
                        # Database only has 1min data, so we'll fetch that and resample
                        interval = "1min"
                        
                        market_data = fetch_ohlcv(
                            symbol=symbol,
                            interval=interval,
                            start=start_date,
                            end=end_date,
                            asset_class=asset_class
                        )
                        
                        if market_data is None or market_data.empty:
                            st.error(f"❌ No market data available for {symbol} in the specified period.")
                        else:
                            st.success(f"✅ Fetched {len(market_data)} candles of 1min data")
                            
                            # Ensure timestamp is index
                            if 'timestamp' in market_data.columns:
                                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
                                market_data = market_data.set_index('timestamp')
                            
                            # Resample 1min data to the desired timeframe
                            agg_dict = {
                                'Open': 'first',
                                'High': 'max',
                                'Low': 'min',
                                'Close': 'last'
                            }
                            if 'Volume' in market_data.columns:
                                agg_dict['Volume'] = 'sum'
                            
                            if regime_timeframe == "1h":
                                # Resample 1min data to 1h
                                market_data = market_data.resample('1h').agg(agg_dict).dropna()
                                st.info(f"📊 Resampled to 1h timeframe: {len(market_data)} candles")
                            elif regime_timeframe == "4h":
                                # Resample 1min data to 4h
                                market_data = market_data.resample('4h').agg(agg_dict).dropna()
                                st.info(f"📊 Resampled to 4h timeframe: {len(market_data)} candles")
                            elif regime_timeframe == "1d":
                                # Resample 1min data to 1d
                                market_data = market_data.resample('1D').agg(agg_dict).dropna()
                                st.info(f"📊 Resampled to 1d timeframe: {len(market_data)} candles")
                            
                            # Initialize analyzer
                            analyzer = SignalAnalyzer()
                            
                            # Step 1: Calculate regime indicators
                            st.info("Step 1/4: Calculating market indicators (ADX, SMA, ATR)...")
                            
                            # Adjust indicator periods based on timeframe to reduce "Unknown" regimes
                            # For daily timeframe, use shorter periods since we have fewer candles
                            if regime_timeframe == "1d":
                                # Daily: Use shorter periods to avoid too many "Unknown" at start
                                sma_short = 20
                                sma_long = 50
                                atr_ma_period = 20
                            elif regime_timeframe == "4h":
                                # 4H: Use moderate periods
                                sma_short = 30
                                sma_long = 100
                                atr_ma_period = 30
                            else:  # 1h
                                # 1H: Use full periods
                                sma_short = 50
                                sma_long = 200
                                atr_ma_period = 50
                            
                            market_with_indicators = analyzer.calculate_regimes(
                                market_df=market_data,
                                adx_period=14,
                                sma_short=sma_short,
                                sma_long=sma_long,
                                atr_period=14,
                                atr_ma_period=atr_ma_period
                            )
                            
                            # Step 2: Define regimes
                            st.info("Step 2/4: Classifying market regimes...")
                            market_with_regimes = analyzer.define_regime(
                                market_df=market_with_indicators,
                                adx_threshold=adx_threshold
                            )
                            
                            # Check for "Unknown" regimes and provide feedback
                            if 'Regime' in market_with_regimes.columns:
                                unknown_count = (market_with_regimes['Regime'] == 'Unknown').sum()
                                total_candles = len(market_with_regimes)
                                if unknown_count > 0:
                                    unknown_pct = (unknown_count / total_candles * 100) if total_candles > 0 else 0
                                    if unknown_pct > 50:
                                        st.warning(f"⚠️ {unknown_pct:.1f}% of candles have 'Unknown' regime. This may be due to insufficient data for indicator calculation. Consider increasing lookback period or using a lower timeframe.")
                                    elif unknown_pct > 10:
                                        st.info(f"ℹ️ {unknown_count} candles ({unknown_pct:.1f}%) have 'Unknown' regime (insufficient data for indicators at start of period). This is normal.")
                            
                            # Step 3: Merge signals with regimes
                            st.info("Step 3/4: Matching signals to market conditions (timezone-aware)...")
                            
                            # Adjust tolerance based on timeframe
                            # For daily candles, we need at least 1 day tolerance
                            # For 4h candles, we need at least 4 hours tolerance
                            # For 1h candles, 1 hour tolerance is fine
                            if regime_timeframe == "1d":
                                merge_tolerance = pd.Timedelta('2 days')  # Allow up to 2 days lookback for daily
                            elif regime_timeframe == "4h":
                                merge_tolerance = pd.Timedelta('8 hours')  # Allow up to 8 hours lookback for 4h
                            else:  # 1h
                                merge_tolerance = pd.Timedelta('2 hours')  # Allow up to 2 hours lookback for 1h
                            
                            signals_with_regimes = analyzer.merge_signals_with_regimes(
                                signals_df=signals_df,
                                market_df=market_with_regimes,
                                entry_time_col='signal_date',
                                timezone='Asia/Dubai',  # Use GMT+4 to match database and signal analysis
                                tolerance=merge_tolerance
                            )
                            
                            # Step 4: Calculate metrics by regime
                            st.info("Step 4/4: Calculating performance metrics by regime...")
                            regime_metrics = analyzer.calculate_metrics_by_regime(
                                signals_df=signals_with_regimes,
                                pnl_col='pips_made',
                                final_status_col='final_status'
                            )
                            
                            if regime_metrics.empty:
                                # Provide detailed diagnostics
                                matched_count = signals_with_regimes['Regime'].notna().sum() if 'Regime' in signals_with_regimes.columns else 0
                                total_signals = len(signals_with_regimes)
                                unmatched_count = total_signals - matched_count
                                
                                st.warning("⚠️ No regime metrics calculated.")
                                st.info(f"""
                                **Diagnostics:**
                                - Total signals: {total_signals}
                                - Signals matched with regime: {matched_count}
                                - Signals without regime: {unmatched_count}
                                
                                **Possible reasons:**
                                1. Signal dates are outside the market data period
                                2. Merge tolerance ({merge_tolerance}) is too strict for {regime_timeframe} timeframe
                                3. Market data gaps or timezone misalignment
                                
                                **Suggestions:**
                                - Try increasing lookback period
                                - Try using a lower timeframe (1h or 4h instead of 1d)
                                - Check if signal dates overlap with market data period
                                """)
                            else:
                                st.success("✅ Regime analysis complete!")
                                
                                # Store in session state
                                st.session_state['regime_metrics'] = regime_metrics
                                st.session_state['signals_with_regimes'] = signals_with_regimes
                
                except Exception as e:
                    st.error(f"❌ Regime analysis failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display regime analysis results
    if 'regime_metrics' in st.session_state and not st.session_state['regime_metrics'].empty:
        st.markdown("---")
        st.markdown("#### 📊 Regime Analysis Results")
        
        regime_metrics = st.session_state['regime_metrics']
        
        # Display metrics table
        st.markdown("##### Performance Metrics by Market Regime")
        st.dataframe(regime_metrics, use_container_width=True)
        
        # Create regime heatmap visualization
        st.markdown("##### 🔥 Win Rate Heatmap")
        
        # Parse regime labels to create heatmap
        regime_metrics['Trend'] = regime_metrics['Regime'].str.split(' - ').str[0]
        regime_metrics['Volatility'] = regime_metrics['Regime'].str.split(' - ').str[1]
        
        # Create pivot table for heatmap
        heatmap_data = regime_metrics.pivot_table(
            values='Win_Rate_%',
            index='Trend',
            columns='Volatility',
            aggfunc='mean'
        )
        
        # Create heatmap using plotly (go is imported at top of file)
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=heatmap_data.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 16},
            colorbar=dict(title="Win Rate %")
        ))
        
        fig.update_layout(
            title="Signal Performance by Market Condition",
            xaxis_title="Volatility",
            yaxis_title="Trend",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("##### 💡 Key Insights")
        
        # Check if we have multiple regimes
        num_regimes = len(regime_metrics)
        
        if num_regimes == 0:
            st.warning("No regime data available for insights.")
            return
        
        # Sort by Win Rate descending to get best, then by Profit Factor as tiebreaker
        regime_metrics_sorted = regime_metrics.sort_values(
            by=['Win_Rate_%', 'Profit_Factor'], 
            ascending=[False, False]
        )
        best_regime = regime_metrics_sorted.iloc[0]
        
        # For worst, filter out the best regime and sort by Win Rate ascending
        regime_metrics_worst = regime_metrics[
            regime_metrics['Regime'] != best_regime['Regime']
        ]
        
        if len(regime_metrics_worst) > 0:
            # Sort by Win Rate ascending, then by Profit Factor ascending (lower is worse)
            worst_regime = regime_metrics_worst.sort_values(
                by=['Win_Rate_%', 'Profit_Factor'], 
                ascending=[True, True]
            ).iloc[0]
        else:
            # If only one regime exists, don't show worst performance
            worst_regime = None
        
        if worst_regime is not None:
            # Show both best and worst if we have multiple regimes
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.success(f"""
                **Best Performance:**
                - Regime: {best_regime['Regime']}
                - Win Rate: {best_regime['Win_Rate_%']:.1f}%
                - Total Trades: {best_regime['Total_Trades']}
                - Profit Factor: {best_regime['Profit_Factor']:.2f}
                """)
            
            with insight_col2:
                st.error(f"""
                **Worst Performance:**
                - Regime: {worst_regime['Regime']}
                - Win Rate: {worst_regime['Win_Rate_%']:.1f}%
                - Total Trades: {worst_regime['Total_Trades']}
                - Profit Factor: {worst_regime['Profit_Factor']:.2f}
                """)
        else:
            # Only one regime - show only best performance
            st.info(f"""
            **📊 Current Performance Analysis:**
            
            Only one market regime detected in your data:
            - **Regime**: {best_regime['Regime']}
            - **Win Rate**: {best_regime['Win_Rate_%']:.1f}%
            - **Total Trades**: {best_regime['Total_Trades']}
            - **Profit Factor**: {best_regime['Profit_Factor']:.2f}
            
            💡 **Tip**: To compare best vs worst performance, you need signals across multiple market regimes (Trending/Ranging × High Vol/Low Vol).
            """)
        
        # Recommendations
        st.markdown("##### 🎯 Trading Recommendations")
        if worst_regime is not None:
            # Multiple regimes - show comparison recommendations
            st.info(f"""
            Based on the regime analysis:
            
            1. **Focus on**: {best_regime['Regime']} conditions where your win rate is highest ({best_regime['Win_Rate_%']:.1f}%)
            2. **Avoid or reduce exposure during**: {worst_regime['Regime']} conditions (only {worst_regime['Win_Rate_%']:.1f}% win rate)
            3. **Sample size check**: Ensure adequate trades in each regime for statistical significance
            """)
        else:
            # Single regime - show general recommendations
            st.info(f"""
            Based on the regime analysis:
            
            1. **Current Performance**: Your signals in {best_regime['Regime']} conditions show a {best_regime['Win_Rate_%']:.1f}% win rate
            2. **Profit Factor**: {best_regime['Profit_Factor']:.2f} - {'Consider reviewing strategy' if best_regime['Profit_Factor'] < 1.0 else 'Strategy shows positive edge'}
            3. **Sample Size**: With only {best_regime['Total_Trades']} trades, collect more data for statistical significance
            4. **Next Steps**: Analyze signals across different market conditions (Trending/Ranging × High Vol/Low Vol) to identify optimal trading environments
            """)
        
        # Export option
        if st.button("📥 Export Regime Analysis to CSV", key="export_regime_csv"):
            csv = regime_metrics.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col_c:
        kpi_options = ["ATR", "Volume", "VWAP", "EMA", "SMA", "RSI", "MACD", "Bollinger Bands", "Stochastic", "Momentum", "Add..."] + (st.session_state.get("kpi_indicators", []) or [])
        selected_kpi = st.selectbox(
            "KPI Indicator",
            options=kpi_options,
            index=0,
            key="select_kpi"
        )
        if selected_kpi == "Add...":
            with st.expander("Upload KPI Indicators via Excel/CSV"):
                st.write("Expected columns: kpi_name")
                kpi_file = st.file_uploader("Choose file", type=["csv", "xls", "xlsx"], key="upload_kpis_file")
                if kpi_file:
                    dfk = _read_df(kpi_file)
                    if dfk is None or dfk.empty:
                        st.error("Unable to read file or no rows found")
                    else:
                        required_cols = ["kpi_name"]
                        if not all(col in dfk.columns for col in required_cols):
                            st.error(f"File must include: {', '.join(required_cols)}")
                        else:
                            st.dataframe(dfk[[c for c in dfk.columns if c in ["kpi_name"]]], use_container_width=True)
                            if st.button("Add KPIs to selection", key="save_kpis_btn"):
                                names = [str(x).strip() for x in dfk["kpi_name"].tolist() if str(x).strip()]
                                st.session_state.kpi_indicators = sorted(set((st.session_state.get("kpi_indicators") or []) + names))
                                st.success(f"Added {len(names)} KPIs")
        
        # KPI Matrix - Connected to Financial Instrument Category and Signal Provider
        if selected_kpi and selected_kpi != "Add...":
            st.markdown("---")
            st.markdown("#### KPI Matrix")
            st.info(f"Calculate {selected_kpi} for selected instrument and provider")
            
            # Use selected instrument from col_a (Financial Instrument Category)
            if selected_category and selected_instrument_item and selected_instrument_item != "Add...":
                # Determine symbol based on category and instrument
                if selected_category == "Commodities" and selected_instrument_item == "GOLD":
                    symbol = "C:XAUUSD"
                    instrument_display = "GOLD (C:XAUUSD)"
                elif selected_category == "Indices" and selected_instrument_item == "S&P 500":
                    symbol = "^SPX"
                    instrument_display = "S&P 500 (^SPX)"
                else:
                    symbol = selected_instrument_item
                    instrument_display = selected_instrument_item
                
                # Use selected provider from col_b (Signal Provider)
                provider_display = selected_provider if selected_provider and selected_provider != "Add..." else "None"
                
                # Show current selections
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.caption(f"**Instrument:** {instrument_display}")
                with col_info2:
                    st.caption(f"**Provider:** {provider_display}")
                
                # Calculate KPI button
                if st.button(f"Calculate {selected_kpi}", key="calculate_kpi_btn", type="primary", use_container_width=True):
                    try:
                        # Fetch data for the selected instrument
                        supabase = get_supabase()
                        if supabase:
                            # Determine table based on category/symbol
                            if selected_category == "Commodities" or "^" in symbol or symbol.upper() == "GOLD":
                                table_name = "market_data_commodities_1min"
                            elif selected_category == "Indices":
                                table_name = "market_data_indices_1min"
                            elif selected_category == "Currencies":
                                table_name = "market_data_currencies_1min"
                            else:
                                table_name = "market_data_stocks_1min"
                            
                            # Fetch recent data (last 200 bars for calculation)
                            # Support multiple symbol formats for fallback
                            symbols_to_try = [symbol]
                            if symbol == "C:XAUUSD":
                                symbols_to_try = ["C:XAUUSD", "^XAUUSD", "GOLD", "XAUUSD"]
                            elif symbol == "^SPX":
                                symbols_to_try = ["^SPX", "S&P 500", "I:SPX"]
                            
                            result_data = []
                            used_symbol = symbol
                            
                            for try_sym in symbols_to_try:
                                query_symbol = try_sym.upper() if not try_sym.startswith("^") else try_sym
                                result = supabase.table(table_name)\
                                    .select("timestamp,open,high,low,close,volume")\
                                    .eq("symbol", query_symbol)\
                                    .order("timestamp", desc=True)\
                                    .limit(200)\
                                    .execute()
                                
                                if result.data and len(result.data) > 0:
                                    result_data = result.data
                                    used_symbol = try_sym
                                    break
                            
                            if result_data and len(result_data) > 0:
                                # Convert to DataFrame
                                df = pd.DataFrame(result_data)
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                df = df.sort_values('timestamp')
                                
                                # Rename columns to match expected format
                                df = df.rename(columns={
                                    'open': 'Open',
                                    'high': 'High',
                                    'low': 'Low',
                                    'close': 'Close',
                                    'volume': 'Volume'
                                })
                                
                                # Map display name to internal name
                                internal_kpi_name = selected_kpi
                                if selected_kpi == "Bollinger Bands":
                                    internal_kpi_name = "BB"
                                
                                # Calculate KPI
                                kpi_result = calculate_kpi(internal_kpi_name, df)
                                
                                if kpi_result is not None:
                                    st.success(f"✓ {selected_kpi} calculated successfully")
                                    
                                    # Data Provenance Proof
                                    with st.expander("🔍 Data Source Verification", expanded=False):
                                        st.write(f"**Source:** Database ({table_name})")
                                        st.write(f"**Symbol Found:** {used_symbol}")
                                        st.write(f"**Records Used:** {len(df)} 1-minute bars")
                                        st.write(f"**Latest Data Point:** {df['timestamp'].max()}")
                                        st.dataframe(df.tail(3)[['timestamp', 'Open', 'Close', 'Volume']], use_container_width=True)

                                    # Display results based on KPI type
                                    if isinstance(kpi_result, dict):
                                        st.markdown("##### KPI Results")
                                        
                                        # Handle different dictionary structures
                                        if "current_volume" in kpi_result:
                                            # Volume
                                            col_res1, col_res2, col_res3 = st.columns(3)
                                            with col_res1:
                                                st.metric("Current Volume", f"{kpi_result.get('current_volume', 0):,.0f}")
                                            with col_res2:
                                                st.metric("Average Volume", f"{kpi_result.get('avg_volume', 0):,.0f}")
                                            with col_res3:
                                                st.metric("Volume Ratio", f"{kpi_result.get('volume_ratio', 0):.2f}")
                                        elif "macd" in kpi_result:
                                            # MACD
                                            col_res1, col_res2, col_res3 = st.columns(3)
                                            with col_res1:
                                                st.metric("MACD Line", f"{kpi_result.get('macd', 0):.4f}")
                                            with col_res2:
                                                st.metric("Signal Line", f"{kpi_result.get('signal', 0):.4f}")
                                            with col_res3:
                                                st.metric("Histogram", f"{kpi_result.get('histogram', 0):.4f}")
                                        elif "upper" in kpi_result:
                                            # Bollinger Bands
                                            col_res1, col_res2, col_res3 = st.columns(3)
                                            with col_res1:
                                                st.metric("Upper Band", f"{kpi_result.get('upper', 0):.4f}")
                                            with col_res2:
                                                st.metric("Middle Band", f"{kpi_result.get('middle', 0):.4f}")
                                            with col_res3:
                                                st.metric("Lower Band", f"{kpi_result.get('lower', 0):.4f}")
                                        elif "k_line" in kpi_result:
                                            # Stochastic
                                            col_res1, col_res2 = st.columns(2)
                                            with col_res1:
                                                st.metric("%K Line", f"{kpi_result.get('k_line', 0):.2f}")
                                            with col_res2:
                                                st.metric("%D Line", f"{kpi_result.get('d_line', 0):.2f}")
                                        else:
                                            # Generic dictionary fallback
                                            st.write(kpi_result)
                                    else:
                                        # Single value result
                                        st.markdown("##### KPI Result")
                                        st.metric(selected_kpi, f"{kpi_result:.4f}" if isinstance(kpi_result, float) else str(kpi_result))
                                    
                                    # Show signal provider context if selected
                                    if provider_display and provider_display != "None":
                                        st.info(f"**Context:** {selected_kpi} calculated for {instrument_display} with signal provider {provider_display}")
                                        
                                        # Fetch and display signals for this provider/symbol
                                        try:
                                            from tradingagents.database.db_service import get_provider_signals
                                            
                                            # Use the same symbol logic as above
                                            query_symbol = symbol.upper() if not symbol.startswith("^") else symbol
                                            
                                            # Fetch signals
                                            prov_signals = get_provider_signals(
                                                symbol=query_symbol, 
                                                provider=provider_display if provider_display != "PipXpert" else "PipXpert", # Handle naming if needed
                                                limit=50
                                            )
                                            
                                            if prov_signals:
                                                with st.expander(f"Recent Signals from {provider_display} ({len(prov_signals)})", expanded=False):
                                                    p_df = pd.DataFrame(prov_signals)
                                                    if not p_df.empty:
                                                        # Select relevant columns
                                                        cols = ['signal_date', 'action', 'entry_price', 'target_1', 'stop_loss']
                                                        disp_cols = [c for c in cols if c in p_df.columns]
                                                        st.dataframe(p_df[disp_cols], use_container_width=True)
                                            else:
                                                st.caption(f"No recent signals found for {query_symbol} from {provider_display}")
                                        except Exception as e:
                                            st.warning(f"Could not fetch signals: {e}")
                                else:
                                    st.warning(f"Could not calculate {selected_kpi}. Please ensure sufficient data is available.")
                            else:
                                st.error(f"No data found for {symbol}. Please ingest data first.")
                        else:
                            st.error("Database connection not available")
                    except Exception as e:
                        st.error(f"Error calculating KPI: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.warning("Please select a Financial Instrument from the Category dropdown above to calculate KPIs.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Trade Efficiency Analysis Section - After Market Regime Analysis
    st.markdown("---")
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Trade Efficiency Analysis (MFE/MAE)")
    st.markdown("""
    Analyze trade execution efficiency by calculating:
    - **MAE (Maximum Adverse Excursion)**: Maximum price movement against the trade
    - **MFE (Maximum Favorable Excursion)**: Maximum price movement in favor of the trade
    """)
    
    with st.expander("📖 About Trade Efficiency Analysis", expanded=False):
        st.markdown("""
        **Trade Efficiency Analysis** helps answer critical questions:
        
        > *"Are stop losses too loose?"* - Analyze MAE to see if trades experienced more pain than necessary
        
        > *"Are we exiting too early?"* - Analyze MFE to see if we're leaving money on the table
        
        > *"What if we used tighter stop losses?"* - Simulate different stop loss scenarios
        
        **Key Metrics:**
        - **MAE (Maximum Adverse Excursion)**: How much did price go against us while trade was open?
        - **MFE (Maximum Favorable Excursion)**: How much did price go in our favor while trade was open?
        - **R-Multiple**: Normalized excursion divided by stop loss distance (MAE_R = 1.0 means price hit stop loss)
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 1: Select Provider and Symbol
    st.markdown("#### Step 1: Select Data Source")
    
    eff_col1, eff_col2 = st.columns(2)
    with eff_col1:
        providers = get_available_providers()
        selected_eff_provider = st.selectbox(
            "Signal Provider",
            options=providers if providers else ["No providers available"],
            key="efficiency_provider"
        )
    
    with eff_col2:
        if selected_eff_provider and selected_eff_provider != "No providers available":
            # Get instruments for this specific provider
            supabase = get_supabase()
            instruments = []
            if supabase:
                try:
                    result = supabase.table('signal_provider_signals')\
                        .select('symbol')\
                        .eq('provider_name', selected_eff_provider)\
                        .execute()
                    if result.data:
                        raw_instruments = [row.get('symbol', '').strip() for row in result.data if row.get('symbol')]
                        # Normalize symbols to remove duplicates (e.g., C:XAUUSD and XAUUSD)
                        normalized_instruments = {}
                        for inst in raw_instruments:
                            inst_upper = inst.upper().strip()
                            if not inst_upper:
                                continue
                            
                            # Handle C: prefix (currencies/commodities)
                            if inst_upper.startswith("C:") and len(inst_upper) > 2:
                                base = inst_upper[2:]  # Remove "C:" prefix
                                # Prefer base symbol without prefix
                                if base not in normalized_instruments:
                                    normalized_instruments[base] = base
                            # Handle ^ prefix (indices)
                            elif inst_upper.startswith("^"):
                                normalized_instruments[inst_upper] = inst_upper
                            # Handle I: prefix (indices)
                            elif inst_upper.startswith("I:"):
                                base = inst_upper[2:]
                                if base not in normalized_instruments:
                                    normalized_instruments[base] = base
                            else:
                                # No prefix - use as-is
                                normalized_instruments[inst_upper] = inst_upper
                        
                        instruments = sorted(list(normalized_instruments.values()))
                except Exception as e:
                    print(f"Error fetching instruments for provider: {e}")
            
            selected_eff_symbol = st.selectbox(
                "Symbol",
                options=instruments if instruments else ["No instruments available"],
                key="efficiency_symbol"
            )
        else:
            selected_eff_symbol = None
    
    # Step 2: Load Backtest Results
    if st.button("Load Trades & Calculate Efficiency", key="load_efficiency_trades"):
        if not selected_eff_provider or selected_eff_provider == "No providers available":
            st.error("Please select a provider")
        elif not selected_eff_symbol or selected_eff_symbol == "No instruments available":
            st.error("Please select a symbol")
        else:
            with st.spinner("Loading trades and calculating efficiency metrics..."):
                try:
                    # First, try to use signal analysis results from session state
                    trades_df = pd.DataFrame()
                    
                    if 'latest_analysis_results' in st.session_state and st.session_state['latest_analysis_results']:
                        analysis_results = st.session_state['latest_analysis_results']
                        
                        # Debug: Show what we have
                        if len(analysis_results) > 0:
                            sample_result = analysis_results[0]
                            available_providers = list(set([r.get('provider_name', '') for r in analysis_results if r.get('provider_name')]))
                            available_symbols = list(set([r.get('symbol', '') for r in analysis_results if r.get('symbol')]))
                            
                            st.info(f"📊 Found {len(analysis_results)} analysis results in session. Providers: {', '.join(available_providers[:3])}, Symbols: {', '.join(available_symbols[:5])}")
                        
                        # Filter by provider and symbol if they match
                        filtered_results = []
                        for result in analysis_results:
                            result_provider = str(result.get('provider_name', '')).strip()
                            result_symbol = str(result.get('symbol', '')).strip()
                            
                            # Normalize symbols for comparison (handle C: prefix and variations)
                            def normalize_symbol(sym):
                                if not sym:
                                    return ''
                                sym_upper = sym.upper().strip()
                                # Remove C: prefix for comparison
                                if sym_upper.startswith('C:'):
                                    return sym_upper[2:]
                                return sym_upper
                            
                            normalized_result_symbol = normalize_symbol(result_symbol)
                            normalized_selected_symbol = normalize_symbol(selected_eff_symbol) if selected_eff_symbol else ''
                            
                            # Normalize providers (case-insensitive, strip whitespace)
                            normalized_result_provider = result_provider.upper().strip()
                            normalized_selected_provider = selected_eff_provider.upper().strip() if selected_eff_provider else ''
                            
                            # Check if matches selected provider and symbol
                            provider_match = (not selected_eff_provider or 
                                            selected_eff_provider == "No providers available" or 
                                            result_provider == selected_eff_provider or
                                            normalized_result_provider == normalized_selected_provider)
                            
                            symbol_match = (not selected_eff_symbol or 
                                          selected_eff_symbol == "No instruments available" or 
                                          result_symbol.upper() == selected_eff_symbol.upper() or
                                          normalized_result_symbol == normalized_selected_symbol)
                            
                            if provider_match and symbol_match:
                                filtered_results.append(result)
                        
                        # If no filtered results but we have analysis results, use all of them with a warning
                        if not filtered_results and analysis_results:
                            st.warning(f"""
                            ⚠️ **No exact match found** for:
                            - Provider: {selected_eff_provider}
                            - Symbol: {selected_eff_symbol}
                            
                            **Using all {len(analysis_results)} analysis results from session.**
                            
                            **Available in session:**
                            - Providers: {', '.join(available_providers[:3])}
                            - Symbols: {', '.join(available_symbols[:5])}
                            
                            **Tip:** Make sure you select the same provider/symbol that you used for Signal Analysis.
                            """)
                            filtered_results = analysis_results
                        
                        if filtered_results:
                            # Convert analysis results to backtest-like format
                            converted_trades = []
                            for result in filtered_results:
                                # Determine exit datetime and exit price
                                exit_datetime = None
                                exit_price = None
                                
                                # Check which TP/SL was hit (in order of priority)
                                if result.get('tp3_hit_datetime'):
                                    exit_datetime = result.get('tp3_hit_datetime')
                                    exit_price = result.get('target_3')
                                elif result.get('tp2_hit_datetime'):
                                    exit_datetime = result.get('tp2_hit_datetime')
                                    exit_price = result.get('target_2')
                                elif result.get('tp1_hit_datetime'):
                                    exit_datetime = result.get('tp1_hit_datetime')
                                    exit_price = result.get('target_1')
                                elif result.get('sl_hit_datetime'):
                                    exit_datetime = result.get('sl_hit_datetime')
                                    exit_price = result.get('stop_loss')
                                
                                # If no exit datetime, use a default (expired/open trades)
                                if not exit_datetime:
                                    # For expired/open trades, use signal_date + 30 days as exit
                                    import pytz
                                    gmt4_tz = pytz.timezone('Asia/Dubai')
                                    signal_date = pd.to_datetime(result.get('signal_date'))
                                    if signal_date.tzinfo is None:
                                        signal_date = gmt4_tz.localize(signal_date)
                                    elif signal_date.tzinfo != gmt4_tz:
                                        signal_date = signal_date.astimezone(gmt4_tz)
                                    exit_datetime = signal_date + pd.Timedelta(days=30)
                                    # Use entry price as exit price for expired trades
                                    exit_price = result.get('entry_price')
                                
                                # Calculate profit_loss from pips_made if available
                                profit_loss = result.get('pips_made', 0)
                                if profit_loss == 0 and exit_price and result.get('entry_price'):
                                    # Calculate manually
                                    is_buy = result.get('action', '').upper() == 'BUY'
                                    if is_buy:
                                        profit_loss = (exit_price - result.get('entry_price')) * 10000  # Convert to pips
                                    else:
                                        profit_loss = (result.get('entry_price') - exit_price) * 10000
                                
                                # Parse and ensure timestamps are in GMT+4
                                import pytz
                                gmt4_tz = pytz.timezone('Asia/Dubai')
                                
                                entry_dt = pd.to_datetime(result.get('signal_date'))
                                if entry_dt.tzinfo is None:
                                    entry_dt = gmt4_tz.localize(entry_dt)
                                elif entry_dt.tzinfo != gmt4_tz:
                                    entry_dt = entry_dt.astimezone(gmt4_tz)
                                
                                exit_dt = pd.to_datetime(exit_datetime)
                                if exit_dt.tzinfo is None:
                                    exit_dt = gmt4_tz.localize(exit_dt)
                                elif exit_dt.tzinfo != gmt4_tz:
                                    exit_dt = exit_dt.astimezone(gmt4_tz)
                                
                                trade_dict = {
                                    'id': result.get('signal_id'),  # Use signal_id as identifier
                                    'provider_name': result.get('provider_name'),
                                    'symbol': result.get('symbol'),
                                    'entry_datetime': entry_dt,
                                    'exit_datetime': exit_dt,
                                    'entry_price': result.get('entry_price'),
                                    'exit_price': exit_price,
                                    'direction': result.get('action', 'BUY').upper(),
                                    'stop_loss': result.get('stop_loss'),
                                    'profit_loss': profit_loss,
                                    'net_profit_loss': profit_loss,
                                    'final_status': result.get('final_status', 'EXPIRED')
                                }
                                converted_trades.append(trade_dict)
                            
                            if converted_trades:
                                trades_df = pd.DataFrame(converted_trades)
                                st.success(f"✅ Using {len(trades_df)} trades from signal analysis results")
                            else:
                                st.warning(f"⚠️ Could not convert {len(filtered_results)} analysis results to trades format")
                        else:
                            if analysis_results:
                                st.warning(f"⚠️ Found {len(analysis_results)} analysis results in session, but none matched Provider: {selected_eff_provider}, Symbol: {selected_eff_symbol}")
                            else:
                                st.info("ℹ️ No analysis results found in session state")
                    
                    # If no session state results, try database
                    if trades_df.empty:
                        trades_df = get_backtest_results_with_efficiency(
                            provider_name=selected_eff_provider,
                            symbol=selected_eff_symbol,
                            limit=1000
                        )
                    
                    if trades_df.empty:
                        # Check if there are any backtest results at all (without filters)
                        all_results = get_backtest_results_with_efficiency(limit=10)
                        
                        # Also check session state one more time for debugging
                        has_session_results = 'latest_analysis_results' in st.session_state and st.session_state.get('latest_analysis_results')
                        session_count = len(st.session_state.get('latest_analysis_results', [])) if has_session_results else 0
                        
                        if all_results.empty and not has_session_results:
                            st.warning("""
                            **No backtest results or signal analysis results found.**
                            
                            To use Trade Efficiency Analysis:
                            
                            1. Go to the **Signal Analysis** section in Phase 1 (in col_b)
                            2. Select a provider and symbol
                            3. Click **"Run Analysis"** to generate analysis results
                            4. **Keep the same provider/symbol selected** when you come here
                            5. Click **"Load Trades & Calculate Efficiency"** (results will be used from session)
                            
                            **Note:** Make sure you run analysis in the same browser session and don't refresh the page.
                            
                            Or run backtesting to save results to database.
                            """)
                        elif has_session_results:
                            st.error(f"""
                            **Found {session_count} analysis results in session, but couldn't match them.**
                            
                            **Selected:**
                            - Provider: {selected_eff_provider}
                            - Symbol: {selected_eff_symbol}
                            
                            **Please ensure:**
                            1. You selected the SAME provider and symbol that you used for Signal Analysis
                            2. The analysis completed successfully (check the Signal Analysis section)
                            3. You haven't refreshed the page (which clears session state)
                            
                            Try running Signal Analysis again with the same provider/symbol, then come back here.
                            """)
                        else:
                            # Show what's available
                            available_providers = all_results['provider_name'].unique() if 'provider_name' in all_results.columns else []
                            available_symbols = all_results['symbol'].unique() if 'symbol' in all_results.columns else []
                            
                            st.warning(f"""
                            **No backtest results found for:**
                            - Provider: {selected_eff_provider}
                            - Symbol: {selected_eff_symbol}
                            
                            **Available in database:**
                            - Providers: {', '.join(available_providers[:5])}{'...' if len(available_providers) > 5 else ''}
                            - Symbols: {', '.join(available_symbols[:5])}{'...' if len(available_symbols) > 5 else ''}
                            
                            Please select a provider/symbol combination that has backtest results, or run signal analysis first.
                            """)
                    else:
                        # Check if efficiency metrics already exist
                        needs_calculation = True
                        if 'mfe' in trades_df.columns and 'mae' in trades_df.columns:
                            if not trades_df['mfe'].isna().all() and not trades_df['mae'].isna().all():
                                needs_calculation = False
                        
                        if needs_calculation:
                            st.info("Calculating MFE/MAE for all trades...")
                            
                            # Initialize analyzer
                            analyzer = TradeEfficiencyAnalyzer()
                            
                            # Determine asset class
                            asset_class = None
                            if selected_eff_symbol.startswith("C:") or '/' in selected_eff_symbol:
                                asset_class = "Currencies"
                            elif selected_eff_symbol.startswith("I:") or selected_eff_symbol.startswith("^"):
                                asset_class = "Indices"
                            elif "*" in selected_eff_symbol or "XAU" in selected_eff_symbol or "XAG" in selected_eff_symbol:
                                asset_class = "Commodities"
                            else:
                                asset_class = "Stocks"
                            
                            # Calculate efficiency
                            trades_df = analyzer.calculate_excursions_batch(
                                trades_df=trades_df,
                                symbol=selected_eff_symbol,
                                interval='1min',
                                asset_class=asset_class
                            )
                            
                            # Save to database
                            for idx, row in trades_df.iterrows():
                                if 'id' in row and pd.notna(row.get('mfe')) and pd.notna(row.get('mae')):
                                    update_backtest_efficiency(
                                        backtest_id=int(row['id']),
                                        mfe=float(row.get('mfe', 0)),
                                        mae=float(row.get('mae', 0)),
                                        mfe_pips=float(row.get('mfe_pips', 0)),
                                        mae_pips=float(row.get('mae_pips', 0)),
                                        mae_r=float(row.get('mae_r')) if pd.notna(row.get('mae_r')) else None,
                                        mfe_r=float(row.get('mfe_r')) if pd.notna(row.get('mfe_r')) else None,
                                        confidence=str(row.get('confidence', 'HIGH'))
                                    )
                            
                            st.success(f"✅ Calculated efficiency metrics for {len(trades_df)} trades")
                        else:
                            st.success(f"✅ Loaded {len(trades_df)} trades with existing efficiency metrics")
                        
                        # Store in session state (use different keys to avoid widget conflict)
                        st.session_state['efficiency_trades_df'] = trades_df
                        st.session_state['efficiency_provider_selected'] = selected_eff_provider
                        st.session_state['efficiency_symbol_selected'] = selected_eff_symbol
                        
                except Exception as e:
                    st.error(f"❌ Error calculating efficiency: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Step 3: Display Efficiency Analysis
    if 'efficiency_trades_df' in st.session_state:
        trades_df = st.session_state['efficiency_trades_df']
        
        if not trades_df.empty:
            st.markdown("---")
            st.markdown("#### Step 2: Efficiency Metrics Overview")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_mae = trades_df['mae_pips'].mean() if 'mae_pips' in trades_df.columns else 0
                st.metric("Average MAE", f"{avg_mae:.2f} pips")
            with col2:
                avg_mfe = trades_df['mfe_pips'].mean() if 'mfe_pips' in trades_df.columns else 0
                st.metric("Average MFE", f"{avg_mfe:.2f} pips")
            with col3:
                avg_mae_r = trades_df['mae_r'].mean() if 'mae_r' in trades_df.columns and not trades_df['mae_r'].isna().all() else 0
                st.metric("Average MAE_R", f"{avg_mae_r:.2f}R" if avg_mae_r > 0 else "N/A")
            with col4:
                low_confidence = (trades_df['confidence'] == 'LOW').sum() if 'confidence' in trades_df.columns else 0
                st.metric("Low Confidence Trades", f"{low_confidence}")
            
            # Step 4: Efficiency Map (Scatter Plot)
            st.markdown("---")
            st.markdown("#### Step 3: Efficiency Map")
            st.markdown("""
            **Visual Insight**: 
            - X-axis: MAE (Pain) - How much price went against us
            - Y-axis: MFE (Potential) - How much price went in our favor
            - Green dots: Winning trades
            - Red dots: Losing trades
            """)
            
            # Prepare data for scatter plot
            profit_col = 'profit_loss' if 'profit_loss' in trades_df.columns else 'net_profit_loss'
            if profit_col in trades_df.columns:
                winning_trades = trades_df[trades_df[profit_col] > 0]
                losing_trades = trades_df[trades_df[profit_col] <= 0]
            else:
                winning_trades = pd.DataFrame()
                losing_trades = trades_df
            
            # Create scatter plot using plotly (go is imported at top of file)
            fig = go.Figure()
            
            # Add winning trades (green)
            if not winning_trades.empty and 'mae_pips' in winning_trades.columns and 'mfe_pips' in winning_trades.columns:
                fig.add_trace(go.Scatter(
                    x=winning_trades['mae_pips'],
                    y=winning_trades['mfe_pips'],
                    mode='markers',
                    name='Winning Trades',
                    marker=dict(
                        color='green',
                        size=8,
                        opacity=0.6,
                        line=dict(width=1, color='darkgreen')
                    ),
                    text=winning_trades.get('symbol', ''),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'MAE: %{x:.2f} pips<br>' +
                                  'MFE: %{y:.2f} pips<br>' +
                                  '<extra></extra>'
                ))
            
            # Add losing trades (red)
            if not losing_trades.empty and 'mae_pips' in losing_trades.columns and 'mfe_pips' in losing_trades.columns:
                fig.add_trace(go.Scatter(
                    x=losing_trades['mae_pips'],
                    y=losing_trades['mfe_pips'],
                    mode='markers',
                    name='Losing Trades',
                    marker=dict(
                        color='red',
                        size=8,
                        opacity=0.6,
                        line=dict(width=1, color='darkred')
                    ),
                    text=losing_trades.get('symbol', ''),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'MAE: %{x:.2f} pips<br>' +
                                  'MFE: %{y:.2f} pips<br>' +
                                  '<extra></extra>'
                ))
            
            fig.update_layout(
                title="Trade Efficiency Map (MFE vs MAE)",
                xaxis_title="MAE (Maximum Adverse Excursion) - Pips",
                yaxis_title="MFE (Maximum Favorable Excursion) - Pips",
                height=600,
                hovermode='closest',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Step 5: Stop Loss Optimizer
            st.markdown("---")
            st.markdown("#### Step 4: Stop Loss Optimizer")
            st.markdown("""
            **Simulate tighter stop losses** to see how it would affect overall PnL.
            """)
            
            # Get max MAE for slider range
            if 'mae_pips' in trades_df.columns:
                max_mae = trades_df['mae_pips'].max()
                
                # Calculate current average stop loss distance
                if 'entry_price' in trades_df.columns and 'stop_loss' in trades_df.columns:
                    analyzer = TradeEfficiencyAnalyzer()
                    current_avg_sl = trades_df.apply(
                        lambda row: abs(row['entry_price'] - row['stop_loss']) / analyzer._calculate_pip_value(row['entry_price'], st.session_state.get('efficiency_symbol_selected', selected_eff_symbol if 'selected_eff_symbol' in locals() else '')),
                        axis=1
                    ).mean() if not trades_df.empty else max_mae
                else:
                    current_avg_sl = max_mae
                
                # Slider for proposed stop loss
                proposed_sl = st.slider(
                    "Proposed Stop Loss (Pips)",
                    min_value=0.0,
                    max_value=float(max_mae * 1.5) if max_mae > 0 else 100.0,
                    value=float(current_avg_sl) if current_avg_sl > 0 else float(max_mae * 0.5) if max_mae > 0 else 50.0,
                    step=1.0,
                    key="sl_optimizer_slider"
                )
                
                # Calculate optimization
                if st.button("Calculate Optimization", key="calc_optimization"):
                    analyzer = TradeEfficiencyAnalyzer()
                    optimization_result = analyzer.simulate_stop_loss_optimization(
                        trades_df,
                        proposed_sl
                    )
                    
                    st.session_state['optimization_result'] = optimization_result
                
                # Display optimization results
                if 'optimization_result' in st.session_state:
                    opt_result = st.session_state['optimization_result']
                    
                    opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
                    with opt_col1:
                        st.metric(
                            "Original Total PnL",
                            f"${opt_result['original_total_pnl']:,.2f}"
                        )
                    with opt_col2:
                        delta_val = opt_result['pnl_difference']
                        st.metric(
                            "Projected Total PnL",
                            f"${opt_result['projected_total_pnl']:,.2f}",
                            delta=f"${delta_val:,.2f}"
                        )
                    with opt_col3:
                        st.metric(
                            "Trades Stopped Out",
                            f"{opt_result['stopped_out_trades']} / {opt_result['total_trades']}"
                        )
                    with opt_col4:
                        win_rate_delta = opt_result['win_rate_projected'] - opt_result['win_rate_original']
                        st.metric(
                            "Projected Win Rate",
                            f"{opt_result['win_rate_projected']:.1f}%",
                            delta=f"{win_rate_delta:.1f}%"
                        )
                    
                    # Interpretation
                    if opt_result['pnl_difference'] < 0:
                        st.warning(f"⚠️ Tighter stop loss would reduce PnL by ${abs(opt_result['pnl_difference']):,.2f}")
                    else:
                        st.success(f"✅ Tighter stop loss would improve PnL by ${opt_result['pnl_difference']:,.2f}")
            
            # Step 6: Detailed Table
            st.markdown("---")
            st.markdown("#### Step 5: Detailed Efficiency Table")
            
            # Filter options
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                show_only_low_confidence = st.checkbox("Show only low confidence trades", key="filter_confidence")
            with filter_col2:
                min_mae_filter = st.number_input("Minimum MAE (pips)", min_value=0.0, value=0.0, key="filter_mae")
            
            # Apply filters
            display_df = trades_df.copy()
            if show_only_low_confidence and 'confidence' in display_df.columns:
                display_df = display_df[display_df['confidence'] == 'LOW']
            if min_mae_filter > 0 and 'mae_pips' in display_df.columns:
                display_df = display_df[display_df['mae_pips'] >= min_mae_filter]
            
            # Select columns to display
            columns_to_show = [
                'symbol', 'entry_datetime', 'exit_datetime', 
                'entry_price', 'exit_price', 'profit_loss', 'net_profit_loss',
                'mae_pips', 'mfe_pips', 'mae_r', 'mfe_r', 'confidence'
            ]
            available_columns = [col for col in columns_to_show if col in display_df.columns]
            
            if available_columns:
                st.dataframe(
                    display_df[available_columns],
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No efficiency columns available to display")
    
    # Data ingestion
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Data Ingestion Pipeline")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Ingest All Historical Daily Data (Oct 1, 2023 → Now)",use_container_width=True, key="ingest_all_data"):
            pipeline = DataIngestionPipeline()
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            symbols = get_watchlist_symbols()
            results = {}
            
            # Explicit date range start with dynamic end -> run time
            start_date = datetime(2023, 10, 1)
            end_date = datetime.utcnow()
            
            for i, symbol in enumerate(symbols):
                status_text.text(f"Processing {symbol}... ({i+1}/{len(symbols)})")
                progress_bar.progress((i + 1) / len(symbols))
                
                result = pipeline.ingest_historical_data(
                    symbol, 
                    interval='daily',
                    start_date=start_date,
                    end_date=end_date
                )
                results[symbol] = result
                
                if result:
                    st.success(f"✅ {symbol} processed successfully")
                else:
                    st.warning(f"⚠️ {symbol} had issues (check console for details)")
            
            pipeline.close()
            
            # Final results
            success_count = sum(1 for success in results.values() if success)
            failed_count = len(results) - success_count
            
            if success_count > 0:
                st.success(f"✅ Successfully processed {success_count}/{len(results)} stocks")
                st.balloons()
                
                # Automatically save ingestion metadata to database
                try:
                    from tradingagents.database.db_service import update_data_health, log_event
                    
                    # Update data health for each symbol
                    for symbol, success in results.items():
                        update_data_health(
                            symbol=symbol,
                            data_fetch_status={"status": "success" if success else "failed", "source": "phase1_ingestion"},
                            health_score=10.0 if success else 0.0
                        )
                    
                    # Log completion
                    log_event("data_ingestion_completed", {
                        "success_count": success_count,
                        "failed_count": failed_count,
                        "total": len(results)
                    })
                    st.info("✅ Ingestion metadata automatically saved to database (data health & event logs)")
                except Exception as e:
                    st.warning(f"⚠️ Market data saved successfully, but failed to save metadata: {str(e)}")
            else:
                st.error(f"❌ Failed to process all {len(results)} stocks")
            
            if failed_count > 0:
                st.warning(f"⚠️ {failed_count} stocks had issues")
                failed_stocks = [symbol for symbol, success in results.items() if not success]
                st.write("Stocks with issues:", ", ".join(failed_stocks))
            
            # Show detailed results
            st.subheader("Detailed Results")
            results_df = pd.DataFrame([
                {"Symbol": symbol, "Status": "✅ Success" if success else "⚠️ Issues"}
                for symbol, success in results.items()
            ])
            st.dataframe(results_df, use_container_width=True)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

    # 5-minute intraday ingestion (Oct 1, 2023 - Oct 31, 2025) with resume capability
    with col1:
        if st.button("Ingest All Historical 5-min Data (Oct 1, 2023 → Now)", use_container_width=True, key="ingest_all_5min"):
            pipeline = DataIngestionPipeline()
            progress_bar = st.progress(0)
            status_text = st.empty()

            symbols = get_watchlist_symbols()
            results_5min = {}
            
            # Explicit date range start with dynamic end -> run time
            start_date = datetime(2023, 10, 1)
            now_utc = datetime.utcnow()
            end_date = max(start_date, now_utc - timedelta(minutes=15))

            for i, symbol in enumerate(symbols):
                status_text.text(f"Processing 5-min {symbol}... ({i+1}/{len(symbols)})")
                progress_bar.progress((i + 1) / len(symbols))

                # 2 years of 5-min data with resume capability, chunk_days=30 for faster ingestion
                result = pipeline.ingest_historical_data(
                    symbol, 
                    interval='5min', 
                    chunk_days=30,
                    start_date=start_date,
                    end_date=end_date
                )
                results_5min[symbol] = result

                if result:
                    st.success(f"✅ 5-min {symbol} processed successfully")
                else:
                    st.warning(f"⚠️ 5-min {symbol} had issues (check console for details)")

            pipeline.close()

            success_count = sum(1 for success in results_5min.values() if success)
            failed_count = len(results_5min) - success_count

            if success_count > 0:
                st.success(f"✅ Successfully processed 5-min data for {success_count}/{len(results_5min)} stocks")
                st.balloons()
            else:
                st.error(f"❌ Failed to process 5-min data for all {len(results_5min)} stocks")

            if failed_count > 0:
                st.warning(f"⚠️ {failed_count} stocks had issues for 5-min ingestion")
                failed_stocks = [symbol for symbol, success in results_5min.items() if not success]
                st.write("Stocks with issues:", ", ".join(failed_stocks))

            # Show detailed results
            st.subheader("5-min Detailed Results")
            results_df = pd.DataFrame([
                {"Symbol": symbol, "Status": "✅ Success" if success else "⚠️ Issues"}
                for symbol, success in results_5min.items()
            ])
            st.dataframe(results_df, use_container_width=True)

            progress_bar.empty()
            status_text.empty()

    # New: 1-minute intraday ingestion (Aug-Oct 2025)
    with col1:
        if st.button("Ingest 1-min Data (Aug 2025 → Now)", use_container_width=True, key="ingest_all_1min"):
            pipeline = DataIngestionPipeline()
            progress_bar = st.progress(0)
            status_text = st.empty()

            symbols = get_watchlist_symbols()
            results_1min = {}
            start_range = datetime(2025, 8, 1)
            # Ingest up to current time when button is clicked
            # Set end_date to tomorrow to ensure today's data is included
            # (ingestion pipeline normalizes to midnight, so tomorrow ensures today is processed)
            now_utc = datetime.utcnow()
            end_range = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

            for i, symbol in enumerate(symbols):
                status_text.text(f"Processing 1-min {symbol}... ({i+1}/{len(symbols)})")
                progress_bar.progress((i + 1) / len(symbols))

                # Ingest 3 months of 1-min data (Aug-Oct 2025)
                result = pipeline.ingest_historical_data(
                    symbol,
                    interval='1min',
                    chunk_days=3,
                    start_date=start_range,
                    end_date=end_range
                )
                results_1min[symbol] = result

                if result:
                    st.success(f"✅ 1-min {symbol} processed successfully (Aug-Oct 2025)")
                else:
                    st.warning(f"⚠️ 1-min {symbol} had issues (check console for details)")

            pipeline.close()

            success_count = sum(1 for success in results_1min.values() if success)
            failed_count = len(results_1min) - success_count

            if success_count > 0:
                st.success(f"✅ Successfully processed 1-min data (Aug-Oct 2025) for {success_count}/{len(results_1min)} stocks")
                st.balloons()
            else:
                st.error(f"❌ Failed to process 1-min data for all {len(results_1min)} stocks")

            if failed_count > 0:
                st.warning(f"⚠️ {failed_count} stocks had issues for 1-min ingestion")
                failed_stocks = [symbol for symbol, success in results_1min.items() if not success]
                st.write("Stocks with issues:", ", ".join(failed_stocks))

            # Show detailed results
            st.subheader("1-min Detailed Results (Aug-Oct 2025)")
            results_df = pd.DataFrame([
                {"Symbol": symbol, "Status": "✅ Success" if success else "⚠️ Issues"}
                for symbol, success in results_1min.items()
            ])
            st.dataframe(results_df, use_container_width=True)

            progress_bar.empty()
            status_text.empty()
    
    with col2:
        if st.button("Check Data Status", use_container_width=True, key="main_data_status"):
            sb = get_supabase()
            if sb:
                # Supabase path: compute counts per symbol from market_data
                rows = []
                for symbol in WATCHLIST_STOCKS.keys():
                    try:
                        resp = sb.rpc("get_symbol_stats", {"p_symbol": symbol}).execute()
                        # If RPC not available, fallback to simple count query
                    except Exception:
                        resp = sb.table("market_data").select("date").eq("symbol", symbol).execute()
                    data = resp.data if hasattr(resp, "data") else []
                    count = len(data) if isinstance(data, list) else 0
                    rows.append({
                        "Symbol": symbol,
                        "Records": count,
                        "Latest Date": "-",
                        "Completion %": f"{min(100, (count/1260)*100):.1f}%"  # 5 years = 1260 trading days
                    })
                status_df = pd.DataFrame(rows)
                st.dataframe(status_df, use_container_width=True)

                # Also show 5-min data counts if table exists
                rows_5min = []
                for symbol in WATCHLIST_STOCKS.keys():
                    try:
                        resp5 = sb.table("market_data_5min").select("timestamp").eq("symbol", symbol).limit(1).execute()
                        # If query succeeds, get count via RPC or select
                        try:
                            resp5c = sb.rpc("get_symbol_stats_5min", {"p_symbol": symbol}).execute()
                            data5 = resp5c.data if hasattr(resp5c, "data") else []
                            count5 = len(data5) if isinstance(data5, list) else 0
                        except Exception:
                            resp5all = sb.table("market_data_5min").select("timestamp").eq("symbol", symbol).execute()
                            data5all = resp5all.data if hasattr(resp5all, "data") else []
                            count5 = len(data5all) if isinstance(data5all, list) else 0
                    except Exception:
                        count5 = 0
                    rows_5min.append({
                        "Symbol": symbol,
                        "Records (5min)": count5,
                        "Latest Timestamp": "-"
                    })
                status_df_5min = pd.DataFrame(rows_5min)
                st.subheader("5-minute data status")
                st.dataframe(status_df_5min, use_container_width=True)

                # Also show 1-min data counts if table exists
                rows_1min = []
                for symbol in WATCHLIST_STOCKS.keys():
                    try:
                        resp1 = sb.table("market_data_stocks_1min").select("timestamp").eq("symbol", symbol).limit(1).execute()
                        # If query succeeds, get count via RPC or select
                        try:
                            resp1c = sb.rpc("get_symbol_stats_1min", {"p_symbol": symbol}).execute()
                            data1 = resp1c.data if hasattr(resp1c, "data") else []
                            count1 = len(data1) if isinstance(data1, list) else 0
                        except Exception:
                            resp1all = sb.table("market_data_stocks_1min").select("timestamp").eq("symbol", symbol).execute()
                            data1all = resp1all.data if hasattr(resp1all, "data") else []
                            count1 = len(data1all) if isinstance(data1all, list) else 0
                    except Exception:
                        count1 = 0
                    rows_1min.append({
                        "Symbol": symbol,
                        "Records (1min)": count1,
                        "Latest Timestamp": "-"
                    })
                status_df_1min = pd.DataFrame(rows_1min)
                st.subheader("1-minute data status")
                st.dataframe(status_df_1min, use_container_width=True)
            else:
                pipeline = DataIngestionPipeline()
                status = pipeline.get_data_completion_status()
                pipeline.close()
                status_df = pd.DataFrame([
                    {
                        "Symbol": symbol,
                        "Records": info["historical_records"],
                        "Latest Date": info["latest_date"].strftime("%Y-%m-%d") if info["latest_date"] else "N/A",
                        "Completion %": f"{info['completion_percentage']:.1f}%"
                    }
                    for symbol, info in status.items()
                ])
                st.dataframe(status_df, use_container_width=True)
    
    st.markdown("---")
    with st.expander("🛠️ Manual Gap Fixer (Advanced)", expanded=False):
        st.write("Manually ingest a specific date range to fix gaps in the middle of your data.")
        
        mf_col1, mf_col2, mf_col3, mf_col4 = st.columns(4)
        with mf_col1:
            fix_symbol = st.selectbox("Symbol", options=list(WATCHLIST_STOCKS.keys()), key="fix_symbol")
        with mf_col2:
            fix_interval = st.selectbox("Interval", ["daily", "5min", "1min"], key="fix_interval")
        with mf_col3:
            fix_start = st.date_input("Start Date", value=datetime(2023, 6, 1), key="fix_start")
        with mf_col4:
            fix_end = st.date_input("End Date", value=datetime(2023, 6, 30), key="fix_end")
            
        if st.button("Run Targeted Ingestion", key="fix_ingest_btn"):
            st.info(f"Starting manual ingestion for {fix_symbol} ({fix_interval}) from {fix_start} to {fix_end}...")
            
            # Convert dates to datetime
            start_dt = datetime.combine(fix_start, datetime.min.time())
            # For end date, we want the end of that day
            end_dt = datetime.combine(fix_end, datetime.max.time().replace(microsecond=0))
            
            pipeline = DataIngestionPipeline()
            try:
                # Use resume_from_latest=False to force check this specific range
                # regardless of what data exists after it.
                # The pipeline handles duplicate detection automatically.
                success = pipeline.ingest_historical_data(
                    symbol=fix_symbol,
                    interval=fix_interval,
                    start_date=start_dt,
                    end_date=end_dt,
                    resume_from_latest=False
                )
                
                if success:
                    st.success(f"✅ Successfully ingested data for {fix_symbol}")
                else:
                    st.error(f"❌ Ingestion failed for {fix_symbol} (check logs)")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                pipeline.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # Data coverage guardrail & feature lab
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Data Coverage Guardrail & Feature Lab")
    coverage_service = DataCoverageService()
    feature_lab = FeatureLab()

    coverage_description = """
    **Required Coverage Windows**
    - **1-minute**: Aug 1, 2025 → current time minus 15 minutes (EAT reference)
    - **5-minute**: Oct 1, 2023 → current time minus 15 minutes
    - **Daily**: Oct 1, 2023 → previous trading day close

    Use the buttons below to audit the Supabase tables and backfill any gaps
    before triggering downstream analysis or LLM prompts.
    """
    st.info(coverage_description)

    coverage_col1, coverage_col2 = st.columns(2)
    with coverage_col1:
        if st.button("🔍 Run Coverage Audit", key="coverage_audit"):
            with st.spinner("Checking Supabase tables for required ranges..."):
                report_records = coverage_service.build_watchlist_report()
                st.session_state.coverage_records = report_records
                st.session_state.coverage_rows = [rec.to_dict() for rec in report_records]
                st.success("Coverage audit completed")
    with coverage_col2:
        records = st.session_state.get("coverage_records") or []
        missing_records = [rec for rec in records if rec.needs_backfill()]
        if st.button("🛠️ Auto Backfill Missing Data", key="coverage_backfill"):
            if not records:
                st.warning("Run the coverage audit first to identify gaps.")
            elif not missing_records:
                st.success("No gaps detected. Nothing to backfill.")
            else:
                # Check API key before attempting backfill
                import os
                polygon_key = os.getenv("POLYGON_API_KEY")
                if not polygon_key:
                    st.error("❌ POLYGON_API_KEY not found in environment variables. Cannot backfill data.")
                    st.info("💡 Please add POLYGON_API_KEY to your .env file. Get a key from: https://polygon.io/dashboard/api-keys")
                else:
                    with st.spinner("Running targeted ingestion jobs..."):
                        try:
                            pipeline = DataIngestionPipeline()
                            logs = coverage_service.backfill_missing(pipeline, missing_records)
                            pipeline.close()
                            st.session_state.coverage_backfill_logs = logs
                            
                            # Check for 401 errors in logs
                            has_401_error = False
                            for log in logs:
                                if isinstance(log, dict) and "error" in str(log.get("message", "")).lower():
                                    if "401" in str(log.get("message", "")) or "unauthorized" in str(log.get("message", "")).lower():
                                        has_401_error = True
                                        break
                            
                            if has_401_error:
                                st.error("❌ Backfill failed: Polygon API authentication error (401 Unauthorized)")
                                st.warning("💡 Your POLYGON_API_KEY appears to be invalid or expired. Please:")
                                st.markdown("""
                                1. Check your API key at: https://polygon.io/dashboard/api-keys
                                2. Update your `.env` file with: `POLYGON_API_KEY=your_valid_key_here`
                                3. Restart the Streamlit app
                                """)
                            else:
                                success_count = sum(1 for log in logs if log.get("success", False))
                                total_count = len(logs)
                                if success_count > 0:
                                    st.success(f"✅ Backfill completed: {success_count}/{total_count} tasks succeeded")
                                else:
                                    st.warning(f"⚠️ Backfill completed but no tasks succeeded. Check logs below for details.")
                        except ValueError as e:
                            # Catch 401 errors raised from polygon_integration
                            error_msg = str(e)
                            if "401" in error_msg or "unauthorized" in error_msg.lower():
                                st.error("❌ Polygon API Authentication Failed")
                                st.warning("💡 Your POLYGON_API_KEY is invalid or expired. Please:")
                                st.markdown("""
                                1. Get a valid API key from: https://polygon.io/dashboard/api-keys
                                2. Update your `.env` file: `POLYGON_API_KEY=your_valid_key_here`
                                3. Restart the Streamlit app
                                """)
                            else:
                                st.error(f"❌ Backfill error: {error_msg}")
                        except Exception as e:
                            error_msg = str(e)
                            if "401" in error_msg or "unauthorized" in error_msg.lower():
                                st.error("❌ Polygon API Authentication Failed (401 Unauthorized)")
                                st.warning("💡 Please check your POLYGON_API_KEY in the .env file")
                            else:
                                st.error(f"❌ Backfill failed: {error_msg}")

    coverage_rows = st.session_state.get("coverage_rows")
    if coverage_rows:
        coverage_df = pd.DataFrame(coverage_rows)
        st.dataframe(coverage_df, use_container_width=True, hide_index=True)
        statuses = coverage_df["status"].value_counts().to_dict()
        st.caption(f"Status summary: {statuses}")

    backfill_logs = st.session_state.get("coverage_backfill_logs")
    if backfill_logs:
        log_df = pd.DataFrame(backfill_logs)
        st.markdown("#### Backfill Activity")
        st.dataframe(log_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### QUANTUMTRADER v0.1 Prompt")
    symbols_for_lab = get_watchlist_symbols()
    
    # Auto-select the instrument chosen in the top section
    lab_index = 0
    if selected_instrument_item and selected_instrument_item != "Add...":
        # Add to list if not present
        if selected_instrument_item not in symbols_for_lab:
             symbols_for_lab.insert(0, selected_instrument_item)
             lab_index = 0
        else:
             lab_index = symbols_for_lab.index(selected_instrument_item)
             
    lab_symbol = st.selectbox("Symbol", symbols_for_lab, index=lab_index, key="quantum_symbol")
    
    # Command timestamp is now auto-stamped (label only, not a filter)
    with st.expander("ℹ️ Command Timestamp (Auto Label)", expanded=False):
        st.markdown("""
        **Auto label only — never a data filter.**
        
        The system always fetches the most recent data in the database. The
        command timestamp is just stamped into the prompt output so you can see
        when you ran the command. No manual input is required.
        """)
    
    quantum_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    st.text_input(
        "Command Timestamp (auto, UTC)",
        value=f"{quantum_timestamp}",
        help="Label only. Data pull always uses the freshest records available.",
        key="quantum_ts_input",
        disabled=True,
    )
    
    # Check data freshness before allowing QUANTUMTRADER to run
    data_ready = False  # Initialize as False, will be set to True if symbol selected and data is ready
    if lab_symbol:
        # Add refresh button to re-check data freshness
        col_refresh1, col_refresh2 = st.columns([3, 1])
        with col_refresh2:
            if st.button("🔄 Refresh Data Check", key="refresh_data_check"):
                st.session_state.quantum_show_freshness_once = True
                st.rerun()
        
        show_check = st.session_state.get("quantum_show_freshness_once", False)
        if show_check:
            supabase = get_supabase()
            db_check_info = {}
            if supabase:
                try:
                    # Determine table based on asset class
                    table_1min_name = "market_data_stocks_1min"
                    table_5min_name = "market_data_5min"
                    
                    if selected_category == "Commodities":
                        table_1min_name = "market_data_commodities_1min"
                        # table_5min_name = "market_data_commodities_5min" # Not yet implemented, fallback or ignore
                    elif selected_category == "Indices":
                        table_1min_name = "market_data_indices_1min"
                    elif selected_category == "Currencies":
                        table_1min_name = "market_data_currencies_1min"
                        
                    check_1min = supabase.table(table_1min_name)\
                        .select("symbol,timestamp")\
                        .eq("symbol", lab_symbol.upper())\
                        .order("timestamp", desc=True)\
                        .limit(5)\
                        .execute()
                    db_check_info["1min_raw"] = check_1min.data if hasattr(check_1min, "data") else []
                    
                    # 5min check (only for stocks currently, or if tables exist)
                    try:
                        check_5min = supabase.table(table_5min_name)\
                            .select("symbol,timestamp")\
                            .eq("symbol", lab_symbol.upper())\
                            .order("timestamp", desc=True)\
                            .limit(5)\
                            .execute()
                        db_check_info["5min_raw"] = check_5min.data if hasattr(check_5min, "data") else []
                    except Exception:
                        db_check_info["5min_raw"] = []

                    # Debug info for available symbols
                    try:
                        all_symbols_1min = supabase.table(table_1min_name)\
                            .select("symbol")\
                            .limit(100)\
                            .execute()
                        unique_symbols_1min = list(set([r.get("symbol") for r in (all_symbols_1min.data if hasattr(all_symbols_1min, "data") else []) if r.get("symbol")]))
                        db_check_info["all_symbols_1min"] = sorted(unique_symbols_1min)[:20]
                    except Exception:
                        db_check_info["all_symbols_1min"] = []
                        
                    # 5min symbols
                    try:
                        all_symbols_5min = supabase.table(table_5min_name)\
                            .select("symbol")\
                            .limit(100)\
                            .execute()
                        unique_symbols_5min = list(set([r.get("symbol") for r in (all_symbols_5min.data if hasattr(all_symbols_5min, "data") else []) if r.get("symbol")]))
                        db_check_info["all_symbols_5min"] = sorted(unique_symbols_5min)[:20]
                    except Exception:
                         db_check_info["all_symbols_5min"] = []

                except Exception as e:
                    import traceback
                    db_check_info["error"] = str(e)
                    db_check_info["error_traceback"] = traceback.format_exc()
            
            latest_1min = fetch_latest_bar(lab_symbol, "1min", asset_class=selected_category)
            latest_5min = fetch_latest_bar(lab_symbol, "5min", asset_class=selected_category)
            data_ready = True
            freshness_warnings = []
            freshness_cutoff_minutes = 20
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            current_day = now_utc.weekday()
            current_hour_utc = now_utc.hour
            is_weekend = current_day >= 5
            is_after_hours = current_hour_utc < 14 or current_hour_utc >= 21
            market_status_msg = None
            if is_weekend or (current_day == 4 and current_hour_utc >= 21) or (current_day == 0 and current_hour_utc < 14):
                freshness_cutoff_minutes = 2880
                market_status_msg = "Market is closed (weekend/after hours). Allowing older data."
            st.caption(f"🕐 Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
            with st.expander("🔍 Debug: Latest Data Query Results", expanded=False):
                st.write(f"**Direct Database Query Results:**")
                if db_check_info.get("error"):
                    st.error(f"Database query error: {db_check_info['error']}")
                st.write(f"**1-minute table - Top 5 records for {lab_symbol.upper()}:**")
                if db_check_info.get("1min_raw"):
                    for i, record in enumerate(db_check_info["1min_raw"][:5], 1):
                        st.write(f"{i}. Symbol: {record.get('symbol')}, Timestamp: {record.get('timestamp')}")
                else:
                    st.write("❌ No records found in 1-min table")
                st.write(f"**5-minute table - Top 5 records for {lab_symbol.upper()}:**")
                if db_check_info.get("5min_raw"):
                    for i, record in enumerate(db_check_info["5min_raw"][:5], 1):
                        st.write(f"{i}. Symbol: {record.get('symbol')}, Timestamp: {record.get('timestamp')}")
                else:
                    st.write("❌ No records found in 5-min table")
                st.write(f"**fetch_latest_bar() function results:**")
                st.write(f"**1-minute data query result:**")
                if latest_1min:
                    st.json({k: str(v) for k, v in latest_1min.items()})
                else:
                    st.write("❌ No 1-minute data returned from fetch_latest_bar()")
                st.write(f"**5-minute data query result:**")
                if latest_5min:
                    st.json({k: str(v) for k, v in latest_5min.items()})
                else:
                    st.write("❌ No 5-minute data returned from fetch_latest_bar()")
            if not latest_1min:
                data_ready = False
                freshness_warnings.append(f"❌ No 1-minute data found for {lab_symbol}")
            else:
                latest_1min_ts = pd.to_datetime(latest_1min.get("timestamp")).replace(tzinfo=None)
                age_minutes = (now_utc - latest_1min_ts).total_seconds() / 60
                if age_minutes > freshness_cutoff_minutes:
                    data_ready = False
                    freshness_warnings.append(f"⚠️ 1-minute data is {age_minutes:.0f} minutes old (latest: {latest_1min_ts.strftime('%Y-%m-%d %H:%M:%S')})")
                else:
                    st.caption(f"✅ 1-min data fresh: {latest_1min_ts.strftime('%Y-%m-%d %H:%M:%S')} ({age_minutes:.0f}m ago)")
            if not latest_5min:
                data_ready = False
                freshness_warnings.append(f"❌ No 5-minute data found for {lab_symbol}")
            else:
                latest_5min_ts = pd.to_datetime(latest_5min.get("timestamp")).replace(tzinfo=None)
                age_minutes = (now_utc - latest_5min_ts).total_seconds() / 60
                if age_minutes > freshness_cutoff_minutes:
                    data_ready = False
                    freshness_warnings.append(f"⚠️ 5-minute data is {age_minutes:.0f} minutes old (latest: {latest_5min_ts.strftime('%Y-%m-%d %H:%M:%S')})")
                else:
                    st.caption(f"✅ 5-min data fresh: {latest_5min_ts.strftime('%Y-%m-%d %H:%M:%S')} ({age_minutes:.0f}m ago)")
            if not data_ready:
                if market_status_msg:
                    st.warning(f"**📊 Market Status: {market_status_msg}**")
                    st.info("""
                    **QUANTUMTRADER can still run with older data during market closure.**
                    
                    The data freshness check has been relaxed because markets are currently closed.
                    You can proceed with analysis using the available historical data.
                    """)
                    col_override1, col_override2 = st.columns([2, 1])
                    with col_override1:
                        st.write("**Option:** Proceed with available data despite age warnings.")
                    with col_override2:
                        if st.button("✅ Override & Allow QUANTUMTRADER", key="override_freshness", type="primary"):
                            st.session_state.quantum_data_override = True
                            st.rerun()
                else:
                    st.error("**⚠️ Data Not Ready - Cannot Run QUANTUMTRADER**")
                    for warning in freshness_warnings:
                        st.warning(warning)
                if freshness_warnings and not market_status_msg:
                    for warning in freshness_warnings:
                        st.warning(warning)
                col_ingest1, col_ingest2 = st.columns([2, 1])
                with col_ingest1:
                    status_msg = """
                    **QUANTUMTRADER needs recent, consecutive market bars before analysis.**
                    
                    We compute intraday metrics from the last 20 consecutive 1‑minute bars, and validation uses recent 5‑minute bars for alignment and volatility. If either feed is stale or missing, results will be inaccurate.
                    
                    **Current status**\n"""
                    if latest_1min:
                        latest_1min_ts_display = pd.to_datetime(latest_1min.get("timestamp")).replace(tzinfo=None)
                        age_1min_display = (now_utc - latest_1min_ts_display).total_seconds() / 60
                        status_msg += f"- Latest `1‑min` bar: {latest_1min_ts_display.strftime('%Y-%m-%d %H:%M:%S')} ({age_1min_display:.0f} minutes ago)\n"
                    else:
                        status_msg += "- Latest `1‑min` bar: None\n"
                    if latest_5min:
                        latest_5min_ts_display = pd.to_datetime(latest_5min.get("timestamp")).replace(tzinfo=None)
                        age_5min_display = (now_utc - latest_5min_ts_display).total_seconds() / 60
                        status_msg += f"- Latest `5‑min` bar: {latest_5min_ts_display.strftime('%Y-%m-%d %H:%M:%S')} ({age_5min_display:.0f} minutes ago)\n"
                    else:
                        status_msg += "- Latest `5‑min` bar: None\n"
                    status_msg += """- Requirement: During market hours, bars must be ≤ 20 minutes old; off‑hours relax to ≤ 48 hours.
                    
                    **Action**
                    Click ‘📥 Ingest Latest Data’ to pull `1‑min` (3 days) and `5‑min` (7 days) up to now, then we’ll re‑check freshness automatically.
                    """
                    st.info(status_msg)
                with col_ingest2:
                    st.markdown("#### Quick Ingest")
                    ingest_disabled = (selected_category != "Stocks")
                    if st.button(f"📥 Ingest Latest Data for {lab_symbol}", key="quick_ingest_quantum", type="primary", disabled=ingest_disabled):
                        latest_before_1min = fetch_latest_bar(lab_symbol, "1min", asset_class=selected_category)
                        latest_before_5min = fetch_latest_bar(lab_symbol, "5min", asset_class=selected_category)
                        before_1min_ts = pd.to_datetime(latest_before_1min.get("timestamp")).replace(tzinfo=None) if latest_before_1min else None
                    if ingest_disabled:
                        st.caption("⚠️ Automated ingestion is only available for Stocks. Use manual upload for others.")
                        before_5min_ts = pd.to_datetime(latest_before_5min.get("timestamp")).replace(tzinfo=None) if latest_before_5min else None
                        with st.spinner(f"Ingesting latest 1-min and 5-min data for {lab_symbol} up to current time..."):
                            try:
                                pipeline = DataIngestionPipeline()
                                now_utc_end = datetime.now(timezone.utc).replace(tzinfo=None)
                                st.write(f"**Ingestion Details:**")
                                st.write(f"- Target end time: {now_utc_end.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                                if before_1min_ts:
                                    st.write(f"- Current 1-min latest: {before_1min_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                                if before_5min_ts:
                                    st.write(f"- Current 5-min latest: {before_5min_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                                st.info(f"📥 Ingesting 1-minute data for {lab_symbol}...")
                                success_1min = pipeline.ingest_historical_data(
                                    symbol=lab_symbol,
                                    interval="1min",
                                    days_back=3,
                                    end_date=now_utc_end,
                                    resume_from_latest=True
                                )
                                st.write(f"1-min ingestion result: {'✅ Success' if success_1min else '❌ Failed'}")
                                stats_1 = {}
                                try:
                                    stats_1 = pipeline.recent_stats.get((lab_symbol, "1min"), {}) or {}
                                except Exception:
                                    stats_1 = {}
                                st.info(f"📥 Ingesting 5-minute data for {lab_symbol}...")
                                success_5min = pipeline.ingest_historical_data(
                                    symbol=lab_symbol,
                                    interval="5min",
                                    days_back=7,
                                    end_date=now_utc_end,
                                    resume_from_latest=True
                                )
                                st.write(f"5-min ingestion result: {'✅ Success' if success_5min else '❌ Failed'}")
                                stats_5 = {}
                                try:
                                    stats_5 = pipeline.recent_stats.get((lab_symbol, "5min"), {}) or {}
                                except Exception:
                                    stats_5 = {}
                                if stats_1 or stats_5:
                                    st.markdown("**Inserted Records**")
                                    st.write(f"- 1-min: {stats_1.get('inserted', 0)} new, {stats_1.get('skipped', 0)} duplicates skipped")
                                    st.write(f"- 5-min: {stats_5.get('inserted', 0)} new, {stats_5.get('skipped', 0)} duplicates skipped")
                                pipeline.close()
                                import time
                                time.sleep(2)
                                latest_after_1min = fetch_latest_bar(lab_symbol, "1min", asset_class=selected_category)
                                latest_after_5min = fetch_latest_bar(lab_symbol, "5min", asset_class=selected_category)
                                after_1min_ts = pd.to_datetime(latest_after_1min.get("timestamp")).replace(tzinfo=None) if latest_after_1min else None
                                after_5min_ts = pd.to_datetime(latest_after_5min.get("timestamp")).replace(tzinfo=None) if latest_after_5min else None
                                st.write(f"**Post-Ingestion Check:**")
                                if before_1min_ts and after_1min_ts:
                                    if after_1min_ts > before_1min_ts:
                                        st.success(f"✅ 1-min data updated: {before_1min_ts.strftime('%Y-%m-%d %H:%M:%S')} → {after_1min_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                                    else:
                                        st.warning(f"⚠️ 1-min data unchanged: {after_1min_ts.strftime('%Y-%m-%d %H:%M:%S')} (no new data found or already up to date)")
                                elif after_1min_ts:
                                    st.info(f"ℹ️ 1-min data now exists: {after_1min_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                                if before_5min_ts and after_5min_ts:
                                    if after_5min_ts > before_5min_ts:
                                        st.success(f"✅ 5-min data updated: {before_5min_ts.strftime('%Y-%m-%d %H:%M:%S')} → {after_5min_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                                    else:
                                        st.warning(f"⚠️ 5-min data unchanged: {after_5min_ts.strftime('%Y-%m-%d %H:%M:%S')} (no new data found or already up to date)")
                                elif after_5min_ts:
                                    st.info(f"ℹ️ 5-min data now exists: {after_5min_ts.strftime('%Y-%m-%d %H:%M:%S')}")
                                if after_1min_ts and after_5min_ts:
                                    age_1min_final = (now_utc_end - after_1min_ts).total_seconds() / 60
                                    age_5min_final = (now_utc_end - after_5min_ts).total_seconds() / 60
                                    if age_1min_final <= 20 and age_5min_final <= 20:
                                        st.success(f"🎉 Data is now fresh! Refreshing page...")
                                        st.balloons()
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.warning(f"⚠️ Data ingested but still not fresh enough (1-min: {age_1min_final:.0f}m, 5-min: {age_5min_final:.0f}m). Market may be closed or no recent data available.")
                                        st.info("💡 Check the console/terminal logs for detailed ingestion output. You may need to wait for market hours or check Polygon API availability.")
                                else:
                                    st.warning(f"⚠️ Ingestion completed but couldn't verify new data. Check console logs for details.")
                                    st.info("💡 Click '🔄 Refresh Data Check' button above to see current database state.")
                            except Exception as e:
                                st.error(f"❌ Ingestion error: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                    st.markdown("---")
                    st.markdown("**Or manually ingest:**")
                    st.markdown("Go to **'Ingest Historical Data'** section above")
            else:
                if latest_1min and latest_5min:
                    latest_1min_ts = pd.to_datetime(latest_1min.get("timestamp")).replace(tzinfo=None)
                    latest_5min_ts = pd.to_datetime(latest_5min.get("timestamp")).replace(tzinfo=None)
                    age_1min = (now_utc - latest_1min_ts).total_seconds() / 60
                    age_5min = (now_utc - latest_5min_ts).total_seconds() / 60
                    st.success(f"✅ Data is fresh! Latest 1-min: {latest_1min_ts.strftime('%Y-%m-%d %H:%M:%S')} ({age_1min:.0f}m ago) | Latest 5-min: {latest_5min_ts.strftime('%Y-%m-%d %H:%M:%S')} ({age_5min:.0f}m ago)")
            st.session_state.quantum_show_freshness_once = False
        else:
            st.caption("Press 'Refresh Data Check' to view freshness status.")
    
    # Check if user has overridden freshness check
    override_active = st.session_state.get("quantum_data_override", False)
    
    # If override is active, show cancel option
    if override_active:
        st.success("✅ Override Active: QUANTUMTRADER can run with older data")
        if st.button("❌ Cancel Override", key="cancel_override"):
            st.session_state.quantum_data_override = False
            st.rerun()
        # Allow running despite freshness warnings when override is active
        effective_data_ready = True
    else:
        effective_data_ready = data_ready
    
    if st.button("⚙️ Run QUANTUMTRADER Prompt", key="quantum_prompt_button", disabled=not (lab_symbol and effective_data_ready)):
        with st.spinner("Calculating QUANTUMTRADER metrics..."):
            try:
                qt_result = feature_lab.run_quantum_screen(lab_symbol, quantum_timestamp, asset_class=selected_category)
                st.session_state.quantum_result = qt_result
                st.success("QUANTUMTRADER packet ready!")
            except Exception as exc:
                st.error(f"Quantum run failed: {exc}")

    quantum_result = st.session_state.get("quantum_result")
    if quantum_result:
        st.success(quantum_result["summary"])
        metrics_df = pd.DataFrame([quantum_result["metrics"]])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.markdown("###### QUANTUMTRADER Prompt Payload")
        st.code(quantum_result["prompt"], language="markdown")
        
        # Trade Decision Engine
        st.markdown("---")
        st.markdown("##### 🎯 TRADE DECISION ENGINE")
        st.markdown("""
        **CONDITIONS FOR "TRADE YES":**
        
        1. Composite_Score ≥ 6.5
        2. All Phase 1 gates passed ✓
        3. R:R ratio achievable ≥ 1:2
        4. Position size calculable within $2,000 exposure limit
        5. No conflicting daily trend (avoid counter-trend)
        
        **DIRECTION DECISION:**
        - If 5-min trend = UP and aligned with higher timeframes → BUY
        - If 5-min trend = DOWN and aligned with higher timeframes → SELL
        - If conflicting → "NO TRADE - Trend conflict"
        
        **RISK MANAGEMENT OVERLAY:**
        - Max daily loss: $400 (4% of $10k)
        - Max concurrent trades: 3
        - Auto-close all positions at 4:00 PM ET
        """)
        
        if st.button("🎯 Evaluate Trade Decision", key="evaluate_trade_decision", type="primary"):
            with st.spinner("Evaluating all trade conditions..."):
                try:
                    decision_engine = TradeDecisionEngine()
                    decision = decision_engine.evaluate_trade_decision(lab_symbol, quantum_timestamp, asset_class=selected_category)
                    st.session_state.trade_decision = decision
                    decision_engine.close()
                    st.success("✅ Trade decision evaluation complete!")
                except Exception as exc:
                    st.error(f"Trade decision evaluation failed: {exc}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display decision results
        trade_decision = st.session_state.get("trade_decision")
        if trade_decision and trade_decision.get("symbol") == lab_symbol:
            st.markdown("---")
            st.markdown("###### Trade Decision Results")
            
            # Decision header
            if trade_decision["trade_decision"] == "TRADE YES":
                st.success(f"## ✅ **TRADE YES** - {lab_symbol}")
            else:
                st.error(f"## ❌ **NO TRADE** - {lab_symbol}")
            
            # Direction
            direction = trade_decision.get("direction")
            if direction:
                direction_emoji = "📈" if direction == "UP" else "📉" if direction == "DOWN" else "⚠️"
                st.info(f"{direction_emoji} **Direction**: {direction}")
            
            # Reason
            st.markdown(f"**Reason**: {trade_decision.get('reason', 'N/A')}")
            
            # Conditions breakdown
            st.markdown("#### Condition Evaluation:")
            conditions = trade_decision.get("conditions", {})
            
            # Create conditions table
            cond_rows = []
            
            # Condition 1: Composite Score
            cond1 = conditions.get("composite_score", {})
            c_metrics = cond1.get("details", {})
            
            c_details = f"Threshold: {cond1.get('threshold', 6.5)}"
            if c_metrics:
                c_details += f" | Vol: {c_metrics.get('volume_score')} | VWAP: {c_metrics.get('vwap_score')} (Val: {c_metrics.get('vwap')}) | Mom: {c_metrics.get('momentum_score')} | Cat: {c_metrics.get('catalyst_score')}"
            
            cond_rows.append({
                "Condition": "1. Composite_Score ≥ 6.5",
                "Status": "✅ PASS" if cond1.get("pass") else "❌ FAIL",
                "Value": f"{cond1.get('value', 'N/A'):.2f}" if cond1.get('value') else "N/A",
                "Details": c_details
            })
            
            # Condition 2: Phase 1 Gates
            cond2 = conditions.get("phase1_gates", {})
            cond_rows.append({
                "Condition": "2. All Phase 1 gates passed",
                "Status": "✅ PASS" if cond2.get("pass") else "❌ FAIL",
                "Value": cond2.get("reason", "N/A"),
                "Details": ""
            })
            
            # Condition 3: R:R Ratio
            cond3 = conditions.get("rr_ratio", {})
            cond_rows.append({
                "Condition": "3. R:R ratio ≥ 1:2",
                "Status": "✅ PASS" if cond3.get("achievable") else "❌ FAIL",
                "Value": f"{cond3.get('ratio', 'N/A')}:1",
                "Details": f"Stop: ${cond3.get('stop_loss', 'N/A'):.2f}" if cond3.get('stop_loss') else "N/A"
            })
            
            # Condition 4: Position Size
            cond4 = conditions.get("position_size", {})
            cond_rows.append({
                "Condition": "4. Position size ≤ $2,000",
                "Status": "✅ PASS" if cond4.get("calculable") else "❌ FAIL",
                "Value": f"${cond4.get('exposure', 'N/A'):.2f}" if cond4.get('exposure') else "N/A",
                "Details": f"Shares: {cond4.get('recommended_shares', 'N/A')}"
            })
            
            # Condition 5: Trend Alignment
            cond5 = conditions.get("trend_alignment", {})
            cond_rows.append({
                "Condition": "5. No conflicting daily trend",
                "Status": "✅ PASS" if not cond5.get("conflict") else "❌ FAIL",
                "Value": f"{cond5.get('m5_trend', 'N/A')} vs {cond5.get('daily_trend', 'N/A')}",
                "Details": cond5.get("conflict_reason", "") if cond5.get("conflict") else "Aligned"
            })
            
            cond_df = pd.DataFrame(cond_rows)
            # Streamlit's dataframe API does not support hide_columns; drop display-only columns instead.
            st.dataframe(cond_df.drop(columns=["Details"], errors="ignore"), use_container_width=True, hide_index=True)
            
            # Risk Management Overlay
            st.markdown("#### Risk Management Overlay:")
            risk_metrics = trade_decision.get("risk_metrics", {})
            col_risk1, col_risk2, col_risk3 = st.columns(3)
            
            with col_risk1:
                st.metric("Active Trades", f"{risk_metrics.get('active_trades', 0)} / {risk_metrics.get('max_concurrent', 3)}")
            with col_risk2:
                st.metric("Max Daily Loss", f"${risk_metrics.get('max_daily_loss', 400)}")
            with col_risk3:
                st.metric("Auto-Close Time", "4:00 PM ET")
            
            can_trade = risk_metrics.get("can_trade", False)
            if not can_trade:
                st.warning(f"⚠️ {risk_metrics.get('reason', 'Cannot open new trade')}")
            
            # Recommendation (if TRADE YES)
            if trade_decision["trade_decision"] == "TRADE YES" and trade_decision.get("recommendation"):
                rec = trade_decision["recommendation"]
                st.markdown("---")
                st.markdown("#### 📋 Trade Recommendation:")
                
                col_rec1, col_rec2, col_rec3, col_rec4 = st.columns(4)
                
                with col_rec1:
                    st.metric("Action", rec.get("action", "N/A"))
                    st.metric("Entry Price", f"${rec.get('entry_price', 0):.2f}")
                
                with col_rec2:
                    st.metric("Stop Loss", f"${rec.get('stop_loss', 0):.2f}")
                    st.metric("Target 1", f"${rec.get('target1', 0):.2f}")
                
                with col_rec3:
                    st.metric("Target 2", f"${rec.get('target2', 0):.2f}")
                    st.metric("R:R Ratio", f"1:{rec.get('rr_ratio', 0):.2f}")
                
                with col_rec4:
                    st.metric("Shares", rec.get("position_size_shares", 0))
                    st.metric("Exposure", f"${rec.get('exposure', 0):.2f}")
                    st.metric("Risk", f"${rec.get('risk_amount', 0):.2f}")

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Document management
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Document Management")
    doc_manager = DocumentManager()
    
    # Show available stocks
    st.info(f"**Available Stocks:** {', '.join(list(WATCHLIST_STOCKS.keys())[:10])}... (19 total)")
    
    # Upload document
    uploaded_file = st.file_uploader("Upload Research Document", type=['pdf', 'txt', 'docx'])
    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox("Select Stock (Optional)", [""] + list(WATCHLIST_STOCKS.keys()), 
                                 help="Choose from 19 high-beta US stocks in our watchlist")
        with col2:
            # Create a better default title
            default_title = uploaded_file.name.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
            title = st.text_input("Document Title", value=default_title)
        
        # Show file info
        st.info(f"📄 **File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024:.1f} KB")
        
        if st.button("Upload Document", key="upload_doc_phase1", type="primary"):
            if not title.strip():
                st.error("Please enter a document title")
            else:
                with st.spinner("Uploading and processing document..."):
                    result = doc_manager.upload_document(
                        uploaded_file, 
                        uploaded_file.name, 
                        title,
                        symbol=symbol if symbol else None
                    )
                    if result["success"]:
                        st.success(f"✅ Document '{title}' uploaded successfully!")
                        st.balloons()
                    else:
                        st.error(f"Upload failed: {result['message']}")
    
    # Display documents
    st.markdown("#### Uploaded Documents")
    documents = doc_manager.get_documents()
    if documents:
        # Add document management controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**Total Documents:** {len(documents)}")
        with col2:
            if st.button("🔄 Refresh", key="refresh_docs"):
                st.rerun()
        with col3:
            # Filter by symbol
            symbols = ["All"] + list(set([doc.get("symbol", "N/A") for doc in documents if doc.get("symbol")]))
            selected_symbol = st.selectbox("Filter by Symbol", symbols, key="doc_filter")
        
        # Transform documents for display
        display_docs = []
        filtered_docs = documents
        if selected_symbol != "All":
            filtered_docs = [doc for doc in documents if doc.get("symbol") == selected_symbol]
        
        for i, doc in enumerate(filtered_docs):
            # Create a better display name
            file_name = doc.get("file_name", f"Document_{i+1}")
            symbol = doc.get("symbol", "N/A")
            
            # Create a descriptive name
            if symbol != "N/A":
                display_name = f"{file_name} ({symbol})"
            else:
                display_name = file_name
            
            # Format creation date
            created_at = doc.get("uploaded_at", doc.get("created_at", ""))
            if created_at:
                try:
                    parsed_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    formatted_date = parsed_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    formatted_date = created_at[:10] if len(created_at) >= 10 else "Unknown"
            else:
                formatted_date = "Unknown"
            
            # Get content preview
            content = doc.get("file_content", "")
            content_preview = content[:150] + "..." if len(content) > 150 else content
            
            display_docs.append({
                "ID": f"#{i+1}",
                "File Name": display_name,
                "Symbol": symbol,
                "Content Preview": content_preview,
                "Created": formatted_date,
                "Size": f"{len(content)} chars"
            })
        
        if display_docs:
            # Compact dropdown instead of bulky table
            st.markdown("#### Documents")
            selected_doc_idx = st.selectbox(
                "Select Document",
                range(len(filtered_docs)),
                format_func=lambda x: display_docs[x]["File Name"],
                key="doc_dropdown",
            )

            selected_doc = filtered_docs[selected_doc_idx]
            selected_display = display_docs[selected_doc_idx]

            # Summary line
            st.write(
                f"**File:** {selected_display['File Name']}  |  "
                f"**Symbol:** {selected_display['Symbol']}  |  "
                f"**Created:** {selected_display['Created']}  |  "
                f"**Size:** {selected_display['Size']}"
            )
            st.caption(selected_display["Content Preview"])

            # Download button for selected document
            file_content = selected_doc.get("file_content", "")
            file_name = selected_doc.get("file_name", f"document_{selected_doc_idx+1}")

            if file_name.lower().endswith(('.pdf', '.docx', '.doc')):
                base_name = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
                download_filename = f"{base_name}.txt"
            elif not file_name.lower().endswith('.txt'):
                download_filename = f"{file_name}.txt"
            else:
                download_filename = file_name

            st.download_button(
                label="⬇️ Download",
                data=file_content.encode('utf-8') if isinstance(file_content, str) else file_content,
                file_name=download_filename,
                mime="text/plain",
                key=f"download_doc_selected_{selected_doc.get('id', selected_doc_idx)}",
                use_container_width=False,
            )

            # Actions
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 View Full Content", key="view_content"):
                    st.subheader(f"Content: {selected_doc.get('file_name', 'Unknown')}")
                    st.text_area(
                        "Document Content",
                        selected_doc.get("file_content", ""),
                        height=300,
                    )

            with col2:
                if st.button("🗑️ Delete Document", key="delete_doc"):
                    doc_id = selected_doc.get("id")
                    if doc_id and doc_manager.delete_document(doc_id):
                        st.success("Document deleted successfully!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Failed to delete document")
        else:
            st.info(f"No documents found for symbol: {selected_symbol}")
    else:
        st.info("No documents uploaded yet")
    
    doc_manager.close()

def phase2_master_data_ai():
    """Phase 2: Master Data & AI Integration"""
    
    # Initialize AI analyzer
    ai_analyzer = AIResearchAnalyzer()
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Master Data Dashboard", 
        "AI Document Analysis", 
        "Instrument Profiles", 
        "Research Insights",
        "Raw Data Tools (Macro & Instrument)"
    ])
    
    with tab1:
        st.subheader("Master Data Dashboard")
        
        # Get watchlist symbols
        symbols = get_watchlist_symbols()
        
        # Compute dynamic document count from research_documents table
        try:
            doc_manager_tmp = DocumentManager()
            all_docs_tmp = doc_manager_tmp.get_documents() or []
            docs_count = len(all_docs_tmp)
            doc_manager_tmp.close()
        except Exception as e:
            docs_count = 0
            print(f"Error counting documents: {e}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Instruments", len(symbols))
        with col2:
            # AI Analysis Status: "Active" if GROQ_API_KEY is configured, "Inactive" otherwise
            groq_key = os.getenv("GROQ_API_KEY", "") or os.getenv("GROK_API_KEY", "")
            ai_status = "Active" if groq_key and groq_key.startswith('gsk_') else "Inactive"
            st.metric("AI Analysis Status", ai_status)
        with col3:
            # Documents Processed: Count from research_documents table
            st.metric("Documents Processed", docs_count)
        
        # Master data summary (only generate when button is clicked)
        st.info("""
        **How Master Data is Generated:**
        1. Click "Generate/Refresh Master Data Summary" button below
        2. System calls LLM (Groq) to analyze all 19 instruments
        3. For each instrument: analyzes market data, document insights, and generates AI sentiment
        4. Results stored in session state (temporary, lost on page refresh)
        5. Click "💾 Save Master Data" button to permanently save to `master_data` table with RAG embeddings
        """)
        regenerate = st.button("Generate/Refresh Master Data Summary", type="primary", key="generate_master_data")
        if regenerate:
            with st.spinner("Generating comprehensive analysis..."):
                summary = ai_analyzer.get_master_data_summary()
                if isinstance(summary, dict) and "error" not in summary:
                    st.session_state.master_data_summary = summary
                    st.success("Master data analysis completed!")
                else:
                    st.error(f"Error generating summary: {summary.get('error','Unknown error') if isinstance(summary, dict) else 'Unknown error'}")
        
        if "master_data_summary" in st.session_state:
            summary = st.session_state.master_data_summary
            st.subheader("Analysis Summary")
            col1s, col2s, col3s, col4s = st.columns(4)
            with col1s:
                st.metric("Total Instruments", summary.get("total_instruments", len(symbols)))
            with col2s:
                # Bullish Signals Calculation:
                # 1. For each instrument, LLM (Groq/Llama-3.1-8b-instant) generates analysis text
                # 2. _extract_sentiment_from_analysis() searches LLM response for "bullish" keyword
                # 3. If found → returns "Bullish", else checks for "bearish" → "Bearish", else → "Neutral"
                # 4. Counts all instruments where overall_sentiment == "Bullish"
                bullish_count = sum(1 for inst in summary.get("instruments", []) 
                                   if inst.get("ai_analysis", {}).get("overall_sentiment") == "Bullish")
                st.metric("Bullish Signals", bullish_count)
            with col3s:
                # Bearish Signals Calculation:
                # 1. For each instrument, LLM (Groq/Llama-3.1-8b-instant) generates analysis text
                # 2. _extract_sentiment_from_analysis() searches LLM response for "bearish" keyword
                # 3. If found → returns "Bearish", else checks for "bullish" → "Bullish", else → "Neutral"
                # 4. Counts all instruments where overall_sentiment == "Bearish"
                bearish_count = sum(1 for inst in summary.get("instruments", []) 
                                   if inst.get("ai_analysis", {}).get("overall_sentiment") == "Bearish")
                st.metric("Bearish Signals", bearish_count)
            with col4s:
                # Avg Confidence Calculation:
                # For each instrument, gets confidence from:
                # - confidence_score (calculated from doc_insights + news_sentiment) OR
                # - ai_analysis.confidence (extracted from LLM response via regex: "confidence: X")
                # Then calculates: mean([all confidence values])
                confs = [inst.get("confidence_score", inst.get("ai_analysis", {}).get("confidence", 5)) for inst in summary.get("instruments", [])]
                avg_confidence = float(np.mean(confs)) if confs else 0.0
                st.metric("Avg Confidence", f"{avg_confidence:.1f}/10")
            
            # Detailed calculation breakdown
            st.markdown("---")
            with st.expander("📊 How These Metrics Are Calculated", expanded=False):
                st.markdown("""
                ### Calculation Methodology:
                
                **1. Bullish Signals (e.g., 10):**
                - **Step-by-Step**:
                  1. For each of 19 instruments, LLM (Groq/Llama-3.1-8b-instant) is called
                  2. LLM receives prompt asking for "Overall Assessment (Bullish/Bearish/Neutral)"
                  3. LLM generates analysis text response
                  4. `_extract_sentiment_from_analysis()` searches response for keyword "bullish"
                  5. If "bullish" found → `overall_sentiment = "Bullish"`, else checks "bearish" → "Bearish", else → "Neutral"
                  6. **Final Count**: Sum of instruments where `overall_sentiment == "Bullish"`
                - **Example**: 10 instruments have "bullish" in LLM response → **Bullish Signals = 10**
                - **Code**: `app.py` lines 1044-1046, extraction in `ai_analysis.py` lines 290-298
                
                **2. Bearish Signals:**
                - **Source**: LLM (Groq/Llama-3.1-8b-instant) analysis response
                - **Process**: 
                  1. LLM is called with prompt (see `_generate_comprehensive_analysis()` in `ai_analysis.py`)
                  2. LLM response text is analyzed by `_extract_sentiment_from_analysis()`
                  3. If response contains "bearish" keyword → returns "Bearish"
                  4. Counts instruments where `ai_analysis.overall_sentiment == "Bearish"`
                - **LLM Prompt Location**: `tradingagents/dataflows/ai_analysis.py` lines 192-210
                - **Sentiment Extraction**: `tradingagents/dataflows/ai_analysis.py` lines 290-298
                - If AI analysis fails or returns "Neutral"/"Bullish", it's not counted
                
                **3. Average Confidence (e.g., 3.7/10):**
                - **Step-by-Step**:
                  1. For each instrument, get confidence from:
                     - **Option A**: `confidence_score` = mean([doc_insights confidence, news_count/10])
                     - **Option B**: `ai_analysis.confidence` = regex extract "confidence: X" from LLM text (default 5)
                  2. For each instrument: `confidence = confidence_score OR ai_analysis.confidence OR 5`
                  3. **Final Calculation**: `mean([confidence for all 19 instruments])`
                - **Example**: Confidences [3,4,3,4,4,3,4,3,4,3,4,3,4,3,4,3,4,3,4] → **Avg = 3.7/10**
                - **Code**: `app.py` lines 1055-1057, calculation in `ai_analysis.py` lines 319-346
                
                **4. Why You Might See 0 Signals:**
                - AI analysis may have failed (no GROQ_API_KEY or API error)
                - All instruments returned "Neutral" sentiment
                - AI analysis text didn't contain clear sentiment keywords
                """)
                
                # Show detailed breakdown per instrument
                st.markdown("### 📈 Per-Instrument Breakdown:")
                st.caption("**Sentiment Source**: Extracted from LLM (Groq) response text via `_extract_sentiment_from_analysis()`. Looks for 'bullish' or 'bearish' keywords in the LLM analysis text.")
                instruments = summary.get("instruments", [])
                
                # Create a summary table
                breakdown_data = []
                for inst in instruments:
                    symbol = inst.get("symbol", "Unknown")
                    ai_analysis = inst.get("ai_analysis", {})
                    # Sentiment comes from: LLM response → _extract_sentiment_from_analysis() → overall_sentiment field
                    sentiment = ai_analysis.get("overall_sentiment", "N/A")
                    confidence = inst.get("confidence_score", ai_analysis.get("confidence", "N/A"))
                    has_error = "error" in ai_analysis
                    error_msg = ai_analysis.get("error", "")
                    
                    breakdown_data.append({
                        "Symbol": symbol,
                        "Sentiment": sentiment,
                        "Confidence": confidence if isinstance(confidence, (int, float)) else "N/A",
                        "Status": "❌ Error" if has_error else "✅ Success",
                        "Error": error_msg[:50] + "..." if error_msg and len(error_msg) > 50 else error_msg
                    })
                
                if breakdown_data:
                    df_breakdown = pd.DataFrame(breakdown_data)
                    # Ensure proper types for Arrow serialization
                    for col in df_breakdown.columns:
                        if df_breakdown[col].dtype == 'object':
                            df_breakdown[col] = df_breakdown[col].astype(str)
                    st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
                    
                    # Summary statistics
                    st.markdown("### 📊 Summary Statistics:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        successful = sum(1 for inst in instruments if "error" not in inst.get("ai_analysis", {}))
                        failed = len(instruments) - successful
                        st.metric("Successful Analyses", successful)
                        st.metric("Failed Analyses", failed)
                    with col2:
                        sentiment_counts = {}
                        for inst in instruments:
                            sent = inst.get("ai_analysis", {}).get("overall_sentiment", "Unknown")
                            sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
                        st.write("**Sentiment Distribution:**")
                        for sent, count in sentiment_counts.items():
                            st.write(f"- {sent}: {count}")
                    with col3:
                        confs = [inst.get("confidence_score", inst.get("ai_analysis", {}).get("confidence", 5)) 
                                for inst in instruments 
                                if isinstance(inst.get("confidence_score", inst.get("ai_analysis", {}).get("confidence", 5)), (int, float))]
                        if confs:
                            st.write("**Confidence Stats:**")
                            st.write(f"- Min: {min(confs):.1f}/10")
                            st.write(f"- Max: {max(confs):.1f}/10")
                            st.write(f"- Median: {np.median(confs):.1f}/10")

            # Save master data to database with RAG embeddings
            st.markdown("---")
            st.info("💾 Save master data to the **`master_data`** table with RAG embeddings for semantic search.")
            
            if st.button("💾 Save Master Data", type="primary", key="save_master_data_rag"):
                with st.spinner("💾 Saving master data to database with embeddings..."):
                    try:
                        save_result = ai_analyzer.save_master_data_summary_with_rag(summary)
                        if save_result.get("success"):
                            saved_count = save_result.get("saved_count", 0)
                            failed_count = save_result.get("failed_count", 0)
                            if failed_count == 0:
                                st.success(f"✅ Master data saved to `master_data` table with RAG embeddings! ({saved_count} instruments)")
                            else:
                                st.warning(f"⚠️ Master data partially saved: {saved_count} succeeded, {failed_count} failed. Check errors below.")
                                if save_result.get("errors"):
                                    with st.expander("View Errors", expanded=False):
                                        for error in save_result.get("errors", [])[:5]:
                                            st.error(error)
                    except Exception as e:
                        st.error(f"❌ Failed to save master data: {e}")

    with tab5:
        st.subheader("Raw Data Tools – Macro Environment & Per-Instrument")
        st.write("""
        **Goal**: Fetch non-price data (macro news, company-specific news, insider sentiment, and fundamentals)
        directly from free/vendor APIs (Alpha Vantage, OpenAI-powered tools, Google News, Reddit/Finnhub local data)
        **without** going through Supabase.
        """)

        mode = st.radio(
            "Select data focus",
            ["Global Macro Environment", "Per-Instrument News & Fundamentals"],
            horizontal=True,
        )

        # Common symbol list for instrument mode
        symbols = get_watchlist_symbols()

        if mode == "Global Macro Environment":
            st.markdown("##### Global / Macro News Snapshot")
            col1, col2, col3 = st.columns(3)
            with col1:
                curr_date = st.date_input(
                    "Current date",
                    value=datetime.utcnow().date(),
                    key="raw_macro_curr_date",
                    help="Used as the reference date for global news lookback.",
                )
            with col2:
                look_back_days = st.number_input(
                    "Look back (days)",
                    min_value=1,
                    max_value=30,
                    value=7,
                    step=1,
                    key="raw_macro_lookback",
                )
            with col3:
                limit = st.number_input(
                    "Max articles",
                    min_value=1,
                    max_value=20,
                    value=10,
                    step=1,
                    key="raw_macro_limit",
                )

            if st.button("🌍 Fetch Global Macro News", type="primary", key="btn_raw_macro"):
                curr_date_str = curr_date.strftime("%Y-%m-%d")
                try:
                    with st.spinner("Fetching global macro news via configured news vendors..."):
                        # Route directly through the vendor interface (OpenAI, Google, local Reddit/Finnhub, etc.)
                        macro_news = route_to_vendor(
                            "get_global_news",
                            curr_date_str,
                            int(look_back_days),
                            int(limit),
                        )
                    if macro_news:
                        st.markdown("###### Global Macro – Groq Summary")
                        summary = _summarize_with_groq(
                            macro_news,
                            f"Global macro news (last {int(look_back_days)} days, max {int(limit)} articles)",
                        )
                        st.markdown(summary)

                        with st.expander("Raw vendor output", expanded=False):
                            st.code(
                                macro_news if isinstance(macro_news, str) else str(macro_news),
                                language="json",
                            )
                    else:
                        st.info("No global news returned for the selected window.")
                except Exception as e:
                    st.error(f"Error fetching global news: {e}")

        else:
            st.markdown("##### Per-Instrument Non-Financial Data")
            if not symbols:
                st.warning("No symbols in watchlist – configure `WATCHLIST_STOCKS` to use this tool.")
            else:
                col_sym, col_sd, col_ed = st.columns(3)
                with col_sym:
                    ticker = st.selectbox(
                        "Select symbol",
                        symbols,
                        key="raw_instr_symbol",
                    )
                with col_sd:
                    start_date = st.date_input(
                        "Start date",
                        value=datetime.utcnow().date() - timedelta(days=7),
                        key="raw_instr_start",
                    )
                with col_ed:
                    end_date = st.date_input(
                        "End date",
                        value=datetime.utcnow().date(),
                        key="raw_instr_end",
                    )

                curr_date_str = end_date.strftime("%Y-%m-%d")
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                st.markdown("###### Choose raw data sources")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    use_company_news = st.checkbox("Company News / Social", value=True, key="raw_use_news")
                with col_b:
                    use_insider = st.checkbox("Insider Sentiment", value=True, key="raw_use_insider")
                with col_c:
                    use_fundamentals = st.checkbox("Fundamentals Snapshot", value=True, key="raw_use_fund")

                if st.button("🔎 Fetch Instrument Data (Raw Vendor Output)", type="primary", key="btn_raw_instr"):
                    with st.spinner(f"Fetching non-financial data for {ticker}..."):
                        outputs = []

                        # 1) Company / social/news – may hit Alpha Vantage, OpenAI, Google, or local (Reddit/Finnhub)
                        if use_company_news:
                            try:
                                news_text = route_to_vendor(
                                    "get_news",
                                    ticker,
                                    start_str,
                                    end_str,
                                )
                                if news_text:
                                    outputs.append(("Company / Social News", news_text))
                            except Exception as e:
                                # If vendor fails, just mark as unavailable
                                st.info(f"Company/news feed not available for {ticker}: {e}")

                        # 2) Insider sentiment – local vendor (e.g., Finnhub data cache), routed via interface
                        if use_insider:
                            try:
                                insider_text = route_to_vendor(
                                    "get_insider_sentiment",
                                    ticker,
                                    curr_date_str,
                                )
                                if insider_text:
                                    outputs.append(("Insider Sentiment", insider_text))
                            except Exception as e:
                                st.info(f"Insider sentiment not available for {ticker}: {e}")

                        # 3) Fundamentals – Alpha Vantage / OpenAI fundamental tools via interface routing
                        if use_fundamentals:
                            try:
                                fundamentals_text = route_to_vendor(
                                    "get_fundamentals",
                                    ticker,
                                    curr_date_str,
                                )
                                if fundamentals_text:
                                    outputs.append(("Fundamentals Snapshot", fundamentals_text))
                            except Exception as e:
                                st.info(f"Fundamentals feed not available for {ticker}: {e}")

                    if outputs:
                        for section_title, text_value in outputs:
                            with st.expander(section_title, expanded=False):
                                # Show Groq summary first
                                summary_title = f"{section_title} for {ticker}"
                                summary = _summarize_with_groq(
                                    text_value if isinstance(text_value, str) else str(text_value),
                                    summary_title,
                                )
                                st.markdown("**Groq Summary**")
                                st.markdown(summary)

                                # Raw vendor data in a nested expander
                                with st.expander("Raw vendor output", expanded=False):
                                    st.code(
                                        text_value if isinstance(text_value, str) else str(text_value),
                                        language="json",
                                    )
                    else:
                        st.info("No raw data returned from the selected vendors for this symbol / date range.")
    
    with tab2:
        st.subheader("AI Document Analysis")
        
        # Explanation about embeddings and pgvector
        with st.expander("📖 About Embeddings & Vector Storage", expanded=False):
            st.markdown("""
            ### **Embedding Storage:**
            
            **Where Embeddings Are Stored:**
            - **Documents**: `research_documents.embedding_vector` (JSONB column)
            - **Master Data**: `master_data.embedding_vector` (JSONB column)
            - **Format**: Stored as JSONB array of floats: `[0.123, -0.456, ...]`
            
            **Do You Need pgvector?**
            - **Current Setup**: Using **JSONB** to store embeddings (works fine for storage)
            - **pgvector Extension**: NOT required for basic storage, but **recommended for similarity search**
            
            **pgvector Benefits:**
            - ✅ Fast similarity search using cosine distance
            - ✅ Indexed vector operations (much faster than JSONB)
            - ✅ Built-in functions: `<=>` (cosine distance), `<->` (L2 distance)
            
            **Without pgvector (Current):**
            - ✅ Embeddings are stored and can be retrieved
            - ⚠️ Similarity search requires loading all vectors into Python
            - ⚠️ Slower for large datasets (no vector indexes)
            
            **To Enable pgvector (Optional):**
            1. Run in Supabase SQL Editor:
               ```sql
               CREATE EXTENSION IF NOT EXISTS vector;
               ALTER TABLE research_documents 
                 ADD COLUMN embedding_vector_pgvector vector(768);
               ALTER TABLE master_data 
                 ADD COLUMN embedding_vector_pgvector vector(768);
               ```
            2. Update code to use `vector` type instead of `jsonb`
            3. Use `<=>` operator for similarity search
            
            **Current Status:**
            - ✅ Embeddings are generated (Gemini API)
            - ✅ Embeddings are saved to JSONB columns
            - ⚠️ Similarity search uses Python (not optimized)
            - 💡 Consider pgvector for production RAG queries
            """)
        
        # Document upload with enhanced processing
        st.write("Upload research documents for AI analysis:")
        st.info(f"📊 **Available Stocks:** {', '.join(symbols[:10])}... (19 total)")
        
        uploaded_file = st.file_uploader(
            "Choose a document", 
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files for AI analysis"
        )
        
        if uploaded_file:
            col1, col2, col3 = st.columns(3)
            with col1:
                symbol = st.selectbox("Select Symbol", [""] + symbols, 
                                     help="Choose from 19 high-beta US stocks in our watchlist")
            with col2:
                title = st.text_input("Document Title", uploaded_file.name)
            with col3:
                doc_type = st.selectbox("Document Type", ["research", "earnings", "news", "analysis"])
            
            if st.button("Analyze Document", type="primary", key="analyze_document"):
                if symbol and title:
                    doc_manager = DocumentManager()
                    
                    # Extract text from file without saving to DB
                    with st.spinner("Extracting document text..."):
                        file_extension = os.path.splitext(uploaded_file.name)[1]
                        content_text = doc_manager._extract_text_from_stream(uploaded_file, file_extension)
                    
                    if content_text:
                        # Perform AI analysis on extracted text (without saving to DB)
                        with st.spinner("Analyzing with AI..."):
                            # Create a temporary document ID for analysis
                            temp_doc_id = f"temp_{datetime.now().timestamp()}"
                            
                            # Store content in session state for later saving
                            # Store file bytes instead of the uploader object (not serializable)
                            uploaded_file.seek(0)
                            file_bytes = uploaded_file.read()
                            uploaded_file.seek(0)  # Reset for potential display
                            
                            st.session_state.pending_document = {
                                "file_bytes": file_bytes,
                                "filename": uploaded_file.name,
                                "title": title,
                                "symbol": symbol,
                                "document_type": doc_type,
                                "content_text": content_text
                            }
                            
                            # Analyze the document content directly (without needing a DB doc_id)
                            # Extract signals from content first
                            bullish_keywords = [
                                "buy", "bullish", "positive", "growth", "outperform", "upgrade",
                                "strong", "beat", "exceed", "increase", "rise", "gain"
                            ]
                            bearish_keywords = [
                                "sell", "bearish", "negative", "decline", "underperform", "downgrade",
                                "weak", "miss", "fall", "decrease", "drop", "loss"
                            ]
                            
                            content_lower = content_text.lower()
                            bullish_signals = [word for word in bullish_keywords if word in content_lower]
                            bearish_signals = [word for word in bearish_keywords if word in content_lower]
                            sentiment_score = len(bullish_signals) - len(bearish_signals)
                            
                            if sentiment_score > 0:
                                overall_sentiment = "Bullish"
                            elif sentiment_score < 0:
                                overall_sentiment = "Bearish"
                            else:
                                overall_sentiment = "Neutral"
                            
                            signals = {
                                "success": True,
                                "bullish_signals": bullish_signals,
                                "bearish_signals": bearish_signals,
                                "sentiment_score": sentiment_score,
                                "overall_sentiment": overall_sentiment,
                                "confidence": min(abs(sentiment_score) * 2, 10)
                            }
                            
                            # Analyze the document content directly using Groq
                            try:
                                from groq import Groq
                                groq_key = os.getenv("GROQ_API_KEY", "") or os.getenv("GROK_API_KEY", "")
                                
                                # Debug: Check if key is loaded (show masked version)
                                if groq_key:
                                    # Mask the API key for display (show first 8 and last 4 chars)
                                    masked_key = f"{groq_key[:8]}...{groq_key[-4:]}" if len(groq_key) > 12 else "***"
                                    
                                    # Validate key format (Groq keys typically start with 'gsk_')
                                    if not groq_key.startswith('gsk_'):
                                        analysis = {
                                            "success": False,
                                            "error": "Invalid API Key Format",
                                            "details": f"GROQ_API_KEY should start with 'gsk_'. Found: {masked_key}. Please check your .env file and ensure you have a valid Groq API key from https://console.groq.com/"
                                        }
                                    else:
                                        try:
                                            client = Groq(api_key=groq_key)
                                            analysis_prompt = f"""
                                            Analyze the following research document for trading insights related to {symbol}.
                                            
                                            Document Content:
                                            {content_text[:4000]}
                                            
                                            Please provide:
                                            1. Key financial metrics mentioned
                                            2. Bullish signals (positive indicators)
                                            3. Bearish signals (negative indicators)
                                            4. Overall sentiment (Bullish/Bearish/Neutral)
                                            5. Confidence level (1-10)
                                            6. Key risks mentioned
                                            7. Investment recommendation (BUY/SELL/HOLD)
                                            
                                            Format your response as a structured analysis.
                                            """
                                            chat = client.chat.completions.create(
                                                model="llama-3.1-8b-instant",
                                                messages=[
                                                    {"role": "system", "content": "You are a professional financial analyst."},
                                                    {"role": "user", "content": analysis_prompt},
                                                ],
                                                temperature=0.2,
                                            )
                                            analysis_text = chat.choices[0].message.content if chat.choices else ""
                                            analysis = {
                                                "success": True,
                                                "analysis": analysis_text,
                                                "document_id": None,
                                                "symbol": symbol,
                                                "timestamp": datetime.now().isoformat()
                                            }
                                        except Exception as api_error:
                                            error_str = str(api_error)
                                            if "401" in error_str or "invalid_api_key" in error_str.lower() or "Invalid API Key" in error_str:
                                                analysis = {
                                                    "success": False,
                                                    "error": "Invalid API Key",
                                                    "details": f"The GROQ_API_KEY in your .env file appears to be invalid. Key format: {masked_key}. Please:\n1. Get a valid API key from https://console.groq.com/\n2. Update your .env file with: GROQ_API_KEY=gsk_your_actual_key_here\n3. Restart the application"
                                                }
                                            elif "403" in error_str or "permission" in error_str.lower() or "credits" in error_str.lower():
                                                analysis = {
                                                    "success": False,
                                                    "error": "Permission/credits issue with Groq",
                                                    "details": f"Your API key may have insufficient credits or access. Key: {masked_key}. Check your Groq account at https://console.groq.com/"
                                                }
                                            else:
                                                analysis = {
                                                    "success": False,
                                                    "error": f"API Error: {error_str[:100]}",
                                                    "details": error_str
                                                }
                                else:
                                    analysis = {
                                        "success": False,
                                        "error": "No working provider. Set GROQ_API_KEY",
                                        "details": "GROQ_API_KEY environment variable is not set. Please add it to your .env file: GROQ_API_KEY=gsk_your_key_here"
                                    }
                            except Exception as e:
                                msg = str(e)
                                if "401" in msg or "invalid_api_key" in msg.lower():
                                    groq_key = os.getenv("GROQ_API_KEY", "") or os.getenv("GROK_API_KEY", "")
                                    masked_key = f"{groq_key[:8]}...{groq_key[-4:]}" if groq_key and len(groq_key) > 12 else "Not set"
                                    analysis = {
                                        "success": False,
                                        "error": "Invalid API Key",
                                        "details": f"401 Unauthorized - Invalid API Key detected. Key format: {masked_key}. Please verify your GROQ_API_KEY in the .env file is correct. Get a key from https://console.groq.com/"
                                    }
                                elif "403" in msg or "permission" in msg.lower() or "credits" in msg.lower():
                                    analysis = {
                                        "success": False,
                                        "error": "Permission/credits issue with Groq. Ensure GROQ_API_KEY has access.",
                                        "details": msg
                                    }
                                else:
                                    analysis = {
                                        "success": False,
                                        "error": str(e),
                                        "details": msg
                                    }
                            
                            # Store analysis results in session state
                            st.session_state.pending_document["analysis"] = analysis
                            st.session_state.pending_document["signals"] = signals
                            
                            # Display results
                            st.subheader("AI Analysis Results")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Document Analysis:**")
                                if analysis.get("success"):
                                    st.write(analysis.get("analysis", "Analysis completed"))
                                else:
                                    st.warning(f"AI analysis failed: {analysis.get('error', 'Unknown error')}")
                                    details = analysis.get('details')
                                    if details:
                                        st.code(details)
                                    st.info("💡 Set GROQ_API_KEY and ensure provider access is valid.")
                            
                            with col2:
                                st.write("**Trading Signals:**")
                                if signals.get("success"):
                                    st.write(f"**Sentiment:** {signals['overall_sentiment']}")
                                    st.write(f"**Confidence:** {signals['confidence']}/10")
                                    st.write(f"**Bullish Signals:** {len(signals['bullish_signals'])}")
                                    st.write(f"**Bearish Signals:** {len(signals['bearish_signals'])}")
                                else:
                                    st.warning(f"Signal extraction failed: {signals.get('error', 'Unknown error')}")
                            
                            st.success("✅ Document analyzed! Review the results above and click 'Save to Database' when ready.")
                    else:
                        st.error("Failed to extract text from document")
                else:
                    st.warning("Please select a symbol and enter a title")
            
            # Save to database button (only show if analysis is complete)
            if "pending_document" in st.session_state and st.session_state.pending_document:
                st.markdown("---")
                if st.button("💾 Save to Database", type="primary", key="save_document_to_db"):
                    doc_manager = DocumentManager()
                    pending = st.session_state.pending_document
                    
                    with st.spinner("Saving document to database..."):
                        # Create a BytesIO object from stored bytes
                        file_stream = io.BytesIO(pending["file_bytes"])
                        
                        result = doc_manager.upload_document(
                            file_stream,
                            pending["filename"],
                            pending["title"],
                            document_type=pending["document_type"],
                            symbol=pending["symbol"]
                        )
                        
                        if result["success"]:
                            doc_id = result.get("document_id")
                            if doc_id:
                                # Update the document with analysis results and embedding
                                try:
                                    analysis = pending.get("analysis", {})
                                    signals = pending.get("signals", {})
                                    
                                    # Generate embedding from content text
                                    embedding_vector = doc_manager._generate_embedding(pending["content_text"])
                                    
                                    # Build update payload - only include fields that exist in schema
                                    update_payload = {}
                                    
                                    # Always try to update embedding (embedding_vector column)
                                    if embedding_vector is not None and len(embedding_vector) > 0:
                                        update_payload["embedding_vector"] = embedding_vector
                                        st.info(f"📊 Generated embedding vector ({len(embedding_vector)} dimensions)")
                                    else:
                                        gemini_key = os.getenv("GEMINI_API_KEY", "")
                                        if not gemini_key:
                                            st.warning("⚠️ Embedding vector not generated. Please set GEMINI_API_KEY in your .env file to enable embeddings.")
                                        else:
                                            st.warning("⚠️ Embedding vector not generated. Check console for error details.")
                                    
                                    # Try optional fields - handle gracefully if columns don't exist
                                    optional_fields = {}
                                    
                                    if analysis.get("success"):
                                        # Store analysis in file_content or metadata if ai_analysis_text doesn't exist
                                        # Try ai_analysis_text first, fallback to appending to file_content
                                        optional_fields["ai_analysis_text"] = analysis.get("analysis")
                                    
                                    if signals.get("success"):
                                        optional_fields.update({
                                            "overall_sentiment": signals.get("overall_sentiment"),
                                            "signal_confidence": signals.get("confidence"),
                                            "bullish_signals": signals.get("bullish_signals"),
                                            "bearish_signals": signals.get("bearish_signals"),
                                        })
                                    
                                    # Add timestamp if column exists
                                    optional_fields["last_analyzed_at"] = datetime.now().isoformat()
                                    
                                    # Try to update with all fields
                                    try:
                                        full_payload = {**update_payload, **optional_fields}
                                        doc_manager.supabase.table("research_documents").update(full_payload).eq("id", doc_id).execute()
                                        st.info("✅ Document saved with analysis and embedding!")
                                    except Exception as update_error:
                                        # If full update fails, try just embedding (columns don't exist)
                                        error_str = str(update_error)
                                        if "PGRST204" in error_str or "Could not find" in error_str:
                                            # Column doesn't exist, store everything in file_content
                                            try:
                                                # Only update embedding (embedding_vector should exist)
                                                if embedding_vector is not None and len(embedding_vector) > 0:
                                                    doc_manager.supabase.table("research_documents").update({
                                                        "embedding_vector": embedding_vector
                                                    }).eq("id", doc_id).execute()
                                                    st.info(f"✅ Embedding vector saved ({len(embedding_vector)} dimensions)")
                                                else:
                                                    gemini_key = os.getenv("GEMINI_API_KEY", "")
                                                    if not gemini_key:
                                                        st.warning("⚠️ No embedding vector to save. Set GEMINI_API_KEY in .env file to enable embeddings.")
                                                    else:
                                                        st.warning("⚠️ No embedding vector to save. Check console for error details.")
                                                
                                                # Store full analysis in file_content as JSON
                                                current_doc = doc_manager.get_document_content(doc_id)
                                                if current_doc:
                                                    import json
                                                    analysis_metadata = {
                                                        "ai_analysis": analysis.get("analysis") if analysis.get("success") else None,
                                                        "signals": signals if signals.get("success") else None,
                                                        "analyzed_at": datetime.now().isoformat()
                                                    }
                                                    # Append analysis metadata to file_content as JSON
                                                    analysis_json_str = json.dumps(analysis_metadata, indent=2)
                                                    updated_content = current_doc + f"\n\n--- AI ANALYSIS METADATA ---\n{analysis_json_str}"
                                                    doc_manager.supabase.table("research_documents").update({
                                                        "file_content": updated_content
                                                    }).eq("id", doc_id).execute()
                                                
                                                st.info("✅ Document saved with embedding! Analysis metadata stored in file_content (some columns not found in schema).")
                                            except Exception as e2:
                                                # If even embedding fails, at least document is saved
                                                st.info("✅ Document saved! Some metadata couldn't be updated, but document is stored.")
                                        else:
                                            st.warning(f"⚠️ Document saved but analysis update failed: {error_str}")
                                except Exception as e:
                                    st.warning(f"⚠️ Document saved but analysis update failed: {e}")
                            
                            st.success("✅ Document saved to database successfully!")
                            st.balloons()
                            
                            # Clear pending document
                            del st.session_state.pending_document
                            
                            # Store in document analyses history
                            if "document_analyses" not in st.session_state:
                                st.session_state.document_analyses = []
                            st.session_state.document_analyses.append({
                                "symbol": pending["symbol"],
                                "title": pending["title"],
                                "analysis": pending.get("analysis", {}),
                                "signals": pending.get("signals", {})
                            })
                        else:
                            st.error(f"Failed to save document: {result.get('error', 'Unknown error')}")
        
        # Display existing analyses
        if "document_analyses" in st.session_state and st.session_state.document_analyses:
            st.subheader("Recent Document Analyses")
            for analysis in st.session_state.document_analyses[-5:]:  # Show last 5
                with st.expander(f"{analysis['symbol']} - {analysis['title']}", expanded=False):
                    if analysis['analysis'].get('success'):
                        st.write("**Analysis:**", analysis['analysis']['analysis'])
                    else:
                        st.write("**Analysis:**", f"Failed: {analysis['analysis'].get('error', 'Unknown error')}")
                    if analysis['signals'].get('success'):
                        st.write("**Signals:**", analysis['signals']['overall_sentiment'])
                    else:
                        st.write("**Signals:**", f"Failed: {analysis['signals'].get('error', 'Unknown error')}")
    
    with tab3:
        st.subheader("Instrument Profiles")
        
        # Explanation of data sources
        with st.expander("📖 How Instrument Profiles Work - Data Sources Explained", expanded=False):
            st.markdown("""
            ### **Data Flow & Sources:**
            
            **1. Market Data** (`profile['market_data']`):
            - **Source**: Supabase tables (`market_data`, `market_data_5min`, `market_data_1min`) populated by the ingestion pipeline
            - **Method**: `_get_market_data()` in `ai_analysis.py` lines 59-138
            - **Data Retrieved**: 
              - Last 30 days of OHLCV data
              - Calculates: current_price, price_change_pct, volume, high_30d, low_30d, volatility
            - **Location**: `tradingagents/dataflows/ai_analysis.py` → `_get_market_data()`
            
            **2. Document Insights** (`profile['document_insights']`):
            - **Source**: `research_documents` table in Supabase
            - **Method**: `document_manager.get_document_insights(symbol)` 
            - **Data Retrieved**:
              - All documents uploaded for this symbol
              - Extracts: filename, signals (bullish/bearish keywords), sentiment, confidence
            - **Location**: `tradingagents/dataflows/document_manager.py` → `get_document_insights()`
            
            **3. News Sentiment** (`profile['news_sentiment']`):
            - **Source**: Placeholder (not yet implemented with real news API)
            - **Method**: `_analyze_news_sentiment()` in `ai_analysis.py` lines 140-154
            - **Returns**: Default structure with sentiment_score=0, news_count=0
            - **Location**: `tradingagents/dataflows/ai_analysis.py` → `_analyze_news_sentiment()`
            
            **4. AI Analysis** (`profile['ai_analysis']`):
            - **Source**: LLM (Groq/Llama-3.1-8b-instant)
            - **Method**: `_generate_comprehensive_analysis()` in `ai_analysis.py` lines 156-269
            - **Input**: Combines market_data + doc_insights + news_sentiment
            - **LLM Prompt**: See `ai_analysis.py` lines 192-210
            - **Output**: 
              - `analysis_text`: Full LLM response
              - `overall_sentiment`: Extracted from text ("Bullish"/"Bearish"/"Neutral")
              - `recommendation`: Extracted from text ("BUY"/"SELL"/"HOLD")
              - `confidence`: Extracted from text (1-10)
            - **Location**: `tradingagents/dataflows/ai_analysis.py` → `_generate_comprehensive_analysis()`
            
            **5. Confidence Score** (`profile['confidence_score']`):
            - **Source**: Calculated from doc_insights + news_sentiment
            - **Method**: `_calculate_confidence_score()` in `ai_analysis.py` lines 319-346
            - **Formula**: `mean([doc_insights confidence scores, news_count/10])`
            - **Location**: `tradingagents/dataflows/ai_analysis.py` → `_calculate_confidence_score()`
            
            ### **Complete Flow:**
            1. User clicks "Generate Profile" → calls `analyze_instrument_profile(symbol)`
            2. System fetches: Market Data (Supabase tables) + Document Insights (DB) + News Sentiment (placeholder)
            3. All data combined into prompt → sent to LLM (Groq)
            4. LLM generates comprehensive analysis text
            5. System extracts sentiment, recommendation, confidence from LLM response
            6. Profile displayed in UI with all components
            7. When you click **"Save Instrument Profile"**, the profile is stored in Supabase table `instrument_profiles` with:
               - `profile_text` (LLM-ready text for embeddings)
               - `embedding_vector` (Gemini-generated RAG vector)
               - `profile_data` (full JSON payload for later retrieval)
            """)
        
        # Select symbol for detailed analysis
        selected_symbol = st.selectbox("Select Instrument for Analysis", symbols)
        
        if selected_symbol and st.button("Generate Profile", type="primary", key="generate_profile"):
            with st.spinner(f"Analyzing {selected_symbol}..."):
                profile = ai_analyzer.analyze_instrument_profile(selected_symbol)
                
                if "error" not in profile:
                    st.session_state.current_profile = profile  # Store for saving
                    st.success(f"Profile generated for {selected_symbol}")
                    st.balloons()
                    
                    # Display comprehensive profile
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Market Data")
                        st.caption("**Source**: Supabase (`market_data`) - 30-day historical data ingested via Polygon/DB pipelines → `_get_market_data()`")
                        market_data = profile.get("market_data", {})
                        if "error" not in market_data:
                            data_source = market_data.get("source", "unknown")
                            st.caption(f"Data source: {data_source}")
                            st.metric("Current Price", f"${market_data.get('current_price', 'N/A')}")
                            st.metric("30-day Change", f"{market_data.get('price_change_pct', 'N/A')}%")
                            st.metric("Volume", f"{market_data.get('volume', 'N/A'):,}")
                            st.metric("30-day High", f"${market_data.get('high_30d', 'N/A')}")
                            st.metric("30-day Low", f"${market_data.get('low_30d', 'N/A')}")
                        else:
                            st.error(f"Market data unavailable: {market_data.get('error', 'Unknown error')}")
                    
                    with col2:
                        st.subheader("AI Analysis")
                        st.caption("**Source**: LLM (Groq/Llama-3.1-8b-instant) → `_generate_comprehensive_analysis()`")
                        ai_analysis = profile.get("ai_analysis", {})
                        if "error" not in ai_analysis:
                            st.write("**Overall Assessment:**", ai_analysis.get("overall_sentiment", "N/A"))
                            st.write("**Recommendation:**", ai_analysis.get("recommendation", "N/A"))
                            st.write("**Confidence:**", f"{ai_analysis.get('confidence', 'N/A')}/10")
                            
                            # Display full analysis
                            st.subheader("Detailed Analysis")
                            st.write(ai_analysis.get("analysis_text", "No analysis available"))
                        else:
                            error_msg = ai_analysis.get("error", "Unknown error")
                            details = ai_analysis.get("details", "")
                            st.error(f"AI analysis unavailable: {error_msg}")
                            if details:
                                with st.expander("Error Details", expanded=False):
                                    st.code(details)
                    
                    # Document insights
                    doc_insights = profile.get("document_insights", [])
                    if doc_insights:
                        st.subheader("Document Insights")
                        st.caption(f"**Source**: `research_documents` table → {len(doc_insights)} document(s) found for {selected_symbol}")
                        for insight in doc_insights:
                            with st.expander(f"📄 {insight.get('filename', 'Unknown')}", expanded=False):
                                signals = insight.get("signals", {})
                                if signals.get("success"):
                                    st.write(f"**Sentiment:** {signals.get('overall_sentiment', 'N/A')}")
                                    st.write(f"**Confidence:** {signals.get('confidence', 'N/A')}/10")
                                    st.write(f"**Bullish Signals:** {', '.join(signals.get('bullish_signals', []))}")
                                    st.write(f"**Bearish Signals:** {', '.join(signals.get('bearish_signals', []))}")
                    else:
                        st.info(f"No documents found for {selected_symbol} in `research_documents` table")
                    
                    # Save profile to database with RAG
                    st.markdown("---")
                    st.info("💾 Save this instrument profile to `instrument_profiles` table with RAG embeddings for semantic search.")
                    if st.button("💾 Save Instrument Profile to Database (RAG)", type="primary", key="save_instrument_profile"):
                        with st.spinner("💾 Saving instrument profile with embeddings..."):
                            try:
                                from tradingagents.database.db_service import save_instrument_profile_with_rag
                                
                                # Convert profile to text for embedding
                                content_text = ai_analyzer._master_data_to_text(profile)
                                
                                # Generate embedding
                                embedding_vector = ai_analyzer.document_manager._generate_embedding(content_text)
                                
                                # Save to database
                                result = save_instrument_profile_with_rag(
                                    symbol=selected_symbol,
                                    profile_text=content_text,
                                    embedding_vector=embedding_vector,
                                    profile_data=profile,
                                    analysis_timestamp=profile.get("analysis_timestamp")
                                )
                                
                                if result:
                                    # Show detailed confirmation
                                    st.success("✅ **Instrument Profile Saved Successfully!**")
                                    st.info(f"""
                                    **📊 Saved to Database:**
                                    - **Table**: `instrument_profiles`
                                    - **Symbol**: {selected_symbol}
                                    - **Storage Type**: RAG (Retrieval-Augmented Generation)
                                    - **Record ID**: {result.get('id', 'N/A')}
                                    - **Analysis Timestamp**: {result.get('analysis_timestamp', 'N/A')}
                                    - **Generated At**: {result.get('generated_at', 'N/A')}
                                    
                                    **🔍 What Was Saved:**
                                    - Full instrument profile (market data, AI analysis, document insights)
                                    - Text representation for semantic search
                                    - Embedding vector for RAG queries
                                    - Complete JSON data in `profile_data` column
                                    """)
                                    
                                    if embedding_vector:
                                        st.success(f"📊 **Embedding Vector Generated**: {len(embedding_vector)} dimensions (stored in `embedding_vector` JSONB column)")
                                    else:
                                        gemini_key = os.getenv("GEMINI_API_KEY", "")
                                        if not gemini_key:
                                            st.warning("⚠️ **Embedding vector not generated**. Set `GEMINI_API_KEY` in `.env` file to enable RAG embeddings.")
                                        else:
                                            st.warning("⚠️ **Embedding vector not generated**. Check console for error details.")
                                    
                                    st.balloons()
                                else:
                                    st.error("❌ Failed to save instrument profile to `instrument_profiles` table")
                            except Exception as e:
                                error_msg = str(e)
                                if "does not exist" in error_msg.lower() or "instrument_profiles" in error_msg.lower():
                                    st.error(f"❌ Database table missing: {error_msg}")
                                    st.info("💡 Please create the `instrument_profiles` table first (see SQL migration below).")
                                else:
                                    st.error(f"❌ Error saving profile: {error_msg}")
                                    import traceback
                                    st.code(traceback.format_exc())
                else:
                    st.error(f"Error generating profile: {profile['error']}")
    
    with tab4:
        st.subheader("Research Insights Dashboard")
        
        # Explanation of how insights are extracted
        with st.expander("📖 How Research Insights Are Extracted - Data Flow Explained", expanded=False):
            st.markdown("""
            ### **Data Source & Extraction Process:**
            
            **All insights are dynamically extracted from `st.session_state.master_data_summary`** - NOT hardcoded!
            
            **1. Portfolio Overview:**
            - **Source**: `master_data_summary["instruments"]` (list of 19 instrument profiles)
            - **Extraction Method**: Iterates through each instrument in the summary
            - **Data Retrieved**:
              ```python
              for inst in summary["instruments"]:
                  ai_analysis = inst.get("ai_analysis", {})
                  # Extracts:
                  - Symbol: inst["symbol"]
                  - Sentiment: ai_analysis.get("overall_sentiment", "N/A")
                  - Recommendation: ai_analysis.get("recommendation", "N/A")
                  - Confidence: ai_analysis.get("confidence", "N/A")
                  - Documents: len(inst.get("document_insights", []))
              ```
            - **Location**: `app.py` lines 1903-1912
            - **Origin of Data**:
              - `overall_sentiment`: Extracted from LLM response text via `_extract_sentiment_from_analysis()` (looks for "bullish"/"bearish" keywords) → `ai_analysis.py` line 290-298
              - `recommendation`: Extracted from LLM response text via `_extract_recommendation()` (looks for "BUY"/"SELL"/"HOLD" keywords) → `ai_analysis.py` line 300-308
              - `confidence`: Extracted from LLM response text via `_extract_confidence()` (extracts number 1-10 using regex) → `ai_analysis.py` line 310-317
            - **NOT Hardcoded**: All values come from LLM analysis stored in `master_data_summary`
            
            **2. Sentiment Distribution:**
            - **Source**: `insights_df["Sentiment"]` column (from Portfolio Overview dataframe)
            - **Extraction Method**: `pandas.value_counts()` - counts occurrences of each sentiment value
            - **Calculation**:
              ```python
              sentiment_counts = insights_df["Sentiment"].value_counts()
              # Returns: {"Bullish": 8, "Bearish": 5, "Neutral": 6}
              ```
            - **Location**: `app.py` lines 1923-1935
            - **Dynamic**: Automatically counts whatever sentiment values exist (Bullish/Bearish/Neutral/N/A)
            - **NOT Hardcoded**: Distribution changes based on actual LLM responses
            
            **3. Recommendation Distribution:**
            - **Source**: `insights_df["Recommendation"]` column (from Portfolio Overview dataframe)
            - **Extraction Method**: `pandas.value_counts()` - counts occurrences of each recommendation value
            - **Calculation**:
              ```python
              rec_counts = insights_df["Recommendation"].value_counts()
              # Returns: {"BUY": 10, "HOLD": 6, "SELL": 3}
              ```
            - **Location**: `app.py` lines 1940-1952
            - **Dynamic**: Automatically counts whatever recommendation values exist (BUY/SELL/HOLD/N/A)
            - **NOT Hardcoded**: Distribution changes based on actual LLM responses
            
            ### **Complete Data Flow:**
            1. User clicks "Generate/Refresh Master Data Summary"
            2. System calls `ai_analyzer.get_master_data_summary()` → analyzes all 19 instruments
            3. For each instrument:
               - Fetches market data (Polygon API)
               - Fetches document insights (from `research_documents` table)
               - Calls LLM (Groq) with combined context
               - Extracts sentiment, recommendation, confidence from LLM response
            4. All results stored in `st.session_state.master_data_summary`
            5. Research Insights tab reads from this session state
            6. Portfolio Overview: Extracts data from `summary["instruments"]` list
            7. Sentiment/Recommendation Distribution: Counts values using `pandas.value_counts()`
            
            ### **Key Points:**
            - ✅ **NOT Hardcoded**: All values come from LLM analysis
            - ✅ **Dynamic**: Changes based on actual market data and LLM responses
            - ✅ **Real-time**: Reflects current analysis when master data is regenerated
            - ✅ **Data Source**: `st.session_state.master_data_summary` (generated by clicking "Generate Master Data Summary")
            """)
        
        # Overall insights summary
        if "master_data_summary" in st.session_state:
            summary = st.session_state.master_data_summary
            
            st.subheader("Portfolio Overview")
            st.caption("**Source**: Extracted from `master_data_summary['instruments']` → Each instrument's `ai_analysis` dict (from LLM responses)")
            
            # Create insights dataframe
            insights_data = []
            for inst in summary["instruments"]:
                ai_analysis = inst.get("ai_analysis", {})
                insights_data.append({
                    "Symbol": inst["symbol"],
                    "Sentiment": ai_analysis.get("overall_sentiment", "N/A"),
                    "Recommendation": ai_analysis.get("recommendation", "N/A"),
                    "Confidence": ai_analysis.get("confidence", "N/A"),
                    "Documents": len(inst.get("document_insights", []))
                })
            
            if insights_data:
                insights_df = pd.DataFrame(insights_data)
                # Ensure proper types for Arrow serialization
                for col in insights_df.columns:
                    if insights_df[col].dtype == 'object':
                        insights_df[col] = insights_df[col].astype(str)
                st.dataframe(insights_df, use_container_width=True)
                
                # Sentiment distribution
                st.subheader("Sentiment Distribution")
                st.caption("**Source**: Counts sentiment values from Portfolio Overview using `pandas.value_counts()` → NOT hardcoded, dynamically calculated from LLM responses")
                sentiment_counts = insights_df["Sentiment"].value_counts()
                if len(sentiment_counts) > 0:
                    st.bar_chart(sentiment_counts)
                    # Show breakdown
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Bullish", sentiment_counts.get("Bullish", 0))
                    with col2:
                        st.metric("Bearish", sentiment_counts.get("Bearish", 0))
                    with col3:
                        st.metric("Neutral", sentiment_counts.get("Neutral", 0))
                else:
                    st.info("No sentiment data available")
                
                # Recommendation distribution
                st.subheader("Recommendation Distribution")
                st.caption("**Source**: Counts recommendation values from Portfolio Overview using `pandas.value_counts()` → NOT hardcoded, dynamically calculated from LLM responses")
                rec_counts = insights_df["Recommendation"].value_counts()
                if len(rec_counts) > 0:
                    st.bar_chart(rec_counts)
                    # Show breakdown
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("BUY", rec_counts.get("BUY", 0))
                    with col2:
                        st.metric("HOLD", rec_counts.get("HOLD", 0))
                    with col3:
                        st.metric("SELL", rec_counts.get("SELL", 0))
                else:
                    st.info("No recommendation data available")
            else:
                st.warning("No insights data available in master_data_summary")
        else:
            st.info("💡 **Generate master data summary first** to see research insights. Click 'Generate/Refresh Master Data Summary' in the Master Data Dashboard tab.")
    
    # Close AI analyzer
    ai_analyzer.close()

def _calc_sma(series, window):
    return series.rolling(window=window).mean()

def _calc_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _calc_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def _fetch_history(symbol, period="6mo", interval="1d"):
    lookback_days = period_to_days(period, default=180)
    data = fetch_ohlcv(symbol, interval=interval, lookback_days=lookback_days)
    if data is None or data.empty:
        return None
    return data

def run_volume_screening(symbols):
    rows = []
    for sym in symbols:
        hist = _fetch_history(sym, period="3mo")
        if hist is None or hist.empty or len(hist) < 20:
            rows.append({
                "Symbol": sym,
                "AvgVol20": np.nan,
                "TodayVol": np.nan,
                "VolSpike": np.nan,
                "SMA50_SMA200": np.nan,
                "RSI": np.nan,
                "Pass": False
            })
            continue
        avg20 = hist['Volume'].tail(20).mean()
        today_vol = hist['Volume'].iloc[-1]
        vol_spike = today_vol / avg20 if avg20 and not np.isnan(avg20) and avg20 > 0 else np.nan
        sma50 = _calc_sma(hist['Close'], 50)
        sma200 = _calc_sma(hist['Close'], 200)
        rsi = _calc_rsi(hist['Close']).iloc[-1]
        sma_spread = (sma50.iloc[-1] - sma200.iloc[-1]) if not (pd.isna(sma50.iloc[-1]) or pd.isna(sma200.iloc[-1])) else np.nan
        passed = (not np.isnan(vol_spike) and vol_spike >= 1.5) and (not np.isnan(sma_spread) and sma_spread > 0) and (30 < rsi < 70)
        rows.append({
            "Symbol": sym,
            "AvgVol20": int(avg20) if not np.isnan(avg20) else np.nan,
            "TodayVol": int(today_vol) if not np.isnan(today_vol) else np.nan,
            "VolSpike": round(vol_spike, 2) if not np.isnan(vol_spike) else np.nan,
            "SMA50_SMA200": round(sma_spread, 2) if not np.isnan(sma_spread) else np.nan,
            "RSI": round(float(rsi), 1) if not np.isnan(rsi) else np.nan,
            "Pass": passed
        })
    return pd.DataFrame(rows)

def run_seven_stage_fire_test(symbol):
    result = {"symbol": symbol, "stages": [], "score": 0, "max_score": 7}
    hist = _fetch_history(symbol, period="6mo")
    if hist is None or hist.empty:
        result["error"] = "No price data"
        return result

    # Helper to safely extract a 1-D Series for a given OHLCV column, handling MultiIndex
    def _get_col_series(df, col_name):
        if not isinstance(df, pd.DataFrame):
            return None
        s = None
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
                        s = None
        except Exception:
            s = None
        if s is None:
            return None
        if isinstance(s, pd.DataFrame):
            if s.shape[1] >= 1:
                s = s.iloc[:, 0]
            else:
                return None
        # Ensure 1-D series
        if not isinstance(s, pd.Series):
            try:
                arr = np.ravel(s)
                s = pd.Series(arr, index=df.index[: len(arr)])
            except Exception:
                return None
        return s

    open_s = _get_col_series(hist, 'Open')
    high_s = _get_col_series(hist, 'High')
    low_s = _get_col_series(hist, 'Low')
    close_s = _get_col_series(hist, 'Close')
    vol_s = _get_col_series(hist, 'Volume')

    # If essential columns missing, abort gracefully
    if close_s is None or high_s is None or low_s is None or vol_s is None:
        result["error"] = "Insufficient OHLCV data"
        return result

    # Stage 1: Liquidity (volume adequacy)
    vol_tail = pd.to_numeric(vol_s.tail(20), errors='coerce')
    avg20 = float(vol_tail.mean()) if len(vol_tail) else np.nan
    pass1 = pd.notna(avg20) and avg20 >= 1_000_000
    result["stages"].append({
        "name": "Liquidity",
        "pass": pass1,
        "detail": f"Avg20Vol={int(avg20)}" if pd.notna(avg20) else "Avg20Vol=NA"
    })
    result["score"] += 1 if pass1 else 0

    # Stage 2: Volatility sanity (ATR relative to price)
    price = close_s.iloc[-1]
    atr_df = pd.DataFrame({
        'High': pd.to_numeric(high_s, errors='coerce'),
        'Low': pd.to_numeric(low_s, errors='coerce'),
        'Close': pd.to_numeric(close_s, errors='coerce'),
    })
    atr = _calc_atr(atr_df)
    atr_last = atr.iloc[-1] if len(atr) else np.nan
    atr_pct = (atr_last / price) * 100 if pd.notna(atr_last) and pd.notna(price) and price != 0 else np.nan
    pass2 = pd.notna(atr_pct) and 1 <= atr_pct <= 8
    result["stages"].append({
        "name": "Volatility",
        "pass": pass2,
        "detail": f"ATR%={atr_pct:.2f}%" if pd.notna(atr_pct) else "ATR%=NA"
    })
    result["score"] += 1 if pass2 else 0

    # Stage 3: Trend (SMA50 > SMA200)
    sma50 = _calc_sma(close_s, 50)
    sma200 = _calc_sma(close_s, 200)
    sma50_last = sma50.iloc[-1] if len(sma50) else np.nan
    sma200_last = sma200.iloc[-1] if len(sma200) else np.nan
    pass3 = pd.notna(sma50_last) and pd.notna(sma200_last) and sma50_last > sma200_last
    detail3 = f"SMA50-SMA200={(sma50_last - sma200_last):.2f}" if pd.notna(sma50_last) and pd.notna(sma200_last) else "SMA50-SMA200=NA"
    result["stages"].append({"name": "Trend", "pass": pass3, "detail": detail3})
    result["score"] += 1 if pass3 else 0

    # Stage 4: Momentum (RSI between 40 and 65)
    rsi_series = _calc_rsi(close_s)
    rsi = rsi_series.iloc[-1] if len(rsi_series) else np.nan
    pass4 = pd.notna(rsi) and 40 <= rsi <= 65
    result["stages"].append({"name": "Momentum", "pass": pass4, "detail": f"RSI={rsi:.1f}" if pd.notna(rsi) else "RSI=NA"})
    result["score"] += 1 if pass4 else 0

    # Stage 5: Breakout check (close above recent range high)
    recent_high = pd.to_numeric(high_s.tail(20), errors='coerce').max()
    pass5 = pd.notna(price) and pd.notna(recent_high) and price > recent_high * 0.995  # within 0.5% of breakout
    detail5 = f"Px={price:.2f} vs 20dHigh={recent_high:.2f}" if pd.notna(price) and pd.notna(recent_high) else "Px/High=NA"
    result["stages"].append({"name": "Breakout", "pass": pass5, "detail": detail5})
    result["score"] += 1 if pass5 else 0

    # Stage 6: Risk (support proximity using SMA20)
    sma20 = _calc_sma(close_s, 20)
    support = sma20.iloc[-1] if len(sma20) and pd.notna(sma20.iloc[-1]) else price
    risk_pct = ((price - support) / price) * 100 if pd.notna(price) and price != 0 and pd.notna(support) else np.nan
    pass6 = pd.notna(risk_pct) and 0 <= risk_pct <= 5
    result["stages"].append({"name": "Risk", "pass": pass6, "detail": f"Risk%={risk_pct:.2f}" if pd.notna(risk_pct) else "Risk%=NA"})
    result["score"] += 1 if pass6 else 0

    # Stage 7: AI sentiment integration (from Phase 2 analyzer if available)
    ai_pass = False
    ai_detail = "No AI"
    try:
        analyzer = AIResearchAnalyzer()
        profile = analyzer.analyze_instrument_profile(symbol)
        analyzer.close()
        ai = profile.get("ai_analysis", {}) if isinstance(profile, dict) else {}
        sentiment = ai.get("overall_sentiment")
        confidence = ai.get("confidence", 5)
        ai_pass = sentiment in ["Bullish", "Neutral"] and float(confidence) >= 5
        ai_detail = f"Sent={sentiment}, Conf={confidence}"
    except Exception as _:
        ai_pass = False
        ai_detail = "AI error"
    result["stages"].append({"name": "AI Sentiment", "pass": ai_pass, "detail": ai_detail})
    result["score"] += 1 if ai_pass else 0

    return result

def ai_enhanced_recommendation(symbol):
    fire = run_seven_stage_fire_test(symbol)
    score_ratio = fire["score"] / fire["max_score"] if fire.get("max_score") else 0
    base_rec = "Hold"
    if score_ratio >= 0.85:
        base_rec = "Strong Buy"
    elif score_ratio >= 0.6:
        base_rec = "Buy"
    elif score_ratio <= 0.3:
        base_rec = "Sell"

    ai_conf = 5.0
    try:
        analyzer = AIResearchAnalyzer()
        profile = analyzer.analyze_instrument_profile(symbol)
        analyzer.close()
        ai = profile.get("ai_analysis", {}) if isinstance(profile, dict) else {}
        ai_conf = float(ai.get("confidence", 5))
    except Exception:
        ai_conf = 5.0

    final_conf = round(0.7 * (score_ratio * 10) + 0.3 * ai_conf, 1)
    return {"symbol": symbol, "recommendation": base_rec, "confidence": final_conf, "fire_test": fire}


def _render_datascan_output(datascan_result: dict):
    """Display the DataScan engine output in the UI."""
    timeframe_stats = datascan_result.get("timeframe_stats") or {}
    if timeframe_stats:
        rows = []
        for tf_label, info in timeframe_stats.items():
            metric = info.get("trend")
            if not metric:
                rsi_val = info.get("rsi14")
                metric = f"RSI14={rsi_val}" if rsi_val is not None else ""
            rows.append({
                "Timeframe": tf_label.upper(),
                "Records": info.get("records", "N/A"),
                "Start": info.get("start", "-"),
                "End": info.get("end", "-"),
                "Key Metric": metric or ""
            })
        stats_df = pd.DataFrame(rows)
        st.dataframe(stats_df, use_container_width=True)

    if datascan_result.get("report"):
        st.markdown("##### DataScan Report")
        st.markdown(datascan_result["report"])

    if datascan_result.get("prompt"):
        with st.expander("View Generated Prompt Payload", expanded=False):
            st.code(datascan_result["prompt"], language="markdown")

    samples = datascan_result.get("samples") or {}
    if samples:
        with st.expander("Recent Candles Preview (D1 / M5 / M1)", expanded=False):
            for label in ("daily", "m5", "m1"):
                sample_table = samples.get(label)
                if sample_table:
                    st.markdown(f"**{label.upper()}**")
                    st.markdown(sample_table)

    if datascan_result.get("message"):
        st.warning(datascan_result["message"])

def phase3_trading_engine_core():
    """Phase 3: Trading Engine Core - Complete Phase 1 + Phase 2 Analysis Workflow"""
    
    # Initialize engines
    if 'volume_screener' not in st.session_state:
        st.session_state.volume_screener = VolumeScreeningEngine()
    
    if 'fire_tester' not in st.session_state:
        st.session_state.fire_tester = FireTestingEngine()
    
    if 'ai_scorer' not in st.session_state:
        st.session_state.ai_scorer = AIEnhancedScoringEngine()

    if 'datascan_engine' not in st.session_state:
        st.session_state.datascan_engine = HistoricalDataScanEngine()
    
    symbols = get_watchlist_symbols()
    
    # Main tabs for Phase 3
    tab1, tab2, tab3 = st.tabs([
        "📊 Phase 1: Volume Screening", 
        "🔥 Phase 2: Deep Analysis", 
        "⚡ Trading Workflow"
    ])
    
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### Phase 1: Volume Screening Engine")
        st.write("""
        **Purpose**: Identify stocks with volume spikes and positive momentum.
        
        **Criteria**:
        - Volume spike >= 1.3x average (20-day)
        - SMA50 > SMA200 (or SMA20 > SMA50 if SMA200 unavailable) - uptrend
        - RSI between 30-70 (momentum)
        - Average volume >= 1M shares (liquidity)
        """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Run Volume Screening", type="primary", key="run_phase1_screening", use_container_width=True):
                with st.spinner("Screening watchlist for volume spikes..."):
                    screening_df = st.session_state.volume_screener.screen_watchlist(symbols)
                    
                    # Store results in session state
                    st.session_state.phase1_results = screening_df
                    
                    # Show results
                    st.success(f"✅ Screening complete! Found {len(screening_df[screening_df['Pass'] == '✅'])} candidates")
                    
                    # Display results
                    st.dataframe(screening_df, use_container_width=True, height=400)
                    
                    # Show passed symbols
                    passed_symbols = screening_df[screening_df['Pass'] == '✅']['Symbol'].tolist()
                    if passed_symbols:
                        st.info(f"📈 **Passed Symbols**: {', '.join(passed_symbols)}")
                        st.session_state.phase1_passed = passed_symbols
                    else:
                        st.warning("⚠️ No symbols passed Phase 1 screening")
                        st.session_state.phase1_passed = []
                    
                    # Save button for volume screening results
                    st.markdown("---")
                    if st.button("💾 Save Screening Results to Database", key="save_screening_results"):
                        try:
                            from tradingagents.database.config import get_supabase
                            from tradingagents.database.db_service import (
                                _make_json_serializable,
                                save_volume_screening_with_rag
                            )

                            supabase = get_supabase()
                            if not supabase:
                                st.error("❌ Database not configured! Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
                            else:
                                # Build summary text for RAG embedding
                                summary_sections = [
                                    f"Volume Screening Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                                    f"Total Symbols Screened: {len(screening_df)}",
                                    f"Passed Symbols ({len(passed_symbols)}): {', '.join(passed_symbols) if passed_symbols else 'None'}"
                                ]
                                summary_sections.append("Top Results:")
                                top_rows = screening_df.head(10).to_dict('records')
                                for row in top_rows:
                                    summary_sections.append(
                                        f"- {row['Symbol']}: VolSpike={row['VolSpike']}, AvgVol20={row['AvgVol20']}, RSI={row['RSI']}, Pass={row['Pass']}"
                                    )
                                summary_text = "\n".join(summary_sections)

                                # Generate embedding via DocumentManager
                                doc_manager = DocumentManager()
                                try:
                                    embedding_vector = doc_manager._generate_embedding(summary_text)
                                finally:
                                    doc_manager.close()

                                screening_payload = {
                                    "total_screened": int(len(screening_df)),
                                    "passed_count": int(len(passed_symbols)),
                                    "failed_count": int(len(screening_df) - len(passed_symbols)),
                                    "passed_symbols": list(passed_symbols),
                                    "screening_results": screening_df.to_dict('records')
                                }

                                metadata = {
                                    "watchlist_size": len(symbols),
                                    "run_source": "phase3_tab1",
                                    "generated_by": "volume_screening_engine"
                                }

                                result = save_volume_screening_with_rag(
                                    screening_results=_make_json_serializable(screening_payload),
                                    summary_text=summary_text,
                                    embedding_vector=embedding_vector,
                                    run_metadata=metadata
                                )

                                if result:
                                    st.success("✅ Screening results saved to `volume_screening_runs` with RAG embedding!")
                                    st.balloons()
                                else:
                                    st.error("❌ Failed to save screening results")
                        except Exception as e:
                            st.error(f"❌ Error saving: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        with col2:
            if st.button("🔄 Clear Results", key="clear_phase1"):
                if 'phase1_results' in st.session_state:
                    del st.session_state.phase1_results
                if 'phase1_passed' in st.session_state:
                    del st.session_state.phase1_passed
                st.rerun()
        
        # Removed Previous Screening Results section per request
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### Phase 2: 7-Stage Fire Testing & AI Analysis")
        st.write("""
        **Purpose**: Deep analysis combining technical indicators with AI insights from research documents.
        
        **7 Stages**:
        1. **Liquidity**: Average volume >= 1M shares
        2. **Volatility**: ATR between 1-8% of price
        3. **Trend**: SMA50 > SMA200 (uptrend)
        4. **Momentum**: RSI between 40-65
        5. **Breakout**: Price near 20-day high
        6. **Risk**: Support distance <= 5%
        7. **AI Sentiment**: Research document analysis
        """)
        
        # Symbol selection for Phase 2
        phase2_symbols = st.session_state.get('phase1_passed', symbols)
        
        if not phase2_symbols:
            st.warning("⚠️ No symbols passed Phase 1. Select any symbol for Phase 2 analysis.")
            phase2_symbols = symbols
        
        selected_symbol = st.selectbox(
            "Select Symbol for Phase 2 Analysis",
            phase2_symbols,
            key="phase2_symbol_selector",
            help="Ideally choose from Phase 1 passed symbols"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔥 Run 7-Stage Fire Test", type="primary", key="run_phase2_firetest", use_container_width=True):
                with st.spinner(f"Running comprehensive analysis for {selected_symbol}..."):
                    fire_result = st.session_state.fire_tester.run_fire_test(selected_symbol)
                    
                    # Store results
                    st.session_state.phase2_results = fire_result
        
        with col2:
            if st.button("🔄 Clear Results", key="clear_phase2"):
                if 'phase2_results' in st.session_state:
                    del st.session_state.phase2_results
                st.rerun()

        # Display results (Persistent View)
        if 'phase2_results' in st.session_state:
            fire_result = st.session_state.phase2_results
            
            if fire_result.get("error"):
                st.error(f"❌ Error: {fire_result['error']}")
            else:
                # Display results
                score_ratio = fire_result["score"] / fire_result["max_score"]
                score_color = "green" if score_ratio >= 0.7 else "orange" if score_ratio >= 0.5 else "red"
                
                col_score1, col_score2, col_score3 = st.columns(3)
                with col_score1:
                    st.metric("Fire Test Score", f"{fire_result['score']} / {fire_result['max_score']}")
                with col_score2:
                    st.metric("Score Percentage", f"{score_ratio*100:.1f}%")
                with col_score3:
                    recommendation = "✅ PASS" if score_ratio >= 0.7 else "⚠️ REVIEW" if score_ratio >= 0.5 else "❌ FAIL"
                    st.metric("Recommendation", recommendation)
                
                # Stage details
                st.markdown("#### 📊 Stage Breakdown")
                stages_data = []
                for stage in fire_result["stages"]:
                    stages_data.append({
                        "Stage": stage["name"],
                        "Status": "✅ PASS" if stage["pass"] else "❌ FAIL",
                        "Detail": stage["detail"],
                        "Description": stage.get("description", "")
                    })
                
                stages_df = pd.DataFrame(stages_data)
                st.dataframe(stages_df, use_container_width=True)
                
                # Visual stage status
                st.markdown("#### 📈 Stage Performance")
                for stage in fire_result["stages"]:
                    status_icon = "✅" if stage["pass"] else "❌"
                    status_color = "green" if stage["pass"] else "red"
                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px 0; border-left: 4px solid {status_color}; background: rgba(255,255,255,0.5); border-radius: 5px;">
                        <strong>{status_icon} {stage['name']}</strong>: {stage['detail']}<br>
                        <small style="color: #666;">{stage.get('description', '')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Save button for Phase 2 Deep Analysis (Fire Test)
                st.markdown("---")
                if st.button("💾 Save Phase 2 Analysis to Database", key="save_phase2_analysis"):
                    try:
                        from tradingagents.database.config import get_supabase
                        from tradingagents.database.db_service import (
                            _make_json_serializable,
                            save_fire_test_with_rag
                        )
                        
                        # Check database connection first
                        supabase = get_supabase()
                        if not supabase:
                            st.error("❌ Database not configured! Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
                        else:
                            # Build summary text for RAG
                            summary_lines = [
                                f"Fire Test for {selected_symbol}",
                                f"Score: {fire_result['score']} / {fire_result['max_score']} ({score_ratio*100:.1f}%)"
                            ]
                            for stg in fire_result.get("stages", [])[:10]:
                                status = "PASS" if stg.get("pass") else "FAIL"
                                summary_lines.append(f"- {stg.get('name')}: {status} | {stg.get('detail','')}")
                            summary_text = "\n".join(summary_lines)

                            # Generate embedding using DocumentManager
                            doc_manager = DocumentManager()
                            try:
                                embedding_vector = doc_manager._generate_embedding(summary_text)
                            finally:
                                doc_manager.close()

                            saved = save_fire_test_with_rag(
                                symbol=selected_symbol,
                                fire_result=_make_json_serializable(fire_result),
                                summary_text=summary_text,
                                embedding_vector=embedding_vector,
                                run_metadata={"source": "phase3_tab2"}
                            )

                            if saved:
                                st.success("✅ Phase 2 Deep Analysis saved to `fire_test_runs` with RAG embedding!")
                                st.balloons()
                            else:
                                st.error("❌ Failed to save - check console for details")
                    except Exception as e:
                        st.error(f"❌ Error saving: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        st.markdown("---")
        st.markdown("#### Historical DataScan Prompt")
        st.write(
            "Run the multi-timeframe DataScan prompt to validate that the ingested D1/M5/M1 datasets are aligned "
        )

        if st.button("🧠 Run Historical DataScan", key="run_datascan_prompt", type="primary"):
            with st.spinner(f"Gathering historical slices for {selected_symbol}..."):
                datascan_result = st.session_state.datascan_engine.run_analysis(selected_symbol)
                st.session_state.datascan_result = datascan_result

                if datascan_result.get("error"):
                    st.error(datascan_result["error"])
                else:
                    _render_datascan_output(datascan_result)
        elif 'datascan_result' in st.session_state and not st.session_state.datascan_result.get("error"):
            st.info("Showing the most recent DataScan output for this session.")
            _render_datascan_output(st.session_state.datascan_result)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### ⚡ Complete Trading Workflow")
        st.write("""
        **Step-by-Step Trading Interface**: Complete workflow from analysis to execution recommendation.
        
        **Workflow**:
        1. **Phase 1 Analysis**: Volume screening results
        2. **Phase 2 Analysis**: 7-stage fire test + AI insights
        3. **AI-Enhanced Recommendation**: Combined scoring
        4. **Risk & Entry Planning**: Stop-loss and targets
        5. **Final Decision**: Execute or pass
        """)
        
        # Workflow state management
        if 'trading_workflow_state' not in st.session_state:
            st.session_state.trading_workflow_state = {
                'current_step': 1,
                'selected_symbol': None,
                'phase1_complete': False,
                'phase2_complete': False,
                'recommendation': None
            }
        
        workflow = st.session_state.trading_workflow_state
        
        # Step 1: Select Symbol
        st.markdown("#### Step 1: Select Symbol")
        workflow_symbols = workflow.get('phase1_passed_symbols', symbols)
        workflow['selected_symbol'] = st.selectbox(
            "Choose Symbol for Trading Analysis",
            workflow_symbols,
            key="workflow_symbol",
            index=0 if workflow['selected_symbol'] not in workflow_symbols else workflow_symbols.index(workflow['selected_symbol']) if workflow['selected_symbol'] in workflow_symbols else 0
        )
        
        if workflow['selected_symbol']:
            selected_sym = workflow['selected_symbol']
            
            # Step 2: Phase 1 Results
            st.markdown("---")
            st.markdown("#### Step 2: Phase 1 Volume Screening")
            
            if st.button("🔍 Check Phase 1 Status", key="check_phase1_workflow"):
                with st.spinner("Checking Phase 1 screening..."):
                    phase1_result = st.session_state.volume_screener.screen_symbol(selected_sym)
                    
                    if phase1_result['pass']:
                        st.success(f"✅ {selected_sym} PASSED Phase 1 screening!")
                        workflow['phase1_complete'] = True
                        st.json(phase1_result['metrics'])
                    else:
                        st.warning(f"⚠️ {selected_sym} did NOT pass Phase 1: {phase1_result['reason']}")
                        workflow['phase1_complete'] = False
                        st.info(f"**Metrics**: {phase1_result['metrics']}")
            elif workflow.get('phase1_complete'):
                st.success("✅ Phase 1 screening completed")
            
            # Step 3: Phase 2 Analysis
            st.markdown("---")
            st.markdown("#### Step 3: Phase 2 Deep Analysis")
            
            if st.button("🔥 Run Phase 2 Analysis", type="primary", key="run_phase2_workflow"):
                with st.spinner(f"Running Phase 2 analysis for {selected_sym}..."):
                    phase2_result = st.session_state.fire_tester.run_fire_test(selected_sym)
                    
                    if phase2_result.get("error"):
                        st.error(f"❌ Phase 2 Error: {phase2_result['error']}")
                        workflow['phase2_complete'] = False
                    else:
                        workflow['phase2_complete'] = True
                        workflow['phase2_result'] = phase2_result
                        
                        # Display Phase 2 results
                        score_ratio = phase2_result["score"] / phase2_result["max_score"]
                        st.success(f"✅ Phase 2 Complete! Score: {phase2_result['score']}/{phase2_result['max_score']} ({score_ratio*100:.1f}%)")
                        
                        # Show stages
                        with st.expander("View Stage Details", expanded=False):
                            for stage in phase2_result["stages"]:
                                status = "✅" if stage["pass"] else "❌"
                                st.write(f"{status} **{stage['name']}**: {stage['detail']}")
            
            # Step 4: AI-Enhanced Recommendation
            st.markdown("---")
            st.markdown("#### Step 4: AI-Enhanced Recommendation")
            
            if st.button("🧠 Generate AI-Enhanced Recommendation", type="primary", key="generate_recommendation"):
                with st.spinner(f"Generating AI-enhanced analysis for {selected_sym}..."):
                    recommendation_result = st.session_state.ai_scorer.calculate_enhanced_score(selected_sym)
                    
                    workflow['recommendation'] = recommendation_result
                    
                    # Display recommendation
                    enhanced_score = recommendation_result.get('enhanced_score', 5.0)
                    rec = recommendation_result.get('recommendation', 'Hold')
                    
                    # Color coding
                    if enhanced_score >= 7.0:
                        color = "green"
                        emoji = "🟢"
                    elif enhanced_score >= 5.0:
                        color = "orange"
                        emoji = "🟡"
                    else:
                        color = "red"
                        emoji = "🔴"
                    
                    st.markdown(f"""
                    <div style="padding: 20px; background: linear-gradient(135deg, #{color}40 0%, #{color}20 100%); 
                                border-radius: 15px; border: 2px solid #{color}; margin: 10px 0;">
                        <h2 style="margin: 0; color: #{color};">
                            {emoji} {rec} - Confidence: {enhanced_score}/10
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Score breakdown
                    col_tech, col_ai = st.columns(2)
                    with col_tech:
                        st.metric("Technical Score", f"{recommendation_result.get('technical_score', 0):.2f}/10")
                        st.write("**Breakdown**: 70% weight")
                    with col_ai:
                        st.metric("AI Score", f"{recommendation_result.get('ai_score', 0):.2f}/10")
                        st.write(f"**Sentiment**: {recommendation_result.get('ai_sentiment', 'Neutral')}")
                        st.write(f"**Confidence**: {recommendation_result.get('ai_confidence', 5)}/10")
                        st.write("**Breakdown**: 30% weight")
                    
                    # Show AI analysis if available
                    ai_analysis = recommendation_result.get('ai_analysis', {})
                    if ai_analysis and not ai_analysis.get('error'):
                        with st.expander("View AI Analysis", expanded=False):
                            st.write(ai_analysis.get('analysis_text', 'No analysis available'))
                    
                    # Show fire test summary
                    fire_test = recommendation_result.get('fire_test', {})
                    if fire_test and not fire_test.get('error'):
                        with st.expander("View Fire Test Summary", expanded=False):
                            st.metric("Fire Test Score", f"{fire_test.get('score', 0)}/{fire_test.get('max_score', 7)}")
                            for stage in fire_test.get('stages', []):
                                status = "✅" if stage['pass'] else "❌"
                                st.write(f"{status} {stage['name']}: {stage['detail']}")
                    
                    # Save button for trade signal
                    st.markdown("---")
                    if st.button("💾 Save Trade Signal to Database", key="save_trade_signal"):
                        try:
                            from tradingagents.database.db_service import create_trade_signal, log_event
                            from tradingagents.database.config import get_supabase
                            
                            # Check database connection first
                            supabase = get_supabase()
                            if not supabase:
                                st.error("❌ Database not configured! Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
                            else:
                                rec = recommendation_result.get('recommendation', 'Hold')
                                confidence = recommendation_result.get('enhanced_score', 5.0)
                                
                                # Convert recommendation to signal type
                                signal_type_map = {
                                    'Strong Buy': 'buy',
                                    'Buy': 'buy',
                                    'Hold': 'hold',
                                    'Sell': 'sell',
                                    'Strong Sell': 'sell'
                                }
                                signal_type = signal_type_map.get(rec, 'hold')
                                
                                # Save signal
                                signal_details = {
                                    'technical_score': recommendation_result.get('technical_score', 0),
                                    'ai_score': recommendation_result.get('ai_score', 0),
                                    'ai_sentiment': recommendation_result.get('ai_sentiment', 'Neutral'),
                                    'fire_test_score': fire_test.get('score', 0) if fire_test else 0
                                }
                                
                                signal_result = create_trade_signal(
                                    symbol=selected_sym,
                                    signal_type=signal_type,
                                    confidence=confidence / 10.0,  # Normalize to 0-1
                                    details=signal_details
                                )
                                
                                if signal_result:
                                    log_event(
                                        "trade_recommendation_generated",
                                        {
                                            "symbol": selected_sym,
                                            "recommendation": rec,
                                            "confidence": confidence,
                                            "signal_type": signal_type
                                        }
                                    )
                                    st.success("✅ Trade signal saved to database! (Saved to trade_signals and system_logs tables)")
                                else:
                                    st.error("❌ Failed to save trade signal - check console for details")
                        except Exception as e:
                            st.error(f"❌ Error saving: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # Step 5: Risk & Entry Planning
            st.markdown("---")
            st.markdown("#### Step 5: Risk & Entry Planning")
            
            if workflow.get('recommendation'):
                if st.button("📊 Calculate Entry & Risk", key="calculate_entry_risk"):
                    with st.spinner("Calculating optimal entry and risk parameters..."):
                        hist = _fetch_history(selected_sym, period="3mo")
                        
                        if hist is None or hist.empty:
                            st.error("No price data available")
                        else:
                            # Calculate ATR-based stop loss and targets
                            atr = float(_calc_atr(hist).iloc[-1])
                            current_price = float(hist['Close'].iloc[-1])
                            
                            # Stop loss: 1.5x ATR below entry
                            stop_loss = current_price - (1.5 * atr)
                            stop_loss_pct = ((current_price - stop_loss) / current_price) * 100
                            
                            # Target 1: 2x risk (1:2 risk/reward)
                            target1 = current_price + (2 * (current_price - stop_loss))
                            target1_pct = ((target1 - current_price) / current_price) * 100
                            
                            # Target 2: 2.5x risk (1:2.5 risk/reward)
                            target2 = current_price + (2.5 * (current_price - stop_loss))
                            target2_pct = ((target2 - current_price) / current_price) * 100
                            
                            # Display risk/reward metrics
                            col_price, col_stop, col_t1, col_t2 = st.columns(4)
                            with col_price:
                                st.metric("Entry Price", f"${current_price:.2f}")
                            with col_stop:
                                st.metric("Stop Loss", f"${stop_loss:.2f}", f"-{stop_loss_pct:.2f}%")
                            with col_t1:
                                st.metric("Target 1", f"${target1:.2f}", f"+{target1_pct:.2f}%")
                                st.caption("1:2 R/R")
                            with col_t2:
                                st.metric("Target 2", f"${target2:.2f}", f"+{target2_pct:.2f}%")
                                st.caption("1:2.5 R/R")
                            
                            # Risk summary
                            st.markdown("#### 💼 Risk Summary")
                            risk_amount = current_price - stop_loss
                            reward1 = target1 - current_price
                            reward2 = target2 - current_price
                            
                            st.info(f"""
                            **Risk per Share**: ${risk_amount:.2f} ({stop_loss_pct:.2f}%)
                            
                            **Potential Rewards**:
                            - Target 1: ${reward1:.2f} per share ({target1_pct:.2f}%) - **Risk/Reward: 1:2**
                            - Target 2: ${reward2:.2f} per share ({target2_pct:.2f}%) - **Risk/Reward: 1:2.5**
                            
                            **ATR**: ${atr:.2f} (used for stop-loss calculation)
                            """)
                            
                            # Store risk parameters
                            workflow['risk_params'] = {
                                'entry': current_price,
                                'stop_loss': stop_loss,
                                'target1': target1,
                                'target2': target2,
                                'atr': float(atr)
                            }
            
            # Final step: Decision
            st.markdown("---")
            st.markdown("#### Step 6: Final Decision")
            
            if workflow.get('recommendation') and workflow.get('risk_params'):
                rec = workflow['recommendation'].get('recommendation', 'Hold')
                enhanced_score = workflow['recommendation'].get('enhanced_score', 5.0)
                
                if rec in ['Strong Buy', 'Buy'] and enhanced_score >= 7.0:
                    st.success(f"""
                    ## ✅ READY TO EXECUTE
                    
                    **Symbol**: {selected_sym}
                    **Recommendation**: {rec}
                    **Confidence**: {enhanced_score}/10
                    
                    **Next Steps**:
                    1. Review entry, stop-loss, and targets above
                    2. Execute trade with proper position sizing
                    3. Monitor trade and adjust if needed
                    """)
                    
                    if st.button("✅ Mark as Ready to Trade", type="primary", key="mark_ready"):
                        st.balloons()
                        st.success(f"✅ {selected_sym} marked as ready to trade!")
                
                # Save complete trading workflow/session
                st.markdown("---")
                st.markdown("#### 💾 Save Trading Session")
                if st.button("💾 Save Complete Trading Flow to Database", key="save_trading_flow"):
                    try:
                        from tradingagents.database.db_service import log_event
                        from tradingagents.database.config import get_supabase
                        
                        # Check database connection first
                        supabase = get_supabase()
                        if not supabase:
                            st.error("❌ Database not configured! Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
                        else:
                            # Compile complete workflow data
                            workflow_data = {
                                "symbol": selected_sym,
                                "session_date": datetime.now().isoformat(),
                                "phase1_complete": bool(workflow.get('phase1_complete', False)),  # Ensure Python bool
                                "phase1_result": workflow.get('phase1_result'),
                                "phase2_complete": bool(workflow.get('phase2_complete', False)),  # Ensure Python bool
                                "phase2_result": workflow.get('phase2_result'),
                                "recommendation": workflow.get('recommendation'),
                                "risk_params": workflow.get('risk_params'),
                                "final_decision": rec,
                                "final_score": float(enhanced_score),  # Ensure Python float
                                "workflow_status": "ready_to_trade" if (rec in ['Strong Buy', 'Buy'] and enhanced_score >= 7.0) else "review_required" if enhanced_score >= 5.0 else "not_recommended"
                            }
                            
                            # Convert workflow_data to ensure all nested numpy/pandas types are converted
                            from tradingagents.database.db_service import _make_json_serializable
                            workflow_data = _make_json_serializable(workflow_data)
                            
                            result = log_event("trading_workflow_completed", workflow_data)
                            if result:
                                st.success("✅ Complete trading flow saved to database! (Saved to system_logs table)")
                            else:
                                st.error("❌ Failed to save - check console for details")
                    except Exception as e:
                        st.error(f"❌ Error saving: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                elif enhanced_score >= 5.0:
                    st.warning(f"""
                    ## ⚠️ REVIEW REQUIRED
                    
                    **Symbol**: {selected_sym}
                    **Recommendation**: {rec}
                    **Confidence**: {enhanced_score}/10
                    
                    **Consideration**: Review analysis before executing
                    """)
                    
                    # Save complete trading workflow/session (for review required)
                    st.markdown("---")
                    st.markdown("#### 💾 Save Trading Session")
                    if st.button("💾 Save Complete Trading Flow to Database", key="save_trading_flow_review"):
                        try:
                            from tradingagents.database.db_service import log_event
                            from tradingagents.database.config import get_supabase
                            
                            # Check database connection first
                            supabase = get_supabase()
                            if not supabase:
                                st.error("❌ Database not configured! Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
                            else:
                                # Compile complete workflow data
                                workflow_data = {
                                    "symbol": selected_sym,
                                    "session_date": datetime.now().isoformat(),
                                    "phase1_complete": bool(workflow.get('phase1_complete', False)),
                                    "phase1_result": workflow.get('phase1_result'),
                                    "phase2_complete": bool(workflow.get('phase2_complete', False)),
                                    "phase2_result": workflow.get('phase2_result'),
                                    "recommendation": workflow.get('recommendation'),
                                    "risk_params": workflow.get('risk_params'),
                                    "final_decision": rec,
                                    "final_score": float(enhanced_score),
                                    "workflow_status": "review_required"
                                }
                                
                                # Convert workflow_data to ensure all nested numpy/pandas types are converted
                                from tradingagents.database.db_service import _make_json_serializable
                                workflow_data = _make_json_serializable(workflow_data)
                                
                                result = log_event("trading_workflow_completed", workflow_data)
                                if result:
                                    st.success("✅ Complete trading flow saved to database! (Saved to system_logs table)")
                                else:
                                    st.error("❌ Failed to save - check console for details")
                        except Exception as e:
                            st.error(f"❌ Error saving: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.error(f"""
                    ## ❌ NOT RECOMMENDED
                    
                    **Symbol**: {selected_sym}
                    **Recommendation**: {rec}
                    **Confidence**: {enhanced_score}/10
                    
                    **Recommendation**: Pass on this trade
                    """)
                    
                    # Save complete trading workflow/session (for not recommended)
                    st.markdown("---")
                    st.markdown("#### 💾 Save Trading Session")
                    if st.button("💾 Save Complete Trading Flow to Database", key="save_trading_flow_not_rec"):
                        try:
                            from tradingagents.database.db_service import log_event
                            from tradingagents.database.config import get_supabase
                            
                            # Check database connection first
                            supabase = get_supabase()
                            if not supabase:
                                st.error("❌ Database not configured! Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
                            else:
                                # Compile complete workflow data
                                workflow_data = {
                                    "symbol": selected_sym,
                                    "session_date": datetime.now().isoformat(),
                                    "phase1_complete": bool(workflow.get('phase1_complete', False)),
                                    "phase1_result": workflow.get('phase1_result'),
                                    "phase2_complete": bool(workflow.get('phase2_complete', False)),
                                    "phase2_result": workflow.get('phase2_result'),
                                    "recommendation": workflow.get('recommendation'),
                                    "risk_params": workflow.get('risk_params'),
                                    "final_decision": rec,
                                    "final_score": float(enhanced_score),
                                    "workflow_status": "not_recommended"
                                }
                                
                                # Convert workflow_data to ensure all nested numpy/pandas types are converted
                                from tradingagents.database.db_service import _make_json_serializable
                                workflow_data = _make_json_serializable(workflow_data)
                                
                                result = log_event("trading_workflow_completed", workflow_data)
                                if result:
                                    st.success("✅ Complete trading flow saved to database! (Saved to system_logs table)")
                                else:
                                    st.error("❌ Failed to save - check console for details")
                        except Exception as e:
                            st.error(f"❌ Error saving: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)


def phase4_session_management_execution():
    """Phase 4: Session Management & Execution - Trade tracking and execution"""
    
    # Initialize services
    # Ensure a valid UUID user_id is present (fixes DB uuid type errors)
    if ("user_id" not in st.session_state) or (st.session_state.get("user_id") in [None, "", "default_user"]):
        st.session_state.user_id = str(uuid4())
    user_id = st.session_state.user_id
    
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = TradingSessionManager(user_id)
    
    if 'trade_service' not in st.session_state:
        st.session_state.trade_service = TradeExecutionService(user_id)
    
    session_manager = st.session_state.session_manager
    trade_service = st.session_state.trade_service
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Session Management",
        "📈 Active Trades",
        "⚡ Execute Trade",
        "📊 Trade History"
    ])
    
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Trading Session Management")
        
        # Get or create active session
        active_session = session_manager.get_active_session()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if active_session:
                st.success(f"✅ **Active Session**: {active_session.get('session_name', 'Current Session')}")
                st.info(f"**Started**: {active_session.get('start_date', 'Unknown')}")
                
                if active_session.get('notes'):
                    st.write(f"**Notes**: {active_session.get('notes')}")
            else:
                st.warning("⚠️ No active session. Create a new session to start trading.")
        
        with col2:
            if st.button("🆕 New Session", key="new_session", use_container_width=True):
                session_name = st.text_input("Session Name", value=f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}", key="session_name_input")
                notes = st.text_area("Notes (optional)", key="session_notes_input")
                if st.button("✅ Create", key="create_session_confirm"):
                    try:
                        session = session_manager.create_session(session_name, notes)
                        if session:
                            # Confirmation and DB log
                            st.success("✅ Session created and saved to database!")
                            try:
                                from tradingagents.database.db_service import log_event, _make_json_serializable
                                payload = _make_json_serializable({
                                    "session_id": session.get("id"),
                                    "session_name": session.get("session_name", session_name),
                                    "start_date": session.get("start_date"),
                                    "status": session.get("status", "active"),
                                    "notes": notes
                                })
                                log_event("session_created", payload)
                            except Exception:
                                pass
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error creating session: {str(e)}")
            
            if active_session and active_session.get('id') != "default_session":
                if st.button("🔒 Close Session", key="close_session", use_container_width=True):
                    st.session_state.show_close_session_form = True
                
                if st.session_state.get('show_close_session_form', False):
                    close_notes = st.text_area("Close Notes (optional)", key="close_notes")
                    col_close, col_cancel_close = st.columns(2)
                    with col_close:
                        if st.button("✅ Confirm Close", key="confirm_close"):
                            try:
                                session_manager.close_session(active_session['id'], close_notes)
                                st.session_state.show_close_session_form = False
                                st.success("✅ Session closed and saved to database!")
                                try:
                                    from tradingagents.database.db_service import log_event, _make_json_serializable
                                    payload = _make_json_serializable({
                                        "session_id": active_session.get("id"),
                                        "session_name": active_session.get("session_name"),
                                        "end_date": datetime.now().isoformat(),
                                        "status": "closed",
                                        "close_notes": close_notes
                                    })
                                    log_event("session_closed", payload)
                                except Exception:
                                    pass
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error closing session: {str(e)}")
                    with col_cancel_close:
                        if st.button("❌ Cancel", key="cancel_close"):
                            st.session_state.show_close_session_form = False
                            st.rerun()
        
        # Session history
        st.markdown("---")
        st.markdown("#### 📜 Session History")
        sessions = session_manager.get_all_sessions(limit=10)
        if sessions:
            sessions_df = pd.DataFrame([
                {
                    "Name": s.get('session_name', 'N/A'),
                    "Status": s.get('status', 'N/A'),
                    "Start Date": s.get('start_date', 'N/A'),
                    "End Date": s.get('end_date', 'N/A') or 'Active'
                }
                for s in sessions
            ])
            st.dataframe(sessions_df, use_container_width=True)
        else:
            st.info("No previous sessions found.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Active Trades Dashboard")
        
        # Refresh button
        col_refresh, col_info = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄 Refresh", key="refresh_trades", use_container_width=True):
                st.rerun()
        
        with col_info:
            can_open, message = trade_service.can_open_trade()
            if can_open:
                st.info(f"✅ {message}")
            else:
                st.warning(f"⚠️ {message}")
        
        # Get active trades
        active_trades = trade_service.get_active_trades()
        
        if active_trades:
            st.markdown(f"#### 🎯 Active Trades: {len(active_trades)}")
            
            # Summary metrics
            total_unrealized_pnl = sum(t.get('unrealized_pnl', 0) for t in active_trades)
            total_unrealized_pnl_pct = sum(t.get('unrealized_pnl_pct', 0) for t in active_trades) / len(active_trades) if active_trades else 0
            
            col_pnl1, col_pnl2, col_pnl3 = st.columns(3)
            with col_pnl1:
                pnl_color = "green" if total_unrealized_pnl >= 0 else "red"
                st.metric("Total Unrealized P&L", f"${total_unrealized_pnl:,.2f}", 
                         delta=f"{total_unrealized_pnl_pct:.2f}%")
            with col_pnl2:
                st.metric("Active Positions", len(active_trades))
            with col_pnl3:
                st.metric("Remaining Slots", f"{TradingSessionManager.MAX_CONCURRENT_TRADES - len(active_trades)}")
            
            # Trade cards
            for trade in active_trades:
                symbol = trade['symbol']
                pnl = trade.get('unrealized_pnl', 0)
                pnl_pct = trade.get('unrealized_pnl_pct', 0)
                current_price = trade.get('current_price', 0)
                entry_price = trade.get('entry_price', 0)
                
                pnl_color = "🟢" if pnl >= 0 else "🔴"
                
                with st.expander(f"{pnl_color} {symbol} | Entry: ${entry_price:.2f} | Current: ${current_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)", expanded=False):
                    col_trade1, col_trade2 = st.columns(2)
                    
                    with col_trade1:
                        st.write(f"**Quantity**: {trade.get('quantity', 0)}")
                        st.write(f"**Entry Price**: ${entry_price:.2f}")
                        st.write(f"**Current Price**: ${current_price:.2f}")
                        st.write(f"**Entry Date**: {trade.get('entry_date', 'N/A')}")
                    
                    with col_trade2:
                        stop_loss = trade.get('stop_loss')
                        target1 = trade.get('target1')
                        target2 = trade.get('target2')
                        
                        if stop_loss:
                            st.write(f"**Stop Loss**: ${stop_loss:.2f}")
                        if target1:
                            st.write(f"**Target 1**: ${target1:.2f}")
                        if target2:
                            st.write(f"**Target 2**: ${target2:.2f}")
                    
                    # Close trade section
                    st.markdown("---")
                    st.markdown("#### 🚪 Close Trade")
                    
                    col_close1, col_close2 = st.columns([2, 1])
                    with col_close1:
                        exit_price = st.number_input(f"Exit Price for {symbol}", value=float(current_price), 
                                                     min_value=0.0, step=0.01, key=f"exit_price_{symbol}")
                        close_quantity = st.number_input(f"Quantity to Close", value=float(trade.get('quantity', 0)),
                                                         min_value=0.0, max_value=float(trade.get('quantity', 0)),
                                                         step=1.0, key=f"close_qty_{symbol}")
                        close_notes = st.text_input("Close Notes (optional)", key=f"close_notes_{symbol}")
                    
                    with col_close2:
                        if st.button(f"✅ Close {symbol}", key=f"close_{symbol}", type="primary", use_container_width=True):
                            try:
                                result = trade_service.close_trade(symbol, exit_price, close_quantity, close_notes)
                                if result:
                                    realized_pnl = result.get('realized_pnl', 0)
                                    st.success(f"✅ Trade closed! Realized P&L: ${realized_pnl:.2f}")
                                    st.balloons()
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error closing trade: {str(e)}")
                    
                    # Update P&L button
                    if st.button(f"🔄 Update P&L for {symbol}", key=f"update_pnl_{symbol}"):
                        with st.spinner("Updating P&L..."):
                            updated = trade_service.update_trade_pnl(symbol)
                            if updated:
                                st.success(f"✅ Updated! Current P&L: ${updated['unrealized_pnl']:.2f} ({updated['unrealized_pnl_pct']:+.2f}%)")
                                st.rerun()
            
            # Bulk refresh all P&L
            if st.button("🔄 Refresh All P&L", key="refresh_all_pnl"):
                with st.spinner("Updating all trade P&L..."):
                    for trade in active_trades:
                        trade_service.update_trade_pnl(trade['symbol'])
                    st.success("✅ All P&L updated!")
                    st.rerun()
        else:
            st.info("📭 No active trades. Use the 'Execute Trade' tab to open a new position.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### ⚡ Execute Trade")
        
        # Check if we can open a trade
        can_open, message = trade_service.can_open_trade()
        if not can_open:
            st.error(f"❌ {message}")
            st.stop()
        
        st.success(f"✅ {message}")
        
        # Trade execution form
        st.markdown("#### 📝 Trade Details")
        
        col_exec1, col_exec2 = st.columns(2)
        
        with col_exec1:
            symbols = get_watchlist_symbols()
            symbol = st.selectbox("Symbol", symbols, key="execute_symbol")
            quantity = st.number_input("Quantity", min_value=0.0, step=1.0, value=1.0, key="execute_quantity")
            trade_type = st.selectbox("Trade Type", ["LONG", "SHORT"], key="execute_type")
        
        with col_exec2:
            price_snapshot = get_latest_price(symbol)
            if price_snapshot and price_snapshot.get("price") is not None:
                current_price = float(price_snapshot["price"])
                label = price_snapshot.get("interval", "1d")
                st.info(f"💵 **Current Price ({label})**: ${current_price:.2f}")
            else:
                current_price = 0.0
                st.warning("⚠️ Could not find a stored price for this symbol. Ingest data first.")
            
            entry_price = st.number_input(
                "Entry Price",
                min_value=0.0,
                step=0.01,
                value=float(current_price) if current_price else 0.0,
                key="execute_entry_price"
            )
            
            # Risk parameters
            st.markdown("#### 🛡️ Risk Management (Optional)")
            stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.01, value=0.0, key="execute_stop_loss")
            target1 = st.number_input("Target 1", min_value=0.0, step=0.01, value=0.0, key="execute_target1")
            target2 = st.number_input("Target 2", min_value=0.0, step=0.01, value=0.0, key="execute_target2")
        
        notes = st.text_area("Trade Notes (optional)", key="execute_notes")
        
        # Calculate estimated P&L at targets
        if entry_price > 0 and quantity > 0:
            st.markdown("#### 💰 Risk/Reward Preview")
            col_rr1, col_rr2, col_rr3 = st.columns(3)
            
            if stop_loss > 0:
                risk_per_share = abs(entry_price - stop_loss)
                with col_rr1:
                    st.metric("Risk/Share", f"${risk_per_share:.2f}")
                
                if target1 > 0:
                    reward1 = abs(target1 - entry_price)
                    rr1 = reward1 / risk_per_share if risk_per_share > 0 else 0
                    with col_rr2:
                        st.metric("Target 1 R/R", f"1:{rr1:.2f}", f"${reward1 * quantity:.2f}")
                
                if target2 > 0:
                    reward2 = abs(target2 - entry_price)
                    rr2 = reward2 / risk_per_share if risk_per_share > 0 else 0
                    with col_rr3:
                        st.metric("Target 2 R/R", f"1:{rr2:.2f}", f"${reward2 * quantity:.2f}")
        
        # Execute button
        st.markdown("---")
        if st.button("⚡ Execute Trade", type="primary", key="execute_trade_button", use_container_width=True):
            try:
                # Validate inputs
                if not symbol:
                    st.error("❌ Please select a symbol")
                elif quantity <= 0:
                    st.error("❌ Quantity must be greater than 0")
                elif entry_price <= 0:
                    st.error("❌ Entry price must be greater than 0")
                else:
                    with st.spinner(f"Executing trade for {symbol}..."):
                        # Get active session
                        session = session_manager.get_active_session()
                        session_id = session.get('id') if session else None
                        
                        trade = trade_service.open_trade(
                            symbol=symbol,
                            quantity=quantity,
                            entry_price=entry_price if entry_price > 0 else None,
                            trade_type=trade_type,
                            stop_loss=stop_loss if stop_loss > 0 else None,
                            target1=target1 if target1 > 0 else None,
                            target2=target2 if target2 > 0 else None,
                            notes=notes,
                            session_id=session_id
                        )
                        
                        if trade:
                            st.success(f"✅ Trade executed successfully for {symbol}!")
                            st.balloons()
                            st.info(f"**Entry Price**: ${entry_price:.2f} | **Quantity**: {quantity} | **Total Value**: ${entry_price * quantity:.2f}")
                            st.rerun()
            except Exception as e:
                st.error(f"❌ Error executing trade: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Trade History")
        
        # Get trade history
        history = trade_service.get_trade_history(limit=50)
        
        if history:
            st.markdown(f"#### 📜 Recent Trades ({len(history)})")
            
            # Summary statistics
            closed_trades = [t for t in history if t.get('status') == 'closed']
            if closed_trades:
                total_realized_pnl = sum(t.get('realized_pnl', 0) for t in closed_trades)
                winning_trades = [t for t in closed_trades if t.get('realized_pnl', 0) > 0]
                losing_trades = [t for t in closed_trades if t.get('realized_pnl', 0) < 0]
                
                col_hist1, col_hist2, col_hist3, col_hist4 = st.columns(4)
                with col_hist1:
                    st.metric("Total Realized P&L", f"${total_realized_pnl:,.2f}")
                with col_hist2:
                    st.metric("Winning Trades", len(winning_trades))
                with col_hist3:
                    st.metric("Losing Trades", len(losing_trades))
                with col_hist4:
                    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
            
            # History table
            history_data = []
            for trade in history[:20]:  # Show last 20
                history_data.append({
                    "Symbol": trade.get('symbol', 'N/A'),
                    "Type": trade.get('trade_type', 'N/A'),
                    "Quantity": trade.get('quantity', 0),
                    "Entry Price": f"${trade.get('entry_price', 0):.2f}",
                    "Exit Price": f"${trade.get('exit_price', 0):.2f}" if trade.get('exit_price') else "N/A",
                    "Status": trade.get('status', 'N/A'),
                    "P&L": f"${trade.get('realized_pnl', trade.get('unrealized_pnl', 0)):.2f}",
                    "Entry Date": trade.get('entry_date', 'N/A')[:10] if trade.get('entry_date') else 'N/A',
                    "Exit Date": trade.get('exit_date', 'N/A')[:10] if trade.get('exit_date') else 'N/A'
                })
            
            if history_data:
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True, height=400)
        else:
            st.info("📭 No trade history found.")
        
        st.markdown('</div>', unsafe_allow_html=True)


def phase5_results_analysis():
    """Phase 5: Results & Analysis Modules - Trading journal, analysis, learning loop"""
    # Initialize services (reuse Phase 4 services if available)
    user_id = st.session_state.get("user_id", "default_user")
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = TradingSessionManager(user_id)
    if 'trade_service' not in st.session_state:
        st.session_state.trade_service = TradeExecutionService(user_id)

    session_manager = st.session_state.session_manager
    trade_service = st.session_state.trade_service

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Current Results Dashboard",
        "❌ Failed Trade Analysis",
        "📈 Historical Performance",
        "🧠 Learning Feedback"
    ])

    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Current Results Dashboard")

        history = trade_service.get_trade_history(limit=500) or []
        active = trade_service.get_active_trades() or []

        closed_trades = [t for t in history if t.get('status') == 'closed']
        winning_trades = [t for t in closed_trades if float(t.get('realized_pnl', 0)) > 0]
        losing_trades = [t for t in closed_trades if float(t.get('realized_pnl', 0)) < 0]

        total_realized = sum(float(t.get('realized_pnl', 0)) for t in closed_trades)
        avg_win = np.mean([float(t.get('realized_pnl', 0)) for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([float(t.get('realized_pnl', 0)) for t in losing_trades]) if losing_trades else 0.0
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Closed Trades", len(closed_trades))
        with col_b:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col_c:
            st.metric("Avg Win", f"${avg_win:,.2f}")
        with col_d:
            st.metric("Avg Loss", f"${avg_loss:,.2f}")

        col_e, col_f = st.columns(2)
        with col_e:
            st.metric("Realized P&L", f"${total_realized:,.2f}")
        with col_f:
            total_unrealized = sum(float(t.get('unrealized_pnl', 0)) for t in active)
            st.metric("Unrealized P&L (Active)", f"${total_unrealized:,.2f}")

        # Recent trades table
        if history:
            table = []
            for t in history[:50]:
                table.append({
                    "Symbol": t.get('symbol', 'N/A'),
                    "Type": t.get('trade_type', 'N/A'),
                    "Qty": t.get('quantity', 0),
                    "Entry": f"${float(t.get('entry_price', 0)):.2f}",
                    "Exit": f"${float(t.get('exit_price', 0)):.2f}" if t.get('exit_price') else 'N/A',
                    "Status": t.get('status', 'N/A'),
                    "Realized P&L": f"${float(t.get('realized_pnl', t.get('unrealized_pnl', 0))):.2f}",
                    "Entry Date": (t.get('entry_date') or '')[:19],
                    "Exit Date": (t.get('exit_date') or '')[:19]
                })
            st.markdown("#### Recent Trades")
            st.dataframe(pd.DataFrame(table), use_container_width=True, height=360)
        else:
            st.info("No trades recorded yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### ❌ Failed Trade Analysis")
        st.write("Tag losing trades with root causes and save for learning.")

        history = trade_service.get_trade_history(limit=200) or []
        losing_trades = [t for t in history if t.get('status') == 'closed' and float(t.get('realized_pnl', 0)) < 0]

        if losing_trades:
            options = [f"{t.get('symbol','N/A')} | {t.get('entry_date','')[:10]} → {t.get('exit_date','')[:10]} | P&L ${float(t.get('realized_pnl',0)):.2f}" for t in losing_trades]
            idx = st.selectbox("Select a losing trade to analyze", list(range(len(options))), format_func=lambda i: options[i])

            selected = losing_trades[idx]
            col1, col2 = st.columns(2)
            with col1:
                reason = st.selectbox(
                    "Primary Root Cause",
                    [
                        "stop_loss_hit",
                        "late_entry",
                        "false_breakout",
                        "trend_reversal",
                        "low_liquidity",
                        "news_event",
                        "execution_error",
                        "other"
                    ]
                )
                timeframe = st.selectbox("Timeframe Context", ["intraday", "swing", "multi-day"]) 
                avoid_symbol = st.checkbox("Flag symbol for reduced priority")
            with col2:
                contributing = st.multiselect(
                    "Contributing Factors",
                    [
                        "RSI_overbought",
                        "ATR_too_low",
                        "volume_dried_up",
                        "below_sma20",
                        "below_sma50",
                        "gap_down",
                        "market_weakness",
                        "news_negative"
                    ]
                )
                notes = st.text_area("Notes")

            if st.button("💾 Save Failed Trade Analysis", type="primary"):
                try:
                    from tradingagents.database.db_service import log_event, _make_json_serializable
                    payload = {
                        "symbol": selected.get('symbol'),
                        "entry_date": selected.get('entry_date'),
                        "exit_date": selected.get('exit_date'),
                        "entry_price": selected.get('entry_price'),
                        "exit_price": selected.get('exit_price'),
                        "realized_pnl": selected.get('realized_pnl'),
                        "reason": reason,
                        "timeframe": timeframe,
                        "contributing": contributing,
                        "notes": notes,
                        "avoid_symbol": bool(avoid_symbol)
                    }
                    payload = _make_json_serializable(payload)
                    ok = log_event("failed_trade_analysis", payload)
                    if ok:
                        st.success("Saved analysis.")
                    else:
                        st.error("Failed to save analysis.")
                except Exception as e:
                    st.error(f"Error saving analysis: {str(e)}")
        else:
            st.info("No losing trades found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Historical Performance Tracking")

        history = trade_service.get_trade_history(limit=1000) or []
        if not history:
            st.info("No history available.")
        else:
            # Equity curve (realized P&L cumulative)
            df = pd.DataFrame([
                {
                    "date": pd.to_datetime(t.get('exit_date') or t.get('entry_date') or datetime.now().isoformat()),
                    "pnl": float(t.get('realized_pnl', 0)) if t.get('status') == 'closed' else 0.0
                }
                for t in history
            ]).sort_values("date")
            df["equity"] = df["pnl"].cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["date"], y=df["equity"], mode="lines", name="Equity"))
            fig.update_layout(title="Equity Curve (Realized)", template='plotly_white', height=360)
            st.plotly_chart(fig, use_container_width=True)

            # Per-symbol performance
            perf_rows = []
            by_symbol = {}
            for t in history:
                sym = t.get('symbol', 'N/A')
                by_symbol.setdefault(sym, []).append(t)
            for sym, trades in by_symbol.items():
                closed = [x for x in trades if x.get('status') == 'closed']
                if not closed:
                    continue
                wins = [x for x in closed if float(x.get('realized_pnl', 0)) > 0]
                losses = [x for x in closed if float(x.get('realized_pnl', 0)) < 0]
                realized = sum(float(x.get('realized_pnl', 0)) for x in closed)
                wr = (len(wins) / len(closed) * 100) if closed else 0
                avg_pnl = realized / len(closed) if closed else 0
                perf_rows.append({
                    "Symbol": sym,
                    "Trades": len(closed),
                    "Win Rate %": round(wr, 1),
                    "Realized P&L": round(realized, 2),
                    "Avg P&L": round(avg_pnl, 2)
                })
            if perf_rows:
                st.markdown("#### Per-Symbol Performance")
                st.dataframe(pd.DataFrame(perf_rows).sort_values(["Win Rate %", "Realized P&L"], ascending=[False, False]), use_container_width=True, height=360)
            else:
                st.info("No closed trades to summarize.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Learning Feedback Loop")
        st.write("Data-driven suggestions to improve screening and risk parameters.")

        history = trade_service.get_trade_history(limit=1000) or []
        suggestions = []
        # Suggest exclude or deprioritize symbols with negative expectancy and sufficient sample size
        perf = {}
        for t in history:
            sym = t.get('symbol', 'N/A')
            perf.setdefault(sym, {"n": 0, "realized": 0.0, "wins": 0})
            if t.get('status') == 'closed':
                perf[sym]["n"] += 1
                pnl = float(t.get('realized_pnl', 0))
                perf[sym]["realized"] += pnl
                if pnl > 0:
                    perf[sym]["wins"] += 1
        for sym, stats in perf.items():
            n = stats["n"]
            if n >= 5:
                expectancy = stats["realized"] / n if n else 0.0
                wr = (stats["wins"] / n * 100) if n else 0.0
                if expectancy < 0 and wr < 45:
                    suggestions.append({
                        "symbol": sym,
                        "action": "deprioritize",
                        "reason": f"Negative expectancy (${expectancy:.2f}/trade) and low win rate ({wr:.1f}%)."
                    })

        # Suggest tightening Phase 1 thresholds if high false breakouts (proxy: many small losses)
        avg_loss = np.mean([float(t.get('realized_pnl', 0)) for t in history if t.get('status') == 'closed' and float(t.get('realized_pnl', 0)) < 0]) if history else 0
        if avg_loss > -50:  # frequent small losses: tighten
            suggestions.append({
                "symbol": "global",
                "action": "tighten_screening",
                "reason": "Frequent small losses detected. Consider raising volume spike to 1.6x and RSI band to 45-60."
            })

        if suggestions:
            st.markdown("#### Suggested Adjustments")
            st.dataframe(pd.DataFrame(suggestions), use_container_width=True)

            if st.button("💾 Apply Suggestions (log)", type="primary"):
                try:
                    from tradingagents.database.db_service import log_event, _make_json_serializable
                    payload = {"suggestions": suggestions, "generated_at": datetime.now().isoformat()}
                    payload = _make_json_serializable(payload)
                    ok = log_event("learning_feedback_applied", payload)
                    if ok:
                        st.success("Suggestions logged. Review engine configs separately to apply.")
                    else:
                        st.error("Failed to log suggestions.")
                except Exception as e:
                    st.error(f"Error logging suggestions: {str(e)}")
        else:
            st.info("No suggestions at this time. Keep trading to collect more data.")
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.markdown('<div class="section-header">AlphaAnalyst Trading AI Agent</div>', unsafe_allow_html=True)
    
    # Phase selector buttons
    phases = [
        "Foundation & Data Infrastructure",
        "Master Data & AI Integration", 
        "Trading Engine Core",
        "Session Management & Execution",
        "Results & Analysis Modules",
        "Advanced Features & Polish",
    ]
    if 'active_phase' not in st.session_state:
        st.session_state.active_phase = phases[0]
    
    st.markdown('<div class="phase-nav">', unsafe_allow_html=True)
    row1 = st.columns(3)
    with row1[0]:
        if st.button(phases[0], use_container_width=True, key="phase_1"):
            st.session_state.active_phase = phases[0]
    with row1[1]:
        if st.button(phases[1], use_container_width=True, key="phase_2"):
            st.session_state.active_phase = phases[1]
    with row1[2]:
        if st.button(phases[2], use_container_width=True, key="phase_3"):
            st.session_state.active_phase = phases[2]
    
    row2 = st.columns(3)
    with row2[0]:
        if st.button(phases[3], use_container_width=True, key="phase_4"):
            st.session_state.active_phase = phases[3]
    with row2[1]:
        if st.button(phases[4], use_container_width=True, key="phase_5"):
            st.session_state.active_phase = phases[4]
    with row2[2]:
        if st.button(phases[5], use_container_width=True, key="phase_6"):
            st.session_state.active_phase = phases[5]
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="active-phase">Active Phase: {st.session_state.active_phase}</div>', unsafe_allow_html=True)
    
    # Phase-specific content
    if st.session_state.active_phase == phases[0]:
        phase1_foundation_data()
    elif st.session_state.active_phase == phases[1]:
        phase2_master_data_ai()
    elif st.session_state.active_phase == phases[2]:
        phase3_trading_engine_core()
    elif st.session_state.active_phase == phases[3]:
        phase4_session_management_execution()
    elif st.session_state.active_phase == phases[4]:
        phase5_results_analysis()
    elif st.session_state.active_phase == phases[5]:
        phase6_advanced_features()
    
    # Quick Stock Analysis (only for Phase 1)
    if st.session_state.active_phase == phases[0]:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### Quick Stock Analysis")
        stock_input = st.text_input("Enter Company Name", help="e.g., APPLE, TCS")
        
        if st.button("Analyze", use_container_width=True, key="legacy_analyze"):
            if not stock_input:
                st.error("Please enter a stock name")
                return
            
            symbol = COMMON_STOCKS.get(stock_input.upper()) or stock_input
            
            if initialize_agents():
                with st.spinner("Analyzing..."):
                    info, hist = get_stock_data(symbol)
                    
                    if info and hist is not None:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">${info.get("currentPrice", "N/A")}</div><div class="metric-label">Current Price</div></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{info.get("forwardPE", "N/A")}</div><div class="metric-label">Forward P/E</div></div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{info.get("recommendationKey", "N/A").title()}</div><div class="metric-label">Recommendation</div></div>', unsafe_allow_html=True)
                        
                        # Signal provider retrieval removed as per user request
                        
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.plotly_chart(create_price_chart(hist, symbol, signals=None), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        if 'longBusinessSummary' in info:
                            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                            st.markdown("### Company Overview")
                            st.write(info['longBusinessSummary'])
                            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
                    
def phase6_advanced_features():
    """Phase 6: Advanced Features & Polish - Chatbot, search, performance, UI"""
    user_id = st.session_state.get("user_id", "default_user")
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = TradingSessionManager(user_id)
    if 'trade_service' not in st.session_state:
        st.session_state.trade_service = TradeExecutionService(user_id)

    session_manager = st.session_state.session_manager
    trade_service = st.session_state.trade_service

    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Trading Chatbot",
        "🔎 Advanced Search",
        "⚙️ Performance Tools",
        "🎨 UI & Preferences"
    ])

    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 Trading Chatbot")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "Hi! Ask me about symbols, recent analyses, or trades."}
            ]

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_msg = st.chat_input("Ask about a symbol, strategy, or results...")
        if user_msg:
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.write(user_msg)

            response_text = ""
            try:
                # Lightweight intent: extract a symbol token if present
                import re
                tokens = re.findall(r"[A-Z]{2,6}(?:\\.NS)?", user_msg.upper())
                symbol = tokens[0] if tokens else None

                # Try AIResearchAnalyzer profile if symbol exists
                if symbol:
                    analyzer = AIResearchAnalyzer()
                    profile = analyzer.analyze_instrument_profile(symbol)
                    analyzer.close()
                    ai = profile.get("ai_analysis", {}) if isinstance(profile, dict) else {}
                    md = profile.get("market_data", {}) if isinstance(profile, dict) else {}
                    if ai and "error" not in ai:
                        response_text += f"Symbol: {symbol}\n"
                        response_text += f"Sentiment: {ai.get('overall_sentiment','N/A')} | Confidence: {ai.get('confidence','N/A')}/10\n"
                        if md and "error" not in md:
                            response_text += f"Price: ${md.get('current_price','N/A')} | 30d Δ: {md.get('price_change_pct','N/A')}%\n"
                        analysis_text = ai.get('analysis_text') or ai.get('analysis')
                        if analysis_text:
                            response_text += f"\nKey Points:\n{analysis_text[:600]}"  # concise
                    else:
                        # Fallback to stored historical price
                        info, _ = get_stock_data(symbol)
                        if info and info.get("currentPrice") is not None:
                            price = float(info.get("currentPrice"))
                            response_text += f"{symbol} latest stored price: ${price:.2f}\n(Enable GROQ_API_KEY or GROK_API_KEY for deeper AI insights.)"
                        else:
                            response_text = f"I couldn't analyze {symbol} right now. Try again later."
                else:
                    # Non-symbol general question: summarize recent system state
                    recent = []
                    if 'phase1_results' in st.session_state:
                        df = st.session_state.phase1_results
                        if hasattr(df, 'to_dict'):
                            passed = df[df['Pass'] == '✅']['Symbol'].tolist() if 'Pass' in df.columns else []
                            recent.append(f"Phase 1 passed: {', '.join(passed[:10])}{'...' if len(passed)>10 else ''}")
                    if 'phase2_results' in st.session_state:
                        res = st.session_state.phase2_results
                        recent.append(f"Latest fire test score: {res.get('score','N/A')}/{res.get('max_score','N/A')}")
                    if 'document_analyses' in st.session_state:
                        recent.append(f"Recent document analyses: {len(st.session_state.document_analyses)}")
                    response_text = "\n".join(recent) if recent else "I can help with symbols (e.g., AAPL) or summarize recent analyses."

                # Log chat event
                try:
                    from tradingagents.database.db_service import log_event, _make_json_serializable
                    payload = _make_json_serializable({"q": user_msg, "a": response_text[:1500]})
                    log_event("chatbot_interaction", payload)
                except Exception:
                    pass

            except Exception as e:
                response_text = f"Error: {str(e)}"

            st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.write(response_text)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔎 Advanced Search & Filtering")

        colf1, colf2 = st.columns(2)
        with colf1:
            st.markdown("#### Trades Search")
            symbol = st.text_input("Symbol filter (optional)")
            status = st.selectbox("Status", ["any", "open", "closed"], index=0)
            min_pnl = st.number_input("Min Realized P&L", value=float(0))
            max_pnl = st.number_input("Max Realized P&L", value=float(1_000_000))
            start_date = st.date_input("Start date", value=None)
            end_date = st.date_input("End date", value=None)
            if st.button("Search Trades", key="search_trades"):
                trades = trade_service.get_trade_history(limit=1000) or []
                filtered = []
                for t in trades:
                    if symbol and t.get('symbol','').upper() != symbol.upper():
                        continue
                    if status != "any":
                        if status == "open" and t.get('status') == 'closed':
                            continue
                        if status == "closed" and t.get('status') != 'closed':
                            continue
                    realized = float(t.get('realized_pnl', 0)) if t.get('status') == 'closed' else 0.0
                    if realized < min_pnl or realized > max_pnl:
                        continue
                    dt = t.get('exit_date') or t.get('entry_date')
                    if start_date and dt:
                        try:
                            d = pd.to_datetime(dt).date()
                            if d < start_date:
                                continue
                        except Exception:
                            pass
                    if end_date and dt:
                        try:
                            d = pd.to_datetime(dt).date()
                            if d > end_date:
                                continue
                        except Exception:
                            pass
                    filtered.append(t)
                if filtered:
                    rows = []
                    for t in filtered[:300]:
                        rows.append({
                            "Symbol": t.get('symbol','N/A'),
                            "Status": t.get('status','N/A'),
                            "Realized P&L": float(t.get('realized_pnl', t.get('unrealized_pnl', 0)) or 0),
                            "Entry": t.get('entry_date','')[:19],
                            "Exit": (t.get('exit_date') or '')[:19]
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)
                else:
                    st.info("No matching trades.")

        with colf2:
            st.markdown("#### Research Documents Search")
            try:
                doc_manager = DocumentManager()
                doc_symbol = st.text_input("Symbol filter (optional)", key="doc_symbol")
                keyword = st.text_input("Keyword contains (optional)")
                if st.button("Search Documents", key="search_docs"):
                    docs = doc_manager.get_documents(symbol=doc_symbol if doc_symbol else None) or []
                    results = []
                    for d in docs:
                        content = d.get("file_content", "") or ""
                        title = d.get("file_name", "") or ""
                        if keyword and keyword.lower() not in (content.lower() + " " + title.lower()):
                            continue
                        results.append({
                            "File": title,
                            "Symbol": d.get("symbol","N/A"),
                            "Snippet": (content[:160] + "...") if len(content) > 160 else content,
                            "Uploaded": (d.get("uploaded_at") or d.get("created_at") or "")[:19]
                        })
                    if results:
                        st.dataframe(pd.DataFrame(results), use_container_width=True, height=360)
                    else:
                        st.info("No matching documents.")
                doc_manager.close()
            except Exception as e:
                st.error(f"Document search unavailable: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Performance & Optimization")

        @st.cache_data(ttl=300)
        def cached_price_snapshot(symbols):
            data = []
            for sym in symbols:
                try:
                    info, _ = get_stock_data(sym)
                    price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
                    data.append({"Symbol": sym, "Price": price})
                except Exception:
                    data.append({"Symbol": sym, "Price": None})
            return pd.DataFrame(data)

        colp1, colp2 = st.columns(2)
        with colp1:
            if st.button("🧹 Clear Cache", key="clear_cache"):
                try:
                    st.cache_data.clear()
                    st.success("Cache cleared.")
                except Exception as e:
                    st.error(f"Could not clear cache: {str(e)}")
        with colp2:
            if st.button("⚡ Prefetch Watchlist Prices", key="prefetch_prices"):
                symbols = get_watchlist_symbols()
                df = cached_price_snapshot(symbols)
                st.success(f"Prefetched {len(df)} prices (cached 5 minutes)")
                st.dataframe(df, use_container_width=True, height=300)

        st.markdown("#### Tips")
        st.write("- Use cached snapshots for watchlist.\n- Reduce table sizes with compact mode.\n- Close heavy tabs when not in use.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🎨 UI & Preferences")
        compact = st.checkbox("Compact table mode", value=st.session_state.get("ui_compact", False))
        st.session_state.ui_compact = compact
        st.info("Compact mode reduces table heights and whitespace across views.")

        # Apply lightweight styling adjustments
        if compact:
            st.markdown(
                """
                <style>
                .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; }
                .stDataFrame { font-size: 0.9rem; }
                </style>
                """,
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
