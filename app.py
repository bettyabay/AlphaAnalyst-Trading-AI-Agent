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
from tradingagents.dataflows.ingestion_pipeline import DataIngestionPipeline
from tradingagents.dataflows.document_manager import DocumentManager
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

def create_price_chart(hist_data, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index, open=hist_data['Open'],
        high=hist_data['High'], low=hist_data['Low'],
        close=hist_data['Close'], name='OHLC'
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
            common_tables = ['market_data', 'research_documents', 'users', 'positions', 'instrument_master_data', 'portfolio', 'system_logs', 'trade_signals']
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
    
    
    # Selection UI (Financial Instruments, Signal Provider, KPI Indicator)
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Selection")
    col_a, col_b, col_c = st.columns(3)
    # Financial Instruments dropdown
    with col_a:
        categories = ["Commodities", "Indices", "Currencies", "Stocks", "Add..."]
        selected_category = st.selectbox(
            "Financial Instrument Category",
            options=categories,
            index=0,
            key="select_financial_category"
        )
        if selected_category == "Add...":
            st.info("Add Financial Instrument Category flow will be implemented next.")
        instruments_map = {
            "Commodities": ["GOLD", "Add..."],
            "Indices": ["S&P 500", "Add..."],
            "Currencies": ["EUR/USD", "Add..."],
            "Stocks": [*WATCHLIST_STOCKS.keys(), "Add..."]
        }
        current_instruments = instruments_map.get(selected_category, ["Add..."])
        selected_instrument_item = st.selectbox(
            "Financial Instrument",
            options=current_instruments,
            index=0,
            key="select_financial_instrument_item"
        )
        if selected_instrument_item == "Add...":
            st.info("Add Financial Instrument flow will be implemented next.")
    # Signal Provider dropdown
    with col_b:
        signal_provider_options = [
            "PipXpert",
            "Add..."
        ]
        selected_provider = st.selectbox(
            "Signal Provider",
            options=signal_provider_options,
            index=0,
            key="select_signal_provider"
        )
        if selected_provider == "Add...":
            st.info("Add Signal Provider flow will be implemented next.")
    # KPI Indicator dropdown
    with col_c:
        kpi_options = ["ATR", "Volume", "VAMP", "EMA", "SMA", "Add..."]
        selected_kpi = st.selectbox(
            "KPI Indicator",
            options=kpi_options,
            index=0,
            key="select_kpi"
        )
        if selected_kpi == "Add...":
            st.info("Add KPI flow will be implemented next.")
    st.markdown('</div>', unsafe_allow_html=True)
    
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
                        resp1 = sb.table("market_data_1min").select("timestamp").eq("symbol", symbol).limit(1).execute()
                        # If query succeeds, get count via RPC or select
                        try:
                            resp1c = sb.rpc("get_symbol_stats_1min", {"p_symbol": symbol}).execute()
                            data1 = resp1c.data if hasattr(resp1c, "data") else []
                            count1 = len(data1) if isinstance(data1, list) else 0
                        except Exception:
                            resp1all = sb.table("market_data_1min").select("timestamp").eq("symbol", symbol).execute()
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
    lab_symbol = st.selectbox("Symbol", symbols_for_lab, key="quantum_symbol")
    
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
                    check_1min = supabase.table("market_data_1min")\
                        .select("symbol,timestamp")\
                        .eq("symbol", lab_symbol.upper())\
                        .order("timestamp", desc=True)\
                        .limit(5)\
                        .execute()
                    db_check_info["1min_raw"] = check_1min.data if hasattr(check_1min, "data") else []
                    check_5min = supabase.table("market_data_5min")\
                        .select("symbol,timestamp")\
                        .eq("symbol", lab_symbol.upper())\
                        .order("timestamp", desc=True)\
                        .limit(5)\
                        .execute()
                    db_check_info["5min_raw"] = check_5min.data if hasattr(check_5min, "data") else []
                    all_symbols_1min = supabase.table("market_data_1min")\
                        .select("symbol")\
                        .limit(100)\
                        .execute()
                    unique_symbols_1min = list(set([r.get("symbol") for r in (all_symbols_1min.data if hasattr(all_symbols_1min, "data") else []) if r.get("symbol")]))
                    db_check_info["all_symbols_1min"] = sorted(unique_symbols_1min)[:20]
                    all_symbols_5min = supabase.table("market_data_5min")\
                        .select("symbol")\
                        .limit(100)\
                        .execute()
                    unique_symbols_5min = list(set([r.get("symbol") for r in (all_symbols_5min.data if hasattr(all_symbols_5min, "data") else []) if r.get("symbol")]))
                    db_check_info["all_symbols_5min"] = sorted(unique_symbols_5min)[:20]
                except Exception as e:
                    import traceback
                    db_check_info["error"] = str(e)
                    db_check_info["error_traceback"] = traceback.format_exc()
            latest_1min = fetch_latest_bar(lab_symbol, "1min")
            latest_5min = fetch_latest_bar(lab_symbol, "5min")
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
            with st.expander("🔍 Debug: Latest Data Query Results", expanded=True):
                st.write(f"**Direct Database Query Results:**")
                if db_check_info.get("error"):
                    st.error(f"Database query error: {db_check_info['error']}")
                st.write(f"**1-minute table - Top 5 records for {lab_symbol.upper()}:**")
                if db_check_info.get("1min_raw"):
                    for i, record in enumerate(db_check_info["1min_raw"][:5], 1):
                        st.write(f"{i}. Symbol: {record.get('symbol')}, Timestamp: {record.get('timestamp')}")
                else:
                    st.write("❌ No records found in market_data_1min table")
                st.write(f"**5-minute table - Top 5 records for {lab_symbol.upper()}:**")
                if db_check_info.get("5min_raw"):
                    for i, record in enumerate(db_check_info["5min_raw"][:5], 1):
                        st.write(f"{i}. Symbol: {record.get('symbol')}, Timestamp: {record.get('timestamp')}")
                else:
                    st.write("❌ No records found in market_data_5min table")
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
                    if st.button(f"📥 Ingest Latest Data for {lab_symbol}", key="quick_ingest_quantum", type="primary"):
                        latest_before_1min = fetch_latest_bar(lab_symbol, "1min")
                        latest_before_5min = fetch_latest_bar(lab_symbol, "5min")
                        before_1min_ts = pd.to_datetime(latest_before_1min.get("timestamp")).replace(tzinfo=None) if latest_before_1min else None
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
                                latest_after_1min = fetch_latest_bar(lab_symbol, "1min")
                                latest_after_5min = fetch_latest_bar(lab_symbol, "5min")
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
                qt_result = feature_lab.run_quantum_screen(lab_symbol, quantum_timestamp)
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
                    decision = decision_engine.evaluate_trade_decision(lab_symbol, quantum_timestamp)
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
            cond_rows.append({
                "Condition": "1. Composite_Score ≥ 6.5",
                "Status": "✅ PASS" if cond1.get("pass") else "❌ FAIL",
                "Value": f"{cond1.get('value', 'N/A'):.2f}" if cond1.get('value') else "N/A",
                "Details": f"Threshold: {cond1.get('threshold', 6.5)}"
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
            with st.expander("📊 How These Metrics Are Calculated", expanded=True):
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
                                    with st.expander("View Errors"):
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

                        with st.expander("Raw vendor output"):
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
                                with st.expander("Raw vendor output"):
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
                with st.expander(f"{analysis['symbol']} - {analysis['title']}"):
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
                                with st.expander("Error Details"):
                                    st.code(details)
                    
                    # Document insights
                    doc_insights = profile.get("document_insights", [])
                    if doc_insights:
                        st.subheader("Document Insights")
                        st.caption(f"**Source**: `research_documents` table → {len(doc_insights)} document(s) found for {selected_symbol}")
                        for insight in doc_insights:
                            with st.expander(f"📄 {insight.get('filename', 'Unknown')}"):
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
        with st.expander("View Generated Prompt Payload"):
            st.code(datascan_result["prompt"], language="markdown")

    samples = datascan_result.get("samples") or {}
    if samples:
        with st.expander("Recent Candles Preview (D1 / M5 / M1)"):
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
                        with st.expander("View Stage Details"):
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
                        with st.expander("View AI Analysis"):
                            st.write(ai_analysis.get('analysis_text', 'No analysis available'))
                    
                    # Show fire test summary
                    fire_test = recommendation_result.get('fire_test', {})
                    if fire_test and not fire_test.get('error'):
                        with st.expander("View Fire Test Summary"):
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
                
                with st.expander(f"{pnl_color} {symbol} | Entry: ${entry_price:.2f} | Current: ${current_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)"):
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
                        
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.plotly_chart(create_price_chart(hist, symbol), use_container_width=True)
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
