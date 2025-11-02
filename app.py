import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime
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

# Phase 2 imports
from tradingagents.dataflows.ai_analysis import AIResearchAnalyzer

# Phase 3 imports
from tradingagents.agents.utils.trading_engine import (
    VolumeScreeningEngine,
    FireTestingEngine,
    AIEnhancedScoringEngine
)

# Phase 4 imports
from tradingagents.agents.utils.session_manager import (
    TradingSessionManager,
    TradeExecutionService
)

GROQ_API_KEY = "" 

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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #f6f8fa 0%, #ffffff 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
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
        border: 1px solid rgba(255, 255, 255, 0.3);
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
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
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

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1y")
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
    
    env_vars = {
        "POLYGON_API_KEY": os.getenv("POLYGON_API_KEY"),
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY")
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
    
    # Database & Infrastructure Setup
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Database & Infrastructure Setup")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Initialize Database", use_container_width=True, key="init_database"):
            if initialize_database():
                st.success("Database initialized successfully!")
                st.balloons()
            else:
                st.error("Database initialization failed!")
    
    with col2:
        if st.button("Initialize Stocks", use_container_width=True, key="init_stocks"):
            pipeline = DataIngestionPipeline()
            if pipeline.initialize_stocks():
                st.success("Stocks initialized successfully!")
                st.balloons()
            else:
                st.error("Stock initialization failed!")
            pipeline.close()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Watchlist display
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Watchlist (19 High-Beta US Stocks)")
    watchlist_df = pd.DataFrame([
        {"Symbol": symbol, "Company": name} 
        for symbol, name in WATCHLIST_STOCKS.items()
    ])
    st.dataframe(watchlist_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data ingestion
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### Data Ingestion Pipeline")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Ingest All Historical Data", use_container_width=True, key="ingest_all_data"):
            pipeline = DataIngestionPipeline()
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            symbols = get_watchlist_symbols()
            results = {}
            
            for i, symbol in enumerate(symbols):
                status_text.text(f"Processing {symbol}... ({i+1}/{len(symbols)})")
                progress_bar.progress((i + 1) / len(symbols))
                
                result = pipeline.ingest_historical_data(symbol, days_back=1825)  # 5 years of data
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
                
                # Save button for data ingestion results
                st.markdown("---")
                if st.button("💾 Save Ingestion Results to Database", key="save_ingestion_results"):
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
                        st.success("✅ Ingestion results saved to database!")
                    except Exception as e:
                        st.error(f"❌ Failed to save: {str(e)}")
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

    # New: 5-minute intraday ingestion (past 2 years)
    with col1:
        if st.button("Ingest All Historical 5-min Data (2 years)", use_container_width=True, key="ingest_all_5min"):
            pipeline = DataIngestionPipeline()
            progress_bar = st.progress(0)
            status_text = st.empty()

            symbols = get_watchlist_symbols()
            results_5min = {}

            for i, symbol in enumerate(symbols):
                status_text.text(f"Processing 5-min {symbol}... ({i+1}/{len(symbols)})")
                progress_bar.progress((i + 1) / len(symbols))

                # 2 years of 5-min data -> days_back=730
                result = pipeline.ingest_historical_data(symbol, days_back=730, interval='5min')
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
                    from datetime import datetime
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    formatted_date = dt.strftime("%Y-%m-%d %H:%M")
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
            doc_df = pd.DataFrame(display_docs)
            st.dataframe(doc_df, use_container_width=True)
            
            # Add document actions
            st.markdown("#### Document Actions")
            selected_doc_idx = st.selectbox(
                "Select Document for Actions", 
                range(len(filtered_docs)),
                format_func=lambda x: f"#{x+1} - {display_docs[x]['File Name']}",
                key="doc_actions"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 View Full Content", key="view_content"):
                    selected_doc = filtered_docs[selected_doc_idx]
                    st.subheader(f"Content: {selected_doc.get('file_name', 'Unknown')}")
                    st.text_area("Document Content", selected_doc.get("file_content", ""), height=300)
            
            with col2:
                if st.button("🗑️ Delete Document", key="delete_doc"):
                    selected_doc = filtered_docs[selected_doc_idx]
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
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs([
        "Master Data Dashboard", 
        "AI Document Analysis", 
        "Instrument Profiles", 
        "Research Insights"
    ])
    
    with tab1:
        st.subheader("Master Data Dashboard")
        
        # Get watchlist symbols
        symbols = get_watchlist_symbols()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Instruments", len(symbols))
        with col2:
            st.metric("AI Analysis Status", "Active")
        with col3:
            st.metric("Documents Processed", "0")  # Will be updated dynamically
        
        # Master data summary
        if st.button("Generate Master Data Summary", type="primary", key="generate_master_data"):
            with st.spinner("Generating comprehensive analysis..."):
                summary = ai_analyzer.get_master_data_summary()
                
                if "error" not in summary:
                    st.success("Master data analysis completed!")
                    st.balloons()
                    
                    # Display summary metrics
                    st.subheader("Analysis Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Instruments", summary["total_instruments"])
                    with col2:
                        bullish_count = sum(1 for inst in summary["instruments"] 
                                          if inst.get("ai_analysis", {}).get("overall_sentiment") == "Bullish")
                        st.metric("Bullish Signals", bullish_count)
                    with col3:
                        bearish_count = sum(1 for inst in summary["instruments"] 
                                          if inst.get("ai_analysis", {}).get("overall_sentiment") == "Bearish")
                        st.metric("Bearish Signals", bearish_count)
                    with col4:
                        avg_confidence = np.mean([inst.get("confidence_score", 5) 
                                                for inst in summary["instruments"]])
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}/10")
                    
                    # Store summary in session state
                    st.session_state.master_data_summary = summary
                    
                    # Save button for master data
                    st.markdown("---")
                    if st.button("💾 Save Master Data Summary to Database", key="save_master_data"):
                        try:
                            from tradingagents.database.db_service import log_event
                            log_event("master_data_generated", {
                                "total_instruments": summary.get("total_instruments", 0),
                                "source": "phase2"
                            })
                            st.success("✅ Master data summary saved to database!")
                        except Exception as e:
                            st.error(f"❌ Failed to save: {str(e)}")
                else:
                    st.error(f"Error generating summary: {summary['error']}")
    
    with tab2:
        st.subheader("AI Document Analysis")
        
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
            
            if st.button("Upload & Analyze", type="primary", key="upload_analyze_phase2"):
                if symbol and title:
                    doc_manager = DocumentManager()
                    
                    with st.spinner("Processing document..."):
                        result = doc_manager.upload_document(
                            uploaded_file, 
                            uploaded_file.name, 
                            title,
                            document_type=doc_type,
                            symbol=symbol
                        )
                    
                    if result["success"]:
                        st.success("Document uploaded successfully!")
                        st.balloons()
                        
                        # Perform AI analysis and persist RAG to Supabase
                        with st.spinner("Analyzing with AI and storing RAG..."):
                            documents = doc_manager.get_documents(symbol=symbol)
                            if documents:
                                latest_doc = documents[-1]
                                doc_id = latest_doc.get("id")
                                
                                if doc_id:
                                    result_bundle = doc_manager.analyze_and_store(doc_id, symbol)
                                    analysis = result_bundle.get("analysis", {})
                                    signals = result_bundle.get("signals", {})
                                    
                                    # Always display results, even if analysis failed
                                    st.subheader("AI Analysis Results")
                                    
                                    # Display analysis results
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Document Analysis:**")
                                        if analysis.get("success"):
                                            st.write(analysis["analysis"])
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
                                    
                                    # Store analysis in session state
                                    if "document_analyses" not in st.session_state:
                                        st.session_state.document_analyses = []
                                    st.session_state.document_analyses.append({
                                        "symbol": symbol,
                                        "title": title,
                                        "analysis": analysis,
                                        "signals": signals
                                    })
                    else:
                        st.error(f"Upload failed: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("Please select a symbol and enter a title")
        
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
        
        # Select symbol for detailed analysis
        selected_symbol = st.selectbox("Select Instrument for Analysis", symbols)
        
        if selected_symbol and st.button("Generate Profile", type="primary", key="generate_profile"):
            with st.spinner(f"Analyzing {selected_symbol}..."):
                profile = ai_analyzer.analyze_instrument_profile(selected_symbol)
                
                if "error" not in profile:
                    st.success(f"Profile generated for {selected_symbol}")
                    st.balloons()
                    
                    # Display comprehensive profile
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Market Data")
                        market_data = profile.get("market_data", {})
                        if "error" not in market_data:
                            st.metric("Current Price", f"${market_data.get('current_price', 'N/A')}")
                            st.metric("30-day Change", f"{market_data.get('price_change_pct', 'N/A')}%")
                            st.metric("Volume", f"{market_data.get('volume', 'N/A'):,}")
                            st.metric("30-day High", f"${market_data.get('high_30d', 'N/A')}")
                            st.metric("30-day Low", f"${market_data.get('low_30d', 'N/A')}")
                        else:
                            st.error("Market data unavailable")
                    
                    with col2:
                        st.subheader("AI Analysis")
                        ai_analysis = profile.get("ai_analysis", {})
                        if "error" not in ai_analysis:
                            st.write("**Overall Assessment:**", ai_analysis.get("overall_sentiment", "N/A"))
                            st.write("**Recommendation:**", ai_analysis.get("recommendation", "N/A"))
                            st.write("**Confidence:**", f"{ai_analysis.get('confidence', 'N/A')}/10")
                            
                            # Display full analysis
                            st.subheader("Detailed Analysis")
                            st.write(ai_analysis.get("analysis_text", "No analysis available"))
                        else:
                            st.error("AI analysis unavailable")
                    
                    # Document insights
                    doc_insights = profile.get("document_insights", [])
                    if doc_insights:
                        st.subheader("Document Insights")
                        for insight in doc_insights:
                            with st.expander(f"📄 {insight.get('filename', 'Unknown')}"):
                                signals = insight.get("signals", {})
                                if signals.get("success"):
                                    st.write(f"**Sentiment:** {signals.get('overall_sentiment', 'N/A')}")
                                    st.write(f"**Confidence:** {signals.get('confidence', 'N/A')}/10")
                                    st.write(f"**Bullish Signals:** {', '.join(signals.get('bullish_signals', []))}")
                                    st.write(f"**Bearish Signals:** {', '.join(signals.get('bearish_signals', []))}")
                else:
                    st.error(f"Error generating profile: {profile['error']}")
    
    with tab4:
        st.subheader("Research Insights Dashboard")
        
        # Overall insights summary
        if "master_data_summary" in st.session_state:
            summary = st.session_state.master_data_summary
            
            st.subheader("Portfolio Overview")
            
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
                st.dataframe(insights_df, use_container_width=True)
                
                # Sentiment distribution
                st.subheader("Sentiment Distribution")
                sentiment_counts = insights_df["Sentiment"].value_counts()
                st.bar_chart(sentiment_counts)
                
                # Recommendation distribution
                st.subheader("Recommendation Distribution")
                rec_counts = insights_df["Recommendation"].value_counts()
                st.bar_chart(rec_counts)
        else:
            st.info("Generate master data summary first to see research insights")
    
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
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data is None or data.empty:
            return None
        
        # Handle MultiIndex columns (yfinance returns MultiIndex with symbol name as second level)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)
        
        return data
    except Exception:
        return None

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

def phase3_trading_engine_core():
    """Phase 3: Trading Engine Core - Complete Phase 1 + Phase 2 Analysis Workflow"""
    
    # Initialize engines
    if 'volume_screener' not in st.session_state:
        st.session_state.volume_screener = VolumeScreeningEngine()
    
    if 'fire_tester' not in st.session_state:
        st.session_state.fire_tester = FireTestingEngine()
    
    if 'ai_scorer' not in st.session_state:
        st.session_state.ai_scorer = AIEnhancedScoringEngine()
    
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
                            from tradingagents.database.db_service import log_event
                            from tradingagents.database.config import get_supabase
                            from datetime import datetime
                            
                            # Check database connection first
                            supabase = get_supabase()
                            if not supabase:
                                st.error("❌ Database not configured! Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
                            else:
                                # Prepare data and convert to JSON-serializable
                                from tradingagents.database.db_service import _make_json_serializable
                                
                                screening_data = {
                                    "total_screened": int(len(screening_df)),
                                    "passed_count": int(len(passed_symbols)),
                                    "failed_count": int(len(screening_df) - len(passed_symbols)),
                                    "passed_symbols": list(passed_symbols),
                                    "screening_results": screening_df.to_dict('records'),  # Full results
                                    "timestamp": datetime.now().isoformat(),
                                    "source": "phase3_tab1"
                                }
                                
                                # Convert all numpy/pandas types
                                screening_data = _make_json_serializable(screening_data)
                                
                                # Save to system_logs table with detailed results
                                result = log_event("volume_screening_completed", screening_data)
                                
                                if result:
                                    st.success("✅ Screening results saved to database! (Saved to system_logs table)")
                                else:
                                    st.error("❌ Failed to save - check console for details")
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
        
        # Show cached results if available
        if 'phase1_results' in st.session_state:
            st.markdown("#### 📋 Previous Screening Results")
            st.dataframe(st.session_state.phase1_results, use_container_width=True)
        
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
                                from tradingagents.database.db_service import log_event
                                from tradingagents.database.config import get_supabase
                                
                                # Check database connection first
                                supabase = get_supabase()
                                if not supabase:
                                    st.error("❌ Database not configured! Please set SUPABASE_URL and SUPABASE_KEY in your .env file")
                                else:
                                    # Prepare data and convert to JSON-serializable
                                    from tradingagents.database.db_service import _make_json_serializable
                                    
                                    fire_test_data = {
                                        "symbol": selected_symbol,
                                        "score": int(fire_result["score"]),
                                        "max_score": int(fire_result["max_score"]),
                                        "score_percentage": float(round(score_ratio * 100, 2)),
                                        "stages": fire_result["stages"],
                                        "timestamp": fire_result.get("timestamp"),
                                        "source": "phase3_tab2"
                                    }
                                    
                                    # Convert all numpy/pandas types
                                    fire_test_data = _make_json_serializable(fire_test_data)
                                    
                                    result = log_event("phase2_fire_test_completed", fire_test_data)
                                    
                                    if result:
                                        st.success("✅ Phase 2 Deep Analysis saved to database!")
                                    else:
                                        st.error("❌ Failed to save - check console for details")
                            except Exception as e:
                                st.error(f"❌ Error saving: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
        
        with col2:
            if st.button("🔄 Clear Results", key="clear_phase2"):
                if 'phase2_results' in st.session_state:
                    del st.session_state.phase2_results
                st.rerun()
        
        # Show cached results if available
        if 'phase2_results' in st.session_state:
            result = st.session_state.phase2_results
            if not result.get("error"):
                st.markdown("#### 📋 Previous Fire Test Results")
                score_ratio = result["score"] / result["max_score"]
                st.metric("Previous Score", f"{result['score']} / {result['max_score']} ({score_ratio*100:.1f}%)")
        
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
                        from datetime import datetime
                        
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
                            from datetime import datetime
                            
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
                            from datetime import datetime
                            
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
    user_id = st.session_state.get("user_id", "default_user")
    
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
                            st.success("✅ New session created!")
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
                                st.success("✅ Session closed!")
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
            # Get current price
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose", 0)
                st.info(f"💵 **Current Price**: ${current_price:.2f}")
            except Exception:
                current_price = 0
                st.warning("⚠️ Could not fetch current price")
            
            entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01, 
                                          value=float(current_price) if current_price else 0.0, 
                                          key="execute_entry_price")
            
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


def main():
    st.markdown('<div class="section-header">AlphaAnalyst Trading AI Agent - Phase 1</div>', unsafe_allow_html=True)
    
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
        st.info("Phase 5: Results & Analysis Modules - Coming Soon!")
    elif st.session_state.active_phase == phases[5]:
        st.info("Phase 6: Advanced Features & Polish - Coming Soon!")
    
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
    
                    
if __name__ == "__main__":
    main()
