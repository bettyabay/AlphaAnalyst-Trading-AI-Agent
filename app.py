import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
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
        data = data.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Adj Close': 'Adj Close', 'Volume': 'Volume'
        })
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
    

    symbols = get_watchlist_symbols()

    tab1, tab2, tab3 = st.tabs(["Volume Screening", "7-Stage Fire Test", "Trading Wizard"])

    with tab1:
        st.subheader("Phase 1 Volume Screening")
        if st.button("Run Screening", type="primary", key="run_screening"):
            with st.spinner("Screening watchlist..."):
                df = run_volume_screening(symbols)
            st.dataframe(df, use_container_width=True)
            passed = df[df['Pass'] == True]["Symbol"].tolist()
            st.success(f"Passed: {len(passed)} / {len(df)}")
            if passed:
                st.write(", ".join(passed))

    with tab2:
        st.subheader("7-Stage Fire Testing")
        sel = st.selectbox("Select Symbol", symbols, key="fire_symbol")
        if st.button("Run Fire Test", type="primary", key="run_fire"):
            with st.spinner(f"Testing {sel}..."):
                res = run_seven_stage_fire_test(sel)
            if res.get("error"):
                st.error(res["error"])
            else:
                st.metric("Score", f"{res['score']} / {res['max_score']}")
                stages_df = pd.DataFrame([{
                    "Stage": s["name"],
                    "Pass": "✅" if s["pass"] else "❌",
                    "Detail": s["detail"]
                } for s in res["stages"]])
                st.dataframe(stages_df, use_container_width=True)

    with tab3:
        st.subheader("Step-by-Step Trading Interface")
        sel2 = st.selectbox("Symbol", symbols, key="wiz_symbol")
        step = st.radio("Step", ["1) Analyze", "2) Risk & Entry", "3) Recommendation"], horizontal=True)

        if step.startswith("1"):
            with st.spinner("Computing AI-enhanced analysis..."):
                rec = ai_enhanced_recommendation(sel2)
            st.write(f"Recommendation: {rec['recommendation']}")
            st.write(f"Confidence: {rec['confidence']}/10")
            st.metric("Fire Test", f"{rec['fire_test']['score']} / {rec['fire_test']['max_score']}")

        elif step.startswith("2"):
            hist = _fetch_history(sel2, period="3mo")
            if hist is None or hist.empty:
                st.error("No data")
            else:
                atr = _calc_atr(hist).iloc[-1]
                px = hist['Close'].iloc[-1]
                stop = px - 1.5 * atr if atr == atr else px * 0.97
                target = px + 2 * (px - stop)
                st.write(f"Price: {px:.2f}")
                st.write(f"Suggested Stop: {stop:.2f}")
                st.write(f"Suggested Target: {target:.2f}")

        else:
            with st.spinner("Finalizing recommendation..."):
                rec = ai_enhanced_recommendation(sel2)
            st.success(f"{sel2}: {rec['recommendation']} (Confidence {rec['confidence']}/10)")
            st.write("Use Step 2 to refine entries and risk.")

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
        st.info("Phase 4: Session Management & Execution - Coming Soon!")
    elif st.session_state.active_phase == phases[4]:
        st.info("Phase 5: Results & Analysis Modules - Coming Soon!")
    elif st.session_state.active_phase == phases[5]:
        st.info("Phase 6: Advanced Features & Polish - Coming Soon!")
    
    # Original functionality (keep for backward compatibility)
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
