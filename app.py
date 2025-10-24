import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from phi.agent.agent import Agent
from phi.model.groq import Groq
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
    .main { padding: 2rem; }
    .stApp { max-width: 1400px; margin: 0 auto; }
    .card {
        background: linear-gradient(135deg, #f6f8fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e1e4e8;
    }

    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0366d6;
    }

    .metric-label {
        font-size: 14px;
        color: #586069;
        text-transform: uppercase;
    }
    
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e1e4e8;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_agents():
    if not st.session_state.get('agents_initialized', False):
        try:
            st.session_state.web_agent = Agent(
                name="Web Search Agent",
                role="Search the web for information",
                model=Groq(api_key=GROQ_API_KEY),
                tools=[GoogleSearch(fixed_max_results=5), DuckDuckGo(fixed_max_results=5)]
            )
            st.session_state.finance_agent = Agent(
                name="Financial AI Agent",
                role="Providing financial insights",
                model=Groq(api_key=GROQ_API_KEY),
                tools=[YFinanceTools()]
            )
            st.session_state.multi_ai_agent = Agent(
                name='Stock Market Agent',
                role='Stock market analysis specialist',
                model=Groq(api_key=GROQ_API_KEY),
                team=[st.session_state.web_agent, st.session_state.finance_agent]
            )
            st.session_state.agents_initialized = True
            return True
        except Exception as e:
            st.error(f"Agent initialization error: {str(e)}")
            return False

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

def create_database_tables():
    """Create database tables"""
    try:
        from tradingagents.database.config import get_supabase
        sb = get_supabase()
        
        if not sb:
            st.error("❌ Database: Not configured")
            return False
        
        # SQL to create market_data table
        market_data_sql = """
        CREATE TABLE IF NOT EXISTS public.market_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(10,2),
            high DECIMAL(10,2),
            low DECIMAL(10,2),
            close DECIMAL(10,2),
            volume BIGINT,
            source VARCHAR(50) DEFAULT 'polygon',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(symbol, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON public.market_data(symbol);
        CREATE INDEX IF NOT EXISTS idx_market_data_date ON public.market_data(date);
        """
        
        # SQL to create research_documents table
        documents_sql = """
        CREATE TABLE IF NOT EXISTS public.research_documents (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            title VARCHAR(255),
            file_content TEXT,
            file_type VARCHAR(50),
            symbol VARCHAR(20),
            uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_research_documents_symbol ON public.research_documents(symbol);
        CREATE INDEX IF NOT EXISTS idx_research_documents_filename ON public.research_documents(filename);
        """
        
        # Try to create tables using RPC (if available)
        try:
            sb.rpc('exec_sql', {'sql': market_data_sql}).execute()
            sb.rpc('exec_sql', {'sql': documents_sql}).execute()
            st.success("✅ Database Tables: Created successfully!")
            return True
        except:
            # If RPC doesn't work, provide manual instructions
            st.warning("⚠️ Automatic table creation failed. Please create tables manually in Supabase dashboard.")
            st.markdown("**Manual Steps:**")
            st.markdown("1. Go to your Supabase dashboard")
            st.markdown("2. Navigate to SQL Editor")
            st.markdown("3. Run the following SQL:")
            st.code(market_data_sql, language='sql')
            st.code(documents_sql, language='sql')
            return False
            
    except Exception as e:
        st.error(f"❌ Database Table Creation Error: {str(e)}")
        return False

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
    st.markdown("## Phase 1: Foundation & Data Infrastructure")
    
    # Debug Information Section
    st.markdown("### 🔧 Debug Information")
    debug_col1, debug_col2, debug_col3, debug_col4 = st.columns(4)
    with debug_col1:
        if st.button("Test Polygon API", use_container_width=True):
            test_polygon_connection()
    with debug_col2:
        if st.button("Test Database", use_container_width=True):
            test_database_connection()
    with debug_col3:
        if st.button("Create Tables", use_container_width=True):
            create_database_tables()
    with debug_col4:
        if st.button("List Tables", use_container_width=True):
            list_database_tables()
    
    st.markdown("---")
    
    # Database initialization
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Initialize Database", use_container_width=True):
            if initialize_database():
                st.success("Database initialized successfully!")
            else:
                st.error("Database initialization failed!")
    
    with col2:
        if st.button("Initialize Stocks", use_container_width=True):
            pipeline = DataIngestionPipeline()
            if pipeline.initialize_stocks():
                st.success("Stocks initialized successfully!")
            else:
                st.error("Stock initialization failed!")
            pipeline.close()
    
    # Watchlist display
    st.markdown("### Watchlist (19 High-Beta US Stocks)")
    watchlist_df = pd.DataFrame([
        {"Symbol": symbol, "Company": name} 
        for symbol, name in WATCHLIST_STOCKS.items()
    ])
    st.dataframe(watchlist_df, use_container_width=True)
    
    # Data ingestion
    st.markdown("### Data Ingestion Pipeline")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Ingest All Historical Data", use_container_width=True):
            pipeline = DataIngestionPipeline()
            with st.spinner("Ingesting historical data for all stocks..."):
                results = pipeline.ingest_all_historical_data()
                pipeline.close()
                
                success_count = sum(1 for success in results.values() if success)
                st.success(f"Successfully ingested data for {success_count}/{len(results)} stocks")
    
    with col2:
        if st.button("Check Data Status", use_container_width=True):
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
                        "Completion %": f"{min(100, (count/252)*100):.1f}%"
                    })
                status_df = pd.DataFrame(rows)
                st.dataframe(status_df, use_container_width=True)
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
    
    # Document management
    st.markdown("### Document Management")
    doc_manager = DocumentManager()
    
    # Upload document
    uploaded_file = st.file_uploader("Upload Research Document", type=['pdf', 'txt', 'docx'])
    if uploaded_file:
        symbol = st.selectbox("Select Stock (Optional)", [""] + list(WATCHLIST_STOCKS.keys()))
        title = st.text_input("Document Title", value=uploaded_file.name)
        
        if st.button("Upload Document"):
            result = doc_manager.upload_document(
                uploaded_file, 
                uploaded_file.name, 
                title,
                symbol=symbol if symbol else None
            )
            if result["success"]:
                st.success("Document uploaded successfully!")
            else:
                st.error(f"Upload failed: {result['message']}")
    
    # Display documents
    st.markdown("#### Uploaded Documents")
    documents = doc_manager.get_documents()
    if documents:
        # Transform documents for display
        display_docs = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            display_docs.append({
                "ID": doc.get("id", "")[:8] + "...",  # Show first 8 chars of UUID
                "File Name": metadata.get("file_name", "Unknown"),
                "Symbol": metadata.get("symbol", "N/A"),
                "Content Preview": doc.get("content", "")[:100] + "..." if len(doc.get("content", "")) > 100 else doc.get("content", ""),
                "Created": doc.get("created_at", "Unknown")[:10] if doc.get("created_at") else "Unknown"
            })
        doc_df = pd.DataFrame(display_docs)
        st.dataframe(doc_df, use_container_width=True)
    else:
        st.info("No documents uploaded yet")
    
    doc_manager.close()

def phase2_master_data_ai():
    """Phase 2: Master Data & AI Integration"""
    st.header("🎯 Phase 2: Master Data & AI Integration")
    
    # Initialize AI analyzer
    ai_analyzer = AIResearchAnalyzer()
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Master Data Dashboard", 
        "🤖 AI Document Analysis", 
        "📈 Instrument Profiles", 
        "🔍 Research Insights"
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
        if st.button("Generate Master Data Summary", type="primary"):
            with st.spinner("Generating comprehensive analysis..."):
                summary = ai_analyzer.get_master_data_summary()
                
                if "error" not in summary:
                    st.success("Master data analysis completed!")
                    
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
        
        uploaded_file = st.file_uploader(
            "Choose a document", 
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files for AI analysis"
        )
        
        if uploaded_file:
            col1, col2, col3 = st.columns(3)
            with col1:
                symbol = st.selectbox("Select Symbol", [""] + symbols)
            with col2:
                title = st.text_input("Document Title", uploaded_file.name)
            with col3:
                doc_type = st.selectbox("Document Type", ["research", "earnings", "news", "analysis"])
            
            if st.button("Upload & Analyze", type="primary"):
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
                        
                        # Perform AI analysis
                        with st.spinner("Performing AI analysis..."):
                            # Get the document ID from the result
                            documents = doc_manager.get_documents(symbol=symbol)
                            if documents:
                                latest_doc = documents[-1]  # Get the most recent document
                                doc_id = latest_doc.get("id")
                                
                                if doc_id:
                                    analysis = doc_manager.analyze_document_with_ai(doc_id, symbol)
                                    signals = doc_manager.extract_trading_signals(doc_id)
                                    
                                    if analysis.get("success"):
                                        st.success("AI Analysis completed!")
                                        
                                        # Display analysis results
                                        st.subheader("AI Analysis Results")
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**Document Analysis:**")
                                            st.write(analysis["analysis"])
                                        
                                        with col2:
                                            st.write("**Trading Signals:**")
                                            if signals.get("success"):
                                                st.write(f"**Sentiment:** {signals['overall_sentiment']}")
                                                st.write(f"**Confidence:** {signals['confidence']}/10")
                                                st.write(f"**Bullish Signals:** {len(signals['bullish_signals'])}")
                                                st.write(f"**Bearish Signals:** {len(signals['bearish_signals'])}")
                                        
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
                    st.write("**Analysis:**", analysis['analysis']['analysis'])
                    if analysis['signals'].get('success'):
                        st.write("**Signals:**", analysis['signals']['overall_sentiment'])
    
    with tab3:
        st.subheader("Instrument Profiles")
        
        # Select symbol for detailed analysis
        selected_symbol = st.selectbox("Select Instrument for Analysis", symbols)
        
        if selected_symbol and st.button("Generate Profile", type="primary"):
            with st.spinner(f"Analyzing {selected_symbol}..."):
                profile = ai_analyzer.analyze_instrument_profile(selected_symbol)
                
                if "error" not in profile:
                    st.success(f"Profile generated for {selected_symbol}")
                    
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

def main():
    st.title("Stocks Analysis AI Agent")
    
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
    
    row1 = st.columns(3)
    with row1[0]:
        if st.button(phases[0], use_container_width=True):
            st.session_state.active_phase = phases[0]
    with row1[1]:
        if st.button(phases[1], use_container_width=True):
            st.session_state.active_phase = phases[1]
    with row1[2]:
        if st.button(phases[2], use_container_width=True):
            st.session_state.active_phase = phases[2]
    
    row2 = st.columns(3)
    with row2[0]:
        if st.button(phases[3], use_container_width=True):
            st.session_state.active_phase = phases[3]
    with row2[1]:
        if st.button(phases[4], use_container_width=True):
            st.session_state.active_phase = phases[4]
    with row2[2]:
        if st.button(phases[5], use_container_width=True):
            st.session_state.active_phase = phases[5]
    
    st.markdown(f"<div class='card'><div class='metric-label'>Active Phase</div><div class='metric-value'>{st.session_state.active_phase}</div></div>", unsafe_allow_html=True)
    
    # Phase-specific content
    if st.session_state.active_phase == phases[0]:
        phase1_foundation_data()
    elif st.session_state.active_phase == phases[1]:
        phase2_master_data_ai()
    elif st.session_state.active_phase == phases[2]:
        st.info("Phase 3: Trading Engine Core - Coming Soon!")
    elif st.session_state.active_phase == phases[3]:
        st.info("Phase 4: Session Management & Execution - Coming Soon!")
    elif st.session_state.active_phase == phases[4]:
        st.info("Phase 5: Results & Analysis Modules - Coming Soon!")
    elif st.session_state.active_phase == phases[5]:
        st.info("Phase 6: Advanced Features & Polish - Coming Soon!")
    
    # Original functionality (keep for backward compatibility)
    st.markdown("---")
    st.markdown("### Original Analysis (Legacy)")
    stock_input = st.text_input("Enter Company Name", help="e.g., APPLE, TCS")
    
    if st.button("Analyze", use_container_width=True):
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
                        st.markdown(f"<div class='card'><div class='metric-value'>${info.get('currentPrice', 'N/A')}</div><div class='metric-label'>Current Price</div></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='card'><div class='metric-value'>{info.get('forwardPE', 'N/A')}</div><div class='metric-label'>Forward P/E</div></div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"<div class='card'><div class='metric-value'>{info.get('recommendationKey', 'N/A').title()}</div><div class='metric-label'>Recommendation</div></div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    st.plotly_chart(create_price_chart(hist, symbol), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    if 'longBusinessSummary' in info:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("### Company Overview")
                        st.write(info['longBusinessSummary'])
                        st.markdown("</div>", unsafe_allow_html=True)
                    
if __name__ == "__main__":
    main()
