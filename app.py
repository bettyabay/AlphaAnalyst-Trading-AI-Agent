import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
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

def phase1_foundation_data():
    """Phase 1: Foundation & Data Infrastructure"""
    st.markdown("## Phase 1: Foundation & Data Infrastructure")
    
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
        st.info("Phase 2: Master Data & AI Integration - Coming Soon!")
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
