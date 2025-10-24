# AlphaAnalyst Trading AI Agent

## Phase 1: Foundation & Data Infrastructure ✅
## Phase 2: Master Data & AI Integration ✅

A comprehensive trading AI agent built with Streamlit, PostgreSQL, and Polygon.io integration.

### 🎯 Phase 1 Features

- **PostgreSQL Database**: Complete data infrastructure with tables for stocks, historical data, and documents
- **Polygon.io Integration**: Real-time and historical market data for 19 high-beta US stocks
- **Watchlist Management**: 19 carefully selected high-beta US stocks (COIN, TSLA, NVDA, AVGO, PLTR, LRCX, CRWD, HOOD, APP, MU, SNPS, CDNS, VST, DASH, DELL, NOW, PANW, AXON, URI)
- **Data Ingestion Pipeline**: Automated historical data collection and storage
- **Document Management**: Upload and store research documents
- **Data Completion Tracking**: Real-time status monitoring for all instruments

### 🎯 Phase 2 Features

- **Enhanced Document Processing**: PDF, DOCX, and TXT file processing with text extraction
- **AI-Powered Research Analysis**: LLM integration with Groq/Phi framework for document analysis
- **Master Data Visualization**: Comprehensive instrument profiles with interactive charts
- **AI Insight Extraction**: Automated bullish/bearish signal detection from research documents
- **Sentiment Analysis**: AI-driven sentiment scoring and confidence metrics
- **Portfolio Overview Dashboard**: Real-time analysis of all instruments with AI insights
- **Document Embeddings**: Vector embeddings for semantic search and similarity analysis
- **Comprehensive Instrument Profiles**: Market data + AI analysis + document insights integration

### 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Environment**:
   - Copy `tradingagents/config/env_template.py` to `.env`
   - Fill in your API keys (GROQ_API_KEY, POLYGON_API_KEY)
   - Configure DATABASE_URL for PostgreSQL

3. **Initialize Database**:
   ```bash
   python setup_database.py
   ```

4. **Run Application**:
   ```bash
   streamlit run app.py
   ```

### 📊 Watchlist (19 High-Beta US Stocks)

| Symbol | Company | Sector |
|--------|---------|--------|
| COIN | Coinbase Global, Inc. | Technology |
| TSLA | Tesla, Inc. | Automotive |
| NVDA | NVIDIA Corporation | Technology |
| AVGO | Broadcom Inc. | Technology |
| PLTR | Palantir Technologies Inc. | Technology |
| LRCX | Lam Research Corporation | Technology |
| CRWD | CrowdStrike Holdings, Inc. | Technology |
| HOOD | Robinhood Markets, Inc. | Financial |
| APP | AppLovin Corporation | Technology |
| MU | Micron Technology, Inc. | Technology |
| SNPS | Synopsys, Inc. | Technology |
| CDNS | Cadence Design Systems, Inc. | Technology |
| VST | Vistra Corp. | Utilities |
| DASH | DoorDash, Inc. | Technology |
| DELL | Dell Technologies Inc. | Technology |
| NOW | ServiceNow, Inc. | Technology |
| PANW | Palo Alto Networks, Inc. | Technology |
| AXON | Axon Enterprise, Inc. | Technology |
| URI | United Rentals, Inc. | Industrial |

### 🛠 Tech Stack

- **Frontend**: Streamlit with custom CSS styling
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Data Source**: Polygon.io API for market data
- **AI Integration**: Phi framework with Groq models
- **Document Storage**: Local file system with database tracking

### 📈 Success Criteria (Phase 1)

- ✅ All 19 stocks can be added with complete historical data
- ✅ Data checklist shows 100% completion for each instrument  
- ✅ System can store and retrieve uploaded research documents
- ✅ PostgreSQL database with proper schema
- ✅ Polygon.io integration for real-time and historical data
- ✅ Document management system
- ✅ Data ingestion pipeline with status tracking

### 📈 Success Criteria (Phase 2)

- ✅ Uploaded PDFs are processed and key insights extracted
- ✅ AI can identify bullish/bearish signals from research
- ✅ Master data page shows comprehensive instrument profiles
- ✅ Enhanced document upload with PDF/DOCX processing
- ✅ AI-powered research analysis with Groq/Phi integration
- ✅ Comprehensive visualization dashboard with interactive charts
- ✅ Automated sentiment analysis and confidence scoring
- ✅ Document embeddings for semantic search capabilities

### 🔧 Configuration

**Market Focus**: US Stocks Only (MVP Phase)  
**Trading Session**: US Market Hours (9:30 AM - 4:00 PM EST)  
**Primary Trading Window**: First 2 Hours (9:30-11:30 AM EST)  
**Risk Management**: 1% per trade = $100 max risk on $10,000 account

### 📁 Project Structure

```
AlphaAnalyst-Trading-AI-Agent/
├── app.py                          # Main Streamlit application
├── setup_database.py              # Database initialization script
├── requirements.txt                # Python dependencies
├── tradingagents/
│   ├── database/                   # Database models and config
│   │   ├── config.py              # Database configuration
│   │   └── models.py              # SQLAlchemy models
│   ├── dataflows/                 # Data integration modules
│   │   ├── polygon_integration.py # Polygon.io API client
│   │   ├── ingestion_pipeline.py  # Data ingestion pipeline
│   │   ├── document_manager.py    # Enhanced document management (Phase 2)
│   │   ├── ai_analysis.py         # AI research analysis (Phase 2)
│   │   └── master_data_viz.py     # Master data visualization (Phase 2)
│   └── config/
│       ├── watchlist.py           # Watchlist configuration
│       └── env_template.py        # Environment variables template
└── README.md                      # This file
```

### 🎯 Next Phases

- **Phase 2**: Master Data & AI Integration ✅
- **Phase 3**: Trading Engine Core  
- **Phase 4**: Session Management & Execution
- **Phase 5**: Results & Analysis Modules
- **Phase 6**: Advanced Features & Polish

### 🤝 Contributing

This is Phase 2 of a 6-phase development plan. Each phase builds upon the previous one to create a comprehensive trading AI agent.

**Phase 2 Achievement**: Successfully implemented AI-powered document analysis, enhanced visualization capabilities, and comprehensive instrument profiling with automated sentiment analysis and trading signal extraction.
