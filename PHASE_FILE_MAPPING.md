# Phase 1 & Phase 2 File Mapping Guide

## 📁 File Architecture Overview

This document maps all files used in Phase 1 and Phase 2, explaining their purpose and which phase they belong to. This is essential for scaling and updating phases individually.

## 🎯 Core Application Files

### **Main Application**
| File | Phase | Purpose | Dependencies |
|------|-------|---------|--------------|
| `app.py` | **Both** | Main Streamlit application with phase routing | All modules |
| `requirements.txt` | **Both** | Python dependencies for both phases | - |
| `setup_database.py` | **Phase 1** | Database initialization script | database/config.py |

## 🗄️ Database Layer (Phase 1 Foundation)

### **Database Configuration**
| File | Phase | Purpose | Used By |
|------|-------|---------|---------|
| `tradingagents/database/config.py` | **Phase 1** | Supabase/PostgreSQL connection configuration | app.py, setup_database.py |
| `tradingagents/database/models.py` | **Phase 1** | Database schema definitions and table structures | config.py |

**What they do:**
- `config.py`: Establishes database connections, manages Supabase client
- `models.py`: Defines table schemas for market_data, documents, positions, etc.

## 📊 Data Flow Layer

### **Phase 1 Data Flow Files**
| File | Phase | Purpose | Key Functions |
|------|-------|---------|---------------|
| `tradingagents/dataflows/polygon_integration.py` | **Phase 1** | Polygon.io API client for market data | get_recent_data(), get_historical_data() |
| `tradingagents/dataflows/ingestion_pipeline.py` | **Phase 1** | Data ingestion and storage pipeline | initialize_stocks(), ingest_all_historical_data() |
| `tradingagents/dataflows/document_manager.py` | **Both** | Document upload, storage, and management | upload_document(), get_documents() |

### **Phase 2 Data Flow Files**
| File | Phase | Purpose | Key Functions |
|------|-------|---------|---------------|
| `tradingagents/dataflows/ai_analysis.py` | **Phase 2** | AI-powered research analysis engine | analyze_instrument_profile(), get_master_data_summary() |
| `tradingagents/dataflows/master_data_viz.py` | **Phase 2** | Comprehensive visualization system | create_instrument_profile_chart(), create_portfolio_overview_chart() |

**What they do:**
- **Phase 1**: Basic data collection, storage, and retrieval
- **Phase 2**: AI analysis, sentiment detection, and advanced visualization

## ⚙️ Configuration Layer

### **Configuration Files**
| File | Phase | Purpose | Key Data |
|------|-------|---------|----------|
| `tradingagents/config/watchlist.py` | **Both** | 19 high-beta US stocks configuration | WATCHLIST_STOCKS, get_watchlist_symbols() |
| `tradingagents/config/env_template.py` | **Both** | Environment variables template | API keys, database URLs |

**What they do:**
- `watchlist.py`: Defines the 19 stocks for analysis
- `env_template.py`: Template for environment configuration

## 🚫 Unused Files (Future Phases)

### **Trading Agents (Not Used in Phase 1/2)**
These files exist but are **NOT USED** in Phase 1 or Phase 2:

| Directory | Purpose | Phase |
|-----------|---------|-------|
| `tradingagents/agents/` | Trading agent implementations | **Phase 3+** |
| `tradingagents/graph/` | Trading graph and workflow | **Phase 3+** |
| `tradingagents/dataflows/alpha_vantage_*` | Alpha Vantage API integration | **Not Used** |
| `tradingagents/dataflows/google.py` | Google search integration | **Not Used** |
| `tradingagents/dataflows/reddit_utils.py` | Reddit data collection | **Not Used** |
| `tradingagents/dataflows/y_finance.py` | Yahoo Finance integration | **Not Used** |

## 📋 Phase-Specific File Usage

### **Phase 1: Foundation & Data Infrastructure**

#### **Core Files Used:**
```
app.py (phase1_foundation_data function)
├── tradingagents/database/config.py
├── tradingagents/dataflows/ingestion_pipeline.py
├── tradingagents/dataflows/document_manager.py (basic version)
├── tradingagents/dataflows/polygon_integration.py
├── tradingagents/config/watchlist.py
└── setup_database.py
```

#### **Phase 1 Function Flow:**
1. **Database Initialization** → `database/config.py`
2. **Stock Initialization** → `dataflows/ingestion_pipeline.py`
3. **Watchlist Display** → `config/watchlist.py`
4. **Data Ingestion** → `dataflows/ingestion_pipeline.py`
5. **Document Management** → `dataflows/document_manager.py` (basic)
6. **Data Status** → `dataflows/ingestion_pipeline.py`

### **Phase 2: Master Data & AI Integration**

#### **Core Files Used:**
```
app.py (phase2_master_data_ai function)
├── tradingagents/dataflows/ai_analysis.py (NEW)
├── tradingagents/dataflows/master_data_viz.py (NEW)
├── tradingagents/dataflows/document_manager.py (ENHANCED)
├── tradingagents/dataflows/polygon_integration.py (reused)
├── tradingagents/config/watchlist.py (reused)
└── tradingagents/database/config.py (reused)
```

#### **Phase 2 Function Flow:**
1. **Master Data Dashboard** → `ai_analysis.py` + `master_data_viz.py`
2. **AI Document Analysis** → `document_manager.py` (enhanced) + `ai_analysis.py`
3. **Instrument Profiles** → `ai_analysis.py` + `polygon_integration.py`
4. **Research Insights** → `ai_analysis.py` + `master_data_viz.py`

## 🔄 File Dependencies

### **Phase 1 Dependencies:**
```
app.py
├── database/config.py
├── dataflows/ingestion_pipeline.py
│   ├── database/config.py
│   └── polygon_integration.py
├── dataflows/document_manager.py
│   └── database/config.py
├── dataflows/polygon_integration.py
└── config/watchlist.py
```

### **Phase 2 Dependencies:**
```
app.py
├── dataflows/ai_analysis.py
│   ├── dataflows/document_manager.py
│   ├── dataflows/polygon_integration.py
│   └── database/config.py
├── dataflows/master_data_viz.py
│   ├── dataflows/polygon_integration.py
│   └── dataflows/ai_analysis.py
├── dataflows/document_manager.py (enhanced)
│   ├── database/config.py
│   └── NEW: PyPDF2, docx, textract, sentence-transformers
└── config/watchlist.py
```

## 🛠️ For Scaling and Updates

### **To Update Phase 1:**
Focus on these files:
- `tradingagents/dataflows/ingestion_pipeline.py`
- `tradingagents/dataflows/polygon_integration.py`
- `tradingagents/dataflows/document_manager.py` (basic functions)
- `tradingagents/database/config.py`
- `app.py` (phase1_foundation_data function)

### **To Update Phase 2:**
Focus on these files:
- `tradingagents/dataflows/ai_analysis.py`
- `tradingagents/dataflows/master_data_viz.py`
- `tradingagents/dataflows/document_manager.py` (enhanced functions)
- `app.py` (phase2_master_data_ai function)
- `requirements.txt` (Phase 2 dependencies)

### **To Add Phase 3:**
You would likely use:
- `tradingagents/agents/` (currently unused)
- `tradingagents/graph/` (currently unused)
- Create new files in `tradingagents/dataflows/` for trading logic
- Extend `app.py` with `phase3_trading_engine()` function

## 📦 Requirements by Phase

### **Phase 1 Requirements:**
```python
# Core dependencies
streamlit
pandas
plotly
supabase
polygon-api-client
python-dotenv
requests
```

### **Phase 2 Additional Requirements:**
```python
# AI and document processing
phidata
groq
PyPDF2
python-docx
textract
sentence-transformers
scikit-learn
numpy
matplotlib
seaborn
```

## 🎯 Key Takeaways

1. **Phase 1** is about data infrastructure and basic document management
2. **Phase 2** adds AI analysis and advanced visualization
3. **Unused files** in `agents/` and `graph/` are for future phases
4. **Document manager** is enhanced in Phase 2 with PDF processing and AI
5. **Database layer** is shared between both phases
6. **Configuration** is shared between both phases

This architecture allows you to:
- Update phases independently
- Scale features without breaking existing functionality
- Add new phases by extending the existing structure
- Maintain clean separation of concerns
