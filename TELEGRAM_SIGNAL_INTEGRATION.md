# Real-Time Telegram Signal Integration - Technical Approach

## Overview
This document outlines the technical approach to integrate real-time Telegram channel signal monitoring into the AlphaAnalyst Trading AI Agent system.

## Architecture Components

### 1. Backend Components

#### 1.1 Telegram Client Service
**File**: `tradingagents/dataflows/telegram_signal_service.py`

**Responsibilities**:
- Connect to Telegram API using `telethon` or `python-telegram-bot`
- Monitor specified Telegram channels in real-time
- Listen for new messages
- Parse and extract signal data from messages
- Store signals in database

**Key Functions**:
```python
class TelegramSignalService:
    def __init__(self, api_id, api_hash, phone_number=None, session_file=None):
        # Initialize Telegram client
        
    async def connect(self):
        # Connect to Telegram
        
    async def monitor_channel(self, channel_username: str, provider_name: str):
        # Monitor a specific channel for new messages
        
    async def parse_signal_message(self, message: str) -> Dict:
        # Extract signal data from message text
        # Returns: {
        #   "symbol": "EURUSD",
        #   "action": "buy" or "sell",
        #   "entry_price": 1.0850,
        #   "stop_loss": 1.0800,
        #   "target_1": 1.0900,
        #   "target_2": 1.0950,
        #   "signal_date": datetime
        # }
        
    async def save_signal(self, signal_data: Dict, provider_name: str):
        # Save parsed signal to signal_provider_signals table
```

**Dependencies**:
- `telethon` (recommended) or `python-telegram-bot`
- `asyncio` for async operations
- Existing `get_supabase()` from `tradingagents.database.config`

#### 1.2 Signal Parser/Extractor
**File**: `tradingagents/dataflows/telegram_signal_parser.py`

**Responsibilities**:
- Parse various signal message formats (flexible parsing)
- Extract symbol, direction, entry, stop loss, targets
- Handle different message formats from different providers
- Use regex patterns and/or NLP for extraction

**Key Functions**:
```python
class TelegramSignalParser:
    def parse(self, message_text: str) -> Optional[Dict]:
        # Main parsing function
        # Try multiple parsing strategies:
        # 1. Structured format (e.g., "BUY EURUSD @ 1.0850 SL: 1.0800 TP1: 1.0900")
        # 2. Natural language (e.g., "We recommend buying EURUSD at 1.0850...")
        # 3. Table format (if message contains formatted table)
        
    def _parse_structured_format(self, text: str) -> Optional[Dict]:
        # Parse structured signals with regex patterns
        
    def _parse_natural_language(self, text: str) -> Optional[Dict]:
        # Use NLP/LLM to extract signal from natural language
        
    def normalize_symbol(self, symbol: str) -> str:
        # Normalize symbol format (e.g., "EUR/USD" -> "C:EURUSD")
```

**Parsing Strategies**:
1. **Regex Patterns**: For structured formats
   - Pattern: `(BUY|SELL)\s+(\w+)\s+@\s+([\d.]+)\s+SL[:\s]+([\d.]+)\s+TP[:\s]+([\d.]+)`
2. **LLM Extraction**: For natural language (optional, using Groq API)
3. **Template Matching**: Provider-specific templates

#### 1.3 Background Service/Worker
**File**: `tradingagents/dataflows/telegram_signal_worker.py`

**Responsibilities**:
- Run as background process/service
- Continuously monitor Telegram channels
- Handle reconnections and errors
- Log activity

**Implementation Options**:
1. **Separate Python Script**: Run independently with `python -m tradingagents.dataflows.telegram_signal_worker`
2. **Streamlit Background Thread**: Use `threading` or `asyncio` in Streamlit
3. **System Service**: Run as Windows Service or Linux systemd service

**Key Functions**:
```python
async def run_telegram_monitor():
    service = TelegramSignalService(...)
    await service.connect()
    
    # Monitor multiple channels
    channels = [
        {"username": "@signal_provider_1", "provider": "Provider1"},
        {"username": "@signal_provider_2", "provider": "Provider2"}
    ]
    
    tasks = []
    for channel in channels:
        task = service.monitor_channel(
            channel["username"], 
            channel["provider"]
        )
        tasks.append(task)
    
    await asyncio.gather(*tasks)
```

### 2. Database Schema

#### 2.1 Existing Table: `signal_provider_signals`
Already exists and can be used. No schema changes needed.

#### 2.2 New Table: `telegram_channels` (Optional)
**Purpose**: Store Telegram channel configuration

```sql
CREATE TABLE IF NOT EXISTS telegram_channels (
    id BIGSERIAL PRIMARY KEY,
    channel_username VARCHAR(255) NOT NULL UNIQUE,
    provider_name VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_message_id BIGINT,
    last_check_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 3. UI Components (Streamlit)

#### 3.1 Telegram Channel Configuration
**Location**: Add to `phase1_foundation_data()` in `app.py`

**Features**:
- Add/Remove Telegram channels
- Configure channel username and provider name
- Start/Stop monitoring
- View connection status

**UI Structure**:
```python
with st.expander("📱 Telegram Signal Channels", expanded=False):
    st.markdown("### Configure Real-Time Signal Channels")
    
    # Channel list
    col1, col2 = st.columns([3, 1])
    with col1:
        channel_username = st.text_input("Channel Username", placeholder="@signal_provider")
    with col2:
        provider_name = st.text_input("Provider Name", placeholder="Provider1")
    
    if st.button("➕ Add Channel"):
        # Save channel config to database or session state
        pass
    
    # Active channels list
    st.markdown("#### Active Channels")
    # Display list of configured channels with status
    # Show: Channel, Provider, Status (Connected/Disconnected), Last Signal Time
    
    # Start/Stop monitoring buttons
    if st.button("▶️ Start Monitoring"):
        # Start background worker
        pass
    
    if st.button("⏹️ Stop Monitoring"):
        # Stop background worker
        pass
```

#### 3.2 Real-Time Signal Feed
**Location**: Add to `phase1_foundation_data()` or create new section

**Features**:
- Live signal feed (auto-refresh)
- Filter by provider, symbol, action
- Signal details (entry, SL, TP)
- Link to KPI calculations

**UI Structure**:
```python
st.markdown("### 🔴 Live Signal Feed")

# Auto-refresh every 5 seconds
if st.button("🔄 Refresh"):
    st.rerun()

# Filter options
col_filter1, col_filter2, col_filter3 = st.columns(3)
with col_filter1:
    filter_provider = st.selectbox("Provider", ["All"] + providers)
with col_filter2:
    filter_symbol = st.selectbox("Symbol", ["All"] + symbols)
with col_filter3:
    filter_action = st.selectbox("Action", ["All", "Buy", "Sell"])

# Fetch latest signals
latest_signals = get_provider_signals(
    provider=filter_provider if filter_provider != "All" else None,
    symbol=filter_symbol if filter_symbol != "All" else None,
    limit=50
)

# Display signals in real-time table
if latest_signals:
    df_signals = pd.DataFrame(latest_signals)
    st.dataframe(df_signals, use_container_width=True)
    
    # Show latest signal prominently
    if df_signals:
        latest = df_signals.iloc[0]
        st.success(f"🆕 Latest: {latest['action'].upper()} {latest['symbol']} @ {latest['entry_price']}")
else:
    st.info("No signals received yet. Make sure monitoring is active.")
```

#### 3.3 Signal Notification Badge
**Location**: Top of Streamlit app

**Features**:
- Show count of new signals since last check
- Click to view latest signals

```python
# In main app header
new_signals_count = get_new_signals_count(since=last_check_time)
if new_signals_count > 0:
    st.sidebar.badge(f"🔔 {new_signals_count} New Signals", type="notification")
```

### 4. Real-Time Updates Strategy

#### Option 1: Streamlit Auto-Refresh (Simple)
- Use `st.rerun()` with `time.sleep()` in a loop
- Refresh every 5-10 seconds
- Pros: Simple, no additional infrastructure
- Cons: Full page refresh, not true real-time

#### Option 2: Session State Polling (Better)
- Store last signal timestamp in `st.session_state`
- Poll database every few seconds
- Only update UI if new signals detected
- Pros: More efficient, better UX
- Cons: Still polling-based

#### Option 3: WebSocket + Separate Service (Advanced)
- Run Telegram worker as separate service
- Use WebSocket to push updates to Streamlit
- Pros: True real-time, efficient
- Cons: More complex, requires additional infrastructure

**Recommended**: Start with Option 2 (Session State Polling)

### 5. Implementation Steps

#### Phase 1: Basic Telegram Integration
1. ✅ Install dependencies: `pip install telethon`
2. ✅ Create `TelegramSignalService` class
3. ✅ Create `TelegramSignalParser` with basic regex parsing
4. ✅ Test connection to Telegram channel
5. ✅ Parse and save one signal manually

#### Phase 2: Real-Time Monitoring
1. ✅ Create background worker script
2. ✅ Implement channel monitoring loop
3. ✅ Add error handling and reconnection logic
4. ✅ Test continuous monitoring

#### Phase 3: UI Integration
1. ✅ Add Telegram channel configuration UI
2. ✅ Add real-time signal feed display
3. ✅ Implement auto-refresh mechanism
4. ✅ Add signal filtering and search

#### Phase 4: Advanced Features
1. ✅ Improve signal parsing (LLM-based for natural language)
2. ✅ Add signal validation and quality checks
3. ✅ Implement signal notifications/alerts
4. ✅ Add signal performance tracking

### 6. Configuration & Environment Variables

Add to `.env`:
```env
# Telegram API Credentials
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE_NUMBER=+1234567890  # Optional, for phone auth
TELEGRAM_SESSION_FILE=telegram_session.session  # Optional

# Telegram Channels (comma-separated)
TELEGRAM_CHANNELS=@provider1:Provider1,@provider2:Provider2
```

### 7. Dependencies

Add to `requirements.txt`:
```
telethon>=1.34.0  # Telegram client library
# OR
python-telegram-bot>=20.0  # Alternative Telegram library
```

### 8. Security Considerations

1. **API Credentials**: Store securely in `.env`, never commit to git
2. **Session Files**: Store Telegram session files securely
3. **Rate Limiting**: Respect Telegram API rate limits
4. **Error Handling**: Handle connection failures gracefully
5. **Data Validation**: Validate all parsed signals before saving

### 9. Testing Strategy

1. **Unit Tests**: Test signal parser with various message formats
2. **Integration Tests**: Test Telegram connection and message retrieval
3. **End-to-End Tests**: Test full flow from Telegram → Database → UI
4. **Mock Tests**: Use mock Telegram client for development

### 10. File Structure

```
tradingagents/
├── dataflows/
│   ├── telegram_signal_service.py      # Main Telegram client service
│   ├── telegram_signal_parser.py       # Signal parsing logic
│   ├── telegram_signal_worker.py       # Background worker
│   └── signal_provider_ingestion.py    # Existing (for file uploads)
├── database/
│   └── db_service.py                   # Add telegram channel functions
└── ...

scripts/
└── run_telegram_monitor.py             # Standalone worker script

app.py                                   # Add UI components
```

### 11. Example Signal Message Formats

**Format 1: Structured**
```
BUY EURUSD
Entry: 1.0850
Stop Loss: 1.0800
Take Profit 1: 1.0900
Take Profit 2: 1.0950
```

**Format 2: Compact**
```
BUY EURUSD @ 1.0850 SL: 1.0800 TP1: 1.0900 TP2: 1.0950
```

**Format 3: Natural Language**
```
We recommend buying EURUSD at current market price around 1.0850. 
Set stop loss at 1.0800 and take profit targets at 1.0900 and 1.0950.
```

### 12. Next Steps

1. **Get Telegram API Credentials**:
   - Go to https://my.telegram.org/apps
   - Create new application
   - Get `api_id` and `api_hash`

2. **Choose Implementation Approach**:
   - Start with simple regex parsing
   - Add LLM parsing later if needed

3. **Set Up Development Environment**:
   - Install `telethon`
   - Test connection to a test channel
   - Parse sample messages

4. **Build Incrementally**:
   - Phase 1: Basic connection and parsing
   - Phase 2: Database integration
   - Phase 3: UI integration
   - Phase 4: Real-time updates

## Questions to Consider

1. **Channel Access**: Do you have access to the Telegram channels? (Some channels are private)
2. **Message Format**: What format do the signals use? (Structured, natural language, etc.)
3. **Frequency**: How often are signals posted? (Affects polling frequency)
4. **Multiple Providers**: Will you monitor multiple channels simultaneously?
5. **Signal Validation**: Should signals be validated before saving? (e.g., check if symbol exists)

