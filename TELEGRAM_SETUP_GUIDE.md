# Telegram Signal Integration - Quick Start Guide

## ✅ Implementation Complete

The Telegram signal integration has been successfully implemented! The parser correctly handles your message format:

```
📣GBP/USD📣
Direction: SELL
Entry Price:   1.3493
TP1     1.3478
TP2     1.3458
TP3     1.3426
SL       1.3546
```

## 📋 What's Been Implemented

### 1. Core Components
- ✅ **TelegramSignalParser** (`tradingagents/dataflows/telegram_signal_parser.py`)
  - Parses structured signal messages
  - Normalizes symbols (GBP/USD → C:GBPUSD)
  - Extracts entry, stop loss, and targets

- ✅ **TelegramSignalService** (`tradingagents/dataflows/telegram_signal_service.py`)
  - Connects to Telegram API
  - Monitors channels for new messages
  - Saves signals to database automatically

- ✅ **TelegramSignalWorker** (`tradingagents/dataflows/telegram_signal_worker.py`)
  - Background service for continuous monitoring
  - Can run as standalone script

### 2. Database Integration
- ✅ Database helper functions in `db_service.py`
- ✅ SQL migration file: `sql/create_telegram_channels_table.sql`
- ✅ Uses existing `signal_provider_signals` table

### 3. UI Components
- ✅ Telegram channel configuration panel
- ✅ Real-time signal feed display
- ✅ Channel management (add/remove/pause/resume)

## 🚀 Setup Instructions

### Step 1: Install Dependencies

```bash
pip install telethon
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Step 2: Get Telegram API Credentials

1. Go to https://my.telegram.org/apps
2. Log in with your phone number
3. Create a new application
4. Copy your `api_id` and `api_hash`

### Step 3: Configure Environment Variables

Add to your `.env` file:

```env
TELEGRAM_API_ID=your_api_id_here
TELEGRAM_API_HASH=your_api_hash_here
TELEGRAM_PHONE_NUMBER=+1234567890  # Optional, for phone auth
```

### Step 4: Create Database Table

Run the SQL migration:

```sql
-- Run sql/create_telegram_channels_table.sql in your Supabase SQL editor
```

Or use the Supabase dashboard to execute the SQL from `sql/create_telegram_channels_table.sql`.

### Step 5: Configure Channels in UI

1. Start your Streamlit app: `streamlit run app.py`
2. Navigate to **Phase 1: Foundation Data**
3. In the **Signal Provider** column, expand **"📱 Telegram Signal Channels"**
4. Add your Telegram channel:
   - Channel Username: `@your_signal_channel`
   - Provider Name: `YourProviderName`
5. Click **"➕ Add Channel"**

### Step 6: Start Monitoring

Run the Telegram worker in a separate terminal:

```bash
python -m tradingagents.dataflows.telegram_signal_worker
```

**First time setup:**
- You'll be prompted to enter your phone number
- Telegram will send a verification code
- Enter the code when prompted
- If you have 2FA enabled, enter your password

**The worker will:**
- Connect to Telegram
- Monitor all active channels
- Parse incoming signals automatically
- Save signals to the database

## 📱 Using the UI

### View Signals

1. In the Streamlit app, go to **Phase 1: Foundation Data**
2. Expand **"🔴 Live Signal Feed"**
3. Use filters to view signals by:
   - Provider
   - Symbol
   - Action (Buy/Sell)
4. Click **"🔄 Refresh Feed"** to update

### Manage Channels

- **Add Channel**: Enter username and provider name, click "Add"
- **Pause/Resume**: Click buttons next to each channel
- **Delete**: Click "Delete" button to remove a channel

## 🧪 Testing

Test the parser with your message format:

```bash
python test_telegram_parser.py
```

Expected output:
```
✅ Parsing successful!
✅ All validations passed!
```

## 📝 Message Format Support

The parser supports your exact format:

```
📣GBP/USD📣
Direction: SELL
Entry Price:   1.3493
TP1     1.3478
TP2     1.3458
TP3     1.3426
SL       1.3546
```

**Supported variations:**
- Different currency pairs (EUR/USD, USD/JPY, etc.)
- BUY or SELL directions
- Multiple TP levels (TP1, TP2, TP3, etc.)
- Flexible spacing and formatting

## 🔧 Troubleshooting

### "telethon not installed"
```bash
pip install telethon
```

### "Telegram API credentials not configured"
- Check your `.env` file has `TELEGRAM_API_ID` and `TELEGRAM_API_HASH`
- Get credentials from https://my.telegram.org/apps

### "Table telegram_channels does not exist"
- Run the SQL migration: `sql/create_telegram_channels_table.sql`
- Or create the table manually in Supabase

### "Failed to connect to Telegram"
- Check your internet connection
- Verify API credentials are correct
- Make sure you can access Telegram from your network

### "No signals being saved"
- Check the worker is running
- Verify channel username is correct (include @)
- Check channel is set to "Active" in UI
- Look for error messages in worker terminal

## 📊 Signal Data Structure

Signals are saved to `signal_provider_signals` table with:
- `provider_name`: Name of the signal provider
- `symbol`: Normalized symbol (e.g., C:GBPUSD)
- `action`: "buy" or "sell"
- `entry_price`: Entry price
- `stop_loss`: Stop loss price
- `target_1`, `target_2`, `target_3`: Take profit levels
- `signal_date`: Timestamp in GMT+4

## 🎯 Next Steps

1. **Test with Real Channel**: Add your actual Telegram channel and test
2. **Monitor Performance**: Check that signals are being parsed and saved correctly
3. **Integrate with Trading**: Use signals in your trading engine (Phase 3)
4. **Add Notifications**: Extend the callback function to send alerts

## 📚 Files Created

- `tradingagents/dataflows/telegram_signal_parser.py` - Signal parser
- `tradingagents/dataflows/telegram_signal_service.py` - Telegram client service
- `tradingagents/dataflows/telegram_signal_worker.py` - Background worker
- `sql/create_telegram_channels_table.sql` - Database migration
- `test_telegram_parser.py` - Test script
- `TELEGRAM_SIGNAL_INTEGRATION.md` - Technical documentation
- `TELEGRAM_SETUP_GUIDE.md` - This guide

## 💡 Tips

- Keep the worker running in a separate terminal or as a background service
- Use the UI to manage channels without restarting the worker
- Check the "Live Signal Feed" regularly to verify signals are coming through
- The parser automatically normalizes symbols (GBP/USD → C:GBPUSD)

## 🆘 Support

If you encounter issues:
1. Check the worker terminal for error messages
2. Verify database table exists and is accessible
3. Test the parser with `test_telegram_parser.py`
4. Check Telegram API credentials are correct


