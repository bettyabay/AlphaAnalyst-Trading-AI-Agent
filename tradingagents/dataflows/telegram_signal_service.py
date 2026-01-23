"""
Telegram Signal Service
Handles connection to Telegram and monitoring channels for trading signals.
"""
import asyncio
import os
import re
from typing import Optional, Dict, List, Callable
from datetime import datetime
import pytz

try:
    from telethon import TelegramClient, events
    from telethon.errors import SessionPasswordNeededError
    TELETHON_AVAILABLE = True
except ImportError:
    TELETHON_AVAILABLE = False
    print("⚠️ telethon not installed. Install with: pip install telethon")

from tradingagents.database.config import get_supabase
from tradingagents.dataflows.telegram_signal_parser import TelegramSignalParser


class TelegramSignalService:
    """
    Service for monitoring Telegram channels and extracting trading signals.
    """
    
    def __init__(
        self,
        api_id: Optional[str] = None,
        api_hash: Optional[str] = None,
        phone_number: Optional[str] = None,
        session_file: str = "telegram_session.session"
    ):
        """
        Initialize Telegram Signal Service.
        
        Args:
            api_id: Telegram API ID (from https://my.telegram.org/apps)
            api_hash: Telegram API Hash
            phone_number: Phone number for authentication (optional)
            session_file: Path to store Telegram session
        """
        if not TELETHON_AVAILABLE:
            raise ImportError("telethon is required. Install with: pip install telethon")
        
        # Get credentials from environment if not provided
        self.api_id = api_id or os.getenv("TELEGRAM_API_ID")
        self.api_hash = api_hash or os.getenv("TELEGRAM_API_HASH")
        self.phone_number = phone_number or os.getenv("TELEGRAM_PHONE_NUMBER")
        self.session_file = session_file
        
        if not self.api_id or not self.api_hash:
            raise ValueError(
                "Telegram API credentials required. "
                "Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env or pass as arguments."
            )
        
        # Check if session file should be loaded from base64 environment variable
        session_b64 = os.getenv("TELEGRAM_SESSION_B64")
        if session_b64 and not os.path.exists(self.session_file):
            try:
                import base64
                session_data = base64.b64decode(session_b64)
                with open(self.session_file, "wb") as f:
                    f.write(session_data)
                print("✅ Loaded session file from TELEGRAM_SESSION_B64")
            except Exception as e:
                print(f"⚠️ Failed to load session from TELEGRAM_SESSION_B64: {e}")
        
        # Initialize Telegram client
        self.client = TelegramClient(self.session_file, int(self.api_id), self.api_hash)
        self.parser = TelegramSignalParser()
        self.is_connected = False
        self.monitored_channels = {}  # {channel_username: provider_name}
        self.message_handlers = []  # List of callback functions
        
        # Timezone objects for conversions
        self.utc_tz = pytz.timezone('UTC')
        self.gmt4_tz = pytz.timezone('Asia/Dubai')
    
    async def connect(self) -> bool:
        """
        Connect to Telegram.
        
        Returns:
            True if connected successfully
        """
        try:
            # Check if session file exists (already authenticated)
            import os
            if os.path.exists(self.session_file):
                # Try to connect with existing session
                await self.client.connect()
                if await self.client.is_user_authorized():
                    self.is_connected = True
                    print("✅ Connected to Telegram (using existing session)")
                    return True
            
            # No existing session - need phone number for first-time authentication
            phone = self.phone_number
            if not phone:
                print("📱 First-time setup: Phone number required for authentication")
                # Check if we're in an interactive environment
                try:
                    import sys
                    if sys.stdin.isatty():
                        phone = input("Enter your phone number (with country code, e.g., +1234567890): ").strip()
                    else:
                        print("❌ Non-interactive environment detected. Cannot prompt for phone number.")
                        print("💡 Please set TELEGRAM_PHONE_NUMBER environment variable or upload session file.")
                        return False
                except:
                    print("❌ Cannot read input in this environment.")
                    print("💡 Please set TELEGRAM_PHONE_NUMBER environment variable or upload session file.")
                    return False
                
                if not phone:
                    print("❌ Phone number is required for first-time authentication")
                    return False
            
            # Start authentication process
            await self.client.start(phone=phone)
            self.is_connected = True
            print("✅ Connected to Telegram")
            return True
            
        except SessionPasswordNeededError:
            print("⚠️ Two-factor authentication required.")
            # Check for 2FA password in environment
            password = os.getenv("TELEGRAM_2FA_PASSWORD")
            if not password:
                try:
                    import sys
                    if sys.stdin.isatty():
                        password = input("Enter your 2FA password: ")
                    else:
                        print("❌ Non-interactive environment detected. Cannot prompt for 2FA password.")
                        print("💡 Please set TELEGRAM_2FA_PASSWORD environment variable.")
                        return False
                except:
                    print("❌ Cannot read input in this environment.")
                    print("💡 Please set TELEGRAM_2FA_PASSWORD environment variable.")
                    return False
            
            if not password:
                print("❌ 2FA password is required")
                return False
                
            await self.client.sign_in(password=password)
            self.is_connected = True
            print("✅ Connected to Telegram (with 2FA)")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Telegram: {e}")
            print(f"💡 Make sure you have:")
            print(f"   - Valid TELEGRAM_API_ID and TELEGRAM_API_HASH in .env")
            print(f"   - Phone number (TELEGRAM_PHONE_NUMBER in .env or enter when prompted)")
            print(f"   - Internet connection to Telegram servers")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Telegram."""
        if self.client and self.is_connected:
            try:
                await self.client.disconnect()
                self.is_connected = False
                print("Disconnected from Telegram")
            except Exception as e:
                # Handle database lock errors gracefully (common in deployed environments)
                error_str = str(e).lower()
                if "database is locked" in error_str or "locked" in error_str:
                    # This is safe to ignore - the connection is still closed, just state wasn't saved
                    # In deployed environments, this is common due to concurrent access
                    self.is_connected = False
                    # Don't print warning - it's expected in multi-instance deployments
                else:
                    # Log other errors but don't crash
                    print(f"⚠️ Warning during disconnect: {e}")
                    self.is_connected = False
    
    def add_channel(self, channel_username: str, provider_name: str):
        """
        Add a channel to monitor.
        
        Args:
            channel_username: Telegram channel username (e.g., "@signal_provider")
            provider_name: Name of the signal provider
        """
        # Normalize channel username
        if not channel_username.startswith('@'):
            channel_username = f"@{channel_username}"
        
        self.monitored_channels[channel_username] = provider_name
        print(f"📱 Added channel: {channel_username} (Provider: {provider_name})")
    
    def remove_channel(self, channel_username: str):
        """Remove a channel from monitoring."""
        if not channel_username.startswith('@'):
            channel_username = f"@{channel_username}"
        
        if channel_username in self.monitored_channels:
            del self.monitored_channels[channel_username]
            print(f"Removed channel: {channel_username}")
    
    async def save_signal(self, signal_data: Dict, provider_name: str, channel_username: str = None) -> bool:
        """
        Save parsed signal to database.
        
        Args:
            signal_data: Parsed signal data from parser
            provider_name: Name of the signal provider
            channel_username: Telegram channel username (optional, will be included in provider_name)
            
        Returns:
            True if saved successfully
        """
        try:
            supabase = get_supabase()
            if not supabase:
                print("❌ Supabase not configured")
                return False
            
            # Validate signal has all required fields before saving
            if not signal_data.get('symbol'):
                print(f"⚠️ Skipping signal: Missing symbol")
                return False
            
            if signal_data.get('action') not in ['buy', 'sell']:
                print(f"⚠️ Skipping signal: Invalid action: {signal_data.get('action')}")
                return False
            
            if signal_data.get('entry_price') is None:
                print(f"⚠️ Skipping signal: Missing entry_price")
                return False
            
            if signal_data.get('stop_loss') is None:
                print(f"⚠️ Skipping signal: Missing stop_loss")
                return False
            
            # Check for at least one target
            has_target = any(
                signal_data.get(f'target_{i}') is not None 
                for i in range(1, 6)
            )
            if not has_target:
                print(f"⚠️ Skipping signal: Missing at least one target")
                return False
            
            # Build provider_name with channel username if provided
            final_provider_name = provider_name
            if channel_username:
                # Remove @ if present and format as "ProviderName (@channelname)"
                channel_clean = channel_username.lstrip('@')
                final_provider_name = f"{provider_name} (@{channel_clean})"
            
            # Convert signal_date from UTC to GMT+4 if needed
            signal_date = signal_data.get("signal_date")
            if signal_date:
                # If signal_date is a string, parse it
                if isinstance(signal_date, str):
                    try:
                        # Try parsing ISO format
                        if 'T' in signal_date:
                            # Remove timezone info if present and assume UTC
                            signal_date_str = signal_date.split('+')[0].split('Z')[0].split('.')[0]
                            dt_utc = datetime.strptime(signal_date_str, '%Y-%m-%dT%H:%M:%S')
                            dt_utc = self.utc_tz.localize(dt_utc)
                        else:
                            dt_utc = datetime.strptime(signal_date, '%Y-%m-%d %H:%M:%S')
                            dt_utc = self.utc_tz.localize(dt_utc)
                    except:
                        # If parsing fails, use current time in UTC
                        dt_utc = datetime.now(self.utc_tz)
                elif isinstance(signal_date, datetime):
                    # If already a datetime, ensure it's UTC
                    if signal_date.tzinfo is None:
                        dt_utc = self.utc_tz.localize(signal_date)
                    else:
                        dt_utc = signal_date.astimezone(self.utc_tz)
                else:
                    # Fallback to current time in UTC
                    dt_utc = datetime.now(self.utc_tz)
                
                # Convert UTC to GMT+4 (Asia/Dubai)
                dt_gmt4 = dt_utc.astimezone(self.gmt4_tz)
                signal_date_gmt4 = dt_gmt4.isoformat()
            else:
                # If no signal_date provided, use current time in GMT+4
                signal_date_gmt4 = datetime.now(self.gmt4_tz).isoformat()
            
            # Prepare record for database
            record = {
                "provider_name": final_provider_name,
                "symbol": signal_data.get("symbol", "").upper(),
                "signal_date": signal_date_gmt4,
                "action": signal_data.get("action", "").lower(),
                "entry_price": signal_data.get("entry_price"),
                "stop_loss": signal_data.get("stop_loss"),
                "target_1": signal_data.get("target_1"),
                "target_2": signal_data.get("target_2"),
                "target_3": signal_data.get("target_3"),
                "target_4": signal_data.get("target_4"),
                "target_5": signal_data.get("target_5"),
                "timezone_offset": "+04:00",
                "created_at": datetime.now(self.gmt4_tz).isoformat()
            }
            
            # Remove None values
            record = {k: v for k, v in record.items() if v is not None}
            
            # Upsert to database (handles duplicates)
            result = supabase.table("signal_provider_signals").upsert(record).execute()
            
            if result.data:
                print(f"✅ Saved signal: {record['action'].upper()} {record['symbol']} @ {record['entry_price']}")
                return True
            else:
                print(f"⚠️ Failed to save signal: {record}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving signal to database: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def start_monitoring(self, on_new_signal: Optional[Callable] = None):
        """
        Start monitoring all added channels for new messages.
        
        Args:
            on_new_signal: Optional callback function when new signal is detected
                          Function signature: async def callback(signal_data: Dict, provider_name: str)
        """
        if not self.is_connected:
            print("❌ Not connected to Telegram. Call connect() first.")
            return
        
        if not self.monitored_channels:
            print("⚠️ No channels added for monitoring")
            return
        
        print(f"🔍 Starting to monitor {len(self.monitored_channels)} channel(s)...")
        
        @self.client.on(events.NewMessage)
        async def handler(event):
            try:
                # STEP 1: Check if message is from a monitored channel FIRST
                chat = await event.get_chat()
                channel_username = getattr(chat, 'username', None)
                channel_id = getattr(chat, 'id', None)
                
                # Check if this message is from a monitored channel
                provider_name = None
                is_monitored = False
                
                if channel_username:
                    # Normalize channel username (remove @ if present, then add it)
                    channel_username_clean = channel_username.lower().strip().lstrip('@')
                    channel_key = f"@{channel_username_clean}"
                    
                    # Try exact match first
                    provider_name = self.monitored_channels.get(channel_key)
                    
                    # If not found, try case-insensitive match
                    if not provider_name:
                        for monitored_ch, monitored_provider in self.monitored_channels.items():
                            if monitored_ch.lower().lstrip('@') == channel_username_clean:
                                provider_name = monitored_provider
                                channel_key = monitored_ch  # Use the exact key from monitored_channels
                                break
                    
                    if provider_name:
                        is_monitored = True
                    # Silently skip unmonitored channels (no logging needed)
                
                # Skip messages from non-monitored channels immediately
                if not is_monitored:
                    return
                
                # STEP 2: Now we know it's from a monitored channel - get message text
                message_text = event.message.message
                
                # Log ALL messages from monitored channels for debugging
                if message_text:
                    preview = message_text[:100].replace('\n', ' ')
                    print(f"📩 Message from @{channel_username} ({provider_name}): {preview}...")
                else:
                    # Message has no text - check if it has media
                    if event.message.media:
                        print(f"🖼️  Image/media message from @{channel_username} ({provider_name}) - skipping")
                    else:
                        print(f"⚠️  Empty message from @{channel_username} ({provider_name}) - skipping")
                    return
                
                # STEP 3: Apply filtering to skip non-signal messages
                # Skip if message is empty or only contains media (images, videos, etc.)
                if not message_text.strip():
                    if event.message.media:
                        # Skip image-only messages or messages with media but no text
                        return
                    # Skip empty messages
                    return
                
                # Quick check: Skip if message is too short (likely not a signal)
                if len(message_text.strip()) < 20:
                    print(f"   ⏭️  Too short ({len(message_text.strip())} chars) - skipping")
                    return
                
                # Quick check: Skip if message doesn't contain any price-like numbers
                # Signals should have at least one price (decimal or large integer for indices like US30)
                # Allow both decimals (48467.00) and large integers (48467) for indices
                if not re.search(r'\d+\.\d+|\d{4,}', message_text):
                    print(f"   ⏭️  No price numbers found - skipping")
                    return
                
                # Quick check: Skip if message doesn't contain BUY/SELL or currency pair indicators
                # Updated to include indices with numbers (NAS100, US30, etc.)
                # US30 is 4 chars, NAS100 is 6 chars, so we need to support 3-8 char symbols
                has_trading_keywords = (
                    re.search(r'\b(BUY|SELL|buy|sell)\b', message_text, re.IGNORECASE) or
                    re.search(r'[A-Z]{3,4}/[A-Z]{3,4}|[A-Z0-9]{3,8}|[A-Z]{6,7}', message_text) or
                    re.search(r'📣', message_text) or
                    re.search(r'Direction', message_text, re.IGNORECASE) or
                    re.search(r'Entry', message_text, re.IGNORECASE)
                )
                
                if not has_trading_keywords:
                    print(f"   ⏭️  No trading keywords found - skipping")
                    return
                
                # STEP 4: Message passed all filters - try to parse
                print(f"   ✅ Potential signal detected - parsing...")
                
                # Parse the message
                parsed_signal = self.parser.parse(message_text)
                
                if parsed_signal:
                    print(f"   ✅ Signal parsed successfully:")
                    print(f"      {parsed_signal['action'].upper()} {parsed_signal['symbol']} @ {parsed_signal['entry_price']}")
                    if parsed_signal.get('stop_loss'):
                        print(f"      SL: {parsed_signal['stop_loss']}")
                    for i in range(1, 6):
                        if parsed_signal.get(f'target_{i}'):
                            print(f"      TP{i}: {parsed_signal[f'target_{i}']}")
                    
                    # Get message date from Telegram (UTC) and use it as signal_date
                    message_date_utc = event.message.date
                    if message_date_utc:
                        # Ensure it's timezone-aware (Telegram dates are UTC)
                        if message_date_utc.tzinfo is None:
                            message_date_utc = self.utc_tz.localize(message_date_utc)
                        else:
                            message_date_utc = message_date_utc.astimezone(self.utc_tz)
                        # Update signal_date with Telegram message date (will be converted to GMT+4 in save_signal)
                        parsed_signal['signal_date'] = message_date_utc.isoformat()
                    
                    # Save to database (pass channel_username to include in provider_name)
                    await self.save_signal(parsed_signal, provider_name, channel_username)
                    
                    # Call callback if provided
                    if on_new_signal:
                        await on_new_signal(parsed_signal, provider_name)
                else:
                    # Message doesn't contain a signal - log for debugging
                    print(f"   ❌ Could not parse signal from message")
                    print(f"   Full message (first 10 lines):")
                    lines = message_text.split('\n')
                    for i, line in enumerate(lines[:10], 1):
                        print(f"      {i}. {line}")
                    if len(lines) > 10:
                        remaining = len(lines) - 10
                        print(f"      ... ({remaining} more lines)")
                    print("   ---")
                    
            except Exception as e:
                print(f"❌ Error processing message: {e}")
        
        # Keep the client running
        print("✅ Monitoring active. Press Ctrl+C to stop.")
        await self.client.run_until_disconnected()
    
    async def get_channel_messages(
        self,
        channel_username: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get recent messages from a channel (for testing/debugging).
        
        Args:
            channel_username: Channel username
            limit: Number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            if not channel_username.startswith('@'):
                channel_username = f"@{channel_username}"
            
            messages = []
            async for message in self.client.iter_messages(channel_username, limit=limit):
                messages.append({
                    "id": message.id,
                    "text": message.message,
                    "date": message.date.isoformat() if message.date else None
                })
            
            return messages
            
        except Exception as e:
            print(f"❌ Error getting messages: {e}")
            return []
    
    async def fetch_all_channel_messages(
        self,
        channel_username: str,
        limit: Optional[int] = None,
        min_id: Optional[int] = None,
        filter_signals_only: bool = False
    ) -> List[Dict]:
        """
        Fetch all messages from a Telegram channel.
        Useful for exporting all signal formats to Excel.
        
        Args:
            channel_username: Channel username (e.g., "@signal_provider")
            limit: Maximum number of messages to fetch (None = all available from channel creation)
            min_id: Minimum message ID to fetch from (for pagination)
            filter_signals_only: If True, only return messages that contain signals
            
        Returns:
            List of message dictionaries with full text and metadata (and parsed signal if filter_signals_only=True)
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            if not channel_username.startswith('@'):
                channel_username = f"@{channel_username}"
            
            messages = []
            count = 0
            signal_count = 0
            
            # Prepare parameters for iter_messages (handle None values)
            # If limit is None, fetch all messages from channel creation
            iter_params = {}
            if limit is not None:
                iter_params['limit'] = limit
            if min_id is not None:
                iter_params['min_id'] = min_id
            
            async for message in self.client.iter_messages(channel_username, **iter_params):
                # Get message text (handle empty messages)
                message_text = message.message or ""
                
                # Skip empty messages
                if not message_text.strip():
                    continue
                
                # If filtering for signals only, apply the same filters as start_monitoring
                if filter_signals_only:
                    # Apply signal detection filters
                    if len(message_text.strip()) < 20:
                        continue
                    
                    # Check for price-like numbers (allow both decimals and large integers for indices)
                    if not re.search(r'\d+\.\d+|\d{4,}', message_text):
                        continue
                    
                    # Check for trading keywords
                    # Updated to include indices with numbers (NAS100, US30, etc.)
                    has_trading_keywords = (
                        re.search(r'\b(BUY|SELL|buy|sell)\b', message_text, re.IGNORECASE) or
                        re.search(r'[A-Z]{3,4}/[A-Z]{3,4}|[A-Z0-9]{6,8}|[A-Z]{6,7}', message_text) or
                        re.search(r'📣', message_text) or
                        re.search(r'Direction', message_text, re.IGNORECASE) or
                        re.search(r'Entry', message_text, re.IGNORECASE)
                    )
                    
                    if not has_trading_keywords:
                        continue
                    
                    # Try to parse the signal
                    parsed_signal = self.parser.parse(message_text)
                    if not parsed_signal:
                        continue  # Skip if can't parse as signal
                    
                    signal_count += 1
                else:
                    parsed_signal = None
                
                # Get message date in UTC
                message_date = message.date
                if message_date:
                    if message_date.tzinfo is None:
                        message_date = self.utc_tz.localize(message_date)
                    else:
                        message_date = message_date.astimezone(self.utc_tz)
                    date_str = message_date.isoformat()
                else:
                    date_str = None
                    message_date = None
                
                msg_dict = {
                    "message_id": message.id,
                    "text": message_text,
                    "date": date_str,
                    "has_media": message.media is not None,
                    "is_reply": message.is_reply,
                    "views": getattr(message, 'views', None),
                    "forwards": getattr(message, 'forwards', None)
                }
                
                # Add parsed signal if available
                if parsed_signal:
                    msg_dict["parsed_signal"] = parsed_signal
                    # Add signal_date from message date
                    if message_date:
                        parsed_signal['signal_date'] = message_date.isoformat()
                
                messages.append(msg_dict)
                
                count += 1
                if count % 100 == 0:
                    print(f"   Fetched {count} messages... ({signal_count} signals)" if filter_signals_only else f"   Fetched {count} messages...")
            
            print(f"✅ Fetched {len(messages)} total messages from {channel_username}")
            if filter_signals_only:
                print(f"   📊 Found {signal_count} signal messages")
            return messages
            
        except Exception as e:
            print(f"❌ Error fetching messages from {channel_username}: {e}")
            import traceback
            traceback.print_exc()
            return []


