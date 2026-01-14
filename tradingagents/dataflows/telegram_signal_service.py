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
        
        # Initialize Telegram client
        self.client = TelegramClient(self.session_file, int(self.api_id), self.api_hash)
        self.parser = TelegramSignalParser()
        self.is_connected = False
        self.monitored_channels = {}  # {channel_username: provider_name}
        self.message_handlers = []  # List of callback functions
    
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
                phone = input("Enter your phone number (with country code, e.g., +1234567890): ").strip()
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
            password = input("Enter your 2FA password: ")
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
            await self.client.disconnect()
            self.is_connected = False
            print("Disconnected from Telegram")
    
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
    
    async def save_signal(self, signal_data: Dict, provider_name: str) -> bool:
        """
        Save parsed signal to database.
        
        Args:
            signal_data: Parsed signal data from parser
            provider_name: Name of the signal provider
            
        Returns:
            True if saved successfully
        """
        try:
            supabase = get_supabase()
            if not supabase:
                print("❌ Supabase not configured")
                return False
            
            # Prepare record for database
            record = {
                "provider_name": provider_name,
                "symbol": signal_data.get("symbol", "").upper(),
                "signal_date": signal_data.get("signal_date"),
                "action": signal_data.get("action", "").lower(),
                "entry_price": signal_data.get("entry_price"),
                "stop_loss": signal_data.get("stop_loss"),
                "target_1": signal_data.get("target_1"),
                "target_2": signal_data.get("target_2"),
                "target_3": signal_data.get("target_3"),
                "target_4": signal_data.get("target_4"),
                "target_5": signal_data.get("target_5"),
                "timezone_offset": "+04:00",
                "created_at": datetime.now(pytz.timezone('Asia/Dubai')).isoformat()
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
                # Get channel username
                chat = await event.get_chat()
                channel_username = getattr(chat, 'username', None)
                
                if not channel_username:
                    # Try to get channel ID or title
                    channel_id = getattr(chat, 'id', None)
                    # Check if this channel is in our monitored list
                    # We'll match by checking if the message came from a monitored channel
                    found_provider = None
                    for ch_username, provider in self.monitored_channels.items():
                        # Try to match by username or check if we're monitoring this chat
                        if channel_username == ch_username.replace('@', ''):
                            found_provider = provider
                            break
                    
                    if not found_provider:
                        # Try to match by checking message content or channel title
                        chat_title = getattr(chat, 'title', '')
                        # For now, we'll process all messages and check if they contain signals
                        # This is less efficient but works for channels we can't identify by username
                        pass
                
                # Get message text
                message_text = event.message.message
                
                # Skip if message is empty or only contains media (images, videos, etc.)
                if not message_text or not message_text.strip():
                    # Check if message has media attachments (images, videos, etc.)
                    if event.message.media:
                        # Skip image-only messages or messages with media but no text
                        return
                    # Skip empty messages
                    return
                
                # Quick check: Skip if message is too short (likely not a signal)
                if len(message_text.strip()) < 20:
                    return
                
                # Quick check: Skip if message doesn't contain any price-like numbers
                # Signals should have at least one decimal number (price)
                if not re.search(r'\d+\.\d+', message_text):
                    return
                
                # Quick check: Skip if message doesn't contain BUY/SELL or currency pair indicators
                has_trading_keywords = (
                    re.search(r'\b(BUY|SELL|buy|sell)\b', message_text, re.IGNORECASE) or
                    re.search(r'[A-Z]{3,4}/[A-Z]{3,4}|[A-Z]{6,7}', message_text) or
                    re.search(r'📣', message_text) or
                    re.search(r'Direction', message_text, re.IGNORECASE) or
                    re.search(r'Entry', message_text, re.IGNORECASE)
                )
                
                if not has_trading_keywords:
                    # Not a trading signal, skip silently
                    return
                
                # Check if this message is from a monitored channel
                provider_name = None
                if channel_username:
                    channel_key = f"@{channel_username}"
                    provider_name = self.monitored_channels.get(channel_key)
                
                # If we couldn't identify the provider, try to match by content
                # or process if we're monitoring all channels
                if not provider_name:
                    # Try to find provider by checking all monitored channels
                    # For now, we'll use a default or try to extract from message
                    provider_name = "Telegram_Channel"  # Default
                
                # Debug: Log potential signal messages from monitored channels (first 200 chars)
                if provider_name:
                    preview = message_text[:200].replace('\n', ' ')
                    print(f"📩 Potential signal from {channel_username or 'unknown'} ({provider_name}): {preview}...")
                
                # Parse the message
                parsed_signal = self.parser.parse(message_text)
                
                if parsed_signal:
                    print(f"✅ Signal parsed successfully:")
                    print(f"   {parsed_signal['action'].upper()} {parsed_signal['symbol']} @ {parsed_signal['entry_price']}")
                    if parsed_signal.get('stop_loss'):
                        print(f"   SL: {parsed_signal['stop_loss']}")
                    if parsed_signal.get('target_1'):
                        print(f"   TP1: {parsed_signal['target_1']}")
                    
                    # Save to database
                    await self.save_signal(parsed_signal, provider_name)
                    
                    # Call callback if provided
                    if on_new_signal:
                        await on_new_signal(parsed_signal, provider_name)
                else:
                    # Message doesn't contain a signal - log for debugging
                    if provider_name:
                        print(f"⚠️ Could not parse signal from message (format may not match expected pattern)")
                        # Show a preview to help debug
                        lines = message_text.split('\n')[:5]
                        print(f"   Message preview (first 5 lines):")
                        for i, line in enumerate(lines, 1):
                            print(f"   {i}. {line[:80]}")
                    
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


