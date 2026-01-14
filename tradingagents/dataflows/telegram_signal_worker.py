"""
Telegram Signal Worker
Background service for monitoring Telegram channels and extracting signals.
Can be run as a standalone script or integrated into the main application.
"""
import asyncio
import os
import sys
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from tradingagents.dataflows.telegram_signal_service import TelegramSignalService
from tradingagents.database.config import get_supabase


async def load_channels_from_db() -> List[Dict]:
    """
    Load channel configurations from database.
    Returns list of {channel_username, provider_name} dicts.
    """
    supabase = get_supabase()
    if not supabase:
        return []
    
    try:
        # Try to get from telegram_channels table if it exists
        result = supabase.table("telegram_channels").select("*").eq("is_active", True).execute()
        if result.data:
            return [
                {
                    "channel_username": ch["channel_username"],
                    "provider_name": ch["provider_name"]
                }
                for ch in result.data
            ]
    except Exception:
        # Table doesn't exist, use environment variable or return empty
        pass
    
    # Fallback: Load from environment variable
    channels_env = os.getenv("TELEGRAM_CHANNELS", "")
    if channels_env:
        channels = []
        for channel_config in channels_env.split(','):
            parts = channel_config.strip().split(':')
            if len(parts) == 2:
                channels.append({
                    "channel_username": parts[0].strip(),
                    "provider_name": parts[1].strip()
                })
        return channels
    
    return []


async def callback_new_signal(signal_data: Dict, provider_name: str):
    """
    Callback function called when a new signal is detected.
    Can be extended to send notifications, alerts, etc.
    """
    print(f"🔔 New signal callback: {provider_name} - {signal_data['action'].upper()} {signal_data['symbol']}")


async def run_telegram_monitor():
    """
    Main function to run Telegram monitoring service.
    """
    print("=" * 60)
    print("📱 Telegram Signal Monitor - Starting...")
    print("=" * 60)
    
    # Initialize service
    try:
        service = TelegramSignalService()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\n💡 Please set TELEGRAM_API_ID and TELEGRAM_API_HASH in your .env file")
        print("   Get credentials from: https://my.telegram.org/apps")
        return
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Please install telethon: pip install telethon")
        return
    
    # Connect to Telegram
    connected = await service.connect()
    if not connected:
        print("❌ Failed to connect to Telegram")
        return
    
    # Load channels from database or environment
    channels = await load_channels_from_db()
    
    if not channels:
        print("⚠️ No channels configured.")
        print("💡 Add channels via UI or set TELEGRAM_CHANNELS in .env")
        print("   Format: @channel1:Provider1,@channel2:Provider2")
        await service.disconnect()
        return
    
    # Add channels to service
    for channel in channels:
        service.add_channel(
            channel["channel_username"],
            channel["provider_name"]
        )
    
    print(f"\n✅ Monitoring {len(channels)} channel(s):")
    for channel in channels:
        print(f"   - {channel['channel_username']} ({channel['provider_name']})")
    print()
    
    # Start monitoring
    try:
        await service.start_monitoring(on_new_signal=callback_new_signal)
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping monitor...")
        await service.disconnect()
        print("✅ Monitor stopped")
    except Exception as e:
        print(f"\n❌ Error in monitoring: {e}")
        await service.disconnect()


def main():
    """Entry point for standalone script."""
    try:
        asyncio.run(run_telegram_monitor())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()


