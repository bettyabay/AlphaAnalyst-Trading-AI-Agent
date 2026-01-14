"""Quick test to verify Telegram channel is in database"""
from tradingagents.database.db_service import get_telegram_channels

channels = get_telegram_channels()
print(f"Found {len(channels)} channel(s) in database:")
for c in channels:
    status = "🟢 Active" if c.get("is_active") else "🔴 Inactive"
    print(f"  {status}: {c.get('channel_username')} ({c.get('provider_name')})")

