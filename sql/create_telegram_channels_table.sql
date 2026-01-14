-- Create table for Telegram channel configurations
-- Stores channel usernames and provider names for signal monitoring

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

CREATE INDEX IF NOT EXISTS idx_telegram_channels_username 
    ON telegram_channels(channel_username);

CREATE INDEX IF NOT EXISTS idx_telegram_channels_provider 
    ON telegram_channels(provider_name);

CREATE INDEX IF NOT EXISTS idx_telegram_channels_active 
    ON telegram_channels(is_active);

COMMENT ON TABLE telegram_channels IS 'Configuration for Telegram channels monitored for trading signals.';


