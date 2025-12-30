-- Create table for Signal Provider Signals
-- This table stores trading signals from external signal providers (e.g., PipXpert)

CREATE TABLE IF NOT EXISTS signal_provider_signals (
    id BIGSERIAL PRIMARY KEY,
    provider_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    signal_date TIMESTAMPTZ NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('buy', 'sell')),
    entry_price NUMERIC(18, 6),
    target_1 NUMERIC(18, 6),
    target_2 NUMERIC(18, 6),
    target_3 NUMERIC(18, 6),
    stop_loss NUMERIC(18, 6),
    sl_hit_datetime TIMESTAMPTZ,
    tp1_hit_datetime TIMESTAMPTZ,
    tp2_hit_datetime TIMESTAMPTZ,
    tp3_hit_datetime TIMESTAMPTZ,
    timezone_offset VARCHAR(10) DEFAULT '+04:00',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique constraint to prevent duplicate entries
    UNIQUE(provider_name, symbol, signal_date, action)
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_signal_provider_signals_provider_symbol 
    ON signal_provider_signals(provider_name, symbol);

CREATE INDEX IF NOT EXISTS idx_signal_provider_signals_signal_date 
    ON signal_provider_signals(signal_date DESC);

CREATE INDEX IF NOT EXISTS idx_signal_provider_signals_provider 
    ON signal_provider_signals(provider_name);

CREATE INDEX IF NOT EXISTS idx_signal_provider_signals_symbol 
    ON signal_provider_signals(symbol);

-- Add comments
COMMENT ON TABLE signal_provider_signals IS 'Stores trading signals from external signal providers';
COMMENT ON COLUMN signal_provider_signals.provider_name IS 'Name of the signal provider (e.g., PipXpert)';
COMMENT ON COLUMN signal_provider_signals.symbol IS 'Trading symbol or currency pair';
COMMENT ON COLUMN signal_provider_signals.signal_date IS 'Date/time when the signal was generated';
COMMENT ON COLUMN signal_provider_signals.action IS 'Signal action: buy or sell';
COMMENT ON COLUMN signal_provider_signals.entry_price IS 'Entry price for the signal';
COMMENT ON COLUMN signal_provider_signals.timezone_offset IS 'Timezone offset (e.g., +04:00 for GMT+4)';

