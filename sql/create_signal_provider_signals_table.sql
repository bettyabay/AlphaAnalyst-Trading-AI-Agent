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
    UNIQUE(provider_name, symbol, signal_date, action)
);

CREATE INDEX IF NOT EXISTS idx_signal_provider_signals_symbol ON signal_provider_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signal_provider_signals_provider ON signal_provider_signals(provider_name);
