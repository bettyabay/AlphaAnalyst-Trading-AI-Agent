-- Create table for Gold 5-year 1-minute price data
-- This table stores 1-minute OHLCV data for Gold (^XAUUSD) from BarChart exports
-- Note: Table name is quoted to preserve case. Use exact case when referencing in code.

CREATE TABLE IF NOT EXISTS "Gold_price_5yr_1min" (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL DEFAULT '^XAUUSD',
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(18, 6) NOT NULL,
    high NUMERIC(18, 6) NOT NULL,
    low NUMERIC(18, 6) NOT NULL,
    close NUMERIC(18, 6) NOT NULL,
    volume BIGINT DEFAULT 0,
    open_interest BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique constraint to prevent duplicate entries
    UNIQUE(symbol, timestamp)
);

-- Create index on symbol and timestamp for faster queries
CREATE INDEX IF NOT EXISTS idx_gold_price_5yr_1min_symbol_timestamp 
    ON "Gold_price_5yr_1min"(symbol, timestamp DESC);

-- Create index on timestamp for time-based queries
CREATE INDEX IF NOT EXISTS idx_gold_price_5yr_1min_timestamp 
    ON "Gold_price_5yr_1min"(timestamp DESC);

-- Create index on symbol for symbol-based queries
CREATE INDEX IF NOT EXISTS idx_gold_price_5yr_1min_symbol 
    ON "Gold_price_5yr_1min"(symbol);

-- Enable Row Level Security (RLS) if needed
-- ALTER TABLE "Gold_price_5yr_1min" ENABLE ROW LEVEL SECURITY;

-- Add comment to table
COMMENT ON TABLE "Gold_price_5yr_1min" IS 'Stores 5-year historical 1-minute OHLCV data for Gold (^XAUUSD) from BarChart exports';

-- Add comments to columns
COMMENT ON COLUMN "Gold_price_5yr_1min".symbol IS 'Gold symbol, typically ^XAUUSD';
COMMENT ON COLUMN "Gold_price_5yr_1min".timestamp IS '1-minute bar timestamp in UTC';
COMMENT ON COLUMN "Gold_price_5yr_1min".open IS 'Opening price';
COMMENT ON COLUMN "Gold_price_5yr_1min".high IS 'Highest price in the 1-minute period';
COMMENT ON COLUMN "Gold_price_5yr_1min".low IS 'Lowest price in the 1-minute period';
COMMENT ON COLUMN "Gold_price_5yr_1min".close IS 'Closing price';
COMMENT ON COLUMN "Gold_price_5yr_1min".volume IS 'Trading volume';
COMMENT ON COLUMN "Gold_price_5yr_1min".open_interest IS 'Open interest (if available)';

